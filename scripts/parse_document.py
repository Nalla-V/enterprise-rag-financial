"""
Parse all raw files (PDF, HTML, XLSX) into ONE common element table.

Every output row is one element (heading / paragraph / list item / table) with the
same columns, no matter which file format it came from:

    doc_id, ticker, doc_type, period, source_file, format, route,
    element_type, section, text, table_md, page, location, order

Also:
  * quality routing for PDFs: measures the text layer first, then
      clean   -> Docling without OCR (fast)
      scanned -> Docling with OCR
  * every Excel sheet is also saved as CSV (used later for the text-to-SQL route)
  * per-document caching: if the job times out, rerunning skips finished docs

Output:
    data/processed/elements/<doc_id>.parquet   (one per document)
    data/processed/elements.parquet            (all combined)
    data/processed/documents.parquet           (one row per document + quality stats)
    data/processed/tables/<doc_id>/<sheet>.csv (Excel sheets)

"""
import json
import re
import time
from pathlib import Path

import pandas as pd
import pypdfium2 as pdfium

RAW = Path("data/raw")
OUT = Path("data/processed")
MIN_CHARS_PER_PAGE = 200   # below this a PDF is treated as scanned -> OCR route


# ---------------------------------------------------------------- helpers
def doc_info(path: Path):
    """Derive document-level metadata from the folder layout."""
    ticker = path.parts[2]                      # data/raw/<TICKER>/...
    if path.suffix == ".pdf":                   # e.g. BOEING_2022_10K
        name = path.stem
        m = re.match(r".+?_(\d{4}(?:Q\d)?)_(.+)$", name)
        period, doc_type = (m.group(1), m.group(2)) if m else ("", "")
    else:                                       # edgar/2022_10K/10k.html
        name = f"{ticker}_{path.parent.name}_{path.suffix[1:]}"
        period, doc_type = path.parent.name.split("_")
        meta = path.parent / "meta.json"
        if meta.exists():
            doc_type = json.loads(meta.read_text())["form"].replace("-", "")
    return {"doc_id": name, "ticker": ticker, "period": period,
            "doc_type": doc_type.split("_")[0].upper(), "source_file": str(path)}


def pdf_quality(path: Path):
    """Look at the PDF text layer before parsing -> decides the route."""
    pdf = pdfium.PdfDocument(str(path))
    n_pages = len(pdf)
    chars = sum(len(pdf[i].get_textpage().get_text_range().strip()) for i in range(n_pages))
    cpp = chars / max(n_pages, 1)
    return {"n_pages": n_pages, "chars_per_page": round(cpp, 1),
            "route": "clean" if cpp >= MIN_CHARS_PER_PAGE else "scanned"}


# ---------------------------------------------------------------- Docling (PDF + HTML)
_converters = {}


def get_converter(ocr: bool):
    if ocr not in _converters:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions(do_ocr=ocr, do_table_structure=True)
        _converters[ocr] = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return _converters[ocr]


def parse_with_docling(path: Path, ocr: bool):
    from docling_core.types.doc import ListItem, SectionHeaderItem, TableItem, TextItem, TitleItem

    doc = get_converter(ocr).convert(str(path)).document
    rows, section = [], ""
    for item, _level in doc.iterate_items():
        page = item.prov[0].page_no if getattr(item, "prov", None) else None
        if isinstance(item, (SectionHeaderItem, TitleItem)):
            section = item.text.strip()
            rows.append({"element_type": "heading", "text": section, "table_md": None, "page": page})
        elif isinstance(item, TableItem):
            md = item.export_to_markdown(doc=doc)
            caption = item.caption_text(doc) if hasattr(item, "caption_text") else ""
            rows.append({"element_type": "table", "text": caption or "", "table_md": md, "page": page})
        elif isinstance(item, (ListItem, TextItem)):
            text = item.text.strip()
            if len(text) < 3:
                continue
            etype = "list_item" if isinstance(item, ListItem) else "paragraph"
            rows.append({"element_type": etype, "text": text, "table_md": None, "page": page})
        else:
            continue
        rows[-1]["section"] = section
        rows[-1]["location"] = f"page {page}" if page else f"section '{section[:60]}'"
    return rows


# ---------------------------------------------------------------- Excel
def parse_excel(path: Path, doc_id: str):
    tables_dir = OUT / "tables" / doc_id
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (sheet, df) in enumerate(pd.read_excel(path, sheet_name=None, header=None).items()):
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue
        title = str(df.iat[0, 0]).strip()          # full statement title is in cell A1
        df.to_csv(tables_dir / f"{i:03d}.csv", index=False, header=False)
        md = df.head(60).fillna("").to_markdown(index=False, headers=[""] * df.shape[1])
        rows.append({"element_type": "table", "text": title, "table_md": md,
                     "section": title, "page": None,
                     "location": f"sheet '{sheet}' ({len(df)} rows)",
                     "sheet_csv": str(tables_dir / f"{i:03d}.csv")})
    return rows


# ---------------------------------------------------------------- main
def main():
    (OUT / "elements").mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in RAW.rglob("*") if p.suffix in {".pdf", ".html", ".xlsx"})
    print(f"{len(files)} files to parse")

    doc_rows = []
    for path in files:
        info = doc_info(path)
        cache = OUT / "elements" / f"{info['doc_id']}.parquet"
        fmt = path.suffix[1:]
        quality = pdf_quality(path) if fmt == "pdf" else {"n_pages": None, "chars_per_page": None,
                                                          "route": "structured"}
        t0 = time.time()
        if cache.exists():
            elements = pd.read_parquet(cache)
            status = "cached"
        else:
            try:
                if fmt == "xlsx":
                    rows = parse_excel(path, info["doc_id"])
                else:
                    rows = parse_with_docling(path, ocr=(quality["route"] == "scanned"))
            except Exception as e:  # keep going; one broken file shouldn't kill the job
                print(f"  ERROR {info['doc_id']}: {e}")
                continue
            elements = pd.DataFrame(rows)
            for k, v in info.items():
                elements[k] = v
            elements["format"] = fmt
            elements["route"] = quality["route"]
            elements["order"] = range(len(elements))
            elements.to_parquet(cache, index=False)
            status = "parsed"

        n_tables = int((elements.element_type == "table").sum()) if len(elements) else 0
        doc_rows.append({**info, "format": fmt, **quality,
                         "n_elements": len(elements), "n_tables": n_tables})
        print(f"  {status:<6} {fmt:<4} {quality['route']:<10} {info['doc_id']:<45} "
              f"{len(elements):>5} elements {n_tables:>4} tables  {time.time() - t0:5.0f}s")

    pd.DataFrame(doc_rows).to_parquet(OUT / "documents.parquet", index=False)
    all_el = pd.concat([pd.read_parquet(p) for p in (OUT / "elements").glob("*.parquet")],
                       ignore_index=True)
    all_el.to_parquet(OUT / "elements.parquet", index=False)

    print(f"\nTotal: {len(all_el)} elements from {len(doc_rows)} documents")
    print(all_el.groupby(["format", "element_type"]).size().to_string())


if __name__ == "__main__":
    main()