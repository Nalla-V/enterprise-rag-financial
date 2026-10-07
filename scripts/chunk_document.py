"""
Step 4: Turn parsed elements into search chunks + rule-based metadata.

Input : data/processed/elements.parquet, data/eval_questions.csv
Output: data/processed/chunks.parquet  (one row per chunk)

What it does
  1. Structure: detects 10-K/10-Q "PART" / "ITEM" headings with regex
     (needed for HTML, where SEC filings have no real heading tags) and keeps
     part > item > subsection for every element.
  2. Chunking per element type
       text  -> consecutive paragraphs/list items of the same section, ~250 words max,
                never crossing a section boundary
       table -> one chunk per table; long tables split by rows with header repeated
  3. Metadata with rules (no LLM): company, filing, fiscal year/quarter, 10-K item label,
     page range, financial terms mentioned, years mentioned
  4. filing_id = FinanceBench doc_name, so PDF/HTML/XLSX versions of the same filing
     count as ONE document when computing DRM later
  5. Data-quality flags for tables + a quality report at the end

Usage:
    python -u scripts/chunk_documents.py
"""
import re
from pathlib import Path

import pandas as pd

P = Path("data/processed")
MAX_WORDS = 250          # text chunk size (~330 tokens)
MIN_WORDS = 5            # drop fragments shorter than this
MAX_TABLE_ROWS = 40      # longer tables are split, header repeated

COMPANY = {"PEP": "PepsiCo", "AMCR": "Amcor", "JNJ": "Johnson & Johnson", "MMM": "3M",
           "AMD": "AMD", "BBY": "Best Buy", "BA": "Boeing", "AXP": "American Express",
           "JPM": "JPMorgan", "PFE": "Pfizer"}

STRUCT_RE = re.compile(r"^\s*(?:PART\s+(?P<part>IV|I{1,3})\b|ITEM\s+(?P<item>\d{1,2}[A-C]?)\s*[.:\-\u2013\u2014])",
                       re.IGNORECASE)

ITEMS_10K = {"1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments",
             "1C": "Cybersecurity", "2": "Properties", "3": "Legal Proceedings",
             "4": "Mine Safety Disclosures", "5": "Market for Equity", "6": "Selected Financial Data",
             "7": "MD&A", "7A": "Market Risk", "8": "Financial Statements",
             "9": "Changes in Accountants", "9A": "Controls and Procedures", "9B": "Other Information",
             "9C": "Foreign Jurisdiction", "10": "Directors and Governance",
             "11": "Executive Compensation", "12": "Security Ownership",
             "13": "Related Transactions", "14": "Accountant Fees", "15": "Exhibits",
             "16": "10-K Summary", "EO": "Executive Officers", "APX": "Appendix (after Item 16)"}
ITEMS_10Q = {("I", "1"): "Financial Statements", ("I", "2"): "MD&A", ("I", "3"): "Market Risk",
             ("I", "4"): "Controls and Procedures", ("II", "1"): "Legal Proceedings",
             ("II", "1A"): "Risk Factors", ("II", "2"): "Share Repurchases",
             ("II", "3"): "Defaults", ("II", "4"): "Mine Safety", ("II", "5"): "Other Information",
             ("II", "6"): "Exhibits"}

# canonical term -> regex (rule-based metadata, start small, extend from failed queries)
TERMS = {
    "revenue": r"\brevenues?\b|\bnet sales\b|\btotal sales\b",
    "gross_margin": r"\bgross (?:margin|profit)\b",
    "operating_income": r"\boperating (?:income|profit|margin)\b|\bincome from operations\b",
    "net_income": r"\bnet (?:income|earnings|loss)\b",
    "ebitda": r"\bebitda\b",
    "eps": r"\bearnings per share\b|\beps\b",
    "cash_flow": r"\bcash flows? from operating\b|\bfree cash flow\b|\boperating cash flow\b",
    "capex": r"\bcapital expenditures?\b|\bcapex\b|\bpurchases? of property",
    "dividends": r"\bdividends?\b",
    "buybacks": r"\bshare repurchases?\b|\brepurchases? of (?:common )?stock\b|\bbuybacks?\b",
    "debt": r"\blong-term debt\b|\bborrowings\b|\bnotes payable\b|\btotal debt\b",
    "cash": r"\bcash and cash equivalents\b",
    "inventory": r"\binventor(?:y|ies)\b",
    "receivables": r"\baccounts receivable\b|\breceivables\b",
    "goodwill": r"\bgoodwill\b",
    "restructuring": r"\brestructuring\b",
    "segment": r"\bsegments?\b",
    "guidance": r"\boutlook\b|\bguidance\b",
    "tax_rate": r"\beffective tax rate\b",
    "working_capital": r"\bworking capital\b",
    "capital_ratios": r"\bcet1\b|\bcommon equity tier 1\b|\btier 1\b|\bcapital ratio\b",
    "liquidity": r"\bliquidity\b|\blcr\b",
    "returns": r"\breturn on (?:equity|assets|invested capital)\b|\broe\b|\broa\b|\broic\b",
}
TERMS_RE = {k: re.compile(v, re.IGNORECASE) for k, v in TERMS.items()}
YEAR_RE = re.compile(r"\b(?:19[89]\d|20[0-3]\d)\b")
# Content placed AFTER Item 15/16 (appendix): classify it by its headings
FIN_START_RE = re.compile(r"^\s*(?:report of independent registered public accounting firm"
                          r"|index to (?:the )?(?:consolidated )?financial statements"
                          r"|(?:consolidated )?statements? of (?:income|operations|earnings|cash flows)"
                          r"|(?:consolidated )?balance sheets?\b"
                          r"|notes to (?:the )?(?:consolidated )?financial statements)", re.IGNORECASE)
EXHIBIT_RE = re.compile(r"^\s*(?:index to exhibits|exhibit index|exhibits?\b)", re.IGNORECASE)
# Optional Part I section that many 10-Ks place right after Item 4 (Mine Safety)
EXEC_OFFICERS_RE = re.compile(r"^\s*(?:information about our )?executive officers of the (?:registrant|company)"
                              r"|^\s*information about (?:our )?executive officers", re.IGNORECASE)


# ---------------------------------------------------------------- helpers
def item_label(doc_type, part, item):
    if item is None:
        return None
    if doc_type == "10Q":
        return ITEMS_10Q.get((part or "I", item), f"Item {item}")
    return ITEMS_10K.get(item, f"Item {item}")


def table_shape(md):
    lines = [l for l in (md or "").splitlines() if l.strip().startswith("|")]
    if len(lines) < 2:
        return 0, 0, 1.0
    n_cols = lines[0].count("|") - 1
    body = lines[2:]
    cells = [c.strip() for l in body for c in l.strip().strip("|").split("|")]
    empty = sum(1 for c in cells if not c) / max(len(cells), 1)
    return len(body), n_cols, empty


def table_flag(md, fmt="pdf"):
    rows, cols, empty = table_shape(md)
    if rows <= 2 and cols <= 2:
        # in PDFs a tiny grid is the symptom of TableFormer dropping cells;
        # in HTML/Excel the structure comes from the file itself, so it is just a small table
        return "tiny_grid" if fmt == "pdf" else "small_table"
    if empty > 0.6:
        return "sparse"
    return "ok"


def split_table(md):
    lines = md.splitlines()
    header, body = lines[:2], lines[2:]
    if len(body) <= MAX_TABLE_ROWS:
        return [md]
    return ["\n".join(header + body[i:i + MAX_TABLE_ROWS]) for i in range(0, len(body), MAX_TABLE_ROWS)]


FILLER = {"", "$", "%", ")", "(", "—", "-"}


def clean_table_md(md):
    """Drop layout columns/rows that hold only '$', '%', ')' or nothing.
    SEC HTML filings put every '$' sign and closing bracket in its own spacer cell."""
    lines = [l for l in md.splitlines() if l.strip().startswith("|")]
    if len(lines) < 3:
        return md
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for i, l in enumerate(lines) if i != 1]
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    keep = [j for j in range(n) if any(r[j] not in FILLER for r in rows)]
    if not keep:
        return ""
    rows = [[r[j] for j in keep] for r in rows]
    rows = [rows[0]] + [r for r in rows[1:] if any(c not in FILLER for c in r)]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(keep)]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def is_caps_heading(text):
    letters = [c for c in text if c.isalpha()]
    return (len(letters) >= 4 and text.isupper() and len(text.split()) <= 12
            and not text.rstrip().endswith((".", ":", ";", ",")))


def terms_in(text):
    return sorted(k for k, rx in TERMS_RE.items() if rx.search(text))


def years_in(text):
    return sorted(set(YEAR_RE.findall(text)))[:10]


def parse_period(period):
    m = re.match(r"(\d{4})(Q\d)?", str(period))
    return (int(m.group(1)), m.group(2)) if m else (None, None)


# ---------------------------------------------------------------- chunking one document
def chunk_document(doc: pd.DataFrame, filing_id: str):
    d0 = doc.iloc[0]
    doc_type, fmt = d0.doc_type, d0.format
    fiscal_year, quarter = parse_period(d0.period)
    company = COMPANY.get(d0.ticker, d0.ticker)

    chunks = []
    state = {"part": None, "item": "8" if fmt == "xlsx" else None, "sub": ""}
    buf = {"texts": [], "pages": [], "words": 0}

    def base():
        label = item_label(doc_type, state["part"], state["item"])
        header = " | ".join(x for x in [company, f"{doc_type} {d0.period}", label, state["sub"][:120]] if x)
        return {"doc_id": d0.doc_id, "filing_id": filing_id, "ticker": d0.ticker, "company": company,
                "doc_type": doc_type, "period": d0.period, "fiscal_year": fiscal_year,
                "quarter": quarter, "format": fmt, "route": d0.route,
                "part": state["part"], "item": state["item"], "item_label": label,
                "section": state["sub"] or None, "context_header": header}

    def flush():
        if buf["words"] >= MIN_WORDS:
            text = "\n".join(buf["texts"])
            pages = [p for p in buf["pages"] if pd.notna(p)]
            b = base()
            chunks.append({**b, "chunk_type": "text", "chunk_text": text,
                           "page_start": int(min(pages)) if pages else None,
                           "page_end": int(max(pages)) if pages else None,
                           "location": (f"pages {int(min(pages))}-{int(max(pages))}" if pages
                                        else " > ".join(x for x in [b["item_label"], b["section"]] if x)),
                           "table_flag": None, "sheet_csv": None})
        buf.update(texts=[], pages=[], words=0)

    for row in doc.itertuples():
        text = row.text.strip() if isinstance(row.text, str) else ""
        is_short = row.element_type in ("heading", "paragraph") and len(text) < 120
        m = STRUCT_RE.match(text) if is_short else None
        in_appendix = doc_type == "10K" and state["item"] in ("15", "16", "APX")
        no_item_yet = doc_type == "10K" and state["item"] is None

        if is_short and (in_appendix or no_item_yet) and FIN_START_RE.match(text):
            flush()                             # F-pages after Item 15/16 -> Financial Statements
            state.update(item="8", sub=text)
        elif is_short and state["item"] in ("16", "APX") and EXHIBIT_RE.match(text):
            flush()                             # exhibit index placed after Item 16 -> Exhibits
            state.update(item="15", sub=text)
        elif is_short and doc_type == "10K" and state["item"] == "4" and EXEC_OFFICERS_RE.match(text):
            flush()                             # 'Executive officers' block right after Item 4
            state.update(item="EO", sub="")
        elif m:                                 # PART / ITEM boundary
            flush()
            if m.group("part"):
                state.update(part=m.group("part").upper(), item=None, sub="")
            else:
                state.update(item=m.group("item").upper(), sub="")
        elif row.element_type == "heading" or (fmt == "html" and is_short and is_caps_heading(text)):
            # generic sub-heading (HTML has no heading tags: short ALL-CAPS lines are headings)
            flush()
            if state["item"] == "16":           # Item 16 is one line; anything with its own
                state["item"] = "APX"           # heading after it is unclassified appendix
            state["sub"] = text
        elif row.element_type == "table":
            flush()
            if fmt == "xlsx":
                state["sub"] = text
            md = clean_table_md(row.table_md) if isinstance(row.table_md, str) else ""
            if not md.strip():
                continue                        # empty / pure-layout table
            pieces = split_table(md)
            for i, piece in enumerate(pieces):
                body = "\n".join(x for x in [text, piece] if x)
                page = int(row.page) if pd.notna(row.page) else None
                b = base()
                chunks.append({**b, "chunk_type": "table", "chunk_text": body,
                               "page_start": page, "page_end": page,
                               "location": row.location + (f" (part {i + 1}/{len(pieces)})" if len(pieces) > 1 else ""),
                               "table_flag": table_flag(piece, fmt),
                               "sheet_csv": getattr(row, "sheet_csv", None)})
        else:                                   # paragraph / list item
            words = text.split()                # very long paragraphs are cut into MAX_WORDS pieces
            for i in range(0, len(words), MAX_WORDS):
                piece = words[i:i + MAX_WORDS]
                if buf["words"] + len(piece) > MAX_WORDS:
                    flush()
                buf["texts"].append(" ".join(piece))
                buf["pages"].append(row.page)
                buf["words"] += len(piece)
    flush()
    return chunks


# ---------------------------------------------------------------- main
def main():
    el = pd.read_parquet(P / "elements.parquet").sort_values(["doc_id", "order"])
    fb = pd.read_csv("data/eval_questions.csv")
    tenk = fb[fb.doc_type.str.lower() == "10k"]
    filing_of = {(r.ticker, str(r.doc_period)): r.doc_name for r in tenk.itertuples()}

    all_chunks = []
    for doc_id, doc in el.groupby("doc_id", sort=False):
        d0 = doc.iloc[0]
        filing_id = doc_id if d0.format == "pdf" else filing_of.get((d0.ticker, str(d0.period)), doc_id)
        all_chunks += chunk_document(doc, filing_id)

    ch = pd.DataFrame(all_chunks)
    ch = ch[ch.chunk_text.str.strip() != ""].reset_index(drop=True)
    ch["n_words"] = ch.chunk_text.str.split().str.len()
    ch["financial_terms"] = (ch.context_header + " " + ch.chunk_text).map(terms_in)
    ch["years_mentioned"] = ch.chunk_text.map(years_in)
    ch.insert(0, "chunk_id", ch.doc_id + "::" + ch.groupby("doc_id").cumcount().astype(str).str.zfill(5))
    ch.to_parquet(P / "chunks.parquet", index=False)

    # ------------------------------------------------ quality report (data quality gates)
    print(f"{len(ch)} chunks from {ch.doc_id.nunique()} documents / {ch.filing_id.nunique()} filings\n")
    print(ch.groupby(["format", "chunk_type"]).size().to_string(), "\n")
    print("words per chunk (text):", ch[ch.chunk_type == "text"].n_words.describe()[["mean", "50%", "max"]].round(0).to_dict())
    print("table flags:\n", ch[ch.chunk_type == "table"].groupby(["format", "table_flag"]).size().to_string(), "\n")
    filings = ch[ch.doc_type.isin(["10K", "10Q"])]
    no_item = filings.item.isna().mean()
    print(f"10-K/10-Q chunks without an ITEM section: {no_item:.1%}")
    print("chunks per item (10-K):\n", filings[filings.doc_type == "10K"].item_label.value_counts().head(12).to_string())
    empty = (ch.chunk_text.str.strip() == "").sum()
    print(f"\nempty chunks: {empty}")
    ex = ch[(ch.chunk_type == "text") & (ch.item == "7")]
    if len(ex):
        print("\nexample MD&A chunk:\n", ex.iloc[min(3, len(ex) - 1)][["context_header", "location", "financial_terms"]].to_dict())


if __name__ == "__main__":
    main()