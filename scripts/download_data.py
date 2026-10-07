"""
Download the corpus for the RAG demo.

For the 10 chosen companies it fetches:
  1. Every filing FinanceBench asks about, as PDF (10-K, 10-Q, 8-K, earnings releases)
  2. For each 10-K: the EDGAR HTML version + Financial_Report.xlsx (Excel statements)

Result: a mixed-format corpus (PDF + HTML + XLSX) with matching evaluation questions.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

COMPANIES = {  # FinanceBench company name -> stock ticker
    "PepsiCo": "PEP", "Amcor": "AMCR", "Johnson & Johnson": "JNJ", "3M": "MMM",
    "AMD": "AMD", "Best Buy": "BBY", "Boeing": "BA", "American Express": "AXP",
    "JPMorgan": "JPM", "Pfizer": "PFE",
}
PDF_REPO = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"
SEC = "https://www.sec.gov"


def get(session, url):
    for attempt in range(3):
        r = session.get(url, timeout=120)
        if r.status_code == 200:
            return r
        time.sleep(1 + attempt)
    return None


def download_pdfs(session, docs, out):
    for _, row in docs.iterrows():
        target = out / row.ticker / "pdf" / f"{row.doc_name}.pdf"
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        r = get(session, f"{PDF_REPO}/{row.doc_name}.pdf") or get(session, row.doc_link)
        if r is not None and r.content[:4] == b"%PDF":
            target.write_bytes(r.content)
            print(f"  pdf  ok   {row.doc_name}")
        else:
            print(f"  pdf  MISS {row.doc_name}")


def download_edgar_10k(session, ticker, cik, year, out):
    """Fetch 10-K HTML + Financial_Report.xlsx whose report date falls in `year`."""
    subs = get(session, f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    rec = subs["filings"]["recent"]
    for form, acc, doc, rdate in zip(rec["form"], rec["accessionNumber"],
                                     rec["primaryDocument"], rec["reportDate"]):
        if form == "10-K" and rdate.startswith(str(year)):
            base = f"{SEC}/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}"
            d = out / ticker / "edgar" / f"{year}_10K"
            d.mkdir(parents=True, exist_ok=True)
            for name, url in [("10k.html", f"{base}/{doc}"),
                              ("Financial_Report.xlsx", f"{base}/Financial_Report.xlsx")]:
                r = get(session, url)
                time.sleep(0.15)  # SEC allows max 10 requests/second
                if r is not None:
                    (d / name).write_bytes(r.content)
                print(f"  {name:<22} {'ok  ' if r is not None else 'MISS'} {ticker} {year}")
            (d / "meta.json").write_text(json.dumps(
                {"ticker": ticker, "form": "10-K", "report_date": rdate,
                 "accession": acc, "source_url": f"{base}/{doc}"}, indent=2))
            return
    print(f"  edgar MISS {ticker} 10-K {year}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True, help="SEC requires a contact email")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()

    data = Path(args.data)
    raw = data / "raw"
    fb = pd.read_csv(data / "financebench.csv")
    fb = fb[fb.company.isin(COMPANIES)].copy()
    fb["ticker"] = fb.company.map(COMPANIES)
    fb.to_csv(data / "eval_questions.csv", index=False)
    print(f"{len(fb)} evaluation questions saved to {data / 'eval_questions.csv'}")

    session = requests.Session()
    session.headers.update({"User-Agent": f"personal-rag-project {args.email}"})

    docs = fb.drop_duplicates("doc_name")[["ticker", "doc_name", "doc_type", "doc_period", "doc_link"]]
    print(f"\n[1/2] Downloading {len(docs)} PDFs")
    download_pdfs(session, docs, raw)

    print("\n[2/2] Downloading EDGAR HTML + Excel for 10-K filings")
    tickers = get(session, f"{SEC}/files/company_tickers.json").json()
    cik = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in tickers.values()}
    for _, row in docs[docs.doc_type.str.lower() == "10k"].iterrows():
        download_edgar_10k(session, row.ticker, cik[row.ticker], int(row.doc_period), raw)

    print("\nDone.")


if __name__ == "__main__":
    main()