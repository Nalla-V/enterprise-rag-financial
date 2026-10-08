"""
Check the Excel facts table against FinanceBench BEFORE building the SQL tool.

For every FinanceBench question about a 10-K that mentions a metric we have a concept for
(capex, revenue, net income, ...), look up that concept with plain SQL (DuckDB) for the
company + fiscal year(s) in the question, and print it next to the gold answer.

This is a manual, human-in-the-loop check: it shows whether the numbers an SQL tool would get
are the numbers the questions need. No LLM involved.

Usage:  python scripts/check_facts.py
"""
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import config  # noqa: E402

# question keywords -> concepts to look up
KEYWORDS = [
    (r"capital expenditure|capex", ["capex"]),
    (r"revenue|net sales|total sales|top line", ["revenue"]),
    (r"net income|net earnings|net profit", ["net_income"]),
    (r"cash (?:flow )?from operations|operating cash flow|cash flow from operating|operating activities", ["operating_cash_flow"]),
    (r"free cash flow", ["operating_cash_flow", "capex"]),
    (r"total assets|return on assets|\broa\b|asset turnover", ["total_assets"]),
    (r"dividend", ["dividends_paid"]),
    (r"depreciation|d&a|ebitda", ["depreciation_amortization"]),
    (r"cost of goods|cogs|cost of sales|cost of revenue", ["cost_of_revenue"]),
    (r"gross margin|gross profit", ["gross_profit", "revenue"]),
    (r"margin|% of (?:revenue|sales)|as a percent", ["revenue"]),
    (r"operating income|operating margin|operating profit|ebit\b|ebitda", ["operating_income"]),
    (r"inventor", ["inventory"]),
    (r"accounts payable|days payable|dpo", ["accounts_payable"]),
    (r"receivable|days sales outstanding|dso", ["accounts_receivable"]),
    (r"working capital|current ratio|quick ratio", ["total_current_assets", "total_current_liabilities"]),
    (r"quick ratio", ["inventory", "cash", "accounts_receivable"]),
    (r"inventory turnover|days inventory", ["cost_of_revenue"]),
    (r"repurchase|buyback", ["share_repurchases"]),
    (r"\bdebt\b|leverage", ["long_term_debt"]),
    (r"\beps\b|earnings per share", ["eps_diluted"]),
    (r"equity|return on equity|\broe\b", ["total_equity"]),
]
YEAR_RE = re.compile(r"(?:FY\s?|fiscal (?:year )?)?(20[0-3]\d)", re.IGNORECASE)


def main():
    q = pd.read_csv(config.DATA / "eval_questions.csv")
    q = q[q.doc_type.str.lower() == "10k"]
    con = duckdb.connect()
    con.execute(f"CREATE VIEW f AS SELECT * FROM '{config.PROC / 'fin_facts.parquet'}'")
    filings = set(con.execute("SELECT DISTINCT filing_id FROM f").df().filing_id)

    checked, no_data = 0, 0
    for r in q.itertuples():
        concepts = []
        for rx, cs in KEYWORDS:
            if re.search(rx, r.question, re.IGNORECASE):
                concepts += [c for c in cs if c not in concepts]
        if not concepts:
            continue
        found = sorted({int(y) for y in YEAR_RE.findall(r.question)}) or [int(r.doc_period)]
        # 'FY2015 - FY2017 3 year average' -> every year in between; 'inventory turnover' -> prior year too
        years = list(range(found[0], found[-1] + 1))
        if re.search(r"turnover|average|change|improv|declin|consistent", r.question, re.IGNORECASE) and len(years) == 1:
            years = [years[0] - 1, years[0]]
        has_filing = r.doc_name in filings
        rows = con.execute(f"""
            SELECT concept, period_fiscal_year AS fy, line_item, value_musd, value AS raw_value, value_kind
            FROM f
            WHERE ticker = ? AND concept IN ({','.join('?' * len(concepts))})
              AND period_fiscal_year IN ({','.join('?' * len(years))})
              AND (period_months = 12 OR period_months IS NULL)
              AND {'filing_id = ?' if has_filing else 'from_latest_filing'}
            ORDER BY concept, fy
        """, [r.ticker, *concepts, *years, *([r.doc_name] if has_filing else [])]).df()

        checked += 1
        print("=" * 110)
        print(f"[{r.financebench_id}] {r.doc_name}   (concepts: {', '.join(concepts)}; years: {years})")
        print(f"Q   : {r.question[:220]}")
        print(f"GOLD: {str(r.answer)[:220]}")
        if rows.empty:
            no_data += 1
            print("SQL : -- no rows" + ("" if has_filing else "  (no Excel data for this filing)"))
        else:
            rows["value"] = rows.value_musd.where(rows.value_kind == "money", rows.raw_value)
            for x in rows.itertuples():
                unit = "USD m" if x.value_kind == "money" else x.value_kind
                print(f"SQL : {x.concept:<26} FY{x.fy}  {x.value:>12,.2f} {unit:<9}  ({x.line_item[:55]})")

    print("=" * 110)
    print(f"{checked} questions checked, {checked - no_data} with SQL rows, {no_data} without")


if __name__ == "__main__":
    main()