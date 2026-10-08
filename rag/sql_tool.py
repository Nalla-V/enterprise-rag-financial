"""Numbers tool: answers questions from the Excel financial-statement facts (fin_facts.parquet).

Two ways in, safest first:
  1. lookup  (function calling): the LLM only fills a small JSON "tool call"
                {"concepts": [...], "years": [...]}
     which is validated against the allowed concept list and run as a FIXED, parameterised query.
     The LLM never writes SQL -> no wrong joins, no injection, no hallucinated column names.
  2. free SQL (fallback): for metrics outside the concept dictionary, the LLM writes one DuckDB
     SELECT over the table. Guard rails: SELECT only, LIMIT, one retry with the DB error.

Rules learned in issues 27-28 are built in: latest filing only (no duplicates), positive outflows
are the LLM's job (values keep their sign + a 'flow' hint), standard concept names instead of
company-specific line items.
"""
import json
import re

import duckdb
import pandas as pd

from . import config, llm

CONCEPTS = {
    "revenue": "total revenue / net sales / top line", "cost_of_revenue": "cost of sales / COGS / cost of goods sold",
    "gross_profit": "gross profit", "operating_income": "operating income / operating profit / EBIT-like",
    "pretax_income": "income before income taxes", "income_tax_expense": "income tax expense (provision for taxes)",
    "net_income": "net income attributable to the company", "eps_diluted": "diluted EPS (per share)",
    "total_assets": "total assets", "total_liabilities": "total liabilities",
    "total_current_assets": "total current assets", "total_current_liabilities": "total current liabilities",
    "total_equity": "total shareholders' equity", "inventory": "inventories",
    "accounts_receivable": "accounts receivable, net (AR, receivables)", "accounts_payable": "accounts payable (AP)",
    "cash": "cash and cash equivalents", "short_term_investments": "short-term investments / marketable securities",
    "ppe_net": "net property, plant and equipment (PP&E, PPNE, fixed assets)", "long_term_debt": "long-term debt",
    "operating_cash_flow": "net cash from operating activities", "investing_cash_flow": "net cash from investing activities",
    "financing_cash_flow": "net cash from financing activities", "capex": "capital expenditure (cash outflow)",
    "depreciation_amortization": "depreciation and amortization (D&A)", "dividends_paid": "dividends paid (outflow)",
    "share_repurchases": "share repurchases / buybacks (outflow)",
}

PLAN_SYS = ("You plan a lookup in a table of financial-statement facts. Pick the concepts needed to answer "
            "the question (including the ingredients of any ratio, e.g. a margin needs the item AND revenue; "
            "an average or turnover needs the prior year too) and the fiscal years.\n"
            "Allowed concepts:\n" + "\n".join(f"- {k}: {v}" for k, v in CONCEPTS.items()) +
            "\nRecipes: X margin = X / revenue; quick ratio = (cash + short_term_investments + accounts_receivable)"
            " / total_current_liabilities; current ratio = total_current_assets / total_current_liabilities;"
            " effective tax rate = income_tax_expense / pretax_income; inventory turnover = cost_of_revenue /"
            " average inventory (needs prior year); EBITDA = operating_income + depreciation_amortization;"
            " free cash flow = operating_cash_flow - capex; capital intensity = capex / revenue and ppe_net / total_assets."
            '\nReply only as JSON: {"concepts": ["..."], "years": [2022, 2021]}. '
            'If no allowed concept fits, reply {"concepts": [], "years": []}.')

# The free-SQL fallback only ever sees a VIEW that is already filtered to the company in the question
# (issue 36: the LLM guessed tickers like 'AMC' for Amcor and 'BBBY' for Best Buy).
SCHEMA = """Table facts - numbers from ONE company's 10-K financial statements (already filtered to that company):
  period_fiscal_year INT   -- fiscal year the number belongs to (filter on this for the year asked)
  statement TEXT           -- full statement title, e.g. 'Consolidated Statements of Income'
  statement_type TEXT      -- exactly one of: 'income', 'balance', 'cash_flow', 'note'
  line_item TEXT           -- the company's own label, e.g. 'Restructuring charges'
  value_musd DOUBLE        -- USD millions; cash outflows are negative
  value_kind TEXT          -- 'money', 'per_share' or 'shares'"""

SQL_SYS = ("Write ONE DuckDB SELECT over the table below to fetch the numbers needed for the question. Rules: "
           "SELECT period_fiscal_year, statement, line_item, value_musd, value_kind FROM facts; "
           "filter period_fiscal_year for the year(s) asked; match line_item with ILIKE '%keyword%' using "
           "short keywords; put OR conditions in brackets; do not filter on ticker or company; LIMIT 30. "
           "Reply with the SQL only, no explanation.\n\n" + SCHEMA)


class FinFactsSQL:
    def __init__(self, facts: pd.DataFrame | None = None):
        self.facts = facts if facts is not None else pd.read_parquet(config.PROC / "fin_facts.parquet")
        self.con = duckdb.connect()
        self.con.register("fin_facts", self.facts)
        self.coverage = set(map(tuple, self.facts[["ticker", "period_fiscal_year"]].dropna()
                                .drop_duplicates().astype({"period_fiscal_year": int}).values.tolist()))

    def covers(self, tickers, years):
        """Is there Excel data for this company (and year)? The router uses this."""
        if not tickers:
            return False
        if not years:
            return any(t == tk for tk, _ in self.coverage for t in tickers)
        return any((t, y) in self.coverage for t in tickers for y in years)

    # ------------------------------------------------------------ 1. structured lookup
    def plan(self, question, meta):
        p = llm.ask_json(PLAN_SYS, question, default={}, max_tokens=80) or {}
        concepts = [c for c in p.get("concepts", []) if c in CONCEPTS]          # validate: allowed only
        years = sorted({int(y) for y in p.get("years", []) if str(y).isdigit()} | set(meta["years"]))
        return {"concepts": concepts, "years": years, "raw_plan": p}

    def latest_years(self, tickers, n=3):
        """No year in the question -> the latest n fiscal years we have (issue 35: default to most recent)."""
        ys = sorted({y for t, y in self.coverage if t in tickers})
        return ys[-n:]

    def lookup(self, tickers, concepts, years):
        if not (tickers and concepts):
            return pd.DataFrame()
        years = years or self.latest_years(tickers)
        # one row per company / concept / year: the copy from the most recent filing wins
        # (a 2019 10-K restates 2017 under a different label than the 2017 10-K - both are concept net_income)
        sql = f"""
            SELECT * EXCLUDE (rn) FROM (
                SELECT ticker, period_fiscal_year, concept, line_item, value_musd, value, value_kind,
                       statement, filing_id, sheet_chunk_id,
                       ROW_NUMBER() OVER (PARTITION BY ticker, concept, period_fiscal_year
                                          ORDER BY fiscal_year DESC) AS rn
                FROM fin_facts
                WHERE ticker IN ({",".join("?" * len(tickers))})
                  AND concept IN ({",".join("?" * len(concepts))})
                  AND period_fiscal_year IN ({",".join("?" * len(years))})
                  AND (period_months = 12 OR period_months IS NULL)
                  AND from_latest_filing)
            WHERE rn = 1
            ORDER BY ticker, concept, period_fiscal_year"""
        return self.con.execute(sql, [*tickers, *concepts, *years]).df().rename(columns={"period_fiscal_year": "fiscal_year"})

    # ------------------------------------------------------------ 2. free SQL fallback
    def run_sql(self, sql, tickers):
        m = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.DOTALL | re.IGNORECASE)    # code fence -> inner SQL
        sql = (m.group(1) if m else sql).strip().rstrip(";")
        if not re.match(r"(?is)^\s*(select|with)\b", sql) or ";" in sql or \
                re.search(r"(?i)\b(insert|update|delete|drop|create|alter|copy|attach|pragma|fin_facts)\b", sql):
            raise ValueError("only one read-only SELECT on the table 'facts' is allowed")
        # the view the LLM can see: this company only, latest copy of each fact
        ticker_list = ",".join(f"'{t}'" for t in tickers if t in config.COMPANY)       # whitelisted tickers only
        self.con.execute(f"CREATE OR REPLACE TEMP VIEW facts AS SELECT * FROM fin_facts "
                         f"WHERE ticker IN ({ticker_list}) AND from_latest_filing")
        if not re.search(r"(?i)\blimit\s+\d+", sql):
            sql += " LIMIT 30"
        return self.con.execute(sql).df()

    def free_sql(self, question, tickers, retries=1):
        sql, err = "", None
        for _ in range(retries + 1):
            user = f"Question: {question}\nSQL:" if err is None else \
                   f"Question: {question}\nYour SQL:\n{sql}\nfailed: {err}\nFix it. SQL:"
            sql = llm.ask(SQL_SYS, user, max_tokens=300)
            try:
                rows = self.run_sql(sql, tickers)
                if len(rows):
                    return sql, rows
                err = "no rows - loosen the ILIKE patterns"
            except Exception as e:  # noqa: BLE001 - give the DB error back to the model
                err = str(e)[:300]
        return sql, pd.DataFrame()

    # ------------------------------------------------------------ entry point
    def answer_rows(self, question, meta, allow_free_sql=None):
        """-> dict(mode, plan, sql, rows). Structured lookup first; free SQL only for NUMERIC questions
        that the lookup could not serve (issue 36: free SQL on qualitative questions returned junk rows)."""
        p = self.plan(question, meta)
        # ingredients the calculator needs (issue 37: 3M has no gross-profit line -> revenue - cost of sales)
        need = set(p["concepts"])
        if need & {"gross_profit", "operating_income", "net_income", "capex", "operating_cash_flow",
                   "depreciation_amortization", "cost_of_revenue"}:
            need.add("revenue")
        if "gross_profit" in need:
            need.add("cost_of_revenue")
        if "inventory" in need:
            need.add("cost_of_revenue")
        p["concepts"] = [c for c in CONCEPTS if c in need]
        rows = self.lookup(meta["tickers"], p["concepts"], p["years"])
        if len(rows):
            return {"mode": "lookup", "plan": p, "sql": None, "rows": rows}
        if not (meta["numeric"] if allow_free_sql is None else allow_free_sql) or not meta["tickers"]:
            return {"mode": "none", "plan": p, "sql": None, "rows": pd.DataFrame()}
        sql, rows = self.free_sql(question, meta["tickers"])
        return {"mode": "free_sql", "plan": p, "sql": sql, "rows": rows}


def _fmt(v, pct=False):
    return f"{v:,.2%}" if pct else f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"


def calculations(rows: pd.DataFrame):
    """Deterministic calculator: the standard ratios from the [SQL] rows, so the LLM does not do arithmetic
    (issue 37: Qwen-7B computed 9,912 / 46,298 as 21.7% instead of 21.41%)."""
    if rows is None or rows.empty:
        return []
    money = rows[rows.value_kind == "money"]
    v = {(r.ticker, int(r.fiscal_year), r.concept): r.value_musd for r in money.itertuples()}
    out = []
    for t, y in sorted({(t, y) for t, y, _ in v}):
        g = lambda c: v.get((t, y, c))                                          # noqa: E731
        name = config.COMPANY.get(t, t)
        rev, cogs, gp = g("revenue"), g("cost_of_revenue"), g("gross_profit")
        if gp is None and rev is not None and cogs is not None:
            gp = rev - abs(cogs)
            out.append(f"{name} FY{y} gross profit (derived) = revenue {_fmt(rev)} - cost of revenue {_fmt(abs(cogs))} = {_fmt(gp)}")
        if rev:
            for label, val in [("gross margin", gp), ("operating margin", g("operating_income")),
                               ("net margin", g("net_income")), ("pretax margin", g("pretax_income")),
                               ("operating cash flow / revenue", g("operating_cash_flow")),
                               ("capex / revenue", abs(g("capex")) if g("capex") is not None else None)]:
                if val is not None:
                    out.append(f"{name} FY{y} {label} = {_fmt(val)} / {_fmt(rev)} = {_fmt(val / rev, True)}")
        oi, da, capex, ocf = g("operating_income"), g("depreciation_amortization"), g("capex"), g("operating_cash_flow")
        if oi is not None and da is not None:
            ebitda = oi + da
            out.append(f"{name} FY{y} EBITDA = operating income {_fmt(oi)} + D&A {_fmt(da)} = {_fmt(ebitda)}")
            if rev:
                out.append(f"{name} FY{y} EBITDA margin = {_fmt(ebitda)} / {_fmt(rev)} = {_fmt(ebitda / rev, True)}")
            if capex is not None:
                out.append(f"{name} FY{y} EBITDA - capex = {_fmt(ebitda)} - {_fmt(abs(capex))} = {_fmt(ebitda - abs(capex))}")
        if ocf is not None and capex is not None:
            out.append(f"{name} FY{y} free cash flow = {_fmt(ocf)} - {_fmt(abs(capex))} = {_fmt(ocf - abs(capex))}")
        tax, pti = g("income_tax_expense"), g("pretax_income")
        if tax is not None and pti:
            out.append(f"{name} FY{y} effective tax rate = tax {_fmt(tax)} / pretax income {_fmt(pti)} = "
                       f"{_fmt(tax / pti, True)} (signs as reported)")
        ca, cl = g("total_current_assets"), g("total_current_liabilities")
        if ca is not None and cl:
            out.append(f"{name} FY{y} current ratio = {_fmt(ca)} / {_fmt(cl)} = {ca / cl:.2f}")
        cash, sti, ar = g("cash"), g("short_term_investments"), g("accounts_receivable")
        if cl and cash is not None and ar is not None:
            q = cash + (sti or 0) + ar
            out.append(f"{name} FY{y} quick ratio = (cash {_fmt(cash)} + short-term investments {_fmt(sti or 0)}"
                       f" + receivables {_fmt(ar)}) / {_fmt(cl)} = {q / cl:.2f}")
        ppe, ta = g("ppe_net"), g("total_assets")
        if ppe is not None and ta:
            out.append(f"{name} FY{y} net PP&E / total assets = {_fmt(ppe)} / {_fmt(ta)} = {_fmt(ppe / ta, True)}")
        inv, inv_prev = g("inventory"), v.get((t, y - 1, "inventory"))
        if cogs is not None and inv:
            if inv_prev:
                avg = (inv + inv_prev) / 2
                out.append(f"{name} FY{y} inventory turnover = cost of revenue {_fmt(abs(cogs))} / average inventory "
                           f"(({_fmt(inv)} + {_fmt(inv_prev)}) / 2 = {_fmt(avg)}) = {abs(cogs) / avg:.2f}x")
            else:
                out.append(f"{name} FY{y} inventory turnover = {_fmt(abs(cogs))} / ending inventory {_fmt(inv)} = {abs(cogs) / inv:.2f}x")
        for c in ["revenue", "gross_profit", "operating_income", "net_income"]:      # year-over-year growth
            now, prev = g(c), v.get((t, y - 1, c))
            if now is not None and prev:
                out.append(f"{name} FY{y} {c.replace('_', ' ')} growth vs FY{y - 1} = {_fmt(now)} / {_fmt(prev)} - 1 = "
                           f"{_fmt(now / prev - 1, True)}")
    return out


def rows_as_text(rows: pd.DataFrame, n=30):
    """Compact table for the LLM prompt: one 'value' column in the right unit, plus a hint about signs."""
    if rows is None or rows.empty:
        return "(no rows)"
    r = rows.copy()
    if "value" in r:   # money -> USD millions; per-share / share counts -> raw value
        r["value"] = r.value_musd.where(r.value_kind == "money", r.value)
    else:
        r["value"] = r.value_musd
    r["unit"] = r.value_kind.map({"money": "USD m", "per_share": "USD/share", "shares": "shares"}).fillna("")
    yc = "fiscal_year" if "fiscal_year" in r else "period_fiscal_year"
    cols = [c for c in ["ticker", yc, "concept", "line_item", "value", "unit"] if c in r]
    note = ("Cash outflows (capex, dividends, repurchases) are negative in the cash flow statement - "
            "report them as positive amounts.")
    text = note + "\n" + r[cols].head(n).to_markdown(index=False, floatfmt=",.2f")
    calc = calculations(rows)
    if calc:
        text += "\nCalculated from these rows (use these results, do not redo the arithmetic):\n- " + "\n- ".join(calc)
    return text


def plan_json(p):
    return json.dumps({k: v for k, v in p.items() if k != "raw_plan"})