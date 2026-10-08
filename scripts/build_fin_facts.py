"""
Step 5b: Turn the Excel financial statements into ONE clean SQL-ready table.

Each Financial_Report.xlsx sheet looks like:
    row 0:  <statement title - USD ($) $ in Millions> | 12 Months Ended |      |
    row 1:                                            | Dec. 31, 2022   | Dec. 31, 2021 |
    row 2+: <line item>                               | 66,608          | 62,286 |

We melt every sheet into long format:
    ticker | company | fiscal_year | statement | units | line_item | period_label | value

That single table is what the text-to-SQL tool queries (DuckDB on ALICE, Delta on Databricks).

Input : data/processed/chunks.parquet (to find sheets + their filing), data/processed/tables/*/*.csv
Output: data/processed/fin_facts.parquet
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import config  # noqa: E402

NUM_RE = re.compile(r"^\(?-?\$?\s*\(?[\d,]*\.?\d+\)?%?$")


def to_number(x):
    if not isinstance(x, str):
        return float(x) if pd.notna(x) else None
    s = x.strip().replace("$", "").replace(" ", "")
    if not s or not NUM_RE.match(x.strip().replace(" ", "")):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace(",", "").rstrip("%")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


UNITS_RE = re.compile(r"\b(?:USD|EUR|GBP|shares|pure|\$|in (?:Millions|Thousands|Billions|Units))\b", re.IGNORECASE)
DATE_RE = re.compile(r"([A-Z][a-z]{2})\.? (\d{1,2}), (\d{4})")
MONTHS_RE = re.compile(r"(\d{1,2}) Months Ended", re.IGNORECASE)
MON = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
PER_SHARE_RE = re.compile(r"per (?:common |diluted |basic |ordinary )?share|\beps\b|dividends? declared per", re.IGNORECASE)
# a bare "Basic" / "Diluted" row under an EPS heading, in a sheet whose unit is "$ / shares"
BARE_EPS_RE = re.compile(r"^(?:basic|diluted)(?: \(.*\))?$", re.IGNORECASE)
SHARE_COUNT_RE = re.compile(r"\bshares\b|number of|weighted[- ]average", re.IGNORECASE)

# ---------------------------------------------------------------- concept dictionary
# Every company names the same number differently ("Capital spending", "Payments to acquire
# property, plant and equipment", "Additions to property and equipment, net of ..."). Map them
# to one standard name with rules, so the SQL tool can filter `concept = 'capex'`.
# (concept, statement type, regex on the line item - matched from the START, case-insensitive)
# 'Net cash provided by / (used in) / provided/(used) by ... <act> activities' in all its spellings
CASH_ACTIVITY = (r"(?:net |total )?cash (?:\(used\)/|\(used in\)/|\(used by\)/)?(?:provided|generated|flows?|used)(?:/\(used\)|/\(provided\))?"
                 r"(?: by| from| in| for)?(?: \(used (?:in|for|by)\))?(?:/\(used (?:in|for|by)\))?(?: by)? {act} activities")
STMT_TYPES = [
    ("income", r"statements? of (?:consolidated )?(?:operations|income|earnings)(?!.*comprehensive)"),
    ("balance", r"balance sheets?|statements? of financial (?:position|condition)"),
    ("cash_flow", r"statements? of (?:consolidated )?cash flows?"),
]
CONCEPTS = [
    ("revenue", "income", r"(?:total )?(?:net )?(?:revenues?|sales)(?: to customers)?$|total revenues net of interest expense$"
                          r"|net revenues?$|net sales$|sales to customers$"),
    ("cost_of_revenue", "income", r"(?:total )?cost of (?:sales|revenues?|goods sold|products sold|products)$"),
    ("gross_profit", "income", r"gross (?:profit|margin)$"),
    ("operating_income", "income", r"operating (?:income|profit|loss)(?: \(loss\))?$"
                                r"|(?:\(loss\)/)?(?:income|earnings|loss)(?:/\(loss\))? from operations$"),
    ("net_income", "income", r"net (?:\(loss\)/|\(loss\) )?(?:income|earnings|loss)(?: \(loss\)|/\(loss\)|/earnings|/income)?$"
                             r"|net (?:\(loss\)/|\(loss\) )?(?:income|earnings|loss)(?: \(loss\)|/\(loss\)|/earnings|/income)? attributable to (?!non-?controlling)"),
    ("pretax_income", "income", r"(?:\(loss\)/)?(?:income|earnings|loss)(?:/\(loss\))?[^,]{0,45}? before [^,]{0,30}?tax"
                                r"|pre-?tax (?:income|earnings|loss|\(loss\)/income)"),
    ("income_tax_expense", "income", r"\(?provision\)?(?:/\(?benefit\)?)? for (?:income )?tax|\(?benefit\)?(?:/\(?provision\)?)? (?:from|for) (?:income )?tax"
                                     r"|income tax(?:es)?(?: expense| provision| \(?benefit\)?| \(expense\)/benefit| expense \(benefit\)"
                                     r"| \(benefit\)/expense| benefit/\(expense\)| \(provision\)/benefit)?$|taxes on income$"),
    ("eps_diluted", "income", r"diluted|(?:net )?(?:income|earnings) per share.*diluted"),
    ("total_assets", "balance", r"total assets$"),
    ("ppe_net", "balance", r"(?:total )?(?:property|premises),?(?: plant,?)? (?:and|&) equipment(?:[,\s\u2014\u2013-]+net| net)?"
                           r"(?: of (?:accumulated )?depreciation)?$|net property,? plant"),
    ("short_term_investments", "balance", r"short-term (?:investments|marketable securities)$|marketable securities$"),
    ("total_liabilities", "balance", r"total liabilities$"),
    ("total_current_assets", "balance", r"total current assets$"),
    ("total_current_liabilities", "balance", r"total current liabilities$"),
    ("total_equity", "balance", r"total (?:shareholders|stockholders|share ?owners|shareowners)['’]? equity$|total equity$"),
    ("inventory", "balance", r"(?:total )?inventor(?:y|ies)(?:, net)?$|merchandise inventories$"),
    ("accounts_receivable", "balance", r"(?:accounts|trade) receivable|receivables, net"),
    ("accounts_payable", "balance", r"accounts payable$|trade (?:and other )?payables$"),
    ("cash", "balance", r"cash and cash equivalents$|cash and due from banks$"),
    ("long_term_debt", "balance", r"long-term (?:debt|borrowings)(?:, net| obligations)?(?: of current portion)?$|long-term debt, excluding current"),
    ("operating_cash_flow", "cash_flow", CASH_ACTIVITY.format(act="operating")
                                         + r"|(?:net )?cash (?:from|provided by) operations$"),
    ("investing_cash_flow", "cash_flow", CASH_ACTIVITY.format(act="investing")),
    ("financing_cash_flow", "cash_flow", CASH_ACTIVITY.format(act="financing")),
    ("capex", "cash_flow", r"(?:purchases?|payments? (?:for|to acquire)|additions to|capital expenditures? for) (?:of )?"
                           r"(?:property|premises|plant)|capital (?:expenditures?|spending)$"),
    ("depreciation_amortization", "cash_flow", r"depreciation(?:,| and) amortization|depreciation$"),
    ("dividends_paid", "cash_flow", r"(?:cash )?dividends paid|dividends (?:to|paid to) (?:shareholders|stockholders|shareowners)"
                                    r"|payments? (?:of|for) (?:cash )?dividends"),
    ("share_repurchases", "cash_flow", r"(?:purchases?|repurchases?|payments? for repurchase) of (?:common |treasury )?(?:stock|shares)"
                                       r"|share repurchases?$|treasury stock (?:purchases|acquired)"),
]
STMT_RE = [(t, re.compile(rx, re.IGNORECASE)) for t, rx in STMT_TYPES]
CONCEPT_RE = [(c, t, re.compile(rx, re.IGNORECASE)) for c, t, rx in CONCEPTS]
NOT_A_STATEMENT = re.compile(r"parenthetical|\(details\)|\(tables\)|\(policies\)|narrative", re.IGNORECASE)


def statement_type(statement):
    if NOT_A_STATEMENT.search(statement):
        return "note"
    for t, rx in STMT_RE:
        if rx.search(statement):
            return t
    return "note"


# supplemental / non-cash disclosures that look like a concept but are not the number itself
CONCEPT_EXCLUDE = re.compile(r"accrued|not (?:yet )?paid|unpaid|in accounts payable|included in|non-cash investing",
                             re.IGNORECASE)


NOTE_REF_RE = re.compile(r"\s*\((?:see\s+)?notes?\s[^)]*\)", re.IGNORECASE)   # 'Inventories (Notes 1 and 3)'


def concept_of(line_item, stmt_type, kind):
    if CONCEPT_EXCLUDE.search(line_item):
        return None
    # normalise whitespace first: SEC labels can contain non-breaking spaces (\xa0) that look like spaces
    label = re.sub(r"\s+", " ", line_item.replace("\xa0", " "))
    label = NOTE_REF_RE.sub("", label).strip()
    for concept, t, rx in CONCEPT_RE:
        if t == stmt_type and rx.match(label):
            # EPS must be a per-share value; every other concept must be a money amount
            # (stops 'Net income attributable to Pfizer (in dollars per share)' becoming net_income)
            if (concept == "eps_diluted") != (kind == "per_share"):
                continue
            return concept
    return None


def line_priority(line_item, concept):
    """Lower = preferred when several lines of one statement map to the same concept."""
    li = line_item.lower()
    score = 0
    if "continuing operations" in li:
        score += 2                                    # prefer total over continuing-only
    if concept == "net_income" and "attributable to" not in li:
        score += 1                                    # prefer 'attributable to <company>' over incl. NCI
    if concept == "ppe_net" and "net" not in li:
        score += 3                                    # gross PP&E only if no net line exists (e.g. Pfizer)
    if li.startswith("total "):
        score -= 1                                    # 'Total cost of sales' over 'Cost of sales'
    return score + len(li) / 1000                     # tie-break: shorter, more generic label


def split_title(title):
    """'X - Sales by Segment (Details) - USD ($) $ in Millions' -> units are always the LAST part."""
    head, sep, tail = title.rpartition(" - ")
    if sep and UNITS_RE.search(tail):
        return head.strip(), tail.strip()
    return title.strip(), ""


def parse_period(label):
    """'12 Months Ended Dec. 31, 2022' -> (2022-12-31, 12); 'Dec. 31, 2022' -> (2022-12-31, None)."""
    m = DATE_RE.search(str(label))
    if not m or m.group(1) not in MON:
        return pd.NaT, None
    end = pd.Timestamp(int(m.group(3)), MON[m.group(1)], int(m.group(2)))
    months = MONTHS_RE.search(str(label))
    return end, int(months.group(1)) if months else None


def value_kind(line_item, units=""):
    if PER_SHARE_RE.search(line_item) or (BARE_EPS_RE.match(line_item.strip()) and "/ share" in units.lower()):
        return "per_share"
    if SHARE_COUNT_RE.search(line_item) and not re.search(r"\$|amount|value|cost|repurchase", line_item, re.I):
        return "shares"
    return "money"


def money_scale(units):
    """Multiplier that converts the sheet's money values to USD millions."""
    u = units.lower()
    if "$ in millions" in u:
        return 1.0
    if "$ in thousands" in u:
        return 1e-3
    if "$ in billions" in u:
        return 1e3
    if "usd" in u or "$" in u:
        return 1e-6            # plain dollars (e.g. 3M 2022 reports '34,229,000,000')
    return None


def sheet_to_facts(csv_path: Path):
    df = pd.read_csv(csv_path, header=None, dtype=str)
    if df.shape[0] < 2 or df.shape[1] < 2:
        return pd.DataFrame()
    title = str(df.iat[0, 0])
    statement, units = split_title(title)

    # header: row 0 (forward-filled, e.g. "12 Months Ended") + row 1 when row 1 is a pure date row
    top = df.iloc[0, 1:].ffill().fillna("")
    if pd.isna(df.iat[1, 0]) or str(df.iat[1, 0]).strip() == "":
        second = df.iloc[1, 1:].fillna("")
        labels = [f"{a} {b}".strip() for a, b in zip(top, second)]
        body = df.iloc[2:]
    else:
        labels = list(top)
        body = df.iloc[1:]

    rows = []
    for _, r in body.iterrows():
        item = r.iloc[0]
        if pd.isna(item):
            continue
        for label, raw in zip(labels, r.iloc[1:]):
            v = to_number(raw)
            if v is not None:
                rows.append({"statement": statement.strip(), "units": units.strip(),
                             "line_item": str(item).strip(), "period_label": label,
                             "value": v, "raw_value": raw})
    return pd.DataFrame(rows)


def main():
    P = config.PROC
    ch = pd.read_parquet(P / "chunks.parquet")
    sheets = ch[(ch.format == "xlsx") & ch.sheet_csv.notna()].drop_duplicates("sheet_csv")

    parts = []
    for r in sheets.itertuples():
        facts = sheet_to_facts(Path(r.sheet_csv))
        if facts.empty:
            continue
        facts.insert(0, "sheet_chunk_id", r.chunk_id)
        facts.insert(0, "filing_id", r.filing_id)
        facts.insert(0, "fiscal_year", r.fiscal_year)
        facts.insert(0, "company", r.company)
        facts.insert(0, "ticker", r.ticker)
        parts.append(facts)

    out = pd.concat(parts, ignore_index=True)

    # ---------------------------------------------------------------- normalisation
    # 1. period: end date + length ('12 Months Ended Dec. 31, 2022' -> 2022-12-31, 12)
    per = out.period_label.map(parse_period)
    out["period_end"] = [p[0] for p in per]
    out["period_months"] = pd.array([p[1] for p in per], dtype="Int64")
    # 2. fiscal year OF THE COLUMN (a 10-K also has prior-year columns). Companies like J&J or
    #    Best Buy end their year in Jan, so the calendar year in the label != fiscal year.
    #    Anchor = the filing's real fiscal-year end, read from the SEC metadata (meta.json,
    #    'report_date'). Guessing it from dates inside the sheets failed twice (issues 20, 24):
    #    detail sheets contain later dates (subsequent events, forward-looking periods).
    #    Fallback only if meta.json is missing: latest end of a 12-month period.
    fy_end = {}
    for t, fy in out[["ticker", "fiscal_year"]].drop_duplicates().itertuples(index=False):
        meta = config.DATA / "raw" / t / "edgar" / f"{fy}_10K" / "meta.json"
        if meta.exists():
            fy_end[(t, fy)] = pd.Timestamp(json.loads(meta.read_text())["report_date"])
    key = list(zip(out.ticker, out.fiscal_year))
    from_meta = pd.Series([fy_end.get(k, pd.NaT) for k in key], index=out.index)
    guess = out.period_end.where(out.period_months == 12).groupby(out.filing_id).transform("max")
    latest = from_meta.fillna(guess)
    years_back = ((latest - out.period_end).dt.days / 365.25).round()
    print("fiscal year end per filing (source: SEC meta.json, else guessed):")
    chk = pd.DataFrame({"filing": out.filing_id, "meta": from_meta, "guess_from_sheets": guess}).drop_duplicates("filing")
    chk["differs"] = chk.meta.notna() & (chk.meta != chk.guess_from_sheets)
    print(chk.to_string(index=False), "\n")
    out["period_fiscal_year"] = (out.fiscal_year - years_back).astype("Int64")
    # 3. one comparable scale: money in USD millions
    out["value_kind"] = [value_kind(li, u) for li, u in zip(out.line_item, out.units)]
    scale = out.units.map(money_scale)
    #    some sheet titles carry no unit at all -> use the filing's usual money scale
    #    (the most common one among its sheets) and flag it as inferred
    default_scale = scale.groupby(out.filing_id).transform(lambda s: s.mode().iloc[0] if s.notna().any() else None)
    out["units_inferred"] = scale.isna() & default_scale.notna()
    scale = scale.fillna(default_scale)
    out["value_musd"] = (out.value * scale).where(out.value_kind == "money")
    # 4. the same number appears in several 10-Ks (each 10-K repeats 2-3 prior years, sometimes
    #    restated). Mark the copy from the MOST RECENT filing so the SQL tool can avoid duplicates.
    fact_key = ["ticker", "statement", "line_item", "period_fiscal_year", "period_months"]
    newest = out.groupby(fact_key, dropna=False).fiscal_year.transform("max")
    out["from_latest_filing"] = out.fiscal_year == newest
    # 5. statement type + standard concept name (rule-based dictionary above)
    out["statement_type"] = out.statement.map(statement_type)
    out["concept"] = [concept_of(li, st, k) for li, st, k in zip(out.line_item, out.statement_type, out.value_kind)]
    #    one line per concept per period: if several lines match (e.g. 'Net income' and
    #    'Net income attributable to PepsiCo'), keep the preferred one, keep the others as candidates
    has = out.concept.notna()
    out["concept_priority"] = [line_priority(li, c) if c else None for li, c in zip(out.line_item, out.concept)]
    grp = ["filing_id", "concept", "period_fiscal_year", "period_months"]
    best = out[has].groupby(grp, dropna=False).concept_priority.transform("min")
    out["concept_alternative"] = None
    loser = has & (out.concept_priority > best.reindex(out.index))
    out.loc[loser, "concept_alternative"] = out.loc[loser, "concept"]
    out.loc[loser, "concept"] = None
    #    still a tie (same label, e.g. Boeing's 'Total revenues' for products, services AND the
    #    grand total): keep the largest amount - a total is never smaller than its parts
    tied = out.concept.notna()
    biggest = out[tied].value.abs().groupby([out[tied][g] for g in grp], dropna=False).transform("max")
    part = tied & (out.value.abs() < biggest.reindex(out.index))
    out.loc[part, "concept_alternative"] = out.loc[part, "concept"]
    out.loc[part, "concept"] = None
    out = out.drop(columns="concept_priority")
    out.to_parquet(P / "fin_facts.parquet", index=False)

    # ---------------------------------------------------------------- quality report
    print(f"{len(out)} facts from {len(parts)} of {len(sheets)} sheets "
          f"({len(sheets) - len(parts)} sheets had no numbers, e.g. text/policy notes), "
          f"{out.ticker.nunique()} companies\n")
    print("facts per filing:\n", out.groupby(["ticker", "fiscal_year"]).size().to_string(), "\n")
    print("top units:\n", out.units.value_counts().head(6).to_string(), "\n")
    print(f"no units in title : {(out.units == '').mean():.1%} of facts "
          f"({out.units_inferred.mean():.1%} got the filing's usual scale, flagged units_inferred)")
    no_unit = out[out.units == ""].drop_duplicates("statement").statement.head(5).tolist()
    print("  examples of sheets without units:", no_unit)
    print(f"period date parsed: {out.period_end.notna().mean():.1%} of facts")
    print("value kinds:\n", out.value_kind.value_counts().to_string(), "\n")
    print(f"facts repeated in an older filing (from_latest_filing = False): {(~out.from_latest_filing).mean():.1%}\n")
    print("random sample:\n", out.sample(min(8, len(out)), random_state=1)[["ticker", "fiscal_year", "statement", "line_item",
                                                            "period_fiscal_year", "period_months", "value_musd"]]
          .to_string(max_colwidth=40), "\n")

    # spot check: headline revenue per filing, from the main income statement, for the filing's own year
    main_inc = out.statement.str.match(r"(?i)^\s*consolidated statements? of (?:operations|income|earnings)\s*$")
    rev = out[main_inc & (out.period_fiscal_year == out.fiscal_year) & (out.period_months == 12)
              & out.line_item.str.contains(r"(?i)^(?:total )?(?:net )?(?:revenues?|sales)\b|^net sales|sales to customers")]
    first = rev.groupby(["ticker", "fiscal_year"]).head(1)
    print("spot check - revenue for the filing's own fiscal year, USD millions (compare with the 10-K):")
    print(first[["ticker", "fiscal_year", "line_item", "period_label", "value_musd"]]
          .to_string(index=False, max_colwidth=40))
    missing = sorted(set(out.filing_id) - set(first.filing_id))
    if missing:
        print("no revenue line found for:", missing)

    # concept coverage: for each filing's OWN fiscal year, which standard concepts were found?
    own = out[out.concept.notna() & (out.period_fiscal_year == out.fiscal_year)
              & ((out.period_months == 12) | out.period_months.isna())]
    print("\nstatement types:\n", out.statement_type.value_counts().to_string())
    print(f"\nfacts with a concept: {out.concept.notna().sum()}")
    cov = own.groupby(["concept", "filing_id"]).size().unstack(fill_value=0).gt(0)
    print(f"\nconcept coverage for each filing's own year ({len(cov.columns)} filings):")
    print(cov.sum(axis=1).sort_values(ascending=False).astype(str).add(f" / {len(cov.columns)}").to_string())
    # same concept matched by several different line items in one statement = rule too broad
    amb = own.groupby(["filing_id", "concept"]).line_item.nunique()
    amb = amb[amb > 1]
    if len(amb):
        print("\nconcepts matched by more than one line item (check these rules):")
        for (fid, c), _ in amb.head(15).items():
            items = own[(own.filing_id == fid) & (own.concept == c)].line_item.unique()[:3]
            print(f"  {fid:<26} {c:<22} {list(items)}")
    # which filings miss a concept, and what lines did that statement have? (to extend the rules)
    print("\nmissing concepts (filing: closest line items in that statement):")
    hints = {"net_income": "net", "operating_cash_flow": "operating activities", "total_liabilities": "liabilities",
             "operating_income": "operat", "eps_diluted": "diluted"}
    stype = {c: t for c, t, _ in CONCEPTS}
    for c, kw in hints.items():
        have = set(own[own.concept == c].filing_id)
        for fid in sorted(set(out.filing_id) - have)[:4]:
            cand = out[(out.filing_id == fid) & (out.statement_type == stype[c])
                       & out.line_item.str.contains(kw, case=False, regex=False)].line_item.unique()[:3]
            print(f"  {c:<20} {fid:<26} {list(cand)}")

    latest = own[own.fiscal_year == own.groupby("ticker").fiscal_year.transform("max")]
    capex = latest[latest.concept == "capex"].groupby("ticker").head(1)
    print("\ncapex for each company's latest filing year (USD millions, positive):")
    print(capex.assign(capex_musd=capex.value_musd.abs())[["ticker", "fiscal_year", "line_item", "capex_musd", "units_inferred"]]
          .to_string(index=False, max_colwidth=55))


if __name__ == "__main__":
    main()