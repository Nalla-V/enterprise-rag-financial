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
PER_SHARE_RE = re.compile(r"per (?:common |diluted |basic )?share|\beps\b|dividends? declared per", re.IGNORECASE)
SHARE_COUNT_RE = re.compile(r"\bshares\b|number of|weighted[- ]average", re.IGNORECASE)


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


def value_kind(line_item):
    if PER_SHARE_RE.search(line_item):
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
    out["value_kind"] = out.line_item.map(value_kind)
    scale = out.units.map(money_scale)
    out["value_musd"] = (out.value * scale).where(out.value_kind == "money")
    out.to_parquet(P / "fin_facts.parquet", index=False)

    # ---------------------------------------------------------------- quality report
    print(f"{len(out)} facts from {len(parts)} of {len(sheets)} sheets "
          f"({len(sheets) - len(parts)} sheets had no numbers, e.g. text/policy notes), "
          f"{out.ticker.nunique()} companies\n")
    print("facts per filing:\n", out.groupby(["ticker", "fiscal_year"]).size().to_string(), "\n")
    print("top units:\n", out.units.value_counts().head(6).to_string(), "\n")
    print(f"no units in title : {(out.units == '').mean():.1%} of facts")
    print(f"period date parsed: {out.period_end.notna().mean():.1%} of facts")
    print("value kinds:\n", out.value_kind.value_counts().to_string(), "\n")
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


if __name__ == "__main__":
    main()