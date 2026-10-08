"""Rule-based understanding of a question: which company, which years, which kind of question.

No LLM here on purpose: deterministic, fast, easy to debug (the Reddit post's advice).
"""
import re

from . import config

YEAR_RE = re.compile(r"(?<!\d)(19[89]\d|20[0-3]\d)(?!\d)")
SHORT_FY_RE = re.compile(r"\bFY\s?'?(\d{2})\b", re.IGNORECASE)        # 'FY22' -> 2022
QUARTER_RE = re.compile(r"\bQ[1-4]\b|\bquarter(?:ly)?\b|\b10-?Q\b", re.IGNORECASE)
NUMERIC_RE = re.compile(
    r"\bhow much\b|\bhow many\b|\bwhat (?:is|was|were|are) (?:the )?(?:total|amount|value|ratio|margin)\b"
    r"|\bratio\b|\bmargin\b|\bpercent(?:age)?\b|%|\bin (?:usd|millions|billions|thousands)\b"
    r"|\bcalculate\b|\bcompute\b|\bround(?:ed)?\b|\bexact\b|\btable\b|\bchange\b|\bgrowth\b"
    r"|\bamount\b|\bcapex\b|\bcapital expenditure|\bturnover\b|\beps\b|\bebitda\b",
    re.IGNORECASE)


def companies(q: str):
    ql = q.lower()
    return sorted(t for t, names in config.COMPANY_ALIASES.items()
                  if any(re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", ql) for n in names))


def years(q: str):
    found = {int(y) for y in YEAR_RE.findall(q)} | {2000 + int(y) for y in SHORT_FY_RE.findall(q)}
    return sorted(found)


def wants_quarter(q: str) -> bool:
    return bool(QUARTER_RE.search(q))


def is_numeric(q: str) -> bool:
    """The post's 'precision mode' trigger: exact numbers / tables / calculations."""
    return bool(NUMERIC_RE.search(q))


def expand_acronyms(q: str) -> str:
    """Append expansions for finance acronyms found in the question."""
    extra = [exp for ac, exp in config.ACRONYMS.items()
             if re.search(rf"(?<![A-Za-z]){re.escape(ac)}(?![A-Za-z])", q, re.IGNORECASE)
             and exp.lower() not in q.lower()]
    return q if not extra else f"{q} ({'; '.join(extra)})"


def analyse(q: str) -> dict:
    return {"tickers": companies(q), "years": years(q), "quarter": wants_quarter(q),
            "numeric": is_numeric(q)}