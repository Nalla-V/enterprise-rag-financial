"""Central settings. Everything can be overridden with environment variables."""
import os
from pathlib import Path

DATA = Path(os.getenv("RAG_DATA", "data"))
PROC = DATA / "processed"

# Models
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
GEN_MODEL = os.getenv("GEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")   # used by vLLM on ALICE

# Any OpenAI-compatible endpoint: vLLM on ALICE, Databricks serving endpoints, or a hosted API
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_MODEL = os.getenv("LLM_MODEL", GEN_MODEL)

COMPANY = {"PEP": "PepsiCo", "AMCR": "Amcor", "JNJ": "Johnson & Johnson", "MMM": "3M",
           "AMD": "AMD", "BBY": "Best Buy", "BA": "Boeing", "AXP": "American Express",
           "JPM": "JPMorgan", "PFE": "Pfizer"}

# How a user might refer to each company in a question (rule-based query metadata)
COMPANY_ALIASES = {
    "PEP": ["pepsico", "pepsi"], "AMCR": ["amcor"], "JNJ": ["johnson & johnson", "johnson and johnson", "j&j", "jnj"],
    "MMM": ["3m"], "AMD": ["amd", "advanced micro devices"], "BBY": ["best buy", "bestbuy"],
    "BA": ["boeing"], "AXP": ["american express", "amex"], "JPM": ["jpmorgan", "jp morgan", "jpm", "chase"],
    "PFE": ["pfizer"],
}

# Finance acronyms -> expansion (the post's "acronym confusion" fix)
ACRONYMS = {
    "EPS": "earnings per share", "FCF": "free cash flow", "COGS": "cost of goods sold",
    "SG&A": "selling, general and administrative expenses", "D&A": "depreciation and amortization",
    "EBITDA": "earnings before interest, taxes, depreciation and amortization",
    "EBIT": "earnings before interest and taxes", "CAPEX": "capital expenditures",
    "PP&E": "property, plant and equipment", "ROE": "return on equity", "ROA": "return on assets",
    "ROIC": "return on invested capital", "CET1": "common equity tier 1 capital ratio",
    "AOCI": "accumulated other comprehensive income", "R&D": "research and development",
    "YOY": "year over year", "DPO": "days payable outstanding", "DSO": "days sales outstanding",
    "NII": "net interest income", "LCR": "liquidity coverage ratio", "OCF": "operating cash flow",
}