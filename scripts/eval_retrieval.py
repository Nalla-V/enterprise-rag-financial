"""
Step 8: hybrid retrieval + retrieval-only evaluation (no LLM yet).

Two modes:
  1. Try a single question (shows the top hits):
        python scripts/eval_retrieval.py --query "What was 3M's FY2018 capital expenditure?"
  2. Ablation over all 79 FinanceBench questions (GPU job recommended for the reranker):
        python -u scripts/eval_retrieval.py

Metrics per question (then averaged), at k = 1, 5, 10:
  doc_hit@k          is at least one top-k chunk from the RIGHT filing?            (higher = better)
  drm@k              Document-level Retrieval Mismatch: share of top-k chunks
                     from a WRONG filing                                          (lower  = better)
  evidence_recall@k  share of the gold evidence's words found in the top-k chunks
                     of the right filing                                           (higher = better)
  page_hit@k         a top-k PDF chunk covers the gold evidence page (+-1)         (higher = better)
  mrr                1 / rank of the first chunk from the right filing

Output: results/retrieval_ablation.csv          one row per configuration
        results/retrieval_per_question.parquet  one row per (configuration, question) for failure analysis
"""
import argparse
import ast
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import config, query_meta  # noqa: E402
from rag.retrieval import HybridRetriever, RetrievalConfig  # noqa: E402

KS = (1, 5, 10)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
STOP = set("the a an of and or to in for on by with as at is are was were be been this that from its it "
           "which their our we our us".split())

# The ablation: each row switches ONE more component on.
BASE = RetrievalConfig(name="", variant="plain", dense=False, bm25=False, filters=False,
                       acronyms=False, table_boost=False, rerank=False, k=10)
CONFIGS = [
    replace(BASE, name="1 bm25 only (plain)", bm25=True),
    replace(BASE, name="2 dense only (plain)", dense=True),
    replace(BASE, name="3 hybrid RRF (plain)", dense=True, bm25=True),
    replace(BASE, name="4 hybrid + metadata header", variant="header", dense=True, bm25=True),
    replace(BASE, name="5 hybrid + summary (SAC)", variant="sac", dense=True, bm25=True),
    replace(BASE, name="6 header + filters", variant="header", dense=True, bm25=True, filters=True),
    replace(BASE, name="7 header + filters + acronyms + table boost", variant="header", dense=True, bm25=True,
            filters=True, acronyms=True, table_boost=True),
    replace(BASE, name="8 = 7 + reranker", variant="header", dense=True, bm25=True,
            filters=True, acronyms=True, table_boost=True, rerank=True),
    replace(BASE, name="9 SAC + filters + acronyms + table boost + reranker", variant="sac", dense=True, bm25=True,
            filters=True, acronyms=True, table_boost=True, rerank=True),
    # round 2 - fixes suggested by round 1
    replace(BASE, name="10 = 7 with BM25 weight 0.5", variant="header", dense=True, bm25=True,
            filters=True, acronyms=True, table_boost=True, bm25_weight=0.5),
    replace(BASE, name="11 = 8 + metadata kept after rerank", variant="header", dense=True, bm25=True,
            filters=True, acronyms=True, table_boost=True, rerank=True, rerank_keep_meta=True),
    replace(BASE, name="12 = 9 + metadata kept after rerank", variant="sac", dense=True, bm25=True,
            filters=True, acronyms=True, table_boost=True, rerank=True, rerank_keep_meta=True),
]


# ---------------------------------------------------------------- gold data
def parse_evidence(s):
    """eval_questions.csv stores the evidence list as a Python/numpy repr string."""
    s = re.sub(r"array\((\[.*?\])(?:, dtype=object)?\)", r"\1", str(s), flags=re.DOTALL)
    try:
        ev = ast.literal_eval(s)
        return [dict(e) for e in ev]
    except (ValueError, SyntaxError):
        return []


def load_questions():
    q = pd.read_csv(config.DATA / "eval_questions.csv")
    q["evidence"] = q.evidence.map(parse_evidence)
    q["evidence_text"] = q.evidence.map(lambda ev: " ".join(e.get("evidence_text", "") for e in ev))
    q["evidence_pages"] = q.evidence.map(lambda ev: [int(e["evidence_page_num"]) for e in ev
                                                     if str(e.get("evidence_page_num", "")).isdigit()])
    return q.reset_index(drop=True)


# ---------------------------------------------------------------- metrics
def tokens(text):
    return {t for t in TOKEN_RE.findall(str(text).lower()) if t not in STOP}


def metrics(hits: pd.DataFrame, gold_filing, evidence_text, evidence_pages):
    out = {}
    right = (hits.filing_id == gold_filing).to_numpy()
    ev = tokens(evidence_text)
    for k in KS:
        r = right[:k]
        out[f"doc_hit@{k}"] = float(r.any())
        out[f"drm@{k}"] = float(1 - r.mean()) if len(r) else 1.0
        if ev:
            got = set().union(*[tokens(t) for t in hits.chunk_text.iloc[:k][r]]) if r.any() else set()
            out[f"evidence_recall@{k}"] = len(ev & got) / len(ev)
        if evidence_pages:
            top = hits.iloc[:k][r & (hits.format.iloc[:k] == "pdf").to_numpy()]
            # FinanceBench pages are 0-based, Docling's 1-based -> compare page+1, allow +-1
            hit = any(pd.notna(p0) and any(p0 - 1 <= g + 1 <= p1 + 1 for g in evidence_pages)
                      for p0, p1 in zip(top.page_start, top.page_end.fillna(top.page_start)))
            out[f"page_hit@{k}"] = float(hit)
    first = int(right.argmax()) if right.any() else None
    out["mrr"] = 1.0 / (first + 1) if first is not None else 0.0
    return out


# ---------------------------------------------------------------- modes
def show_query(ret, question, cfg):
    meta = query_meta.analyse(question)
    print("question  :", question)
    print("meta      :", meta)
    print("expanded  :", query_meta.expand_acronyms(question) if cfg.acronyms else "(acronyms off)")
    t0 = time.time()
    hits = ret.search(question, cfg, meta)
    print(f"{len(hits)} hits in {time.time() - t0:.1f}s ({cfg.name})\n")
    for h in hits.itertuples():
        print(f"#{h.rank:<2} {h.filing_id:<28} {h.format:<4} {h.chunk_type:<5} {str(h.location)[:28]:<28} {h.context_header[:70]}")
        print(f"     {h.chunk_text[:160].replace(chr(10), ' ')}")


def run_ablation(ret, qs, configs):
    rows, per_q = [], []
    for cfg in configs:
        t0 = time.time()
        res = []
        for q in qs.itertuples():
            meta = query_meta.analyse(q.question)
            hits = ret.search(q.question, cfg, meta)
            m = metrics(hits, q.doc_name, q.evidence_text, q.evidence_pages)
            res.append(m)
            per_q.append({"config": cfg.name, "financebench_id": q.financebench_id, "doc_name": q.doc_name,
                          "question_type": q.question_type, "numeric": meta["numeric"],
                          "top_filings": hits.filing_id.head(5).tolist(), **m})
        avg = pd.DataFrame(res).mean().to_dict()
        rows.append({"config": cfg.name, **avg, "sec_per_query": (time.time() - t0) / len(qs)})
        print(f"{cfg.name:<55} doc_hit@5 {avg['doc_hit@5']:.2f}  drm@5 {avg['drm@5']:.2f}  "
              f"evidence@5 {avg.get('evidence_recall@5', float('nan')):.2f}  page@5 {avg.get('page_hit@5', float('nan')):.2f}  "
              f"mrr {avg['mrr']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(per_q)


def log_mlflow(table):
    try:
        import mlflow
    except ImportError:
        print("(mlflow not installed - skipping tracking)")
        return
    mlflow.set_experiment("retrieval-ablation")
    for r in table.to_dict("records"):
        cfg = next(c for c in CONFIGS if c.name == r["config"])
        with mlflow.start_run(run_name=r["config"]):
            mlflow.log_params({k: v for k, v in cfg.as_params().items() if k != "name"})
            mlflow.log_metrics({k.replace("@", "_at_"): float(v) for k, v in r.items()
                                if k != "config" and pd.notna(v)})
    print("logged to MLflow (./mlruns) - view with: mlflow ui")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="try one question instead of running the ablation")
    ap.add_argument("--config", type=int, default=7, help="config number for --query (1-9)")
    ap.add_argument("--only", type=int, nargs="*", help="run only these config numbers, e.g. --only 1 3 7")
    args = ap.parse_args()

    ret = HybridRetriever()
    print(f"index: {len(ret.chunks)} chunks, device {ret.device}\n")

    if args.query:
        show_query(ret, args.query, CONFIGS[args.config - 1])
        return

    qs = load_questions()
    print(f"{len(qs)} questions, {(qs.evidence_text.str.len() > 0).mean():.0%} with evidence text, "
          f"{qs.evidence_pages.map(len).gt(0).mean():.0%} with evidence pages\n")
    configs = [CONFIGS[i - 1] for i in args.only] if args.only else CONFIGS
    table, per_q = run_ablation(ret, qs, configs)

    out = Path("results")
    out.mkdir(exist_ok=True)
    table.to_csv(out / "retrieval_ablation.csv", index=False)
    per_q.to_parquet(out / "retrieval_per_question.parquet", index=False)
    cols = ["config", "doc_hit@1", "doc_hit@5", "drm@5", "evidence_recall@5", "page_hit@5", "mrr"]
    print("\n" + table[[c for c in cols if c in table]].round(3).to_string(index=False))

    # breakdown of the best config by question type and numeric vs text
    best = table.assign(_d=-table["drm@5"]).sort_values(["evidence_recall@5", "_d"], ascending=False).config.iloc[0]
    b = per_q[per_q.config == best]
    print(f"\nbest config: {best}")
    print(b.groupby("question_type")[["doc_hit@5", "drm@5", "evidence_recall@5"]].mean().round(2).to_string())
    print(b.groupby("numeric")[["doc_hit@5", "drm@5", "evidence_recall@5"]].mean().round(2).to_string())

    # failure analysis: which questions does the best config still miss?
    miss = b[b["doc_hit@5"] == 0]
    print(f"\nquestions where the right filing is NOT in the top 5 ({len(miss)}):")
    for r in miss.itertuples():
        print(f"  {r.financebench_id:<24} gold {r.doc_name:<34} got {r.top_filings[:3]}")
    low = b[(b["doc_hit@5"] == 1) & (b["evidence_recall@5"] < 0.3)]
    print(f"right filing but little of the evidence (<30%) ({len(low)}): {low.financebench_id.tolist()[:12]}")
    log_mlflow(table)


if __name__ == "__main__":
    main()