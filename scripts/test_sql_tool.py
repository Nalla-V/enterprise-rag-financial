"""
Step 9b: test the numbers tool on the FinanceBench 10-K questions (run through jobs/job_with_llm.sh).

For each question it shows the LLM's tool call (concepts + years), whether the structured lookup or
the free-SQL fallback was used, and the rows - next to the gold answer.
Summary: valid tool calls, questions answered by lookup / free SQL / nothing.

    sbatch jobs/job_with_llm.sh scripts/test_sql_tool.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import config, query_meta  # noqa: E402
from rag.sql_tool import FinFactsSQL, plan_json, rows_as_text  # noqa: E402


def main():
    q = pd.read_csv(config.DATA / "eval_questions.csv")
    q = q[q.doc_type.str.lower() == "10k"]
    tool = FinFactsSQL()
    stats = {"lookup": 0, "free_sql": 0, "nothing": 0, "not_numeric": 0, "no_excel": 0}
    t_all = time.time()
    for r in q.itertuples():
        meta = query_meta.analyse(r.question)
        print("=" * 100)
        print(f"[{r.financebench_id}] {r.doc_name} | meta {meta}")
        print("Q   :", r.question[:200])
        print("GOLD:", str(r.answer)[:160].replace("\n", " "))
        if not tool.covers(meta["tickers"], meta["years"]):
            stats["no_excel"] += 1
            print("TOOL: skipped - no Excel data for this company/year (router would send to documents)")
            continue
        t0 = time.time()
        res = tool.answer_rows(r.question, meta)
        mode = "not_numeric" if res["mode"] == "none" else (res["mode"] if len(res["rows"]) else "nothing")
        stats[mode] += 1
        note = "  -> no matching concept, not a numeric question: router sends it to documents" if mode == "not_numeric" else ""
        print(f"TOOL: {mode} | plan {plan_json(res['plan'])} | {time.time() - t0:.1f}s{note}")
        if res["sql"]:
            print("SQL :", " ".join(res["sql"].split())[:300])
        if len(res["rows"]):
            print(rows_as_text(res["rows"], n=8).split("\n", 1)[-1])

    print("=" * 100)
    n = sum(stats.values())
    print(f"{n} questions in {time.time() - t_all:.0f}s: " + ", ".join(f"{k} {v}" for k, v in stats.items()))


if __name__ == "__main__":
    main()