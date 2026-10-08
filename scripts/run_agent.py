"""
Step 9c: run the agent (through jobs/job_with_llm.sh, which provides the LLM server).

    sbatch jobs/job_with_llm.sh scripts/run_agent.py --demo
    sbatch jobs/job_with_llm.sh scripts/run_agent.py --question "What was 3M's FY2018 capex?"

--demo runs a few typical single questions plus one multi-turn conversation, and prints for each:
the standalone question, the route, the numbers-tool call, the top filings, the answer, the sources,
and the time per step.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.graph import RAGAgent  # noqa: E402

DEMO = [
    "What is the FY2018 capital expenditure amount (in USD millions) for 3M?",                  # numbers
    "What is the FY2022 unadjusted EBITDA less capex for PepsiCo? Define unadjusted EBITDA as "  # calculation
    "unadjusted operating income + depreciation and amortization from the cash flow statement.",
    "Has Boeing reported any materially important ongoing legal battles from FY2022?",           # documents
    "Are Best Buy's gross margins historically consistent?",                                     # no year given
    "Does AMEX have an improving operating margin profile as of 2022? If operating margin is "   # 'not relevant'
    "not a useful metric for a company like this, then state that and explain why.",
]
CONVERSATION = [                                                                                # multi-turn
    "What was Boeing's total revenue in FY2022?",
    "And its gross margin that year?",
    "How does that compare with 3M in the same year?",
]


def show(res, question):
    t = {x["node"]: x for x in res["trace"]}
    print("=" * 100)
    print("USER      :", question)
    if res["standalone"] != question:
        print("STANDALONE:", res["standalone"])
    print(f"ROUTE     : {res['route']} (LLM said: {res.get('llm_route')}, Excel data: {t['route']['covered']})")
    if "sql" in t:
        print(f"NUMBERS   : {t['sql']['mode']} {t['sql']['plan']} -> {t['sql']['rows']} rows")
    print("FILINGS   :", t["retrieve"]["top_filings"], "| retry" if res["attempts"] == 11 else "")
    print("ANSWER    :", res["answer"].replace("\n", "\n            "))
    print("SOURCES   :")
    for src in res["sources"]:
        if src["n"] == "SQL":
            print(f"   [SQL] concepts {src['concepts']} from {src['filings']}")
        else:
            print(f"   [{src['n']}] {src['filing_id']} ({src['format']}, {src['location']})")
    print("TIME      :", " | ".join(f"{x['node']} {x.get('sec', 0)}s" for x in res["trace"] if "sec" in x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    agent = RAGAgent()
    print(f"agent ready in {time.time() - t0:.0f}s")

    if args.question:
        show(agent.ask(args.question), args.question)
    if args.demo:
        for q in DEMO:
            show(agent.ask(q), q)
        print("\n" + "#" * 40 + " multi-turn conversation " + "#" * 40)
        history = []
        for q in CONVERSATION:
            res = agent.ask(q, history)
            show(res, q)
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": res["answer"]}]


if __name__ == "__main__":
    main()