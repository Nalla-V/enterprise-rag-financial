"""
Step 9a: smoke test for the local LLM server (run through jobs/job_with_llm.sh).

Checks the three things the agent needs from the LLM:
  1. a normal answer
  2. a JSON answer that can be parsed (used for routing and the judge)
  3. an answer grounded in provided context, with a citation
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import config, llm  # noqa: E402

print("endpoint:", config.LLM_BASE_URL, "| model:", config.LLM_MODEL)

t0 = time.time()
print("\n1) plain answer:")
print(llm.ask("You are concise.", "In one sentence: what is a 10-K filing?", max_tokens=80))
print(f"   ({time.time() - t0:.1f}s)")

t0 = time.time()
print("\n2) JSON answer (router-style):")
r = llm.ask_json('Classify the question. Reply only as JSON: {"route": "documents" | "numbers" | "both"}',
                 "What was Boeing's total revenue in FY2022?", default={"route": "PARSE FAILED"}, max_tokens=30)
print("  ", r, f"({time.time() - t0:.1f}s)")

t0 = time.time()
print("\n3) grounded answer with citation:")
context = ("[1] 3M | 10K 2018 | Financial Statements | Consolidated Statement of Cash Flows (pdf, page 60)\n"
           "Purchases of property, plant and equipment (PP&E) (1,577) (1,373) (1,420)")
print(llm.ask("Answer only from the sources and cite them as [n].",
              f"Sources:\n{context}\n\nQuestion: What was 3M's FY2018 capital expenditure in USD millions?",
              max_tokens=80))
print(f"   ({time.time() - t0:.1f}s)")