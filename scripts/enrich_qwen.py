"""
Step 5: LLM enrichment with Qwen on the ALICE GPU (vLLM offline batch inference).

  1. One short, generic summary per FILING          -> used for Summary-Augmented Chunking (SAC)
  2. One 1-2 sentence description per TABLE chunk   -> "dual embedding" of tables

Input : data/processed/chunks.parquet
Output: data/processed/doc_summaries.parquet       (filing_id, doc_summary)
        data/processed/table_descriptions.parquet  (chunk_id, table_description)

Usage (GPU job):  python -u scripts/enrich_with_qwen.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import config  # noqa: E402

SUMMARY_WORDS_IN = 2500     # words of the filing shown to the model
TABLE_WORDS_IN = 900        # words of a table shown to the model

SUMMARY_SYS = ("You summarise financial filings. Write a generic summary of at most 80 words: "
               "company, document type, fiscal period, and the main topics covered. "
               "No numbers unless they are the headline figures. Plain prose, no lists.")
TABLE_SYS = ("You describe tables from financial filings. In 1-2 sentences say what the table shows: "
             "which metrics, which entity or segment, which periods, and the units. "
             "Do not repeat all the numbers.")


def truncate(text, n_words):
    words = text.split()
    return " ".join(words[:n_words])


def summary_input(group: pd.DataFrame) -> str:
    """Prefer the business overview + start of MD&A; fall back to the first text of the filing."""
    text = group[group.chunk_type == "text"]
    pick = pd.concat([text[text.item == "1"].head(6), text[text.item.isin(["7", "2"])].head(6)])
    if pick.n_words.sum() < 300:
        pick = text.head(15)
    return truncate("\n".join(pick.chunk_text), SUMMARY_WORDS_IN)


def main():
    from vllm import LLM, SamplingParams

    P = config.PROC
    ch = pd.read_parquet(P / "chunks.parquet")
    # one source per filing for the summary: PDF first, else HTML (all versions share filing_id)
    order = {"pdf": 0, "html": 1, "xlsx": 2}
    ch["_fmt_rank"] = ch.format.map(order)

    llm = LLM(model=config.GEN_MODEL, max_model_len=8192, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()

    def prompt(system, user):
        return tok.apply_chat_template([{"role": "system", "content": system},
                                        {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)

    # ---------------------------------------------------------------- 1. filing summaries
    filings = []
    for fid, g in ch.groupby("filing_id"):
        best = g[g._fmt_rank == g._fmt_rank.min()]
        meta = best.iloc[0]
        header = f"{meta.company} | {meta.doc_type} | period {meta.period}"
        filings.append((fid, prompt(SUMMARY_SYS, f"{header}\n\n{summary_input(best)}")))

    outs = llm.generate([p for _, p in filings], SamplingParams(temperature=0, max_tokens=160))
    summaries = pd.DataFrame({"filing_id": [f for f, _ in filings],
                              "doc_summary": [o.outputs[0].text.strip() for o in outs]})
    summaries.to_parquet(P / "doc_summaries.parquet", index=False)
    print(f"{len(summaries)} filing summaries written")
    print("example:", summaries.iloc[0].to_dict(), "\n")

    # ---------------------------------------------------------------- 2. table descriptions
    tables = ch[ch.chunk_type == "table"]
    prompts = [prompt(TABLE_SYS, f"Context: {r.context_header}\n\n{truncate(r.chunk_text, TABLE_WORDS_IN)}")
               for r in tables.itertuples()]
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=90))
    desc = pd.DataFrame({"chunk_id": tables.chunk_id.values,
                         "table_description": [o.outputs[0].text.strip() for o in outs]})
    desc.to_parquet(P / "table_descriptions.parquet", index=False)
    print(f"{len(desc)} table descriptions written")
    print(desc.sample(3, random_state=0).to_string(max_colwidth=160))


if __name__ == "__main__":
    main()