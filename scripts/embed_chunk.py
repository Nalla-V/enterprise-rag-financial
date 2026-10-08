"""
Step 6: Embed every chunk with bge-m3 (ALICE GPU), in three variants for the ablation.

  plain   : chunk text only                                   (baseline)
  header  : context header + chunk text (+ table description) (metadata-aware chunk)
  sac     : filing summary + header + chunk text               (Summary-Augmented Chunking)

Input : chunks.parquet, doc_summaries.parquet, table_descriptions.parquet
Output: data/processed/index/chunks.parquet        (chunks + summaries/descriptions + embed texts)
        data/processed/index/emb_<variant>.npy     (float16, rows aligned with chunks.parquet)

Usage (GPU job):  python -u scripts/embed_chunks.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import config  # noqa: E402

VARIANTS = ["plain", "header", "sac"]


def build_texts(ch: pd.DataFrame) -> pd.DataFrame:
    desc = ch.table_description.fillna("")
    body = np.where(ch.chunk_type == "table", desc + "\n" + ch.chunk_text, ch.chunk_text)
    ch["text_plain"] = ch.chunk_text
    ch["text_header"] = ch.context_header + "\n" + body
    ch["text_sac"] = ch.doc_summary.fillna("") + "\n" + ch.context_header + "\n" + body
    return ch


def main():
    from sentence_transformers import SentenceTransformer

    P = config.PROC
    out = P / "index"
    out.mkdir(exist_ok=True)

    ch = pd.read_parquet(P / "chunks.parquet")
    for name, key in [("doc_summaries.parquet", "filing_id"), ("table_descriptions.parquet", "chunk_id")]:
        if (P / name).exists():
            ch = ch.merge(pd.read_parquet(P / name), on=key, how="left")
        else:
            print(f"! {name} missing - run enrich_with_qwen.py first; continuing without it")
    for col in ["doc_summary", "table_description"]:
        if col not in ch:
            ch[col] = None
    ch = build_texts(ch).reset_index(drop=True)
    ch.to_parquet(out / "chunks.parquet", index=False)

    # ---- input check: did the merges work?
    n_tab = (ch.chunk_type == "table").sum()
    print(f"{len(ch)} chunks loaded ({n_tab} tables)")
    print(f"  with filing summary    : {ch.doc_summary.notna().mean():.1%} of chunks")
    print(f"  with table description : {ch.table_description.notna().sum()} of {n_tab} tables")
    for v in VARIANTS:
        print(f"  avg words, {v:<6}: {ch[f'text_{v}'].str.split().str.len().mean():.0f}")
    print("\nexample 'sac' text (first 400 chars):\n" + ch.text_sac.iloc[len(ch) // 2][:400] + "\n", flush=True)

    t0 = time.time()
    model = SentenceTransformer(config.EMBED_MODEL, device="cuda", model_kwargs={"torch_dtype": "float16"})
    model.max_seq_length = 1024
    print(f"model {config.EMBED_MODEL} loaded in {time.time() - t0:.0f}s "
          f"(dim {model.get_sentence_embedding_dimension()}, max_seq {model.max_seq_length})", flush=True)

    for v in VARIANTS:
        t0 = time.time()
        print(f"[{v}] embedding {len(ch)} texts ...", flush=True)
        texts = ch[f"text_{v}"].tolist()
        # sort by length -> far less padding -> much faster
        order = np.argsort([len(t) for t in texts])
        emb = model.encode([texts[i] for i in order], batch_size=64, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)
        full = np.empty_like(emb)
        full[order] = emb
        np.save(out / f"emb_{v}.npy", full.astype(np.float16))
        print(f"[{v}] done: {full.shape} in {time.time() - t0:.0f}s -> {out / f'emb_{v}.npy'}", flush=True)

    # ---- sanity check: nearest neighbour of one chunk should come from the same filing
    e = np.load(out / "emb_header.npy").astype(np.float32)
    i = len(ch) // 2
    nn = np.argsort(-(e @ e[i]))[1:6]
    same = (ch.filing_id.iloc[nn] == ch.filing_id.iloc[i]).mean()
    print(f"\nsanity check ('header'): chunk {ch.chunk_id.iloc[i]}")
    print(f"  top-5 neighbours from same filing: {same:.0%}")
    print(ch.iloc[nn][["chunk_id", "context_header"]].to_string(index=False, max_colwidth=80))


if __name__ == "__main__":
    main()