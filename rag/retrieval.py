"""Hybrid retriever: BM25 + dense (bge-m3) fused with Reciprocal Rank Fusion,
plus rule-based metadata filters, acronym expansion, a table boost for numeric
questions and an optional cross-encoder reranker.

Every feature can be switched off in RetrievalConfig - that is how the ablation works.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config, query_meta

RRF_K = 60


@dataclass
class RetrievalConfig:
    name: str = "full"
    variant: str = "header"     # which embedding/text variant: plain | header | sac
    dense: bool = True
    bm25: bool = True
    filters: bool = True        # company hard filter + year / quarter boost
    acronyms: bool = True
    table_boost: bool = True    # 'precision mode' for numeric questions
    rerank: bool = False
    rerank_keep_meta: bool = False   # re-apply the metadata boosts AFTER reranking
    bm25_weight: float = 1.0         # weight of the BM25 ranking in the fusion (dense = 1.0)
    k: int = 10
    pool: int = 100             # candidates per retriever before fusion
    rerank_pool: int = 30

    def as_params(self):
        return asdict(self)


class HybridRetriever:
    def __init__(self, index_dir=None, device=None):
        index_dir = index_dir or (config.PROC / "index")
        self.chunks = pd.read_parquet(index_dir / "chunks.parquet")
        self.index_dir = index_dir
        self._emb, self._bm25 = {}, {}
        self._embedder = self._reranker = None
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        self.fy = self.chunks.fiscal_year.to_numpy()
        self.is_table = (self.chunks.chunk_type == "table").to_numpy()
        self.is_quarterly = self.chunks.doc_type.isin(["10Q", "EARNINGS"]).to_numpy()

    # ------------------------------------------------------------ lazy loading
    def emb(self, variant):
        if variant not in self._emb:
            self._emb[variant] = np.load(self.index_dir / f"emb_{variant}.npy").astype(np.float32)
        return self._emb[variant]

    def bm25(self, variant):
        if variant not in self._bm25:
            import bm25s
            texts = self.chunks[f"text_{variant}"].fillna("").tolist()
            r = bm25s.BM25()
            r.index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)
            self._bm25[variant] = r
        return self._bm25[variant]

    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(config.EMBED_MODEL, device=self.device)
        return self._embedder

    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(config.RERANK_MODEL, device=self.device, max_length=512)
        return self._reranker

    def _boosts(self, cfg, meta):
        b = np.zeros(len(self.chunks), dtype=np.float32)
        if cfg.filters and meta["years"]:
            b[np.isin(self.fy, meta["years"])] += 0.4 / RRF_K
        if cfg.filters and meta["quarter"]:
            b[self.is_quarterly] += 0.3 / RRF_K
        if cfg.table_boost and meta["numeric"]:
            b[self.is_table] += 0.3 / RRF_K
        return b

    # ------------------------------------------------------------ search
    def search(self, question: str, cfg: RetrievalConfig = RetrievalConfig(), meta: dict | None = None):
        meta = meta or query_meta.analyse(question)
        q = query_meta.expand_acronyms(question) if cfg.acronyms else question
        n = len(self.chunks)

        # metadata filter (hard on company, falls back to everything if it leaves too little)
        allowed = np.ones(n, dtype=bool)
        if cfg.filters and meta["tickers"]:
            allowed = self.chunks.ticker.isin(meta["tickers"]).to_numpy()
            if allowed.sum() < cfg.k:
                allowed = np.ones(n, dtype=bool)

        fused = np.zeros(n, dtype=np.float32)
        if cfg.dense:
            qv = self.embedder().encode([q], normalize_embeddings=True)[0].astype(np.float32)
            scores = self.emb(cfg.variant) @ qv
            scores[~allowed] = -np.inf
            top = np.argsort(-scores)[:cfg.pool]
            fused[top] += 1.0 / (RRF_K + np.arange(1, len(top) + 1))
        if cfg.bm25:
            import bm25s
            r = self.bm25(cfg.variant)
            tokens = bm25s.tokenize([q], stopwords="en", return_ids=False, show_progress=False)
            vocab = getattr(r, "vocab_dict", None)
            tokens = [[t for t in tokens[0] if vocab is None or t in vocab]]   # unknown words would crash
            if tokens[0]:
                ids, _ = r.retrieve(tokens, k=min(cfg.pool * 10, n), show_progress=False)
                ids = [int(i) for i in ids[0] if allowed[i]][:cfg.pool]
                fused[ids] += cfg.bm25_weight / (RRF_K + np.arange(1, len(ids) + 1))

        # soft boosts (rule-based): right year, right filing kind, tables for numeric questions
        boost = self._boosts(cfg, meta)
        fused[fused > 0] += boost[fused > 0]

        cand = np.argsort(-fused)[:max(cfg.k, cfg.rerank_pool if cfg.rerank else cfg.k)]
        cand = [i for i in cand if fused[i] > 0]

        if cfg.rerank and cand:
            texts = self.chunks.text_header.iloc[cand].tolist()
            rr = self.reranker().predict([(question, t) for t in texts], batch_size=16)
            order = np.argsort(-rr)
            if cfg.rerank_keep_meta:
                # the cross-encoder only reads text, so it happily promotes a near-identical chunk
                # from last year's filing. Turn its ranking into RRF scores and add the boosts again.
                rr_score = np.empty(len(cand))
                rr_score[order] = 1.0 / (RRF_K + np.arange(1, len(cand) + 1))
                order = np.argsort(-(rr_score + boost[cand]))
            cand = [cand[i] for i in order]

        hits = self.chunks.iloc[cand[:cfg.k]].copy()
        hits["rank"] = range(1, len(hits) + 1)
        hits["score"] = fused[cand[:cfg.k]]
        return hits