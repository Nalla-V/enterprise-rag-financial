"""The question-answering agent, as a LangGraph state machine.

    rewrite ─► route ─► retrieve ─┬──────────► generate ─► check ─┬─► END
                                  └─► sql ─────┘                  │
                                          ▲                       │ answer = "not found" (once):
                                          └──── retrieve (broader)┘   retry with a broader search

rewrite : follow-up -> standalone question ("And in 2021?")     (multi-turn)
          + rule-based metadata; no year given -> most recent filing, said so in the answer
route   : documents | numbers | both                            (tool selection: LLM + rule guards)
retrieve: hybrid search (config 12 of the retrieval ablation)    (always: also gives citations)
sql     : numbers tool on the Excel facts (function calling)
generate: grounded answer with [n] / [SQL] citations
check   : "not found" -> one retry without the year boost and with more chunks
"""
import re
import time
from dataclasses import replace
from typing import Any, TypedDict

from . import config, llm, query_meta
from .retrieval import HybridRetriever, RetrievalConfig
from .sql_tool import FinFactsSQL, rows_as_text

NOT_FOUND = "I cannot find this in the documents"
# metrics that are not meaningful for banks / card issuers (issue 38: the sector note alone was ignored by the 7B model)
NOT_FOR_BANKS = re.compile(r"gross margin|gross profit|operating margin|operating income|inventor|quick ratio|"
                           r"cost of (?:goods|sales)|cogs|days payable|capex|capital expenditure|capital.intensive",
                           re.IGNORECASE)
MAX_CHARS_PER_CHUNK = 2500        # keeps 8 sources well inside the 16k-token context

REWRITE_SYS = ("Rewrite the user's last message as ONE standalone question that is understandable without "
               "the conversation. Keep company names, fiscal years and metrics explicit (resolve 'it', "
               "'that', 'the same', 'and in 2021?'). Return only the question.")
ROUTE_SYS = ("Decide which tool should answer a question about company filings.\n"
             "numbers: the answer is a figure from the financial statements (revenue, net income, assets, cash "
             "flow lines, capex, ...) or a ratio computed from such figures.\n"
             "documents: explanations, drivers ('what drove ...'), risks, strategy, segments, customers, legal, "
             "geographies, acquisitions, or anything qualitative.\n"
             "both: a figure AND an explanation, or a judgement based on figures ('is X capital-intensive?').\n"
             'Reply only as JSON: {"route": "documents" | "numbers" | "both"}')
ANSWER_SYS = ("You are a financial analyst assistant answering questions about company filings.\n"
              "Rules:\n"
              "1. Use ONLY the sources. After each fact, cite the source it came from: [2], or [SQL] for figures "
              "from the statement table. Cite only sources you actually used - never a list of all sources.\n"
              "2. For any ratio, margin, growth or total, use the 'Calculated' lines from [SQL] when they are given; "
              "do not redo arithmetic. Otherwise show the formula and the numbers.\n"
              "3. Read figures per company and year carefully: never use one company's number for another.\n"
              "4. Report cash outflows (capex, dividends, buybacks) as positive amounts; keep units (USD millions).\n"
              "5. If the question asks whether a metric is useful and it is not for this type of company (e.g. "
              "gross margin, operating margin, inventory or quick ratios for a bank or card issuer), say so clearly "
              "and explain why.\n"
              "6. Only if NO source contains anything relevant, reply exactly: '" + NOT_FOUND + ".' "
              "Never write that sentence before or next to an answer.\n"
              "Be concise: the answer first, then the supporting figures.")


class State(TypedDict, total=False):
    question: str
    history: list            # [{"role": "user" | "assistant", "content": str}, ...]
    standalone: str
    meta: dict
    assumed_year: Any
    route: str
    llm_route: str
    cfg: Any
    hits: Any                # DataFrame of retrieved chunks
    tool: dict               # numbers-tool result: mode, plan, sql, rows
    answer: str
    sources: list
    attempts: int
    trace: list


class RAGAgent:
    def __init__(self, retriever: HybridRetriever | None = None, sql_tool: FinFactsSQL | None = None,
                 cfg: RetrievalConfig = RetrievalConfig(), k_answer=8, use_sql=True, use_router=True):
        self.retriever = retriever or HybridRetriever()
        self.sql_tool = sql_tool if sql_tool is not None else (FinFactsSQL() if use_sql else None)
        self.cfg, self.k_answer = replace(cfg, k=max(cfg.k, k_answer)), k_answer
        self.use_sql, self.use_router = use_sql and self.sql_tool is not None, use_router
        ch = self.retriever.chunks
        self.latest_year = ch[ch.doc_type == "10K"].groupby("ticker").fiscal_year.max().to_dict()
        self.graph = self._build()

    # ------------------------------------------------------------ nodes
    def rewrite(self, s: State):
        t0, q = time.time(), s["question"]
        if s.get("history"):
            convo = "\n".join(f"{m['role']}: {m['content']}" for m in s["history"][-6:])
            q = llm.ask(REWRITE_SYS, f"{convo}\nuser: {s['question']}", max_tokens=120)
        meta = query_meta.analyse(q)
        assumed = None
        if meta["tickers"] and not meta["years"]:          # issue 35: no year -> most recent filing
            ys = [self.latest_year[t] for t in meta["tickers"] if t in self.latest_year]
            if ys:
                assumed = max(ys)
                meta = {**meta, "years": [assumed]}
        return {"standalone": q, "meta": meta, "assumed_year": assumed, "attempts": 0, "cfg": self.cfg,
                "trace": [{"node": "rewrite", "standalone": q, "meta": meta, "sec": round(time.time() - t0, 2)}]}

    def route(self, s: State):
        t0, meta = time.time(), s["meta"]
        covered = self.use_sql and self.sql_tool.covers(meta["tickers"], meta["years"])
        llm_route = None
        if not covered:
            r = "documents"                                 # no Excel data for this company / year
        elif self.use_router:
            llm_route = (llm.ask_json(ROUTE_SYS, s["standalone"], default={}, max_tokens=30) or {}).get("route")
            r = llm_route if llm_route in ("documents", "numbers", "both") else ("both" if meta["numeric"] else "documents")
            if r == "numbers" and meta["quarter"]:
                r = "both"                                  # quarterly figures live in 10-Qs / releases (PDF)
        else:
            r = "both" if meta["numeric"] else "documents"  # rule-only router (ablation)
        return {"route": r, "llm_route": llm_route,
                "trace": s["trace"] + [{"node": "route", "route": r, "llm_route": llm_route, "covered": covered,
                                        "sec": round(time.time() - t0, 2)}]}

    def retrieve(self, s: State):
        t0 = time.time()
        hits = self.retriever.search(s["standalone"], s["cfg"], s["meta"])
        return {"hits": hits, "trace": s["trace"] + [{"node": "retrieve", "attempt": s["attempts"],
                                                      "top_filings": hits.filing_id.head(5).tolist(),
                                                      "sec": round(time.time() - t0, 2)}]}

    def sql(self, s: State):
        t0 = time.time()
        res = self.sql_tool.answer_rows(s["standalone"], s["meta"])
        return {"tool": res, "trace": s["trace"] + [{"node": "sql", "mode": res["mode"], "plan": {
            k: v for k, v in res["plan"].items() if k != "raw_plan"}, "rows": len(res["rows"]),
            "sec": round(time.time() - t0, 2)}]}

    def generate(self, s: State):
        t0 = time.time()
        blocks, sources = [], []
        for i, r in enumerate(s["hits"].head(self.k_answer).itertuples(), 1):
            blocks.append(f"[{i}] {r.context_header} ({r.format}, {r.location})\n{r.chunk_text[:MAX_CHARS_PER_CHUNK]}")
            sources.append({"n": i, "chunk_id": r.chunk_id, "filing_id": r.filing_id, "format": r.format,
                            "location": r.location, "header": r.context_header})
        tool = s.get("tool") or {}
        rows = tool.get("rows")
        if rows is not None and len(rows):
            blocks.append("[SQL] Figures from the financial statements (Excel version of the 10-K):\n" + rows_as_text(rows))
            sources.append({"n": "SQL", "mode": tool["mode"], "sql": tool.get("sql"),
                            "concepts": tool["plan"].get("concepts"),
                            "filings": sorted(set(rows.get("filing_id", rows.get("ticker", [])).astype(str)))})
        tickers = s["meta"]["tickers"]
        note = "".join(f"\n(Company context: {config.COMPANY[t]} is a {config.SECTOR[t]}.)"
                       for t in tickers if t in config.SECTOR)
        for t in tickers:
            m = NOT_FOR_BANKS.search(s["standalone"])
            if t in config.FINANCIAL and m:
                note += (f"\n(Important: {config.COMPANY[t]} is a financial institution. '{m.group(0)}' is NOT a "
                         f"standard or meaningful metric for banks and card issuers - their revenue is mainly interest "
                         f"and fees, and they have no cost of goods sold or inventory. State this clearly first; "
                         f"you may then mention the metrics such companies use instead, e.g. net interest income, "
                         f"efficiency ratio, return on equity.)")
        if s.get("assumed_year"):
            note += (f"\n(No fiscal year was given. The most recent filing, FY{s['assumed_year']}, was used - "
                     f"mention this assumption in one short sentence.)")
        context = "\n\n".join(blocks) if blocks else "(no sources)"
        answer = llm.ask(ANSWER_SYS, f"Sources:\n{context}\n\nQuestion: {s['standalone']}{note}", max_tokens=500)
        return {"answer": answer, "sources": sources,
                "trace": s["trace"] + [{"node": "generate", "n_sources": len(sources), "sec": round(time.time() - t0, 2)}]}

    def check(self, s: State):
        # retry only if the model really found nothing (issue 37: "I cannot find ... However, ..." is an answer)
        not_found = s["answer"].strip().lower().startswith(NOT_FOUND.lower()) and len(s["answer"]) < 160
        if not_found and s["attempts"] == 0:
            # broaden once: drop the year boost (keep the company filter), more candidates
            meta = {**s["meta"], "years": [], "quarter": False}
            return {"attempts": 1, "meta": meta, "cfg": replace(s["cfg"], k=s["cfg"].k + 5, pool=s["cfg"].pool * 2),
                    "trace": s["trace"] + [{"node": "check", "retry": True}]}
        return {"attempts": s["attempts"] + 10, "trace": s["trace"] + [{"node": "check", "retry": False}]}

    # ------------------------------------------------------------ graph
    def _build(self):
        from langgraph.graph import END, StateGraph

        g = StateGraph(State)
        for name in ["rewrite", "route", "retrieve", "sql", "generate", "check"]:
            g.add_node(name, getattr(self, name))
        g.set_entry_point("rewrite")
        g.add_edge("rewrite", "route")
        g.add_edge("route", "retrieve")
        g.add_conditional_edges("retrieve",
                                lambda s: "sql" if s["route"] != "documents" and s["attempts"] == 0 else "generate",
                                {"sql": "sql", "generate": "generate"})
        g.add_edge("sql", "generate")
        g.add_edge("generate", "check")
        g.add_conditional_edges("check", lambda s: "retrieve" if s["attempts"] == 1 else END,
                                {"retrieve": "retrieve", END: END})
        return g.compile()

    def ask(self, question: str, history: list | None = None) -> State:
        return self.graph.invoke({"question": question, "history": history or []})