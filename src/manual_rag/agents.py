"""End-to-end multi-agent assistant.

  Retriever agent  -> hybrid (BM25 + dense, RRF) + cross-encoder rerank, top-k passages
  Answer agent     -> strict-prompt cited answer, abstains when the specific fact is missing
  Verifier agent   -> a second model (different family) checks the answer against the passages:
                      faithfulness (supported/partial/unsupported) + answer relevance.
                      An 'unsupported' answer is replaced by an abstention; every verdict is kept in the trace.

Citations are also checked deterministically (every cited page must be among the retrieved ones).
"""
from __future__ import annotations
import json, re
from .answer import answer, complete, format_context
from .answer_eval import parse_citations, is_abstention

ABSTAIN_TEXT = "Not enough evidence in the manuals."

VERIFY_PROMPT = """You audit an answer produced by a retrieval-augmented assistant for technical manuals.

Question: {question}

Retrieved excerpts (the only information the assistant was given):
{context}

Assistant's answer: {answer}

Rate two things. Ignore citation markers such as [manual p.3].
1. faithfulness - is every factual claim in the answer stated in, or directly entailed by, the excerpts?
   supported = all claims are; partial = some are not; unsupported = most are not.
2. answer_relevance - does the answer address what the question asks (whatever its correctness)?
   relevant = yes; partial = only partly; irrelevant = no.

Reply with JSON only: {{"faithfulness": "supported" | "partial" | "unsupported", "answer_relevance": "relevant" | "partial" | "irrelevant", "reason": "<one short sentence>"}}"""


def parse_verdict(raw: str):
    m = re.search(r"\{.*\}", raw or "", re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if d.get("faithfulness") in ("supported", "partial", "unsupported") and \
               d.get("answer_relevance") in ("relevant", "partial", "irrelevant"):
                return d
        except Exception:
            pass
    return None


class Assistant:
    def __init__(self, retriever, answer_cfg: dict, verifier_cfg: dict | None = None,
                 top_k: int = 5, mode: str = "hybrid", rerank: bool = True):
        self.r, self.ans_cfg, self.ver_cfg = retriever, {**answer_cfg, "prompt": "strict"}, verifier_cfg
        self.top_k, self.mode, self.rerank = top_k, mode, rerank

    def verify(self, question, hits, text) -> dict:
        if not self.ver_cfg or self.ver_cfg.get("provider") in (None, "none"):
            return {"faithfulness": None, "answer_relevance": None, "reason": "verifier off"}
        if self.ver_cfg["provider"] == "mock":
            return {"faithfulness": "supported", "answer_relevance": "relevant", "reason": "mock"}
        raw = complete(VERIFY_PROMPT.format(question=question, context=format_context(hits), answer=text),
                       self.ver_cfg, temperature=0.0, max_tokens=1500)
        return parse_verdict(raw) or {"faithfulness": "unparsed", "answer_relevance": "unparsed", "reason": raw[:100]}

    def run(self, question: str) -> dict:
        hits = self.r.search(question, mode=self.mode, rerank=self.rerank, k=self.top_k)
        draft = answer(question, hits, self.ans_cfg)
        trace = {"draft": draft, "verifier": None, "overridden": False}
        final = draft
        if not is_abstention(draft):                      # nothing to verify when the agent abstained
            v = self.verify(question, hits, draft)
            trace["verifier"] = v
            if v.get("faithfulness") == "unsupported":    # verifier veto -> abstain rather than risk a hallucination
                final, trace["overridden"] = ABSTAIN_TEXT, True
        cites = parse_citations(final)
        retrieved = {(c.manual, c.page) for c in hits}
        return {"question": question, "answer": final, "hits": hits, "trace": trace,
                "citations": sorted(cites), "citations_valid": bool(cites) and cites <= retrieved}
