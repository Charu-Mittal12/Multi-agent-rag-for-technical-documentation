"""Answer-level metrics: citation parsing, abstention, LLM-judge correctness, aggregation."""
from __future__ import annotations
import json, re
from collections import defaultdict
from .answer import complete

# Accepts [manual p.N] and the bracket styles some models emit instead: 【manual p.N】, (manual p.N)
CITE_RE = re.compile(r"[\[【(（]\s*([^\[\]【】()（）|]+?)\s+p\.\s*(\d+)\s*[\]】)）]")
ABSTAIN = "not enough evidence"

JUDGE_PROMPT = """You grade answers to questions about a technical manual.

Question: {question}
Reference answer (ground truth from the manual): {reference}
Candidate answer: {candidate}

Ignore citation markers like [manual p.3] and ignore extra safety remarks. Judge only whether the
candidate conveys the key facts of the reference.
- correct: all key facts of the reference are present and nothing contradicts it
- partial: some key facts present, or key facts present with a contradicting detail
- incorrect: wrong, missing the key facts, or says the evidence is insufficient

Reply with JSON only: {{"verdict": "correct" | "partial" | "incorrect", "reason": "<one short sentence>"}}"""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\\n", " ").replace("\n", " ")).strip()


def tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", _norm(s).lower())


def parse_citations(text: str) -> set[tuple[str, int]]:
    return {(m.group(1).strip().removesuffix(".pdf"), int(m.group(2))) for m in CITE_RE.finditer(text)}


def is_abstention(text: str) -> bool:
    return ABSTAIN in text.lower()


def token_recall(reference: str, candidate: str) -> float:
    ref, cand = tokens(reference), set(tokens(candidate))
    return sum(t in cand for t in ref) / len(ref) if ref else 0.0


def judge(question: str, reference: str, candidate: str, judge_cfg: dict) -> dict:
    if judge_cfg["provider"] == "mock":  # offline heuristic, plumbing tests only
        r = token_recall(reference, candidate)
        return {"verdict": "correct" if r >= 0.6 else "partial" if r >= 0.3 else "incorrect", "reason": "mock"}
    raw = complete(JUDGE_PROMPT.format(question=question, reference=_norm(reference), candidate=candidate),
                   judge_cfg, temperature=0.0, max_tokens=1500)   # reasoning models need room before the JSON
    m = re.search(r"\{.*?\}", raw, re.S)
    try:
        d = json.loads(m.group(0))
        v = str(d.get("verdict", "")).lower()
        if v in ("correct", "partial", "incorrect"):
            return {"verdict": v, "reason": d.get("reason", "")}
    except Exception:
        pass
    return {"verdict": "unparsed", "reason": raw[:120]}


def evaluate_question(q: dict, hits, answer_text: str, judge_cfg: dict) -> dict:
    gold = {(g["manual"], int(g["page"])) for g in q.get("gold", [])}
    retrieved = {(c.manual, c.page) for c in hits}
    cites = parse_citations(answer_text)
    abstained = is_abstention(answer_text)
    row = {
        "id": q.get("id"), "category": q.get("category", q.get("type", "?")),
        "question": q["question"], "answerable": bool(gold), "answer": answer_text,
        "abstained": abstained, "n_cites": len(cites),
        "cite_valid": bool(cites) and cites <= retrieved,        # every cited page was actually retrieved
        "retrieval_hit": bool(gold & retrieved) if gold else None,
        "cited_gold": bool(gold & cites) if gold else None,
    }
    if gold:
        ref = q.get("answer", "")
        row["token_recall"] = round(token_recall(ref, answer_text), 3) if ref else None
        j = judge(q["question"], ref, answer_text, judge_cfg) if ref else {"verdict": "n/a", "reason": ""}
        row["verdict"], row["judge_reason"] = j["verdict"], j["reason"]
    return row


def _rate(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarize(rows: list[dict]) -> list[dict]:
    """One summary row per category plus 'ALL answerable' and 'unanswerable'."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["category"]].append(r)
    out = []
    for cat, rs in sorted(groups.items()):
        if not rs[0]["answerable"]:
            out.append({"group": cat, "n": len(rs),
                        "abstain_ok": _rate([{"v": r["abstained"]} for r in rs], "v"),
                        "hallucination": round(1 - sum(r["abstained"] for r in rs) / len(rs), 3)})
        else:
            for r in rs:
                r["_strict"] = r.get("verdict") == "correct"
                r["_lenient"] = r.get("verdict") in ("correct", "partial")
                r["_false_abstain"] = r["abstained"]
            out.append({"group": cat, "n": len(rs), "correct": _rate(rs, "_strict"),
                        "correct+partial": _rate(rs, "_lenient"), "token_recall": _rate(rs, "token_recall"),
                        "retrieval_hit": _rate(rs, "retrieval_hit"), "cite_valid": _rate(rs, "cite_valid"),
                        "cited_gold": _rate(rs, "cited_gold"), "false_abstain": _rate(rs, "_false_abstain")})
    ans = [r for r in rows if r["answerable"]]
    if ans:
        out.append({"group": "ALL answerable", "n": len(ans), "correct": _rate(ans, "_strict"),
                    "correct+partial": _rate(ans, "_lenient"), "token_recall": _rate(ans, "token_recall"),
                    "retrieval_hit": _rate(ans, "retrieval_hit"), "cite_valid": _rate(ans, "cite_valid"),
                    "cited_gold": _rate(ans, "cited_gold"), "false_abstain": _rate(ans, "_false_abstain")})
    return out
