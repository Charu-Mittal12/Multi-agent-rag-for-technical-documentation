import os, sys
from pathlib import Path
os.environ["MANUAL_RAG_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from manual_rag.index import tokenize, rrf


def test_error_code_kept_whole():
    assert "e-417" in tokenize("What does error E-417 mean?")


def test_rrf_prefers_items_ranked_high_in_both_lists():
    fused = dict(rrf([[1, 2, 3], [2, 1, 4]]))
    assert fused[1] > fused[3] and fused[2] > fused[4]


def test_citation_parsing_and_abstention():
    from manual_rag.answer_eval import parse_citations, is_abstention, token_recall
    assert parse_citations("Press OK [bosch_oven p.12]. Then wait [bosch_oven.pdf p. 13].") == {("bosch_oven", 12), ("bosch_oven", 13)}
    assert is_abstention("Not enough evidence in the manuals.")
    assert not is_abstention("Hold the button for ten seconds.")
    assert token_recall("at least 10 cm", "keep at least 10 cm away") == 1.0


def test_placeholder_figures_not_indexed():
    from manual_rag.parse import Chunk
    from manual_rag.pipeline import index_worthy
    assert not index_worthy(Chunk("a", "m", 1, "figure", "[figure on page 1 of m; caption pending]"))
    assert index_worthy(Chunk("b", "m", 1, "figure", "Diagram of the drip tray with labelled screws"))
    assert index_worthy(Chunk("c", "m", 1, "text", "caption pending approval"))


def test_parse_citations_alternate_brackets():
    from manual_rag.answer_eval import parse_citations
    t = "Use 40 W【bosch_oven p.21】 and also [miele_dishwasher p.4] or (makita_drill p. 2)."
    assert parse_citations(t) == {("bosch_oven", 21), ("miele_dishwasher", 4), ("makita_drill", 2)}


def test_manual_restriction_and_router():
    import os
    os.environ["MANUAL_RAG_OFFLINE"] = "1"
    from manual_rag.config import load_config
    from manual_rag.embed import make_embedder
    from manual_rag.index import HybridIndex
    from manual_rag.parse import Chunk
    from manual_rag.retrieve import Retriever, manual_of
    cfg = load_config()
    emb = make_embedder(cfg["embeddings"])
    chunks = [Chunk(id=f"a{i}", manual="oven", page=i, kind="text", text=t) for i, t in
              enumerate(["oven cleaning pyrolytic self-cleaning surfaces", "oven light bulb replacement 40 watt",
                         "oven child lock activation"])] + \
             [Chunk(id=f"b{i}", manual="drill", page=i, kind="text", text=t) for i, t in
              enumerate(["drill chuck torque setting clutch", "drill battery charging indicator", "drill speed gear selector"])]
    r = Retriever(HybridIndex(chunks, emb.embed_documents([c.text for c in chunks])), cfg)
    assert r.route("replace the oven light bulb") == "oven"
    assert r.route("set the drill clutch torque") == "drill"
    assert {c.manual for c in r.search("battery", mode="hybrid", rerank=False, k=5, manual="oven")} <= {"oven"}
    assert manual_of({"id": "bosch_oven_Q003"}) == "bosch_oven"


def test_legend_chunk():
    from manual_rag.parse import legend_chunk_text
    body = ("WARNING: Read all instructions.\n\nFig. A\n\n7\n\n12\n\n1\n2\n\n"
            "1 Trigger switch\n\n2 Main handle\n\n3 Blade lock\n\n4 End cap\n\n6 Bevel adjustment lever\n")
    t = legend_chunk_text(body)
    assert t and "Fig. A" in t and "2 = Main handle" in t and "6 = Bevel adjustment lever" in t
    # ordinary numbered steps and pages without a figure label are not legends
    steps = "Fig. B\n\n1. Remove the screw\n\n2. Clean the guard\n\n3. Fit the blade\n\n4. Tighten\n"
    assert legend_chunk_text(steps) is None
    assert legend_chunk_text("1 Alpha\n2 Beta\n3 Gamma\n4 Delta") is None   # no figure label


def test_strict_prompt_selected():
    from manual_rag import answer as A
    seen = {}
    A.complete = lambda prompt, cfg, **kw: seen.setdefault("p", prompt) or "ok"
    class C: manual, page, kind, text = "m", 1, "text", "x"
    A.answer("q?", [C()], {"provider": "openai", "model": "x", "prompt": "strict"})
    assert "Step 1" in seen["p"]
    seen.clear()
    A.answer("q?", [C()], {"provider": "openai", "model": "x"})
    assert "Step 1" not in seen["p"]


def test_chunk_level_relevance_and_strict_metrics():
    import os
    os.environ["MANUAL_RAG_OFFLINE"] = "1"
    from manual_rag.config import load_config
    from manual_rag.embed import make_embedder
    from manual_rag.index import HybridIndex
    from manual_rag.parse import Chunk
    from manual_rag.retrieve import Retriever
    from manual_rag.evaluate import chunk_relevant, evaluate
    cfg = load_config(); emb = make_embedder(cfg["embeddings"])
    chunks = [Chunk(id="1", manual="oven", page=3, kind="text", text="General warnings and safety symbols for the oven."),
              Chunk(id="2", manual="oven", page=3, kind="text", text="Replace the cooking compartment light bulb: 40 watt, 230 V."),
              Chunk(id="3", manual="oven", page=5, kind="text", text="Cleaning the oven door glass with a soft cloth.")]
    r = Retriever(HybridIndex(chunks, emb.embed_documents([c.text for c in chunks])), cfg)
    q = {"id": "oven_Q001", "question": "light bulb replacement oven", "gold": [{"manual": "oven", "page": 3}],
         "answer": "40 watt", "evidence": "40 watt, 230 V"}
    assert chunk_relevant(chunks[1], q) is True       # right page, contains the evidence
    assert chunk_relevant(chunks[0], q) is False      # right page, wrong chunk
    assert chunk_relevant(chunks[2], q) is False      # wrong page
    s = evaluate(r, [q], "bm25", False)
    assert s["recall@1"] == 1.0 and s["s_recall@1"] == 1.0 and s["s_n"] == 1 and s["unlabelled"] == 0
    # a question whose top chunk is on the gold page but lacks the evidence: page-level hit, chunk-level miss
    q2 = dict(q, question="general warnings safety symbols", evidence="40 watt, 230 V")
    s2 = evaluate(r, [q2], "bm25", False, ks=(1, 3, 5))
    assert s2["recall@1"] == 1.0 and s2["s_recall@1"] == 0.0 and s2["gap@5"] <= 0.0 + 1.0


def test_assistant_end_to_end_and_verifier_veto(monkeypatch):
    import os
    os.environ["MANUAL_RAG_OFFLINE"] = "1"
    from manual_rag.config import load_config
    from manual_rag.embed import make_embedder
    from manual_rag.index import HybridIndex
    from manual_rag.parse import Chunk
    from manual_rag.retrieve import Retriever
    from manual_rag import agents
    cfg = load_config()
    emb = make_embedder(cfg["embeddings"])
    chunks = [Chunk(id=f"a{i}", manual="oven", page=i + 1, kind="text", text=t) for i, t in
              enumerate(["oven light bulb replacement 40 watt", "oven child lock activation"])]
    r = Retriever(HybridIndex(chunks, emb.embed_documents([c.text for c in chunks])), cfg)
    out = agents.Assistant(r, {"provider": "mock"}, {"provider": "mock"}, rerank=False).run("oven light bulb")
    assert out["citations_valid"] and not out["trace"]["overridden"]
    monkeypatch.setattr(agents.Assistant, "verify",
                        lambda self, q, h, t: {"faithfulness": "unsupported", "answer_relevance": "relevant", "reason": ""})
    out = agents.Assistant(r, {"provider": "mock"}, {"provider": "mock"}, rerank=False).run("oven light bulb")
    assert out["trace"]["overridden"] and "not enough evidence" in out["answer"].lower()
    assert agents.parse_verdict('x {"faithfulness":"supported","answer_relevance":"relevant","reason":"r"}')
