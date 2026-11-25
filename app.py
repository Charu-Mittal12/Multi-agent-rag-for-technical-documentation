"""Streamlit UI for the technical-manual RAG assistant.   Run:  streamlit run app.py"""
import os, sys, time
from pathlib import Path

import pandas as pd
import pymupdf
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from manual_rag.config import load_config
from manual_rag.index import HybridIndex
from manual_rag.retrieve import Retriever
from manual_rag.answer import answer
from manual_rag.evaluate import load_questions, evaluate
from manual_rag.pipeline import (build_index, index_exists, index_path, list_manuals, render_page)

st.set_page_config(page_title="Manual Assistant", page_icon="🛠️", layout="wide")

# ---------- styling ----------
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.hero {background: linear-gradient(120deg,#4F46E5 0%,#7C3AED 55%,#06B6D4 100%);
       padding: 1.3rem 1.6rem; border-radius: 16px; color: white; margin-bottom: 1.1rem;}
.hero h1 {margin: 0; font-size: 1.7rem; color: white;}
.hero p {margin: .25rem 0 0 0; opacity: .92; font-size: .98rem;}
.chip {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem;
       font-weight: 600; margin-right: 6px; background:#EEF2FF; color:#4338CA; border:1px solid #C7D2FE;}
.chip.table {background:#ECFDF5; color:#047857; border-color:#A7F3D0;}
.chip.figure {background:#FEF3C7; color:#B45309; border-color:#FDE68A;}
.chip.page {background:#F1F5F9; color:#334155; border-color:#CBD5E1;}
.card {border: 1px solid #E5E7EB; border-radius: 12px; padding: .8rem 1rem; margin-bottom: .6rem; background: #FFFFFF;}
.small {color:#6B7280; font-size:.85rem;}
[data-testid="stSidebar"] {background: #F5F6FA;}
</style>
""", unsafe_allow_html=True)

EXAMPLES = [
    "The status LED is blinking red three times, what does it mean?",
    "What does error E-417 mean?",
    "How do I reset the router to factory defaults?",
]
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}
DEFAULT_MODEL = {"anthropic": "claude-haiku-5-5", "gemini": "gemini-2.0-flash", "openai": "gpt-4o-mini"}


# ---------- cached resources ----------
@st.cache_resource(show_spinner="Loading models and index ...")
def get_retriever(stamp: float):
    cfg = load_config()
    return Retriever(HybridIndex.load(index_path(cfg)), cfg)


def index_stamp(cfg) -> float:
    f = index_path(cfg) / "chunks.jsonl"
    return f.stat().st_mtime if f.exists() else 0.0


def chip(text, cls=""):
    return f'<span class="chip {cls}">{text}</span>'


cfg = load_config()
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("active_sources", [])

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### 🛠️ Manual Assistant")
    page = st.radio("Navigate", ["💬 Ask", "📚 Library", "📊 Evaluate"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Retrieval**")
    mode = st.radio("Search method", ["hybrid", "bm25", "dense"], horizontal=True,
                    help="hybrid = keyword (BM25) + semantic vectors fused with RRF")
    use_rerank = st.toggle("Cross-encoder reranker", value=True,
                           help="Re-scores the top candidates with a more accurate model")
    top_k = st.slider("Passages to use", 1, 10, cfg["retrieval"]["final_k"])
    st.markdown("**Answer writer**")
    provider = st.selectbox("LLM", ["none (extractive)", "anthropic", "gemini", "openai"])
    provider = "none" if provider.startswith("none") else provider
    llm_cfg = {"provider": provider, "model": ""}
    if provider != "none":
        llm_cfg["model"] = st.text_input("Model", DEFAULT_MODEL[provider])
        key = st.text_input("API key", type="password", help="Kept in memory for this session only.")
        if key:
            os.environ[KEY_ENV[provider]] = key
    st.divider()
    if index_exists(cfg):
        n_man = len(list_manuals(cfg))
        st.caption(f"Index ready · {n_man} manual(s)")
    else:
        st.caption("No index yet — open **Library**")
    if st.button("Clear chat", width="stretch"):
        ss.messages, ss.active_sources = [], []
        st.rerun()

st.markdown(
    '<div class="hero"><h1>Technical Manual Assistant</h1>'
    "<p>Ask a question about your equipment. Answers are grounded in manual pages, "
    "with the cited page shown on the right.</p></div>", unsafe_allow_html=True)


# =====================================================================  ASK
def show_sources_inline(sources):
    with st.expander(f"Evidence · {len(sources)} passage(s)", expanded=False):
        for s in sources:
            st.markdown(f'{chip(s["manual"] + " · p." + str(s["page"]), "page")}{chip(s["kind"], s["kind"])}',
                        unsafe_allow_html=True)
            st.markdown(s["text"] if s["kind"] == "table" else s["text"][:700])
            st.divider()


if page == "💬 Ask":
    if not index_exists(cfg):
        st.info("No index yet. Go to **📚 Library**, add manuals and click **Build index**.")
        st.stop()

    retriever = get_retriever(index_stamp(cfg))
    left, right = st.columns([3, 2], gap="large")

    with left:
        if not ss.messages:
            st.markdown("##### Try an example")
            for ex in EXAMPLES:
                if st.button(ex, key=f"ex_{ex}", width="stretch"):
                    ss["pending"] = ex
                    st.rerun()
        for m in ss.messages:
            with st.chat_message(m["role"], avatar="🧑‍🔧" if m["role"] == "user" else "🛠️"):
                st.markdown(m["content"])
                if m["role"] == "assistant":
                    st.caption(m["meta"])
                    show_sources_inline(m["sources"])

    prompt = st.chat_input("Describe the problem or ask about a part, error code or procedure ...")
    prompt = prompt or ss.pop("pending", None)
    if prompt:
        with left:
            with st.chat_message("user", avatar="🧑‍🔧"):
                st.markdown(prompt)
            with st.chat_message("assistant", avatar="🛠️"):
                with st.spinner("Searching manuals ..."):
                    t0 = time.time()
                    hits = retriever.search(prompt, mode=mode, rerank=use_rerank, k=top_k)
                    t_ret = time.time() - t0
                with st.spinner("Writing answer ..."):
                    t1 = time.time()
                    try:
                        text = answer(prompt, hits, llm_cfg)
                    except Exception as e:  # missing key, network, etc.
                        text = f"⚠️ The LLM call failed ({type(e).__name__}: {e}). Showing evidence only."
                    t_ans = time.time() - t1
                meta = (f"{mode}{' + rerank' if use_rerank else ''} · retrieval {t_ret:.2f}s · "
                        f"answer {t_ans:.2f}s · "
                        f"{'extractive (no LLM)' if provider == 'none' else provider}")
                srcs = [c.to_dict() for c in hits]
                st.markdown(text)
                st.caption(meta)
                show_sources_inline(srcs)
        ss.messages += [{"role": "user", "content": prompt},
                        {"role": "assistant", "content": text, "meta": meta, "sources": srcs}]
        ss.active_sources = srcs

    with right:
        st.markdown("##### Source viewer")
        srcs = ss.active_sources
        if not srcs:
            st.markdown('<div class="card small">Cited manual pages will appear here, with the '
                        "matching table or paragraph highlighted.</div>", unsafe_allow_html=True)
        else:
            labels = [f'{s["manual"]} · p.{s["page"]} · {s["kind"]}' for s in srcs]
            choice = st.selectbox("Cited passage", range(len(srcs)), format_func=lambda i: labels[i],
                                  label_visibility="collapsed")
            from manual_rag.parse import Chunk
            try:
                st.image(render_page(cfg, Chunk(**srcs[choice])), width="stretch")
            except Exception as e:
                st.warning(f"Could not render the page: {e}")

# =====================================================================  LIBRARY
elif page == "📚 Library":
    st.subheader("Manual library")
    up = st.file_uploader("Add PDF manuals", type="pdf", accept_multiple_files=True)
    mdir = ROOT / cfg["paths"]["manuals_dir"]
    mdir.mkdir(parents=True, exist_ok=True)
    if up:
        for f in up:
            (mdir / f.name).write_bytes(f.getbuffer())
        st.success(f"Saved {len(up)} file(s). Click **Build index** to include them.")
    pdfs = list_manuals(cfg)
    if pdfs:
        rows = []
        for p in pdfs:
            with pymupdf.open(p) as d:
                rows.append({"manual": p.stem, "pages": len(d), "size (KB)": round(p.stat().st_size / 1024)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No manuals yet. Upload PDFs above.")
    if st.button("🔄 Build index", type="primary", disabled=not pdfs):
        with st.status("Building index ...", expanded=True) as status:
            stats = build_index(cfg, progress=st.write)
            status.update(label=f"Indexed {stats['chunks']} chunks in {stats['seconds']}s", state="complete")
        st.dataframe(pd.DataFrame(stats["manuals"]).T, width="stretch")
        get_retriever.clear()
    st.caption("Tables are extracted as markdown. Figures are placeholders until the VLM-caption layer is added.")

# =====================================================================  EVALUATE
else:
    st.subheader("Retrieval evaluation")
    qfile = ROOT / cfg["paths"]["eval_file"]
    if not index_exists(cfg) or not qfile.exists():
        st.info("Needs an index and a question file at `data/eval/questions.jsonl`.")
        st.stop()
    qs = load_questions(qfile)
    kinds = pd.Series([q.get("type", "?") for q in qs]).value_counts()
    c1, c2 = st.columns([1, 2])
    c1.metric("Questions", len(qs))
    c1.metric("Answerable", sum(bool(q.get("gold")) for q in qs))
    c2.bar_chart(kinds)
    if st.button("▶ Run ablation (6 configurations)", type="primary"):
        retriever = get_retriever(index_stamp(cfg))
        rows = []
        bar = st.progress(0.0)
        combos = [(m, r) for m in ["bm25", "dense", "hybrid"] for r in (False, True)]
        for i, (m, r) in enumerate(combos):
            s = evaluate(retriever, qs, m, r)
            rows.append({"config": f"{m}{' + rerank' if r else ''}", **{k: v for k, v in s.items() if k != "n"}})
            bar.progress((i + 1) / len(combos))
        df = pd.DataFrame(rows).set_index("config")
        ss["ablation"] = df
    if "ablation" in ss:
        df = ss["ablation"]
        st.dataframe(df.style.highlight_max(axis=0, color="#C7F9CC"), width="stretch")
        st.bar_chart(df[["mrr", "ndcg@10"]])
        st.download_button("Download CSV", df.to_csv().encode(), "ablation.csv", "text/csv")
    st.caption("Public benchmark runs: see EVALUATION.md (`python scripts/eval_beir.py`).")
