"""Answer generation. provider='none' returns an extractive, cited answer (no API key needed)."""
from __future__ import annotations
import os

PROMPT = """You are a technical-support assistant. Answer ONLY from the manual excerpts below.
Give ordered steps. After each step cite its source as [manual p.N]. If the excerpts do not
contain the answer, reply exactly: "Not enough evidence in the manuals." Add a safety warning
if the excerpts contain one.

Excerpts:
{context}

Question: {question}
"""


def format_context(chunks) -> str:
    return "\n\n".join(f"[{c.manual} p.{c.page} | {c.kind}]\n{c.text}" for c in chunks)


def answer(question: str, chunks, llm_cfg: dict) -> str:
    provider = llm_cfg.get("provider", "none")
    if provider == "none" or not chunks:
        if not chunks:
            return "Not enough evidence in the manuals."
        lines = [f"- {' '.join(c.text.split())[:260]} [{c.manual} p.{c.page}]" for c in chunks[:3]]
        return "Top matching passages (extractive mode, no LLM configured):\n" + "\n".join(lines)

    prompt = PROMPT.format(context=format_context(chunks), question=question)
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(model=llm_cfg["model"], max_tokens=800,
                                     messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    if provider == "openai":
        from openai import OpenAI
        r = OpenAI().chat.completions.create(model=llm_cfg["model"],
                                             messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content
    if provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        return genai.GenerativeModel(llm_cfg["model"]).generate_content(prompt).text
    raise ValueError(f"unknown provider {provider}")
