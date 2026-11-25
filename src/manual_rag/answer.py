"""LLM calls. provider='none' -> extractive answer; 'mock' -> offline stub for plumbing tests."""
from __future__ import annotations
import os, time

PROMPT = """You are a technical-support assistant. Answer ONLY from the manual excerpts below.
Give ordered steps when the question is procedural. After each claim or step, cite its source as
[manual p.N] using the manual name and page shown in the excerpt header. If the excerpts do not
contain the answer, reply exactly: "Not enough evidence in the manuals." Add a safety warning
if the excerpts contain one.

Excerpts:
{context}

Question: {question}
"""


PROMPT_STRICT = """You are a technical-support assistant. Answer ONLY from the manual excerpts below.

Step 1 - decide whether the excerpts state the SPECIFIC fact the question asks for (right product, right
feature, right value). Related or similar information does not count. Information that belongs to a
different product or manual than the one the question is about does not count. A guess does not count.
Step 2 - if the specific fact is NOT stated, reply with exactly this sentence and nothing else:
Not enough evidence in the manuals.
Step 3 - otherwise answer concisely (ordered steps if the question is procedural). After each claim or step
add its source in plain square brackets using the manual name and page from the excerpt header,
for example [bosch_oven p.21]. Add a safety warning only if the excerpts contain one.

Excerpts:
{context}

Question: {question}
"""


def format_context(chunks) -> str:
    return "\n\n".join(f"[{c.manual} p.{c.page} | {c.kind}]\n{c.text}" for c in chunks)


def complete(prompt: str, llm_cfg: dict, temperature: float = 0.0, max_tokens: int = 800, retries: int = 3) -> str:
    """Single LLM call with simple retry/backoff."""
    provider, model = llm_cfg["provider"], llm_cfg.get("model", "")
    last = None
    for attempt in range(retries):
        try:
            if provider == "anthropic":
                import anthropic
                msg = anthropic.Anthropic().messages.create(
                    model=model, max_tokens=max_tokens, temperature=temperature,
                    messages=[{"role": "user", "content": prompt}])
                return msg.content[0].text
            if provider == "openai":
                from openai import OpenAI
                r = OpenAI().chat.completions.create(
                    model=model, temperature=temperature, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}])
                return r.choices[0].message.content
            if provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=os.environ["GEMINI_API_KEY"])
                cfg = genai.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
                return genai.GenerativeModel(model).generate_content(prompt, generation_config=cfg).text
            raise ValueError(f"unknown provider {provider}")
        except ValueError:
            raise
        except Exception as e:  # rate limit / transient network
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed after {retries} tries: {last}")


def answer(question: str, chunks, llm_cfg: dict) -> str:
    provider = llm_cfg.get("provider", "none")
    if not chunks:
        return "Not enough evidence in the manuals."
    if provider == "none":
        lines = [f"- {' '.join(c.text.split())[:260]} [{c.manual} p.{c.page}]" for c in chunks[:3]]
        return "Top matching passages (extractive mode, no LLM configured):\n" + "\n".join(lines)
    if provider == "mock":
        c = chunks[0]
        return f"{' '.join(c.text.split())[:200]} [{c.manual} p.{c.page}]"
    template = PROMPT_STRICT if llm_cfg.get("prompt") == "strict" else PROMPT
    return complete(template.format(context=format_context(chunks), question=question), llm_cfg)
