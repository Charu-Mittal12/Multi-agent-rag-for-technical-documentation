# Evaluation

Data: **TechManualQA-350** (CC-BY-4.0, LLM-generated questions, filtered and partly human-audited). Eight of the ten manuals
were usable (Hilti TE 60-AVR and the Miele heat-pump dryer are missing): **276 questions = 236 answerable + 40 unanswerable**.
Seven categories: Specification Lookup, Location/Definition, Procedural Step, Safety, Tool/Material, Conditional Logic, Unanswerable.
Gold = (manual, page) plus a short evidence snippet. For 51 questions (DeWalt, Electrolux) the gold page was re-derived by locating
the evidence text. At least one gold label is probably wrong (DeWalt Q009).

## Level 1 - retriever alone (no LLM), n = 236
Page level (a retrieved chunk counts if its page is a gold page):
| Config | recall@5 |
|---|---|
| BM25 / BM25 + rerank | 0.869 / 0.924 |
| Dense / dense + rerank | 0.890 / 0.945 |
| Hybrid / **hybrid + rerank** | 0.890 / **0.953** |

Hybrid + rerank: recall@1 0.725, recall@3 0.928, recall@5 0.953, MRR 0.828, nDCG@10 0.866.
Strict chunk level (chunk must be on a gold page *and* contain the evidence): recall@5 0.928, nDCG@10 0.827, precision@5 0.201
(low because most questions have a single relevant chunk, so at most 1 of 5 can be relevant).
11 retrieval failures remain (DeWalt 3, Electrolux 2, Prusa 3, Dyson/Makita/Miele 1 each).
Negative result: an LLM-free router that picks the manual first (84% accurate) *lowered* recall@5 to 0.809; a perfect router would give 0.958.
The numbered-legend chunk type was added after reading failures (recall@5 0.941 -> 0.953), so that gain is optimistic.

## Level 2 - answer generation (earlier 75-question sample, gpt-oss-120b, two judges)
Hybrid vs hybrid + rerank: correct 0.88 -> 0.93 (judge 1), 0.84 -> 0.92 (judge 2), not significant; retrieval hit 0.907 -> 0.987 (p = 0.031).
Judge agreement 93-95% (Cohen's kappa 0.52-0.79; kappa is low when classes are very unbalanced).
Strict abstention prompt on the 40 unanswerable questions: hallucination 17.5-20% -> 5%, with no new false refusals.
The strict prompt was written after seeing baseline errors.

## Level 3 - full pipeline (all 276 questions)
Retriever (hybrid + rerank, top 5) -> answer agent (gpt-oss-20b, strict prompt) -> verifier agent (Qwen) -> judge (Qwen).
| Metric | Value [95% bootstrap CI] |
|---|---|
| Correct (strict) | 0.801 [0.754, 0.852] |
| Correct or partial | 0.903 [0.864, 0.936] |
| Retrieval hit | 0.941 [0.907, 0.966] |
| Valid citations / cites a gold page | 0.881 / 0.788 |
| False abstain | 0.055 [0.030, 0.085] |
| Hallucination (unanswerable) | 0.075 [0.000, 0.175] |
| Faithfulness supported / answer relevant | 0.982 / 0.973 |

Per category (correct strict): Conditional Logic 0.90, Tool/Material 0.87, Specification 0.85, Location 0.85, Safety 0.75, **Procedural 0.59**.
- **Verifier ablation (same run):** 2 vetoes; correctness 0.801 -> 0.801, hallucination 0.075 -> 0.075, false abstain 0.047 -> 0.055.
  The verifier gave no measurable gain. It does supply the faithfulness and relevance metrics.
- **Procedural steps are weakest:** judge reasons mostly say a step of a longer procedure is missing (the top 5 passages rarely hold every step).
- **Wrong-manual retrieval:** searching all manuals at once can cite another product's manual (e.g. a Bosch question answered from the Electrolux page).
- **Hand check (30 rows):** of 12 non-empty partial/incorrect verdicts I could assess, 11 were fair; the 6 "correct" verdicts checked were right.
  The 3 remaining hallucinations: 1 likely mislabelled (OMRON battery type is stated on p.4), 2 real (cross-manual mix-up, vague answer).

## Bugs found and fixed (worth reporting)
- Judge output was cut off at 200 tokens, scoring correct answers as wrong (0.72 reported vs about 0.88 true). Fixed; cached rows repaired.
- A citation parser missed `【】` brackets, making citation validity look broken.
- A reasoning model used its 800-token budget thinking and returned 8 empty answers (7 counted wrong, 1 counted as a hallucination). Answer budget raised to 2500 and empty replies are retried; affected questions were rerun.

## Limitations
One run, one answer model, one judge family (the verifier and judge are both Qwen, so faithfulness is not independent evidence).
Small unanswerable set (n = 40), so the hallucination interval is wide. Gold-label problems and two missing manuals (70 questions).
Figures are not captioned and there is no visual retrieval. Free-tier API limits forced a sample for some earlier experiments.
Differences between the earlier 120b sample results and the full 20b run are not like-for-like (different model, different question set).

## Reproduce
See the "Run the full evaluated pipeline" section of `README.md`. Every script caches per-question results in `results/` and resumes after quota limits.
