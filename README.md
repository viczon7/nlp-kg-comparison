# Structured Knowledge versus Parameters

Research poster, NLP module, Universität Trier. Victoria Zonova, Emrehan Dalaman.

A knowledge graph (338 triples) built from the abstract, introduction and
experimental-setup section of 23 multilingual-NLP papers (mostly TACL / CL 2025–26, plus Russian & Turkish
benchmarks and Univ. Trier papers), used to test whether Gemma 4 E2B + KG gets close to Gemma 4 12B without it
on 182 multiple-choice questions. Poster: [`poster/poster.pdf`](poster/poster.pdf).

| | All (182) | Old papers (108) | New papers (74) | Answer in the retrieved facts (49) |
|---|---|---|---|---|
| Gemma 4 E2B | 35% | 39% | 28% | 35% |
| Gemma 4 E2B + KG | 49% | 46% | 53% | 84% |
| Gemma 4 12B | 55% | 52% | 61% | 59% |

## Reproduce

| Step | Command | Output |
|---|---|---|
| 1. Get text | `pip install pymupdf` then `python scripts/1_prepare.py` | `data/papers.csv`, `data/passages.jsonl` (116 passages, ~36k words); the PDFs go to `data/pdf/` (not in the repo) |
| 2. Extract triples | `pip install ollama`, `ollama pull qwen3.5:9b`, then `python scripts/2_extract.py` (or paste it into a Kaggle notebook with a free GPU) | `kg/triples_to_curate.csv` |
| 3. Curate | open the CSV, write `1`/`0` in `keep`, fix names | — |
| 4. Build KG | `python scripts/3_finalize.py` | `kg/kg_final.csv` |
| 5. Draft questions | `python scripts/4_draft_questions.py` (from the passages, not the KG), then check by hand | `eval/questions_draft.csv` |
| 6. Evaluate | save the checked questions as `eval/questions_final.csv`; `pip install pandas ollama`, `ollama pull gemma4:e2b` and `gemma4:12b`, then `python scripts/5_evaluate.py` (or paste it into Kaggle) | `results.csv` (put it into `eval/`) |
| 7. Analyze | `python scripts/6_analyze.py` | all numbers on the poster: accuracy by old/new, no-answer rates, answer in the retrieved facts or not |

**Fixed choices:** extractor `qwen3.5:9b` (Ollama, temperature 0, seed 42),
zero-shot prompt (no example); triples are plain (head, relation, tail)
with 8 relations, and both head and tail must occur in the passage.
"old" / "new" = paper first public before / after Gemma 4's knowledge cutoff (Jan 2025).

**Files that document the manual work:** `kg/triples_to_curate.csv` (all 576 extracted triples with the
`keep` decision), `eval/questions_final.csv` (all 221 drafted questions with `keep`),
`eval/results.csv` (every model reply, as produced by the run).

