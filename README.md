# Grounded Conversational Chart Accessibility
A verification layer for reducing LLM hallucination in chart question answering.

An LLM answers questions about chart data. Deterministic pandas code (the **oracle**)
computes the true answer, a **verifier** compares the two, and wrong answers are
re-prompted or replaced before reaching the user. A classifier (Logistic Regression +
Random Forest, explained with SHAP) predicts hallucination risk without the data.

Dataset: [ChartQA](https://github.com/vis-nlp/ChartQA), test split.

## How to run
1. `py -3.12 -m pip install -r requirements.txt`
2. Ollama running with `llama3.2:3b` (and `qwen2.5vl:3b`, `gemma3:4b` for 03b);
   optional Gemini key in the `GEMINI_API_KEY` environment variable.
3. Run the notebooks in `notebooks/` **in order**. Every LLM reply is cached in
   `cache/llm_responses.jsonl`, so reruns are instant and make no new calls.

## Notebooks
| # | Notebook | What it does | Produces |
|---|---|---|---|
| 01 | Data + oracle | Load ChartQA, oracle, validate vs human answers, sample 200 questions | `outputs/eval_sample.csv` |
| 02 | LLM answers | Llama 3.2 3B + Gemini Flash-Lite, short + sentence style (800 answers) | `outputs/llm_answers.csv` |
| 03 | Verification | Verify, re-prompt, template fallback, human audit of the verifier | `outputs/verified.csv` |
| 03b | Chart images (extension) | Two vision models read chart images; cross-check; layer on images | `outputs/image_pipeline.csv` |
| 04 | Classifier + SHAP | Ground-truth-free hallucination risk classifier | `outputs/classifier_predictions.csv` |
| 06 | Evaluation | Figures 1-6 and the final summary table | `outputs/figures/`, `outputs/tables/final_summary.csv` |

Section 5 (self-consistency voting) was optional and not implemented; 03b applies the
same agreement idea to chart reading.

## Code (`src/`)
`config.py` settings · `data_utils.py` loading/cleaning · `oracle.py` question typing + oracle ·
`llm.py` model calls + cache · `verify.py` the verifier · `extract.py` chart-image reading +
cross-check · `features.py` classifier features

## Known limitations
- Chart type is inferred from table shape (it sometimes labels a bar chart "pie").
- Oracle and verifier rules were developed on the same data they are evaluated on;
  the 03b parser was refined on its 40 charts (development results).
- The "after" hallucination rate comes from a 120-answer human audit (95% upper bound reported).
- Two text LLMs, two small vision models, 200 questions from one dataset.
