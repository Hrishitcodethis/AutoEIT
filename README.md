# AutoEIT — Automated Scoring for the Spanish Elicited Imitation Task

**GSoC Test II: Evaluation of Transcribed Data**

This project implements an automated scoring system for the Spanish Elicited
Imitation Task (EIT) using the Ortega (2000) meaning-based rubric. The system
compares learner utterances against prompt sentences and outputs sentence-level
scores (0–4) for each utterance in the sample data.

The scoring system was designed to approximate human EIT scoring by combining
transcription preprocessing, idea unit overlap analysis, fuzzy string matching,
and rule-based scoring logic — with optional semantic similarity for borderline
adjudication.

---

## Project Structure

```
AutoEIT/
├── score_eit.py                ← Quick entry point (python score_eit.py)
├── requirements.txt
│
├── data/
│   ├── raw/                    ← Input: sample transcription Excel file
│   └── output/                 ← Generated: scored Excel, CSV, logs
│
├── src/
│   ├── rubric.py               ← Rubric constants, score descriptors, synonymous rules
│   ├── preprocessing.py        ← Text cleaning per EIT protocol
│   ├── scoring.py              ← Ortega rubric implementation + hybrid scoring
│   ├── utils.py                ← Column detection, fuzzy matching, semantic similarity
│   └── pipeline.py             ← End-to-end orchestrator
│
├── scripts/
│   └── run_scoring.py          ← CLI with flags (--no-semantic, --no-spacy, etc.)
│
└── evaluation/
    └── methodology.md          ← Full approach, evaluation, limitations, future work
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run scoring (simplest)
python score_eit.py

# Or with CLI options
python scripts/run_scoring.py --no-semantic --no-spacy

# Optional: enable POS-based content word detection
pip install spacy && python -m spacy download es_core_news_sm
```

### Output

| File | Description |
|------|-------------|
| `data/output/AutoEIT_Scored_Results.xlsx` | Original sheets with `Score` column added |
| `data/output/preprocessed_transcriptions.csv` | Cleaned text, overlap metrics, explanations |
| `data/output/scoring_log.txt` | Per-sentence scoring detail log |

---

## Scoring Method

### Ortega (2000) Rubric

| Score | Criteria |
|-------|----------|
| **4** | Exact repetition — form and meaning match stimulus exactly |
| **3** | Meaning preserved; grammar errors OK if meaning unchanged; `muy` optional; `y`/`pero` interchangeable |
| **2** | >50% idea units present; meaningful but inexact/incomplete; *when in doubt → score 2* |
| **1** | ~50% idea units; much information missing; or not a self-standing sentence |
| **0** | Silence, garbled, or only 1–2 content words matched |

### Implementation

1. **Preprocessing** — Follows the MFS/CogSLA Lab protocol: removes `[gibberish]`,
   `[pause]`, `xxx`, false starts, stuttering; extracts best final response from
   self-corrections.

2. **Content-word overlap** — Extracts content words (nouns, verbs, adjectives,
   adverbs) via spaCy POS tagging or stopword filtering. Computes fuzzy-matched
   overlap ratio against the target sentence.

3. **Fuzzy string similarity** — Levenshtein-based ratio (accent-normalized)
   with synonymous normalization per rubric rules.

4. **Hybrid semantic adjudication** — For borderline 2 ↔ 3 decisions,
   a multilingual sentence-transformer (`paraphrase-multilingual-MiniLM-L12-v2`)
   computes cosine similarity as a tie-breaker. Falls back gracefully when
   not installed.

---

## Evaluation Approach

- **Console + log output** for every sentence shows target, response, assigned
  score, and reasoning — enabling manual spot-checking against rubric examples.
- **Preprocessed CSV** with all intermediate features (content overlap, fuzzy
  ratio, semantic similarity) for systematic analysis.
- **Recommended validation**: Compute weighted Cohen's kappa between automated
  and human-rated scores.

See [evaluation/methodology.md](evaluation/methodology.md) for the full
approach description, sample analysis, limitations, and future work.

---

## Limitations

- Content-word detection uses a stopword list by default (spaCy POS improves this).
- Meaning preservation is approximated via overlap + similarity, not true
  semantic parsing.
- Self-correction extraction is heuristic (splits on `..` patterns).

## Future Improvements

- Fine-tune the sentence-transformer on EIT-specific transcription pairs.
- Integrate an LLM-based meaning judge for the nuanced 2 ↔ 3 boundary.
- Align self-correction detection with audio timestamps.

---

## Dependencies

```
pandas, openpyxl, thefuzz, python-Levenshtein     # required
sentence-transformers                               # recommended
spacy + es_core_news_sm                            # optional
```
