# AutoEIT — Automated Scoring for the Spanish Elicited Imitation Task

**GSoC Test II: Evaluation of Transcribed Data**

This project implements an automated scoring system for the Spanish Elicited
Imitation Task (EIT) using the Ortega (2000) meaning-based rubric. The system
compares learner utterances against prompt sentences and outputs sentence-level
scores (0–4) for each utterance in the sample data.

The scoring system was designed to approximate human EIT scoring by combining
transcription preprocessing, idea unit overlap analysis, fuzzy string matching,
and rule-based scoring logic — with semantic similarity (always active) for
borderline adjudication.

---

## Project Structure

```
AutoEIT/
├── score_eit.py                ← Quick entry point (python score_eit.py)
├── requirements.txt
│
├── data/
│   ├── raw/                    ← Input Excel + committed scored output
│   └── output/                 ← Generated at runtime (gitignored)
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
    ├── methodology.md          ← Full approach, evaluation, limitations, future work
    └── expected_scores.json    ← Gold-standard expected output for --verify
```

---

## How to Run

Requires **Python 3.10+**.

```bash
# Install dependencies (pinned versions for reproducibility)
pip install -r requirements.txt

# Run scoring (simplest)
python score_eit.py

# Or with CLI options
python score_eit.py --no-semantic --no-spacy
python score_eit.py --quiet          # suppress stdout, write to log only
python score_eit.py --help           # see all options

# Verify output matches expected scores (reproducibility check)
python score_eit.py --no-semantic --no-spacy --quiet --verify

# Optional: enable POS-based content word detection
pip install spacy && python -m spacy download es_core_news_sm
```

### Output

| File | Description |
|------|-------------|
| `data/output/AutoEIT_Scored_Results.xlsx` | Original sheets with `Score` column added |
| `data/output/preprocessed_transcriptions.csv` | Cleaned text, overlap metrics, explanations |
| `data/output/scoring_log.txt` | Per-sentence scoring detail log |

> `data/output/` is git-ignored (generated artifacts). Run the script to reproduce.

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

4. **Semantic similarity (always active)** — For borderline 2 ↔ 3 decisions,
   cosine similarity serves as a tie-breaker. Uses TF-IDF character n-gram
   similarity by default (no heavy dependencies); automatically upgrades to
   neural embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) if
   `sentence-transformers` is installed. Thresholds calibrated per backend.

---

## Evaluation Approach

- **Console + log output** for every sentence shows target, response, assigned
  score, and reasoning — enabling manual spot-checking against rubric examples.
- **Preprocessed CSV** with all intermediate features (content overlap, fuzzy
  ratio, semantic similarity) for systematic analysis.
- **Reproducibility verification**: `evaluation/expected_scores.json` contains
  the expected scores for all 120 utterances. Run `python score_eit.py --verify`
  to confirm output matches.
- **Recommended validation**: Compute weighted Cohen's kappa between automated
  and human-rated scores.

See [evaluation/methodology.md](evaluation/methodology.md) for the full
approach description, sample analysis, limitations, and future work.

---

## Limitations

- Content-word detection uses a stopword list by default (spaCy POS improves this)
- TF-IDF character n-gram similarity captures surface form but not deep semantics
- Self-correction extraction is heuristic (splits on `..` patterns)

## Future Improvements

- Fine-tune a sentence-transformer on EIT-specific transcription pairs
- LLM-based meaning judge for nuanced grammar-changes-meaning cases
- spaCy dependency parsing to detect structurally missing main verbs
- Align self-correction detection with audio timestamps

---

## Dependencies

```
pandas, openpyxl, thefuzz, python-Levenshtein, scikit-learn   # required
sentence-transformers                                          # optional (upgrades semantic backend)
spacy + es_core_news_sm                            # optional
```
