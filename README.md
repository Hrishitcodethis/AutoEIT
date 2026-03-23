# AutoEIT — Automated EIT Scoring System

Reproducible Python pipeline that applies the **Ortega (2000) meaning-based rubric**
to Spanish Elicited Imitation Task (EIT) transcriptions, producing sentence-level
scores (0–4) for each learner utterance.

Developed as part of the **AutoEIT GSoC project** evaluation (Test II).

---

## Project Structure

```
AutoEIT/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                   ← place input Excel files here
│   └── processed/             ← auto-generated CSVs written here
│
├── src/
│   ├── preprocessing.py       ← text cleaning per EIT protocol
│   ├── scoring.py             ← Ortega rubric + hybrid semantic adjudication
│   ├── utils.py               ← column detection, fuzzy matching, sem-sim
│   └── pipeline.py            ← orchestrator (read → score → write)
│
├── scripts/
│   ├── run_scoring.py         ← main CLI entry point
│   └── debug_single_example.py ← interactive single-pair debugger
│
├── outputs/                   ← scored Excel + log written here
│
└── evaluation/
    ├── evaluation_notes.md    ← approach, limitations, validation protocol
    └── sample_analysis.md     ← spot-check analysis of sample data
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt

# Optional: research-grade POS tagging
python -m spacy download es_core_news_sm
```

### 2. Run the full pipeline

```bash
python scripts/run_scoring.py
```

This reads `data/raw/AutoEIT Sample Transcriptions for Scoring.xlsx`
and writes:

| Output | Description |
|--------|-------------|
| `outputs/scored_results.xlsx` | Original sheets + `Score` column added |
| `data/processed/preprocessed_transcriptions.csv` | Cleaned text, overlap, sem_sim, explanation |
| `outputs/logs.txt` | Per-sentence detail log |

### 3. Debug a specific example

```bash
python scripts/debug_single_example.py \
  --target   "Las calles de esta ciudad son muy anchas" \
  --response "Las calles de esta cuidad son anchas"
```

Or run interactively (no arguments) to enter pairs one at a time.

---

## CLI Options

```
python scripts/run_scoring.py [options]

  --input   PATH   Input Excel file  (default: data/raw/...)
  --output  PATH   Output Excel file (default: outputs/scored_results.xlsx)
  --csv     PATH   Output CSV        (default: data/processed/...)
  --log     PATH   Log file          (default: outputs/logs.txt)
  --no-semantic    Disable sentence-transformer (use rule-based only)
  --no-spacy       Disable spaCy POS tagging (use stopword fallback)
  --quiet          Suppress stdout output
```

---

## Scoring Rubric (Ortega, 2000)

| Score | Criteria |
|-------|----------|
| **4** | Exact repetition — form and meaning match stimulus exactly |
| **3** | Full meaning preserved; grammar errors OK if meaning unchanged; 'muy' optional; y/pero interchangeable |
| **2** | >50% idea units present; meaningful but inexact, incomplete, or ambiguous; *when in doubt 2 vs 3 → score 2* |
| **1** | ~50% idea units; lots missing; meaning may be unrelated; OR not a self-standing sentence |
| **0** | Silence, garbled, or only 1–2 content words matched |

---

## Technical Design

### Preprocessing (`src/preprocessing.py`)
Follows the MFS/CogSLA Lab UIC protocol exactly:
- Removes `[gibberish]`, `[pause]`, `xxx`/`XX` markers
- Extracts **best final response** from self-corrections (`"Mis gus..Me gustas"`)
- Removes false starts (`[la-]`), stuttering (`co-co-comerme`), abandoned fragments (`ma-`)
- Strips fillers (`um`, `uh`, `mhh`), punctuation, and normalizes whitespace

### Column Detection (`src/utils.py`)
Automatically detects stimulus and transcription columns regardless of
exact capitalization or whitespace variation — no hardcoded column names.

### Scoring (`src/scoring.py`)

**Rule-based (primary):**
- Content-word overlap (fraction of stimulus content words in response)
- Fuzzy string similarity via Levenshtein ratio (tolerates typos/accent omissions)
- Synonymous normalization before comparison ('muy' optional, y/pero canonical)

**Hybrid semantic adjudication (for borderline 2 ↔ 3):**
Uses `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers, supports Spanish):
- cosine sim ≥ 0.82 → upgrade to score 3 (meaning well-preserved)
- cosine sim < 0.60 → keep at score 2 (meaning has diverged)

This correctly handles paraphrased responses that word-overlap alone would
under-score, without needing a full semantic parser.

> The system falls back gracefully if `sentence-transformers` is not installed.

---

## Evaluation

See [evaluation/evaluation_notes.md](evaluation/evaluation_notes.md) for:
- Full approach description
- Validation protocol (weighted Cohen's kappa)
- Known limitations and improvement roadmap

See [evaluation/sample_analysis.md](evaluation/sample_analysis.md) for:
- Score distributions across the 4 sample participants
- Spot-checked examples at each score level
- Analysis of borderline cases

---

## Extending the System

- **Better content words**: spaCy POS tagging (`--no-spacy` disables)
- **Better semantic signal**: swap the sentence-transformer model for a
  fine-tuned Spanish model or GPT-based meaning judge
- **More synonyms**: expand `SYNONYMOUS_SUBS` in `utils.py`
- **New rubrics**: subclass `EITPipeline` and override `_score_sheet`
