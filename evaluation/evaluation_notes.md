# Evaluation Notes — AutoEIT Scoring System

## Approach Overview

This script applies the **Ortega (2000) meaning-based EIT rubric** automatically
to transcribed learner utterances, comparing them against Spanish prompt sentences.

---

## Rubric Implementation (Ortega, 2000)

| Score | Criteria |
|-------|----------|
| **4** | Exact repetition — form and meaning match stimulus exactly. |
| **3** | Full meaning preserved. Ungrammatical strings are allowed if meaning is retained. 'muy' is optional; 'y'/'pero'/'e'/'sino' are interchangeable. |
| **2** | >50% of idea units present; string is meaningful but content is inexact, incomplete, or ambiguous. Rubric principle: *when in doubt between 2 and 3, score 2.* |
| **1** | ~50% of idea units; lots of information missing; meaning may be unrelated/opposed; OR string is not a self-standing sentence. |
| **0** | Silence, garbled/unintelligible, minimal repetition (only 1 word, only function word(s), or 1–2 content words out of order with extraneous words). |

### Exceptions encoded per protocol
- **False starts** (`[la-]`, `co-co-comerme`) → removed before scoring
- **Pauses** (`...`, `[pause]`) → stripped
- **Self-corrections** (`"Mis gus..Me gustas"`) → best final response extracted
- **Unintelligible markers** (`xxx`, `[gibberish]`, `XX`) → removed
- **Filler words** (`um`, `uh`, `mhh`) → stripped

---

## Technical Approach

### Step 1: Preprocessing
Text is cleaned following the MFS/CogSLA Lab UIC Data Processing Protocol
before any comparison takes place (see `src/preprocessing.py`).

### Step 2: Rule-Based Scoring
Primary signal uses three complementary metrics:

1. **Content-word overlap** — fraction of stimulus content words (nouns, verbs,
   adjectives, adverbs) present in the response; drives the 0/1/2/3 thresholds.

2. **Fuzzy string similarity** — Levenshtein-based ratio tolerates
   spelling errors and accent omissions in transcription.

3. **Synonymous normalization** — before comparison, 'muy' is dropped from
   both strings and conjunctions (y/pero/e/sino) are canonicalized to 'y'.

### Step 3: Semantic Similarity (Hybrid Signal)
For **borderline 2 ↔ 3 decisions** (content overlap 55–85%, syn_ratio 60–85%),
a multilingual sentence-transformer
(`paraphrase-multilingual-MiniLM-L12-v2`) computes cosine similarity:

- sim ≥ 0.82 → score upgraded to **3** (meaning well-preserved)
- sim < 0.60 → score kept at **2** (meaning diverges)

This handles cases word-overlap misses, e.g., near-synonymous phrasing or
slight structural rearrangements that preserve meaning.

> Fallback: if `sentence-transformers` is not installed, the rule-based score
> is used exclusively. The system logs a warning.

### Step 4: Content Word Detection
**Primary (if spaCy available):** POS tagging with `es_core_news_sm` →
extract NOUN, VERB, ADJ, ADV tokens.

**Fallback:** Stopword filtering against a curated Spanish function-word list
(~80 entries covering articles, prepositions, conjunctions, clitics, auxiliaries).

---

## Output Evaluation

### How to assess output quality

1. **Spot-check rubric examples**: The rubric itself provides labelled examples
   for each score level. Run `scripts/debug_single_example.py` to trace
   the scoring decision on any specific pair.

2. **Compare with human rater scores** (if available):
   - Load `outputs/scored_results.xlsx` alongside human-annotated scores.
   - Compute **weighted Cohen's kappa** (adjacent scores cost less than
     distant disagreements) using:
     ```python
     from sklearn.metrics import cohen_kappa_score
     kappa = cohen_kappa_score(human_scores, auto_scores, weights='quadratic')
     ```
   - A kappa ≥ 0.70 is considered good inter-rater agreement in linguistics.

3. **Examine borderline cases**: The CSV (`data/processed/preprocessed_transcriptions.csv`)
   includes `sem_sim` and `borderline_adjusted` columns to identify cases
   where the semantic signal changed the decision.

---

## Known Limitations

| Limitation | Impact | Possible Fix |
|------------|--------|-------------|
| Stopword-based content word detection can misclassify words | Moderate | Use spaCy POS tagging |
| Semantic similarity approximates meaning, not an oracle | Low–Moderate | Fine-tune on EIT-specific data |
| `is_meaningful_sentence` uses verb-ending heuristics, not a parser | Low | Full spaCy dependency parse |
| Paraphrase synonyms beyond rubric list not detected | Low | Expand with embeddings |
| Self-correction extraction is heuristic (`..` split) | Low | Align with audio timestamps |

---

## Recommended Validation Protocol

1. Have two human raters score a 30-utterance sample independently.
2. Run automated scoring on the same sample.
3. Compute:
   - Human–human inter-rater agreement (baseline)
   - Human–auto agreement (target)
4. If human–auto kappa is within 0.10 of human–human kappa, the system
   is considered research-grade.
