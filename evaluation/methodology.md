# Methodology — AutoEIT Automated Scoring

## Task

Implement a reproducible script that applies the Ortega (2000) meaning-based
rubric to Spanish EIT sentence transcriptions, comparing learner utterances
to prompt sentences and outputting sentence-level scores (0–4).

---

## Scoring Approach

The scoring system was designed to approximate human EIT scoring using the
Ortega (2000) meaning-based rubric by combining:

1. **Transcription preprocessing** — following the MFS/CogSLA Lab protocol
2. **Idea unit overlap analysis** — content-word matching between target and response
3. **Fuzzy string matching** — Levenshtein-based similarity to tolerate spelling errors
4. **Rule-based scoring logic** — thresholds calibrated to rubric descriptors
5. **Semantic similarity (optional)** — sentence-transformer for borderline adjudication

### Why this design

The EIT rubric is fundamentally about *meaning preservation*. Pure string
comparison misses cases where grammar changes but meaning stays the same
(score 3), or where words overlap but meaning shifts (score 2). The hybrid
approach addresses both:

- **Rule-based scoring** captures the structural aspects (how many content
  words are preserved, how close the surface form is)
- **Semantic similarity** captures meaning preservation beyond word overlap,
  specifically for the difficult 2 ↔ 3 boundary where the rubric says
  "in case of doubt, score 2"

---

## Preprocessing Pipeline

Each learner transcription is cleaned following the exact conventions
specified in the Data Processing Protocol:

| Step | Operation | Example |
|------|-----------|---------|
| 1 | Detect no-response markers | `[no response]` → empty |
| 2 | Remove bracketed annotations | `[gibberish]`, `[pause]`, `[cough]` → removed |
| 3 | Remove unintelligible markers | `xxx`, `XX` → removed |
| 4 | Remove parenthetical comments | `(tambien?)` → removed |
| 5 | Extract best final response | `"Mis gus..Me gustas"` → `"Me gustas"` |
| 6 | Remove false starts | `[la-]`, `[gustan-]` → removed |
| 7 | Remove syllable stuttering | `co-co-comerme` → `comerme` |
| 8 | Remove word-level stuttering | `se se se` → `se` |
| 9 | Remove abandoned fragments | `ma-` → removed |
| 10 | Remove fillers | `um`, `uh`, `mhh` → removed |
| 11 | Remove pauses | `...` → space |
| 12 | Normalize | lowercase, strip punctuation, collapse whitespace |

Critical rubric exception: **always score the best final response**.
Self-corrections, hesitations, and false starts are removed, not penalized.

---

## Scoring Logic (Ortega, 2000)

### Score 4 — Exact repetition
Both form and meaning match the stimulus exactly.
Comparison is accent-normalized (`está` = `esta`).

### Score 3 — Meaning preserved
Full meaning is preserved even if grammar is incorrect.
Synonymous substitutions per rubric:
- `muy` (very) is optional — present or absent is equivalent
- `y` / `pero` / `e` / `sino` are interchangeable conjunctions

Grammar changes that don't affect meaning → score 3.
Ambiguous changes → score 2.

### Score 2 — Partial meaning, inexact
More than 50% of idea units preserved. String is meaningful but
content is inexact, incomplete, or ambiguous.

### Score 1 — Much information missing
About half of idea units represented. String may not constitute a
self-standing sentence or meaning may be unrelated.

### Score 0 — Minimal or no response
Silence, garbled, or minimal repetition (only 1-2 content words matched).

---

## Content Word Detection

**Primary method** (when spaCy is installed):
POS tagging using `es_core_news_sm` → extract NOUN, VERB, ADJ, ADV tokens.

**Fallback method**:
Filter against a curated list of ~80 Spanish function words (articles,
prepositions, conjunctions, clitics, auxiliaries).

---

## Semantic Similarity for Borderline Cases

For cases where content overlap falls in the 55–85% range and fuzzy
similarity is 60–85% (the 2 ↔ 3 boundary), a multilingual
sentence-transformer (`paraphrase-multilingual-MiniLM-L12-v2`)
computes cosine similarity:

- cosine sim ≥ 0.82 → upgrade to score 3 (meaning well-preserved)
- cosine sim < 0.60 → keep at score 2 (meaning diverges)

This handles paraphrased responses that word-overlap alone would
mis-score, without requiring a full semantic parser.

The system falls back gracefully if `sentence-transformers` is not installed.

---

## Evaluation of Output

### Spot-checking
Every scored utterance includes a detailed explanation (content overlap
ratio, fuzzy similarity, semantic similarity if computed). The debug script
(`scripts/debug_single_example.py`) traces the full scoring decision for
any individual target-response pair.

### Validation protocol
1. Have two human raters score a subset independently.
2. Run automated scoring on the same subset.
3. Compute weighted Cohen's kappa (quadratic weights) for human-human
   and human-auto agreement.
4. If human-auto kappa is within 0.10 of human-human kappa, the system
   meets research-grade reliability.

### Score distributions (sample data, 4 participants)

| Participant | Mean | 0s | 1s | 2s | 3s | 4s |
|-------------|------|-----|-----|-----|-----|-----|
| 38001-1A    | 2.97 |  1 |  0 |  6 | 15 |  8 |
| 38002-2A    | 1.67 |  3 | 13 |  8 |  3 |  3 |
| 38004-2A    | 2.27 |  3 |  3 | 11 |  9 |  4 |
| 38006-2A    | 1.47 |  6 | 11 |  7 |  5 |  1 |

The expected difficulty gradient is captured: early items (7–10 syllables)
consistently score higher than later items (15–17 syllables).

---

## Limitations

| Limitation | Impact | Possible Improvement |
|------------|--------|---------------------|
| Stopword-based content word detection | Moderate | Enable spaCy POS tagging |
| Semantic similarity approximates meaning | Low–Moderate | Fine-tune on EIT data |
| `is_meaningful_sentence` uses verb-ending heuristics | Low | Full dependency parse |
| Self-correction extraction is heuristic | Low | Align with audio timestamps |

---

## Future Work

A semantic similarity model (e.g., Sentence Transformers) could be further
fine-tuned on EIT-specific transcription pairs to better capture meaning
preservation in paraphrased responses where word overlap is low but meaning
is preserved. Additionally, integrating an LLM-based meaning judge could
provide a more nuanced assessment of the 2 ↔ 3 boundary, where the rubric
requires expert linguistic judgment about whether grammar changes alter meaning.
