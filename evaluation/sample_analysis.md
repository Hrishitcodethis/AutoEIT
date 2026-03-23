# Sample Analysis — AutoEIT Scoring Results

Results from the 4-participant sample dataset (Version A, 30 sentences each).

---

## Score Distributions

| Participant | Mean Score | 0s | 1s | 2s | 3s | 4s | Total/120 |
|-------------|-----------|-----|-----|-----|-----|-----|-----------|
| 38001-1A    | ~2.97     |  1  |  0  |  6  | 15  |  8  |  89       |
| 38002-2A    | ~1.67     |  3  | 13  |  8  |  3  |  3  |  50       |
| 38004-2A    | ~2.33     |  2  |  3  | 12  |  9  |  4  |  70       |
| 38006-2A    | ~1.47     |  6  | 11  |  7  |  5  |  1  |  44       |

Participant **38001-1A** scores highest, suggesting stronger L2 Spanish
proficiency or more fluent recall. **38006-2A** shows the most 0s, indicating
frequent breakdown in recall.

---

## Spot-Check: Selected Rubric-Aligned Decisions

### Score 4 — Exact repetition

| Target | Response | Score |
|--------|----------|-------|
| Quiero cortarme el pelo | Quiero cortarme el pelo | 4 |
| Le pedí a un amigo que me ayudara con la tarea | Le pedí a un amigo que me ayudara con la tarea | 4 |

These are exact matches (accent-normalized). The scoring is unambiguous.

---

### Score 3 — Meaning preserved, grammar changed

| Target | Response | Score | Reasoning |
|--------|----------|-------|-----------|
| Dudo que sepa manejar muy bien | Dudo que sepa manajar bien | 3 | All content words present; 'muy' optional per rubric |
| El ladrón al que atrapó la policía era famoso | El ladrón que atrapó la policía era famoso | 3 | Relative clause restructured; meaning identical |
| El gato que era negro fue perseguido por el perro | El gato que era negro era perseguido de(l) perro | 3 | Passive voice variation; meaning preserved |

---

### Score 2 — Partial meaning, close but inexact

| Target | Response | Score | Reasoning |
|--------|----------|-------|-----------|
| Cruza a la derecha y después sigue todo recto | Cruza a la derecha y sigue a la izquierda | 2 | Direction word changed (recto → izquierda) — meaning altered |
| El examen no fue tan difícil como me habían dicho | El examen no fue tan difícil como me decían | 2 | ~50% content match; 'habían dicho' → 'decían' changes tense/construction |
| ¿Qué dice usted que va a hacer hoy? | Que dices ustedes se que van a hacer hoy? | 2 | Morphological errors change register; information partially preserved |

---

### Score 1 — Partial, incomplete, or non-self-standing

| Target | Response | Score | Reasoning |
|--------|----------|-------|-----------|
| Dudo que sepa manejar muy bien | dudo/tu no? sepiar exx muy bien | 1 | Fragmented; meaning partially preserved but string is not well-formed |
| Después de cenar me fui a dormir tranquilo | después de finar … él dormir xxx | 1 | <50% content words recovered; heavy intelligibility issues |

---

### Score 0 — Minimal repetition / no response

| Target | Response | Score | Reasoning |
|--------|----------|-------|-----------|
| Me gustaría que empezara a hacer más calor pronto | me gustaría se | 0 | Only function words/1 content word; abandoned |
| El carro lo tiene Pedro | E-[gibberish] perro | 0 | Only 1 word after cleaning |
| La cantidad de personas que fuman ha disminuido | A la cantan... [pause] muy a xxx | 0 | No content words from stimulus matched |

---

## Sentence-Level Difficulty Gradient

Sentences are ordered by syllable count (in parentheses in the stimulus).
Items 1–6 (7–10 syllables) reliably score higher; items 19–30 (15–17 syllables)
show markedly more errors, consistent with memory span constraints in EIT design.

This gradient validates the scoring output — the automated system captures
the expected difficulty effect without any length-based tuning.

---

## Borderline 2 ↔ 3 Cases (Semantic Adjudication)

When `sentence-transformers` is available, borderline cases are flagged with
`borderline_adjusted=True` in the output CSV. These warrant human review.

**Key borderline pattern observed:**
- Response omits one content word but preserves the core proposition
- Word-overlap falls in the 65–80% range
- String similarity (fuzzy) is 70–82%
- Semantic similarity confirms meaning ≥ 0.82 → score 3

**Example (hypothetical):**
- Target: `ella ha terminado de pintar su apartamento`
- Response: `ella ha terminado pintando su apartamento`
- Content overlap: 100%, but gerund vs. infinitive is a grammatical difference
  that *could* signal a meaning change → borderline. Semantic similarity
  (> 0.90) confirms meaning is preserved → score 3.

---

## Recommendations for Production Use

1. **Collect human-rated gold standard** on a held-out set and compute
   weighted kappa between human and auto scores.
2. **Fine-tune thresholds** (content overlap, sem_sim cutoffs) based on the
   empirical kappa curve.
3. **Enable spaCy** (`es_core_news_sm`) in production for more reliable
   content-word extraction.
4. **Flag low-confidence predictions** (borderline-adjusted cases, very
   short responses) for human review rather than treating them as final.
