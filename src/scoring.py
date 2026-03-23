"""
scoring.py — Ortega (2000) EIT rubric implementation.

Scoring scale (0–4):
  4  Exact repetition: form and meaning match stimulus exactly.
  3  Full meaning preserved. Grammar errors allowed if meaning unchanged.
     'muy' is optional; 'y'/'pero' are interchangeable (per rubric).
  2  >50% of idea units present, string is meaningful, but content is
     inexact, incomplete, or ambiguous. When in doubt between 2 and 3 → 2.
  1  ~50% of idea units, lots of information missing, meaning may be
     unrelated/opposed, OR string is not a self-standing sentence.
  0  Silence, garbled, or only 1–2 content words matched.

Hybrid approach:
  Rule-based scoring (content-word overlap + fuzzy string similarity)
  is the primary signal.  For borderline 2 ↔ 3 decisions, semantic
  similarity from a multilingual sentence-transformer is used as a
  tie-breaker if the model is available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from thefuzz import fuzz

from .utils import (
    normalize_accents,
    get_content_words,
    compute_content_overlap,
    apply_synonymous_normalization,
    words_match,
    semantic_similarity,
)

# ---------------------------------------------------------------------------
# Conjugation-aware verb detection for is_meaningful_sentence
# ---------------------------------------------------------------------------

# Spanish finite verb endings that strongly signal a real conjugated form.
# This is a conservative set — it improves on the word-count heuristic
# without requiring a full parser.
_VERB_ENDINGS = re.compile(
    r"""(
        # indicative / subjunctive endings
        [aeiáéíóú]mos   |   # 1st plural: hablamos, vayamos
        [aeiáéíóú]ron   |   # 3rd plural preterite: hablaron
        [aeiáéíóú]ba    |   # imperfect: hablaba
        [aeiáéíóú]ra    |   # past subjunctive: hablara
        [aeiáéíóú]ría   |   # conditional: hablaría
        [aeiáéíóú]r     |   # infinitive ends in -ar/-er/-ir
        [aeiáéíóú]ndo   |   # gerund: hablando
        [aeiáéíóú]do    |   # participle: hablado
        # common 3rd-singular present/preterite
        (?:ó|io)\b      |   # habló, comió
        (?:iene|ienen|uede|ueden|uiero|uieren|usta|ustan|alta|altan)\b
    )""",
    re.VERBOSE | re.IGNORECASE,
)


def is_meaningful_sentence(text: str) -> bool:
    """
    Return True if the response constitutes a self-standing meaningful sentence.

    Checks:
      1. At least 3 words (necessary but not sufficient)
      2. Presence of at least one verb-like token (via conjugation-ending pattern)
         OR at least 2 content words suggesting a noun phrase / predicate
    """
    words = text.split()
    if len(words) < 3:
        return False

    # Check for verb-like word
    for word in words:
        if _VERB_ENDINGS.search(word):
            return True

    # Fallback: two or more content words implies a predicate exists
    # (e.g., "el gato negro perro" — not a sentence, but likely scored
    #  elsewhere as score 1 anyway)
    content_count = sum(
        1 for w in words
        if len(w) > 2 and not re.fullmatch(r"[aeiouáéíóú]", w, re.IGNORECASE)
    )
    return content_count >= 2


# ---------------------------------------------------------------------------
# Score result container
# ---------------------------------------------------------------------------

@dataclass
class ScoreResult:
    score: int
    explanation: str
    target_clean: str = ""
    response_clean: str = ""
    content_overlap: float = 0.0
    matched_content: int = 0
    total_content: int = 0
    fuzzy_ratio: float = 0.0
    sem_sim: Optional[float] = None
    borderline_adjusted: bool = False


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_utterance(
    target_clean: str,
    response_clean: str,
    nlp=None,
    use_semantic: bool = True,
) -> ScoreResult:
    """
    Apply the Ortega (2000) EIT rubric and return a ScoreResult.

    Parameters
    ----------
    target_clean   : preprocessed stimulus sentence (lowercase, no punctuation)
    response_clean : preprocessed learner response
    nlp            : optional spaCy language model for POS-based content words
    use_semantic   : whether to call the sentence-transformer for borderline cases
    """

    result = ScoreResult(
        score=0,
        explanation="",
        target_clean=target_clean,
        response_clean=response_clean,
    )

    # ------------------------------------------------------------------
    # Score 0 — no response
    # ------------------------------------------------------------------
    if not response_clean:
        result.explanation = "No response or completely unintelligible"
        return result

    t_words = target_clean.split()
    r_words = response_clean.split()

    # ------------------------------------------------------------------
    # Content-word analysis
    # ------------------------------------------------------------------
    t_content = get_content_words(target_clean, nlp)
    r_content = get_content_words(response_clean, nlp)

    matched, total, overlap = compute_content_overlap(t_content, r_content)
    result.matched_content = matched
    result.total_content = total
    result.content_overlap = overlap

    # ------------------------------------------------------------------
    # Score 0 — only 1 word OR minimal content words
    # ------------------------------------------------------------------
    if len(r_words) <= 1:
        result.score = 0
        result.explanation = f"Only {len(r_words)} word(s) produced"
        return result

    if total > 0 and matched <= 1 and len(r_content) <= 2:
        result.score = 0
        result.explanation = (
            f"Minimal repetition: {matched}/{total} content words matched"
        )
        return result

    # ------------------------------------------------------------------
    # Fuzzy string similarity (accent-normalized)
    # ------------------------------------------------------------------
    t_norm = normalize_accents(target_clean)
    r_norm = normalize_accents(response_clean)
    ratio = fuzz.ratio(t_norm, r_norm)
    token_sort = fuzz.token_sort_ratio(t_norm, r_norm)
    result.fuzzy_ratio = ratio

    # After synonymous normalization (muy optional, y/pero interchangeable)
    t_syn, r_syn = apply_synonymous_normalization(t_norm, r_norm)
    syn_ratio = fuzz.ratio(t_syn, r_syn)
    syn_token_sort = fuzz.token_sort_ratio(t_syn, r_syn)

    # Word-level overlap (all words, not just content)
    all_matched = sum(
        1 for tw in t_words if any(words_match(tw, rw) for rw in r_words)
    )
    all_overlap = all_matched / len(t_words) if t_words else 0.0

    # Length ratio — very short responses can't preserve full meaning
    length_ratio = len(r_words) / len(t_words) if t_words else 1.0
    truncated = length_ratio < 0.55

    # ------------------------------------------------------------------
    # Score 4 — exact repetition (accent-normalized)
    # ------------------------------------------------------------------
    if t_norm == r_norm:
        result.score = 4
        result.explanation = "Exact repetition"
        return result

    # Score 4 — exact after synonymous normalization
    if t_syn == r_syn:
        result.score = 4
        result.explanation = "Exact match after 'muy'/conjunction normalization"
        return result

    # ------------------------------------------------------------------
    # Score 3 — full meaning preserved
    # ------------------------------------------------------------------
    is_score_3 = False
    score_3_reason = ""

    if not truncated:
        if overlap >= 0.85 and syn_token_sort >= 80 and all_overlap >= 0.7:
            is_score_3 = True
            score_3_reason = (
                f"Meaning preserved: {matched}/{total} content words, "
                f"syn_sort={syn_token_sort}"
            )
        elif syn_ratio >= 85 and overlap >= 0.7:
            is_score_3 = True
            score_3_reason = (
                f"High similarity (syn_ratio={syn_ratio}), "
                f"content overlap={overlap:.0%}"
            )
        elif ratio >= 80 and overlap >= 0.8:
            is_score_3 = True
            score_3_reason = (
                f"Near-exact (ratio={ratio}), content overlap={overlap:.0%}"
            )

    # ------------------------------------------------------------------
    # Borderline 2 ↔ 3: use semantic similarity as tie-breaker
    # ------------------------------------------------------------------
    # Borderline zone: content overlap 0.65–0.85, syn_ratio 70–85
    is_borderline = (
        not truncated
        and 0.55 <= overlap < 0.85
        and 60 <= syn_ratio < 85
        and is_meaningful_sentence(response_clean)
    )

    sem_sim_val: Optional[float] = None
    if use_semantic and (is_borderline or is_score_3):
        sem_sim_val = semantic_similarity(target_clean, response_clean)
        result.sem_sim = sem_sim_val

    if is_borderline and sem_sim_val is not None:
        # High semantic similarity → upgrade to 3; low → stay at 2
        if sem_sim_val >= 0.82:
            is_score_3 = True
            score_3_reason = (
                f"Borderline upgraded to 3: sem_sim={sem_sim_val:.3f}, "
                f"content={overlap:.0%}"
            )
            result.borderline_adjusted = True
        elif sem_sim_val < 0.60 and is_score_3:
            # Rule-based said 3 but semantics disagree — downgrade to 2
            is_score_3 = False
            result.borderline_adjusted = True
            score_3_reason = (
                f"Downgraded from 3: sem_sim={sem_sim_val:.3f} too low"
            )

    if is_score_3:
        result.score = 3
        result.explanation = score_3_reason
        return result

    # ------------------------------------------------------------------
    # Score 2 — >50% idea units, meaningful, but inexact/incomplete
    # ------------------------------------------------------------------
    if overlap > 0.5 and is_meaningful_sentence(response_clean):
        result.score = 2
        result.explanation = (
            f"Partial meaning: {matched}/{total} content words, "
            "meaningful sentence"
        )
        return result

    if overlap > 0.5 or (all_overlap > 0.5 and syn_ratio >= 50):
        result.score = 2
        result.explanation = (
            f"Related meaning: content={overlap:.0%}, all_overlap={all_overlap:.0%}"
        )
        return result

    # ------------------------------------------------------------------
    # Score 1 — ~50% idea units, lots missing, or not self-standing
    # ------------------------------------------------------------------
    if (
        overlap >= 0.3
        or all_overlap >= 0.35
        or (matched >= 2 and is_meaningful_sentence(response_clean))
    ):
        result.score = 1
        result.explanation = (
            f"Partial/incomplete: {matched}/{total} content words, "
            f"all_overlap={all_overlap:.0%}"
        )
        return result

    if ratio >= 35 or token_sort >= 40:
        result.score = 1
        result.explanation = f"Low similarity (ratio={ratio}, token_sort={token_sort})"
        return result

    # ------------------------------------------------------------------
    # Score 0 — very little preserved
    # ------------------------------------------------------------------
    result.score = 0
    result.explanation = (
        f"Minimal match: {matched}/{total} content words, ratio={ratio}"
    )
    return result
