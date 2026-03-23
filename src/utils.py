"""
utils.py — Shared helpers for AutoEIT scoring.

General-purpose utilities:
  - Robust column detection (handles spacing/capitalization variation)
  - Accent normalization
  - Word-level fuzzy matching & content-word extraction
  - Content-overlap computation
  - Optional semantic similarity using sentence-transformers
"""

import unicodedata
import warnings
from thefuzz import fuzz

from .rubric import SPANISH_FUNCTION_WORDS, apply_synonymous_normalization  # noqa: F401 — re-export


# ---------------------------------------------------------------------------
# Accent normalization
# ---------------------------------------------------------------------------

def normalize_accents(text: str) -> str:
    """Strip diacritics so 'está' == 'esta', 'mañana' == 'manana', etc."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# ---------------------------------------------------------------------------
# Robust column detection
# ---------------------------------------------------------------------------

def detect_columns(df) -> tuple[str, str]:
    """
    Find the stimulus (target) and transcription (learner response) columns
    regardless of exact capitalization or minor naming differences.

    Returns (stimulus_col, transcription_col).
    Raises ValueError if either cannot be found.
    """
    cols = df.columns.tolist()
    lower = {c.lower().strip(): c for c in cols}

    stimulus_col = None
    transcription_col = None

    for key, original in lower.items():
        if "stimulus" in key or "target" in key or "prompt" in key:
            stimulus_col = original
        if "transcription" in key or "response" in key or "utterance" in key:
            if transcription_col is None or "rater 1" in key or "rater1" in key:
                transcription_col = original

    if stimulus_col is None:
        raise ValueError(
            f"Could not find stimulus/target column in: {cols}\n"
            "Expected a column containing 'stimulus', 'target', or 'prompt'."
        )
    if transcription_col is None:
        raise ValueError(
            f"Could not find transcription column in: {cols}\n"
            "Expected a column containing 'transcription', 'response', or 'utterance'."
        )

    return stimulus_col, transcription_col


def detect_sentence_col(df) -> str | None:
    """Return the sentence-number column name if present, else None."""
    for col in df.columns:
        if col.lower().strip() in ("sentence", "item", "sentence #", "sentence number"):
            return col
    return None


# ---------------------------------------------------------------------------
# Word-level matching & content-word extraction
# ---------------------------------------------------------------------------

def get_content_words(text: str, nlp=None) -> list[str]:
    """
    Extract content words from text.

    If a spaCy model is provided (nlp), uses POS tagging (NOUN/VERB/ADJ/ADV).
    Otherwise falls back to filtering against SPANISH_FUNCTION_WORDS.
    """
    if nlp is not None:
        doc = nlp(text)
        return [
            token.text for token in doc
            if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
            and not token.is_punct
            and len(token.text) > 1
        ]

    words = text.split()
    return [w for w in words
            if normalize_accents(w) not in SPANISH_FUNCTION_WORDS and len(w) > 1]


def words_match(w1: str, w2: str, threshold: int = 85) -> bool:
    """
    True if two words are equivalent after accent normalization,
    or if their fuzzy similarity exceeds `threshold`.
    """
    if w1 == w2:
        return True
    if normalize_accents(w1) == normalize_accents(w2):
        return True
    return fuzz.ratio(w1, w2) >= threshold


def compute_content_overlap(target_content: list[str], response_content: list[str]):
    """
    Compute how many of the target's content words appear in the response.
    Returns (matched_count, total_count, overlap_ratio).
    """
    if not target_content:
        return 0, 0, 1.0

    matched = 0
    used = set()
    for tw in target_content:
        for i, rw in enumerate(response_content):
            if i not in used and words_match(tw, rw):
                matched += 1
                used.add(i)
                break

    return matched, len(target_content), matched / len(target_content)


# ---------------------------------------------------------------------------
# Optional semantic similarity (sentence-transformers)
# ---------------------------------------------------------------------------

_sentence_model = None
_model_load_attempted = False


def _load_sentence_model():
    """Lazy-load the multilingual sentence-transformers model."""
    global _sentence_model, _model_load_attempted
    if _model_load_attempted:
        return _sentence_model
    _model_load_attempted = True
    try:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
        return _sentence_model
    except ImportError:
        warnings.warn(
            "sentence-transformers not installed — semantic similarity disabled.\n"
            "Install with: pip install sentence-transformers",
            ImportWarning,
            stacklevel=2,
        )
        return None


def semantic_similarity(text1: str, text2: str) -> float | None:
    """
    Compute cosine similarity between two Spanish sentences.
    Returns float in [0, 1], or None if model unavailable.
    """
    model = _load_sentence_model()
    if model is None:
        return None
    try:
        import numpy as np
        embeddings = model.encode([text1, text2], convert_to_numpy=True)
        a, b = embeddings[0], embeddings[1]
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    except Exception as e:
        warnings.warn(f"Semantic similarity computation failed: {e}", stacklevel=2)
        return None


def is_semantic_model_available() -> bool:
    """Check whether the semantic similarity model is available."""
    return _load_sentence_model() is not None
