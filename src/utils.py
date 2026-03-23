"""
utils.py — Shared helpers for AutoEIT scoring.

General-purpose utilities:
  - Robust column detection (handles spacing/capitalization variation)
  - Accent normalization
  - Word-level fuzzy matching & content-word extraction
  - Content-overlap computation
  - Semantic similarity (TF-IDF always available; neural if sentence-transformers installed)
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
# Semantic similarity
# ---------------------------------------------------------------------------
# Two backends:
#   1. sentence-transformers (if installed) — multilingual neural embeddings
#   2. TF-IDF character n-gram cosine similarity — lightweight fallback,
#      always available, gives a different signal from Levenshtein
# The system always computes semantic similarity via one of these backends.
# ---------------------------------------------------------------------------

_sentence_model = None
_model_load_attempted = False
_USE_NEURAL = None  # cached: True = sentence-transformers, False = TF-IDF


def _try_load_neural_model():
    """Try to load sentence-transformers. Returns model or None."""
    global _sentence_model, _model_load_attempted, _USE_NEURAL
    if _model_load_attempted:
        return _sentence_model
    _model_load_attempted = True
    try:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
        _USE_NEURAL = True
        return _sentence_model
    except (ImportError, Exception):
        _USE_NEURAL = False
        return None


def _tfidf_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity using TF-IDF character n-gram vectors.

    This is a lightweight alternative to neural embeddings that captures
    sub-word similarity patterns beyond what Levenshtein ratio provides:
      - Levenshtein measures edit distance (character-level)
      - TF-IDF n-grams weight distinctive character sequences higher

    Uses character n-grams (2-4) to handle Spanish morphological variation
    (e.g., 'hablaron' vs 'hablar' share trigrams 'hab','abl','bla','lar').
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",     # character n-grams with word boundaries
        ngram_range=(2, 4),     # bigrams through 4-grams
        lowercase=True,
    )
    try:
        tfidf = vectorizer.fit_transform([text1, text2])
        sim = float(cos_sim(tfidf[0:1], tfidf[1:2])[0, 0])
        return max(0.0, sim)
    except ValueError:
        return 0.0


def semantic_similarity(text1: str, text2: str) -> float | None:
    """
    Compute semantic similarity between two Spanish sentences.

    Uses sentence-transformers if installed, otherwise falls back to
    TF-IDF character n-gram cosine similarity (always available).

    Returns float in [0, 1].
    """
    if not text1 or not text2:
        return 0.0

    # Try neural embeddings first
    model = _try_load_neural_model()
    if model is not None:
        try:
            import numpy as np
            embeddings = model.encode([text1, text2], convert_to_numpy=True)
            a, b = embeddings[0], embeddings[1]
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        except Exception:
            pass

    # Fallback: TF-IDF character n-gram similarity
    return _tfidf_similarity(text1, text2)


def get_semantic_backend() -> str:
    """Return which backend is active: 'neural' or 'tfidf'."""
    _try_load_neural_model()
    return "neural" if _USE_NEURAL else "tfidf"
