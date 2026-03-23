"""
utils.py — Shared helpers for AutoEIT scoring.

Includes:
  - Robust column detection (handles spacing/capitalization variation)
  - Accent normalization
  - Word-level fuzzy matching
  - Optional semantic similarity using sentence-transformers
    (multilingual model, works for Spanish; falls back gracefully if not installed)
"""

import unicodedata
import warnings
from thefuzz import fuzz

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
        if ("transcription" in key or "response" in key or "utterance" in key):
            # Prefer "rater 1" but accept any transcription column
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
# Word-level matching
# ---------------------------------------------------------------------------

# Spanish function words — used to identify content vs. function words
SPANISH_FUNCTION_WORDS = {
    # articles
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo",
    # prepositions
    "a", "al", "de", "del", "en", "con", "por", "para", "sin",
    "sobre", "entre", "hacia", "hasta", "desde", "ante",
    # conjunctions
    "y", "e", "o", "u", "pero", "sino", "ni", "que", "como",
    # subject/object pronouns and clitics
    "me", "te", "se", "le", "les", "nos", "os",
    "yo", "tu", "el", "ella", "usted", "nosotros", "ustedes", "ellos", "ellas",
    "mio", "mia", "mi", "mis", "su", "sus", "tus",
    # auxiliaries / very common verbs treated as function words
    "es", "ha", "he", "han", "hay", "fue", "ser", "muy",
    # other function words
    "no", "si", "ya", "mas", "tan", "todo", "toda",
    "este", "esta", "ese", "esa", "esto", "eso",
    "donde", "cuando", "quien", "cuya", "cuyo",
}


def get_content_words(text: str, nlp=None) -> list[str]:
    """
    Extract content words from text.

    If a spaCy model is provided (nlp), uses POS tagging (NOUN/VERB/ADJ/ADV).
    Otherwise falls back to filtering against the SPANISH_FUNCTION_WORDS list.
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
    content = []
    for w in words:
        w_norm = normalize_accents(w)
        if w_norm not in SPANISH_FUNCTION_WORDS and len(w) > 1:
            content.append(w)
    return content


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
# Synonymous substitution normalization (per rubric)
# ---------------------------------------------------------------------------

def apply_synonymous_normalization(target: str, response: str) -> tuple[str, str]:
    """
    Normalize per the rubric's explicit synonymous substitution rules:
      - 'muy' is optional (add/omit without penalty)
      - 'y' / 'e' / 'pero' / 'sino' are interchangeable conjunctions
    Both strings are normalized identically so comparisons are fair.
    """
    def normalize(words):
        # Drop 'muy'
        words = [w for w in words if w != "muy"]
        # Canonicalize conjunctions -> 'y'
        words = ["y" if w in ("pero", "sino", "e") else w for w in words]
        return words

    t_words = normalize(target.split())
    r_words = normalize(response.split())
    return " ".join(t_words), " ".join(r_words)


# ---------------------------------------------------------------------------
# Optional semantic similarity
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
        import numpy as np  # noqa: F401 — ensure it's available
        # paraphrase-multilingual-MiniLM-L12-v2 supports Spanish natively
        _sentence_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
        return _sentence_model
    except ImportError:
        warnings.warn(
            "sentence-transformers not installed. Semantic similarity disabled.\n"
            "Install with: pip install sentence-transformers",
            ImportWarning,
            stacklevel=2,
        )
        return None


def semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two Spanish sentences using a
    multilingual sentence-transformers model.

    Returns a float in [-1, 1] (typically [0, 1] for similar sentences).
    Returns None if the model is unavailable.
    """
    model = _load_sentence_model()
    if model is None:
        return None
    try:
        import numpy as np
        embeddings = model.encode([text1, text2], convert_to_numpy=True)
        # Cosine similarity
        a, b = embeddings[0], embeddings[1]
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        return sim
    except Exception as e:
        warnings.warn(f"Semantic similarity computation failed: {e}", stacklevel=2)
        return None


def is_semantic_model_available() -> bool:
    """Check whether the semantic similarity model is available."""
    return _load_sentence_model() is not None
