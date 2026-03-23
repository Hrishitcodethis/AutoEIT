"""
rubric.py — Ortega (2000) EIT Scoring Rubric: constants and descriptors.

This module centralizes all rubric-specific knowledge so that scoring
logic, preprocessing, and evaluation can reference a single source of truth.

Reference:
    Ortega, L. (2000). Understanding syntactic complexity: The measurement
    of change in the syntax of instructed L2 Spanish learners.
"""

# ---------------------------------------------------------------------------
# Score descriptors  (Tables 1-5 from the rubric)
# ---------------------------------------------------------------------------

SCORE_DESCRIPTORS = {
    4: {
        "label": "Exact repetition",
        "criteria": (
            "String matches stimulus exactly. Both form and meaning are "
            "correct without exception or doubt."
        ),
    },
    3: {
        "label": "Meaning preserved",
        "criteria": (
            "Original, complete meaning is preserved. Ungrammatical strings "
            "can receive a 3 as long as exact meaning is preserved. "
            "Synonymous substitutions are acceptable: 'muy' may be added or "
            "omitted; 'y'/'pero' are interchangeable. Grammar changes that "
            "do not affect meaning score 3. Ambiguous grammar changes that "
            "could be interpreted as meaning changes from a native-speaker "
            "perspective should be scored as 2."
        ),
    },
    2: {
        "label": "Partial meaning — inexact or incomplete",
        "criteria": (
            "Content preserves at least more than half of the idea units in "
            "the original stimulus. String is meaningful and meaning is close "
            "or related to original, but departs in some slight changes in "
            "content, making it inexact, incomplete, or ambiguous. "
            "General principle: in case of doubt → score 2."
        ),
    },
    1: {
        "label": "Partial — much information missing",
        "criteria": (
            "Only about half of idea units are represented, but a lot of "
            "important information is left out. Sometimes the resulting "
            "meaning is unrelated or opposed to the stimulus. Or the string "
            "does not constitute a self-standing sentence with meaning."
        ),
    },
    0: {
        "label": "Minimal or no response",
        "criteria": (
            "Silence, garbled / unintelligible, or minimal repetition: only "
            "1 word repeated; only 1 content word plus function word(s); "
            "only function word(s); or 1-2 content words out of order plus "
            "extraneous words not in the original stimulus."
        ),
    },
}


# ---------------------------------------------------------------------------
# Scoring exceptions (from the rubric's Exceptions section)
# ---------------------------------------------------------------------------

SCORING_EXCEPTIONS = """
1. If a subject responds before the tone → no penalty. Score the best final response.
2. Self-corrections, hesitations, or false starts → no penalty.
   Always judge the best final response to the stimulus.
3. A false start is when a subject starts a word but later corrects the infraction.
   Example: "La cantidad de personas que fuman ha dis disminuido."
"""


# ---------------------------------------------------------------------------
# Spanish function words — stopwords for content-word extraction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Synonymous substitution rules  (explicit in the rubric)
# ---------------------------------------------------------------------------

SYNONYMOUS_RULES = """
Per the rubric, the following substitutions do NOT constitute meaning change:
  - Anything with or without 'muy' ('very') should be considered synonymous.
  - Substitutions of 'y' / 'pero' (and / but) are acceptable.
"""


def apply_synonymous_normalization(target: str, response: str) -> tuple[str, str]:
    """
    Normalize both strings according to the rubric's synonymous-substitution
    rules before comparison:
      - Drop 'muy' (optional per rubric)
      - Canonicalize conjunctions 'pero'/'sino'/'e' → 'y'
    """
    def _normalize(words: list[str]) -> list[str]:
        words = [w for w in words if w != "muy"]
        words = ["y" if w in ("pero", "sino", "e") else w for w in words]
        return words

    t = _normalize(target.split())
    r = _normalize(response.split())
    return " ".join(t), " ".join(r)
