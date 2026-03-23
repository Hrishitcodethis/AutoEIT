"""
preprocessing.py — Transcription cleaning pipeline for AutoEIT scoring.

Follows the EIT Data Processing Protocol (MFS/CogSLA Lab UIC):
  - Remove annotation conventions: [gibberish], [pause], xxx/XX
  - Extract best final response from self-corrections
  - Remove false starts (brackets), stuttering, abandoned word fragments
  - Normalize accents, lowercase, collapse whitespace

Stimulus preprocessing is simpler: just strip the trailing syllable count.
"""

import re
import pandas as pd


# ---------------------------------------------------------------------------
# Stimulus preprocessing
# ---------------------------------------------------------------------------

def preprocess_stimulus(text) -> str:
    """
    Clean a target/prompt sentence:
      - Strip trailing syllable count, e.g. "(7)" or "(14)"
      - Lowercase and normalize whitespace
    """
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # Remove trailing syllable count like "(7)", "(14)", etc.
    text = re.sub(r"\s*\(\d+\)\s*$", "", text)
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Transcription preprocessing
# ---------------------------------------------------------------------------

def preprocess_transcription(text) -> str:
    """
    Full cleaning pipeline for a learner transcription.

    Steps (in order):
    1. Handle no-response markers
    2. Remove bracketed annotations: [gibberish], [pause], [cough], [no response]
    3. Remove unintelligible markers: xxx, XX, X (standalone)
    4. Remove parenthetical comments: (tambien?), (xxx?)
    5. Extract best final response from self-corrections ("Mis gus..Me gustas...")
    6. Remove false starts in brackets: [la-], [gustan-]
    7. Remove syllable-level stuttering: "co-co-comerme" → "comerme"
    8. Remove word-level repetition stuttering: "se se se" → "se"
    9. Remove abandoned word fragments ending with "-": "ma-" → ""
    10. Remove filler words: um, uh, mhh, meh, uf
    11. Remove ellipsis / pause markers
    12. Strip punctuation, lowercase, normalize whitespace
    """
    if pd.isna(text):
        return ""

    text = str(text).strip()

    # 1. No-response markers
    if re.fullmatch(
        r"\[?(?:no response|silence|no\s+response)\]?", text, flags=re.IGNORECASE
    ):
        return ""

    # 2. Remove bracketed annotations
    for pattern in (r"\[gibberish\]", r"\[pause\]", r"\[cough\]"):
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[.*?\]", "", text)  # any remaining bracketed content

    # 3. Remove standalone unintelligible markers (x / xx / xxx / XX, etc.)
    text = re.sub(r"\b[xX]{1,}\b", "", text)

    # 4. Remove parenthetical comments
    text = re.sub(r"\(.*?\)", "", text)

    # 5. Extract best final response from self-corrections
    text = _extract_best_response(text)

    # 6. Remove false starts in brackets, e.g. [la-], [gustan-]
    text = re.sub(r"\[\w+-\]", "", text)

    # 7. Remove syllable stuttering: "co-co-comerme" → "comerme"
    #    Pattern: two or more identical hyphenated syllable fragments
    text = re.sub(r"(\b\w+-)\1+", "", text)

    # 8. Remove word-level stuttering: "se se se" → "se"
    text = re.sub(r"\b(\w+)(?:\s+\1){2,}\b", r"\1", text)

    # 9. Remove abandoned word fragments (word ending with "-")
    text = re.sub(r"\b\w+-(?=\s|$)", "", text)

    # 10. Remove fillers
    text = re.sub(r"\b(?:um|uh|mhh|meh|uf|hmm)\b", "", text, flags=re.IGNORECASE)

    # 11. Remove ellipsis and mid-word pauses
    text = text.replace("...", " ").replace("..", " ")

    # 12. Strip remaining punctuation, lowercase, normalize whitespace
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Best-response extraction
# ---------------------------------------------------------------------------

def _extract_best_response(text: str) -> str:
    """
    Per the rubric: score the best final response.

    When a participant self-corrects, the transcription often shows:
        "Mis gus..Me gustas las películas"
        "Queda una ca..Quiere una casa"

    Strategy:
      - Split on ".." (double-dot restarts)
      - Prefer the longest segment (most complete attempt)
      - If the last segment is at least 3 words, use it preferentially
        (it reflects the final, corrected response)
    """
    parts = re.split(r"(?<!\.)\.\.(?!\.)", text)
    if len(parts) <= 1:
        return text

    # Trim each part
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return text

    last = parts[-1]
    longest = max(parts, key=lambda p: len(p.split()))

    # Prefer the last segment if it's reasonably long (self-correction completed)
    if len(last.split()) >= 3:
        return last

    # Otherwise use the longest (participant likely gave up mid-correction)
    return longest


# ---------------------------------------------------------------------------
# Convenience: process a full DataFrame column
# ---------------------------------------------------------------------------

def preprocess_dataframe(df, stimulus_col: str, transcription_col: str) -> pd.DataFrame:
    """
    Add cleaned columns to a copy of the DataFrame.
    Returns df with two new columns: 'target_clean' and 'response_clean'.
    """
    df = df.copy()
    df["target_clean"] = df[stimulus_col].apply(preprocess_stimulus)
    df["response_clean"] = df[transcription_col].apply(preprocess_transcription)
    return df
