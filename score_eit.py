"""
score_eit.py — Backward-compatible entry point.

For the full modular implementation see src/ and scripts/.
This thin wrapper calls the pipeline with the original file paths so
existing usage (`python score_eit.py`) continues to work unchanged.

AutoEIT Scoring Script — Specific Test II
==========================================
Applies the Ortega (2000) meaning-based EIT scoring rubric to sentence
transcriptions, comparing learner utterances to prompt (stimulus) sentences.

APPROACH
--------
1. **Preprocessing**: Clean transcriptions following the protocol conventions:
   - Remove false starts in brackets, e.g. [gustan-] -> removed
   - Remove stuttering, e.g. "co-co-comerme" -> "comerme"
   - Remove pauses "...", "[pause]"
   - Remove unintelligible markers "[gibberish]", "xxx", "XX"
   - Remove trailing dashes from abandoned words, e.g. "ma-" -> removed
   - Extract the "best final response" when self-corrections are present
   - Normalize: lowercase, strip punctuation, collapse whitespace

2. **Scoring Logic** (Ortega, 2000 rubric):
   - Score 4: Exact repetition — form and meaning match the stimulus exactly.
   - Score 3: Meaning fully preserved. Grammar errors OK if meaning unchanged.
              Synonymous substitutions allowed (muy optional, y/pero interchangeable).
   - Score 2: >50% of idea units preserved, string is meaningful, meaning is
              close/related but inexact, incomplete, or ambiguous. When in doubt
              between 2 and 3, score 2.
   - Score 1: ~50% of idea units, lots of important info missing, meaning may be
              unrelated/opposed, OR string is not a self-standing sentence.
   - Score 0: Silence, garbled/unintelligible, minimal repetition (only 1 word,
              only function words, 1-2 content words out of order + extraneous).

3. **Idea Unit Analysis**: Content words (nouns, verbs, adjectives, adverbs) are
   extracted by filtering out Spanish function words. Overlap is computed via
   fuzzy matching to tolerate minor spelling/accent differences.

4. **Evaluation**: The script outputs per-sentence scoring details to the console
   for manual spot-checking, and writes scores to the "Score" column in the
   output Excel file.

REQUIREMENTS
------------
pip install pandas openpyxl thefuzz python-Levenshtein

USAGE
-----
python score_eit.py
"""

import pandas as pd
import re
import os
import unicodedata
from thefuzz import fuzz


# ---------------------------------------------------------------------------
# Spanish function words — used to identify content vs. function words
# ---------------------------------------------------------------------------
SPANISH_FUNCTION_WORDS = {
    # articles
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo",
    # prepositions
    "a", "al", "de", "del", "en", "con", "por", "para", "sin",
    "sobre", "entre", "hacia", "hasta", "desde", "ante",
    # conjunctions
    "y", "e", "o", "u", "pero", "sino", "ni", "que", "como",
    # pronouns (clitic / subject / demonstrative)
    "me", "te", "se", "le", "les", "nos", "os",
    "yo", "tu", "el", "ella", "usted", "nosotros", "ustedes",
    "ellos", "ellas", "mio", "mia",
    "lo", "la", "los", "las",  # object pronouns overlap with articles
    "mi", "mis", "su", "sus", "tu", "tus",
    # common function words
    "es", "ha", "he", "han", "hay", "fue", "ser", "muy",
    "no", "si", "ya", "mas", "tan", "todo", "toda",
    "este", "esta", "ese", "esa", "esto", "eso",
    "donde", "cuando", "quien",
}

# Synonymous pairs that should NOT cause a score reduction per the rubric
SYNONYMOUS_SUBS = [
    ({"y", "e"}, {"pero", "sino"}),     # y/pero are interchangeable
]


def normalize_accents(text):
    """Strip accent marks for comparison purposes (e.g., está -> esta)."""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def preprocess_text(text, is_stimulus=False):
    """
    Clean transcription text following the EIT protocol conventions.
    For stimuli, only strip the trailing syllable count (e.g., "(7)").
    For transcriptions, apply full cleaning pipeline.
    """
    if pd.isna(text):
        return ""

    text = str(text).strip()

    if is_stimulus:
        # Remove trailing syllable count like (7), (14), etc.
        text = re.sub(r'\s*\(\d+\)\s*$', '', text)
        text = text.strip()
        # Normalize whitespace, lowercase
        text = " ".join(text.lower().split())
        return text

    # Check for no response
    if text.lower() in ("[no response]", "no response", "[silence]", ""):
        return ""

    # Remove annotations: [gibberish], [pause], [cough], etc.
    text = re.sub(r'\[gibberish\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[pause\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[cough\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?\]', '', text)  # any remaining bracketed annotations

    # Remove unintelligible markers: xxx, XX, x (standalone)
    text = re.sub(r'\bx{1,}\b', '', text, flags=re.IGNORECASE)

    # Remove parenthetical comments like (tambien?) or (xxx?)
    text = re.sub(r'\(.*?\)', '', text)

    # Handle self-corrections: pick the best final attempt
    # e.g., "Mis gus..Me gustas las películas" -> "Me gustas las películas"
    # e.g., "Queda una ca..Quiere una casa" -> "Quiere una casa"
    # Strategy: if there's a pattern of abandoned-then-restarted, take the latter
    text = extract_best_response(text)

    # Remove false starts in brackets like [la-], [gustan-]
    text = re.sub(r'\[[\w]+-\]', '', text)

    # Remove stuttering: "co-co-comerme" -> "comerme", "se se se" -> "se"
    # Pattern: repeated syllable fragments with hyphens
    text = re.sub(r'(\b\w+-)(?:\1)+', '', text)  # repeated hyphenated fragments
    # Handle "se se se" type stuttering
    text = re.sub(r'\b(\w+)\s+(?:\1\s+)+', r'\1 ', text)

    # Remove abandoned word fragments ending with dash: "ma-" -> ""
    # But preserve the word if it's a false start that was corrected
    text = re.sub(r'\b\w+-\s', ' ', text)
    text = re.sub(r'\b\w+-$', '', text)

    # Remove "um", "uh", "mhh", "meh" type fillers
    text = re.sub(r'\b(?:um|uh|mhh|meh|uf)\b', '', text, flags=re.IGNORECASE)

    # Remove ellipsis / pauses
    text = text.replace('...', ' ').replace('..', ' ')

    # Remove punctuation (keep letters and spaces)
    text = re.sub(r"[^\w\s]", '', text)

    # Normalize whitespace and lowercase
    text = " ".join(text.lower().split())

    return text


def extract_best_response(text):
    """
    Per the rubric: "Always score the best final response."
    If the participant self-corrects, pick the best (usually last complete) attempt.

    Heuristic: if there's a clear restart pattern (sentence fragment .. new sentence),
    take the longest coherent segment.
    """
    # Split on double-dot restarts that indicate self-correction
    # e.g., "Mis gus..Me gustas las películas que cada bien"
    # e.g., "Queda una ca..Quiere una casa en queda mis animales"
    parts = re.split(r'(?<!\.)\.\.(?!\.)', text)
    if len(parts) > 1:
        # Take the longest part (likely the most complete attempt)
        # But prefer later parts if they're reasonably long
        best = parts[-1].strip()
        for part in reversed(parts[:-1]):
            part = part.strip()
            if len(part.split()) > len(best.split()):
                best = part
        # If the last part is very short, it might be an afterthought
        # Keep original if no clear winner
        if len(best.split()) >= 3:
            return best

    return text


def get_content_words(text):
    """Extract content words (non-function words with length > 1)."""
    words = text.split()
    # Normalize accents for comparison against function word list
    content = []
    for w in words:
        w_norm = normalize_accents(w)
        if w_norm not in SPANISH_FUNCTION_WORDS and len(w) > 1:
            content.append(w)
    return content


def get_all_words(text):
    """Get all words from text."""
    return text.split()


def words_match(w1, w2, threshold=85):
    """Check if two words are similar enough (handles accent/spelling differences)."""
    if w1 == w2:
        return True
    # Compare with accents stripped
    if normalize_accents(w1) == normalize_accents(w2):
        return True
    # Fuzzy match for close misspellings
    return fuzz.ratio(w1, w2) >= threshold


def is_synonymous_substitution(w_target, w_response):
    """Check if the substitution is considered synonymous per the rubric."""
    w1 = normalize_accents(w_target)
    w2 = normalize_accents(w_response)
    for group_a, group_b in SYNONYMOUS_SUBS:
        all_syns = group_a | group_b
        if w1 in all_syns and w2 in all_syns:
            return True
    return False


def compute_idea_unit_overlap(target_content, response_content):
    """
    Compute the fraction of target content words matched in the response.
    Returns (matched_count, total_count, overlap_ratio).
    """
    if not target_content:
        return 0, 0, 1.0  # Edge case: no content words in stimulus

    matched = 0
    used_response = set()
    for tw in target_content:
        for i, rw in enumerate(response_content):
            if i not in used_response and words_match(tw, rw):
                matched += 1
                used_response.add(i)
                break
    return matched, len(target_content), matched / len(target_content)


def is_meaningful_sentence(text):
    """
    Check if the response constitutes a self-standing meaningful sentence.
    A simple heuristic: needs at least a verb-like word and a subject/object.
    """
    words = text.split()
    if len(words) < 3:
        return False
    return True


def apply_synonymous_normalization(target, response):
    """
    Normalize known synonymous substitutions before comparison.
    - 'muy' is optional (with/without should be considered equivalent)
    - 'y'/'pero' are interchangeable
    """
    # Remove 'muy' from both for comparison
    t_words = target.split()
    r_words = response.split()

    t_no_muy = [w for w in t_words if w != 'muy']
    r_no_muy = [w for w in r_words if w != 'muy']

    # Normalize y/pero/e
    def norm_conj(words):
        return ['y' if w in ('pero', 'sino', 'e') else w for w in words]

    t_normed = norm_conj(t_no_muy)
    r_normed = norm_conj(r_no_muy)

    return ' '.join(t_normed), ' '.join(r_normed)


def calculate_score(target_raw, transcription_raw):
    """
    Apply the Ortega (2000) EIT scoring rubric.

    Returns (score, explanation) tuple.
    """
    t_clean = preprocess_text(target_raw, is_stimulus=True)
    r_clean = preprocess_text(transcription_raw, is_stimulus=False)

    # --- Score 0: No response / silence ---
    if not r_clean:
        return 0, "No response or completely unintelligible"

    # Get word lists
    t_words = get_all_words(t_clean)
    r_words = get_all_words(r_clean)
    t_content = get_content_words(t_clean)
    r_content = get_content_words(r_clean)

    # --- Score 0: Minimal repetition checks ---
    # Only 1 word total
    if len(r_words) <= 1:
        return 0, f"Only {len(r_words)} word(s) repeated"

    # Only function words repeated (no content words from stimulus)
    _, _, content_overlap = compute_idea_unit_overlap(t_content, r_content)
    matched_content_count, total_content_count, _ = compute_idea_unit_overlap(
        t_content, r_content
    )

    # Check how many response content words match stimulus content words
    if total_content_count > 0 and matched_content_count <= 1 and len(r_content) <= 2:
        return 0, (f"Minimal repetition: {matched_content_count}/{total_content_count} "
                    f"content words matched")

    # --- Score 4: Exact repetition ---
    # Compare with accent normalization
    t_norm = normalize_accents(t_clean)
    r_norm = normalize_accents(r_clean)

    if t_norm == r_norm:
        return 4, "Exact repetition"

    # --- Score 3/4 boundary: Check with synonymous normalization ---
    t_syn, r_syn = apply_synonymous_normalization(t_norm, r_norm)
    if t_syn == r_syn:
        return 4, "Exact match after synonymous normalization (muy/y-pero)"

    # --- Compute similarity metrics ---
    # Token-level similarity
    token_sort = fuzz.token_sort_ratio(t_norm, r_norm)
    ratio = fuzz.ratio(t_norm, r_norm)

    # Synonymous-normalized similarity
    syn_ratio = fuzz.ratio(t_syn, r_syn)
    syn_token_sort = fuzz.token_sort_ratio(t_syn, r_syn)

    # Word-level overlap (all words, not just content)
    all_matched = sum(1 for tw in t_words
                      if any(words_match(tw, rw) for rw in r_words))
    all_overlap = all_matched / len(t_words) if t_words else 0

    # --- Score 3: Meaning fully preserved ---
    # Guard: if the response is much shorter than the target (truncated),
    # it likely doesn't preserve full meaning — cap at 2
    length_ratio = len(r_words) / len(t_words) if t_words else 1.0
    truncated = length_ratio < 0.55

    # High similarity after synonymous normalization, nearly all content words present
    if (not truncated and content_overlap >= 0.85 and syn_token_sort >= 80 and
            all_overlap >= 0.7 and is_meaningful_sentence(r_clean)):
        return 3, (f"Meaning preserved: {matched_content_count}/{total_content_count} "
                    f"content words, syn_sort={syn_token_sort}")

    if not truncated and syn_ratio >= 85 and content_overlap >= 0.7:
        return 3, f"High similarity (syn_ratio={syn_ratio}), content overlap={content_overlap:.0%}"

    # Near-exact with minor grammar changes
    if not truncated and ratio >= 80 and content_overlap >= 0.8:
        return 3, f"Near-exact (ratio={ratio}), content overlap={content_overlap:.0%}"

    # --- Score 2: >50% idea units, meaningful, close but inexact ---
    if (content_overlap > 0.5 and is_meaningful_sentence(r_clean)):
        return 2, (f"Partial meaning: {matched_content_count}/{total_content_count} "
                    f"content words, meaningful sentence")

    if content_overlap > 0.5 or (all_overlap > 0.5 and syn_ratio >= 50):
        return 2, (f"Related meaning: content={content_overlap:.0%}, "
                    f"all_overlap={all_overlap:.0%}")

    # --- Score 1: ~50% idea units, incomplete, or unrelated meaning ---
    if (content_overlap >= 0.3 or all_overlap >= 0.35 or
            (matched_content_count >= 2 and is_meaningful_sentence(r_clean))):
        return 1, (f"Partial/incomplete: {matched_content_count}/{total_content_count} "
                    f"content words, all_overlap={all_overlap:.0%}")

    if ratio >= 35 or token_sort >= 40:
        return 1, f"Low similarity (ratio={ratio}, token_sort={token_sort})"

    # --- Score 0: Very little preserved ---
    return 0, (f"Minimal match: {matched_content_count}/{total_content_count} "
                f"content words, ratio={ratio}")


def process_file(input_path, output_path):
    """Process all participant sheets and write scores."""
    print(f"Loading: {input_path}")
    print("=" * 80)
    xl = pd.ExcelFile(input_path)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Copy 'Info' sheet as-is
        if 'Info' in xl.sheet_names:
            pd.read_excel(xl, sheet_name='Info').to_excel(
                writer, sheet_name='Info', index=False
            )

        for sheet in xl.sheet_names:
            if sheet == 'Info':
                continue

            print(f"\nScoring participant: {sheet}")
            print("-" * 60)
            df = pd.read_excel(xl, sheet_name=sheet)

            scores = []
            explanations = []

            for _, row in df.iterrows():
                sent_num = int(row.get('Sentence', 0))
                target = str(row.get('Stimulus', ''))
                transcript = str(row.get('Transcription Rater 1', ''))

                score, explanation = calculate_score(target, transcript)
                scores.append(score)
                explanations.append(explanation)

                # Console output for review
                t_clean = preprocess_text(target, is_stimulus=True)
                r_clean = preprocess_text(transcript, is_stimulus=False)
                print(f"  S{sent_num:2d} | Score: {score} | {explanation}")
                print(f"       Target:   {t_clean}")
                print(f"       Response: {r_clean}")
                print()

            df['Score'] = scores

            # Summary statistics
            total = sum(scores)
            max_possible = len(scores) * 4
            print(f"  TOTAL: {total}/{max_possible} "
                  f"(Mean: {total/len(scores):.2f})")
            print(f"  Distribution: "
                  f"0s={scores.count(0)}, 1s={scores.count(1)}, "
                  f"2s={scores.count(2)}, 3s={scores.count(3)}, "
                  f"4s={scores.count(4)}")

            df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"\n{'=' * 80}")
    print(f"Scores written to: {output_path}")


# ---------------------------------------------------------------------------
# EVALUATION APPROACH
# ---------------------------------------------------------------------------
EVALUATION_NOTES = """
EVALUATION APPROACH
===================
1. **Spot-check against rubric examples**: The rubric provides scored examples
   for each level (0-4). I manually verified the script's output against these
   known cases to calibrate thresholds.

2. **Console output for transparency**: Every sentence prints its cleaned
   target, cleaned response, assigned score, and the reasoning. This makes it
   easy for a human rater to verify each decision.

3. **Key rubric rules encoded**:
   - Score 4 requires exact match (accent-normalized)
   - 'muy' presence/absence is treated as synonymous (per rubric)
   - 'y'/'pero' substitutions are accepted (per rubric)
   - False starts, pauses, and self-corrections are removed before scoring
   - Best final response is extracted from self-corrections
   - [gibberish], xxx, [pause] markers are stripped
   - Content word overlap drives the 0-1-2-3 distinctions

4. **Limitations and areas for improvement**:
   - Content word detection uses a stopword list rather than POS tagging;
     a Spanish NLP pipeline (e.g., spaCy with es_core_news_sm) would improve
     content vs. function word classification.
   - Meaning preservation is approximated via word overlap and string
     similarity, not true semantic analysis. An LLM-based approach could
     better judge whether meaning changed.
   - The "self-standing sentence" check is simplistic (word count >= 3).
   - Edge cases where grammar changes create ambiguous meaning shifts
     (rubric says score 2 when in doubt) may not always be caught.

5. **Recommended validation**: Compare automated scores against human rater
   scores on a gold-standard subset. Calculate inter-rater agreement (Cohen's
   kappa or weighted kappa) to quantify alignment with human judgment.
"""


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Delegate to the modular pipeline (src/pipeline.py).
    # Original data/AutoEIT_Scored_Results.xlsx path is preserved for
    # backward compatibility.
    # -----------------------------------------------------------------------
    import sys
    _ROOT = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _ROOT)

    try:
        from src.pipeline import run_pipeline
        run_pipeline(
            input_path=os.path.join(
                _ROOT, "data", "raw",
                "AutoEIT Sample Transcriptions for Scoring.xlsx"
            ),
            output_xlsx=os.path.join(_ROOT, "outputs", "scored_results.xlsx"),
            output_csv=os.path.join(
                _ROOT, "data", "processed", "preprocessed_transcriptions.csv"
            ),
            log_path=os.path.join(_ROOT, "outputs", "logs.txt"),
            use_spacy=True,
            use_semantic=True,
            verbose=True,
        )
    except Exception as e:
        print(f"Modular pipeline failed ({e}), falling back to legacy scorer.")
        input_file = os.path.join(
            _ROOT, 'data', 'AutoEIT Sample Transcriptions for Scoring.xlsx'
        )
        output_file = os.path.join(_ROOT, 'data', 'AutoEIT_Scored_Results.xlsx')
        print(EVALUATION_NOTES)
        if os.path.exists(input_file):
            process_file(input_file, output_file)
        else:
            print(f"Input file not found: {input_file}")
