#!/usr/bin/env python3
"""
debug_single_example.py — Interactively score a single target/response pair.

Useful for understanding exactly how the rubric is applied and for
validating edge cases against human rater judgments.

Usage
-----
  python scripts/debug_single_example.py
  python scripts/debug_single_example.py --target "Las calles de esta ciudad son muy anchas" \
         --response "Las calles de esta cuidad son muy anchas"
  python scripts/debug_single_example.py --no-semantic
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess_stimulus, preprocess_transcription
from src.scoring import score_utterance
from src.utils import (
    normalize_accents,
    get_content_words,
    compute_content_overlap,
    apply_synonymous_normalization,
    semantic_similarity,
    is_semantic_model_available,
)
from thefuzz import fuzz


def parse_args():
    p = argparse.ArgumentParser(description="Debug a single EIT scoring example.")
    p.add_argument("--target",   "-t", default=None, help="Target/stimulus sentence")
    p.add_argument("--response", "-r", default=None, help="Learner transcription")
    p.add_argument("--no-semantic", action="store_true",
                   help="Disable semantic similarity")
    return p.parse_args()


def get_input_interactively():
    print("\n--- AutoEIT Debug: Single Example Scorer ---")
    target   = input("Target sentence   : ").strip()
    response = input("Learner response  : ").strip()
    return target, response


def debug_score(target_raw: str, response_raw: str, use_semantic: bool = True):
    print("\n" + "=" * 64)
    print("INPUT")
    print(f"  Target   : {target_raw}")
    print(f"  Response : {response_raw}")

    target_clean   = preprocess_stimulus(target_raw)
    response_clean = preprocess_transcription(response_raw)

    print("\nAFTER PREPROCESSING")
    print(f"  Target   : {target_clean}")
    print(f"  Response : {response_clean}")

    if not response_clean:
        print("\nSCORE: 0  (no response / completely unintelligible)")
        return

    # Content word analysis
    t_content = get_content_words(target_clean)
    r_content = get_content_words(response_clean)
    matched, total, overlap = compute_content_overlap(t_content, r_content)

    print("\nCONTENT WORD ANALYSIS")
    print(f"  Target content words   : {t_content}")
    print(f"  Response content words : {r_content}")
    print(f"  Matched                : {matched}/{total} ({overlap:.0%})")

    # String similarity
    t_norm = normalize_accents(target_clean)
    r_norm = normalize_accents(response_clean)
    t_syn, r_syn = apply_synonymous_normalization(t_norm, r_norm)

    ratio       = fuzz.ratio(t_norm, r_norm)
    token_sort  = fuzz.token_sort_ratio(t_norm, r_norm)
    syn_ratio   = fuzz.ratio(t_syn, r_syn)
    syn_tsort   = fuzz.token_sort_ratio(t_syn, r_syn)

    print("\nFUZZY SIMILARITY")
    print(f"  ratio            : {ratio}")
    print(f"  token_sort_ratio : {token_sort}")
    print(f"  syn_ratio        : {syn_ratio}  (after muy/y-pero normalization)")
    print(f"  syn_token_sort   : {syn_tsort}")

    # Semantic similarity
    if use_semantic:
        if is_semantic_model_available():
            sim = semantic_similarity(target_clean, response_clean)
            print(f"\nSEMANTIC SIMILARITY (multilingual sentence-transformer)")
            print(f"  cosine similarity : {sim:.4f}" if sim is not None else "  N/A")
        else:
            print("\nSEMANTIC SIMILARITY: model not available (install sentence-transformers)")

    # Full scoring decision
    result = score_utterance(
        target_clean, response_clean, use_semantic=use_semantic
    )

    print("\nSCORING DECISION")
    print(f"  Score       : {result.score}")
    print(f"  Explanation : {result.explanation}")
    if result.borderline_adjusted:
        print("  ** Borderline case — score adjusted by semantic similarity **")

    print("\nRUBRIC REFERENCE")
    rubric = {
        4: "Exact repetition — form and meaning match exactly",
        3: "Full meaning preserved (grammar errors OK, synonyms allowed)",
        2: ">50% idea units, meaningful but inexact/incomplete",
        1: "~50% idea units, lots missing, or not a self-standing sentence",
        0: "Silence, garbled, or only 1-2 content words matched",
    }
    for score, desc in rubric.items():
        marker = " <<<" if score == result.score else ""
        print(f"  {score}: {desc}{marker}")
    print("=" * 64)


def main():
    args = parse_args()

    if args.target and args.response:
        debug_score(args.target, args.response, use_semantic=not args.no_semantic)
    else:
        # Interactive mode
        while True:
            target, response = get_input_interactively()
            if not target:
                break
            debug_score(target, response, use_semantic=not args.no_semantic)
            print("\n(Press Enter with empty target to quit)\n")


if __name__ == "__main__":
    main()
