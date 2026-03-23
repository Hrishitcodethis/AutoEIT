#!/usr/bin/env python3
"""
score_eit.py — Entry point for AutoEIT scoring.

The scoring system was designed to approximate human EIT scoring using
the Ortega (2000) meaning-based rubric by combining transcription
preprocessing, idea unit overlap analysis, fuzzy string matching,
and rule-based scoring logic with optional semantic similarity.

Usage:
    python score_eit.py                        # run with defaults
    python score_eit.py --no-semantic          # skip sentence-transformers
    python score_eit.py --no-spacy             # skip spaCy POS tagging
    python score_eit.py --verify               # run + verify against expected scores
    python score_eit.py --input path/to/file.xlsx --output path/to/out.xlsx
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.pipeline import run_pipeline


def parse_args():
    p = argparse.ArgumentParser(
        description="Score Spanish EIT transcriptions using the Ortega (2000) rubric."
    )
    p.add_argument("--input", "-i",
                   default=os.path.join(ROOT, "data", "raw",
                                        "AutoEIT Sample Transcriptions for Scoring.xlsx"),
                   help="Path to input Excel file")
    p.add_argument("--output", "-o",
                   default=os.path.join(ROOT, "data", "output",
                                        "AutoEIT_Scored_Results.xlsx"),
                   help="Path for scored Excel output")
    p.add_argument("--csv",
                   default=os.path.join(ROOT, "data", "output",
                                        "preprocessed_transcriptions.csv"),
                   help="Path for preprocessed CSV output")
    p.add_argument("--log",
                   default=os.path.join(ROOT, "data", "output", "scoring_log.txt"),
                   help="Path for scoring log file")
    p.add_argument("--no-semantic", action="store_true",
                   help="Disable sentence-transformer semantic similarity")
    p.add_argument("--no-spacy", action="store_true",
                   help="Disable spaCy POS tagging (use stopword fallback)")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress stdout output")
    p.add_argument("--verify", action="store_true",
                   help="After scoring, verify output against evaluation/expected_scores.json")
    return p.parse_args()


def verify_scores(csv_path: str) -> bool:
    """Compare generated scores against the committed expected_scores.json."""
    expected_path = os.path.join(ROOT, "evaluation", "expected_scores.json")
    if not os.path.exists(expected_path):
        print("VERIFY: expected_scores.json not found — skipping verification.")
        return True

    import csv as csv_mod
    with open(expected_path, encoding="utf-8") as f:
        expected = json.load(f)

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv_mod.DictReader(f)
        actual = [
            {"participant": r["participant"], "sentence": int(r["sentence"]),
             "score": int(r["score"])}
            for r in reader
        ]

    expected_scores = [
        {"participant": e["participant"], "sentence": e["sentence"], "score": e["score"]}
        for e in expected["scores"]
    ]

    if len(actual) != len(expected_scores):
        print(f"VERIFY FAILED: expected {len(expected_scores)} utterances, got {len(actual)}")
        return False

    mismatches = []
    for i, (exp, act) in enumerate(zip(expected_scores, actual)):
        if exp["score"] != act["score"]:
            mismatches.append(
                f"  {exp['participant']} S{exp['sentence']}: "
                f"expected {exp['score']}, got {act['score']}"
            )

    if mismatches:
        print(f"VERIFY FAILED: {len(mismatches)} score mismatches:")
        for m in mismatches[:10]:
            print(m)
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
        return False

    print(f"VERIFY PASSED: all {len(actual)} scores match expected_scores.json")
    return True


if __name__ == "__main__":
    args = parse_args()

    run_pipeline(
        input_path=args.input,
        output_xlsx=args.output,
        output_csv=args.csv,
        log_path=args.log,
        use_spacy=not args.no_spacy,
        use_semantic=not args.no_semantic,
        verbose=not args.quiet,
    )

    if args.verify:
        success = verify_scores(args.csv)
        sys.exit(0 if success else 1)
