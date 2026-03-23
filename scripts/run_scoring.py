#!/usr/bin/env python3
"""
run_scoring.py — CLI entry point for the AutoEIT scoring pipeline.

Usage
-----
  python scripts/run_scoring.py
  python scripts/run_scoring.py --input "data/raw/AutoEIT Sample Transcriptions for Scoring.xlsx"
  python scripts/run_scoring.py --no-semantic   # skip sentence-transformers
  python scripts/run_scoring.py --no-spacy      # skip spaCy POS tagging
  python scripts/run_scoring.py --quiet         # suppress stdout (log only)

Outputs
-------
  outputs/scored_results.xlsx                    - original sheets + Score column
  data/processed/preprocessed_transcriptions.csv - cleaned text + all features
  outputs/logs.txt                               - per-sentence detail log
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import run_pipeline

DEFAULT_INPUT = ROOT / "data" / "raw" / "AutoEIT Sample Transcriptions for Scoring.xlsx"
DEFAULT_OUTPUT_XLSX = ROOT / "outputs" / "scored_results.xlsx"
DEFAULT_OUTPUT_CSV  = ROOT / "data" / "processed" / "preprocessed_transcriptions.csv"
DEFAULT_LOG         = ROOT / "outputs" / "logs.txt"


def parse_args():
    p = argparse.ArgumentParser(
        description="Score AutoEIT transcriptions using the Ortega (2000) rubric."
    )
    p.add_argument("--input",  "-i", default=DEFAULT_INPUT,
                   help="Path to input Excel file")
    p.add_argument("--output", "-o", default=DEFAULT_OUTPUT_XLSX,
                   help="Path for scored Excel output")
    p.add_argument("--csv",          default=DEFAULT_OUTPUT_CSV,
                   help="Path for preprocessed CSV")
    p.add_argument("--log",          default=DEFAULT_LOG,
                   help="Path for log file")
    p.add_argument("--no-semantic",  action="store_true",
                   help="Disable sentence-transformer semantic similarity")
    p.add_argument("--no-spacy",     action="store_true",
                   help="Disable spaCy POS tagging (use stopword fallback)")
    p.add_argument("--quiet", "-q",  action="store_true",
                   help="Suppress stdout output (write to log only)")
    return p.parse_args()


def main():
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


if __name__ == "__main__":
    main()
