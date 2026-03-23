#!/usr/bin/env python3
"""
run_scoring.py — CLI entry point for the AutoEIT scoring pipeline.

Usage:
  python scripts/run_scoring.py
  python scripts/run_scoring.py --no-semantic --no-spacy
  python scripts/run_scoring.py --input "path/to/file.xlsx" --quiet
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import run_pipeline

DEFAULT_INPUT = ROOT / "data" / "raw" / "AutoEIT Sample Transcriptions for Scoring.xlsx"
DEFAULT_OUTPUT = ROOT / "data" / "output" / "AutoEIT_Scored_Results.xlsx"
DEFAULT_CSV = ROOT / "data" / "output" / "preprocessed_transcriptions.csv"
DEFAULT_LOG = ROOT / "data" / "output" / "scoring_log.txt"


def main():
    p = argparse.ArgumentParser(
        description="Score Spanish EIT transcriptions using the Ortega (2000) rubric."
    )
    p.add_argument("--input",  "-i", default=DEFAULT_INPUT,  help="Input Excel file")
    p.add_argument("--output", "-o", default=DEFAULT_OUTPUT,  help="Scored Excel output")
    p.add_argument("--csv",          default=DEFAULT_CSV,     help="Preprocessed CSV output")
    p.add_argument("--log",          default=DEFAULT_LOG,     help="Scoring log file")
    p.add_argument("--no-semantic",  action="store_true",     help="Disable semantic similarity")
    p.add_argument("--no-spacy",     action="store_true",     help="Disable spaCy POS tagging")
    p.add_argument("--quiet", "-q",  action="store_true",     help="Suppress stdout")
    args = p.parse_args()

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
