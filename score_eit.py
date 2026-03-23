#!/usr/bin/env python3
"""
score_eit.py — Backward-compatible entry point for AutoEIT scoring.

The scoring system was designed to approximate human EIT scoring using
the Ortega (2000) meaning-based rubric by combining transcription
preprocessing, idea unit overlap analysis, fuzzy string matching,
and rule-based scoring logic with optional semantic similarity.

For the full modular implementation see src/ and scripts/.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline(
        input_path=os.path.join(ROOT, "data", "raw",
                                "AutoEIT Sample Transcriptions for Scoring.xlsx"),
        output_xlsx=os.path.join(ROOT, "data", "output",
                                 "AutoEIT_Scored_Results.xlsx"),
        output_csv=os.path.join(ROOT, "data", "output",
                                "preprocessed_transcriptions.csv"),
        log_path=os.path.join(ROOT, "data", "output", "scoring_log.txt"),
        use_spacy=True,
        use_semantic=True,
        verbose=True,
    )
