"""
pipeline.py — End-to-end AutoEIT scoring pipeline.

Reads the transcription Excel file, detects columns robustly,
preprocesses text, scores each utterance using the Ortega rubric,
and writes results to:
  - data/output/AutoEIT_Scored_Results.xlsx   (original sheets + Score column)
  - data/output/preprocessed_transcriptions.csv  (cleaned text + all features)
  - data/output/scoring_log.txt                  (per-sentence detail log)
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .preprocessing import preprocess_stimulus, preprocess_transcription, preprocess_dataframe
from .scoring import score_utterance, ScoreResult
from .utils import detect_columns, detect_sentence_col

# ---------------------------------------------------------------------------
# Optional dependencies (graceful fallback)
# ---------------------------------------------------------------------------

def _try_load_spacy(model_name: str = "es_core_news_sm"):
    """
    Try to load a spaCy Spanish model for POS-based content-word detection.
    Returns None (with a warning) if spaCy or the model is not installed.

    Install with:
        pip install spacy
        python -m spacy download es_core_news_sm
    """
    try:
        import spacy
        nlp = spacy.load(model_name)
        return nlp
    except ImportError:
        logging.warning(
            "spaCy not installed — using stopword-based content word detection.\n"
            "For research-grade POS tagging: pip install spacy && "
            "python -m spacy download es_core_news_sm"
        )
        return None
    except OSError:
        logging.warning(
            f"spaCy model '{model_name}' not found — using stopword fallback.\n"
            f"Install with: python -m spacy download {model_name}"
        )
        return None


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

class EITPipeline:
    """
    Orchestrates the full scoring workflow for one Excel file.

    Parameters
    ----------
    input_path   : path to the raw transcription Excel file
    output_xlsx  : where to write the scored Excel file
    output_csv   : where to write the preprocessed CSV
    log_path     : where to write the per-sentence log
    use_spacy    : whether to attempt spaCy POS tagging
    use_semantic : whether to use semantic similarity for borderline cases
    verbose      : print progress to stdout
    """

    def __init__(
        self,
        input_path: str | Path,
        output_xlsx: str | Path = "data/output/scored_results.xlsx",
        output_csv: str | Path = "data/processed/preprocessed_transcriptions.csv",
        log_path: str | Path = "data/output/logs.txt",
        use_spacy: bool = True,
        use_semantic: bool = True,
        verbose: bool = True,
    ):
        self.input_path = Path(input_path)
        self.output_xlsx = Path(output_xlsx)
        self.output_csv = Path(output_csv)
        self.log_path = Path(log_path)
        self.use_semantic = use_semantic
        self.verbose = verbose

        # Ensure output directories exist
        self.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Optional spaCy NLP model
        self.nlp = _try_load_spacy() if use_spacy else None

        # Log file handle (opened during run)
        self._log_fh = None

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
        if self._log_fh:
            self._log_fh.write(msg + "\n")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> dict[str, pd.DataFrame]:
        """
        Run the full pipeline. Returns a dict of {sheet_name: scored_df}.
        """
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        all_rows = []   # accumulate for CSV export
        scored_dfs = {} # per-sheet scored DataFrames

        with open(self.log_path, "w", encoding="utf-8") as log_fh:
            self._log_fh = log_fh
            self._log(f"AutoEIT Scoring Pipeline")
            self._log(f"Input : {self.input_path}")
            self._log(f"Output: {self.output_xlsx}")
            self._log("=" * 72)

            xl = pd.ExcelFile(self.input_path)

            with pd.ExcelWriter(self.output_xlsx, engine="openpyxl") as writer:
                for sheet in xl.sheet_names:
                    df = pd.read_excel(xl, sheet_name=sheet)
                    scored_df, sheet_rows = self._score_sheet(sheet, df)
                    scored_df.to_excel(writer, sheet_name=sheet, index=False)
                    scored_dfs[sheet] = scored_df
                    all_rows.extend(sheet_rows)

            self._log_fh = None

        # Write combined CSV of preprocessed transcriptions
        if all_rows:
            csv_df = pd.DataFrame(all_rows)
            csv_df.to_csv(self.output_csv, index=False, encoding="utf-8-sig")
            if self.verbose:
                print(f"\nPreprocessed CSV : {self.output_csv}")

        if self.verbose:
            print(f"Scored Excel     : {self.output_xlsx}")
            print(f"Log              : {self.log_path}")

        return scored_dfs

    # ------------------------------------------------------------------
    # Per-sheet scoring
    # ------------------------------------------------------------------

    def _score_sheet(
        self, sheet_name: str, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[dict]]:
        """Score all utterances in one worksheet."""
        self._log(f"\nParticipant: {sheet_name}")
        self._log("-" * 60)

        # Detect columns robustly
        try:
            stim_col, trans_col = detect_columns(df)
        except ValueError as e:
            self._log(f"  WARNING: {e}\n  Skipping sheet '{sheet_name}'.")
            df["Score"] = None
            return df, []

        sent_col = detect_sentence_col(df)

        df = preprocess_dataframe(df, stim_col, trans_col)

        scores = []
        all_rows = []

        for _, row in df.iterrows():
            sent_num = int(row[sent_col]) if sent_col and pd.notna(row[sent_col]) else "?"
            result = score_utterance(
                row["target_clean"],
                row["response_clean"],
                nlp=self.nlp,
                use_semantic=self.use_semantic,
            )
            scores.append(result.score)

            # Build log entry
            adj_flag = " [sem-adjusted]" if result.borderline_adjusted else ""
            sem_str = f", sem_sim={result.sem_sim:.3f}" if result.sem_sim is not None else ""
            self._log(
                f"  S{str(sent_num):>3} | Score: {result.score} | "
                f"{result.explanation}{adj_flag}{sem_str}"
            )
            self._log(f"         Target  : {result.target_clean}")
            self._log(f"         Response: {result.response_clean}")
            self._log("")

            # Row for CSV
            all_rows.append({
                "participant": sheet_name,
                "sentence": sent_num,
                "target_clean": result.target_clean,
                "response_clean": result.response_clean,
                "score": result.score,
                "content_overlap": round(result.content_overlap, 3),
                "matched_content": result.matched_content,
                "total_content": result.total_content,
                "fuzzy_ratio": round(result.fuzzy_ratio, 1),
                "sem_sim": round(result.sem_sim, 3) if result.sem_sim is not None else "",
                "borderline_adjusted": result.borderline_adjusted,
                "explanation": result.explanation,
            })

        df["Score"] = scores

        # Summary
        total = sum(scores)
        n = len(scores)
        dist = {i: scores.count(i) for i in range(5)}
        self._log(
            f"  TOTAL: {total}/{n * 4}  |  "
            f"Mean: {total / n:.2f}  |  "
            f"Distribution: "
            + "  ".join(f"{i}→{dist[i]}" for i in range(5))
        )

        return df, all_rows


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str | Path,
    output_xlsx: str | Path = "data/output/scored_results.xlsx",
    output_csv: str | Path = "data/processed/preprocessed_transcriptions.csv",
    log_path: str | Path = "data/output/logs.txt",
    use_spacy: bool = True,
    use_semantic: bool = True,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Convenience wrapper around EITPipeline.run()."""
    pipeline = EITPipeline(
        input_path=input_path,
        output_xlsx=output_xlsx,
        output_csv=output_csv,
        log_path=log_path,
        use_spacy=use_spacy,
        use_semantic=use_semantic,
        verbose=verbose,
    )
    return pipeline.run()
