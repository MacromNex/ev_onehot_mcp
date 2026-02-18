"""
Protein fitness prediction using ev+onehot model.

This MCP Server provides 1 tool:
1. ev_onehot_predict_fitness: Predict fitness for protein sequences using a pretrained EV+Onehot model

All tools extracted from combining-evolutionary-and-assay-labelled-data/src/evaluate.py.
"""

# Standard imports
from typing import Annotated, Any
import pandas as pd
import numpy as np
from pathlib import Path
import os
from fastmcp import FastMCP
from datetime import datetime
import sys
from loguru import logger

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_INPUT_DIR = PROJECT_ROOT / "tmp" / "inputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp" / "outputs"

INPUT_DIR = Path(os.environ.get("PRED_INPUT_DIR", DEFAULT_INPUT_DIR))
OUTPUT_DIR = Path(os.environ.get("PRED_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp for unique outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Add repo to path for imports
REPO_PATH = PROJECT_ROOT / "repo" / "ev_onehot"
sys.path.insert(0, str(REPO_PATH))

from predictor import JointPredictor, EVPredictor, OnehotRidgePredictor
from util import spearman, ndcg, auroc

# MCP server instance
pred_mcp = FastMCP(name="pred")


@pred_mcp.tool
def ev_onehot_predict_fitness(
    model_dir: Annotated[str, "Path to model directory containing: wt.fasta (wild-type sequence), plmc/ directory (EVmutation parameters), and ridge_model.joblib (pretrained model)"],
    csv_file: Annotated[str | None, "Path to CSV file with sequences to predict. Must contain a sequence column (default: 'seq'). Optionally contains 'log_fitness' column for evaluation."] = None,
    sequences: Annotated[list[str] | None, "List of protein sequences to predict fitness for. Alternative to csv_file."] = None,
    seq_col: Annotated[str, "Name of the column containing protein sequences in the CSV file"] = "seq",
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Predict fitness for protein sequences using a pretrained EV+Onehot model.
    Input is model directory with pretrained model files and sequences to predict, output is predictions CSV file.
    """
    logger.info("="*70)
    logger.info("Starting Protein Fitness Prediction")
    logger.info("="*70)

    # Input validation
    logger.info(f"Model directory: {model_dir}")

    # Directory existence validation
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Check required files in model directory
    logger.info("\nChecking required files...")
    model_file = model_path / "ridge_model.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"Trained model not found: {model_file}. Please run training first.")
    logger.info(f"  ✓ Trained model: {model_file.name}")

    wt_file = model_path / "wt.fasta"
    if not wt_file.exists():
        raise FileNotFoundError(f"Wild-type FASTA not found: {wt_file}")
    logger.info(f"  ✓ Wild-type FASTA: {wt_file.name}")

    plmc_dir = model_path / "plmc"
    if not plmc_dir.exists():
        raise FileNotFoundError(f"PLMC directory not found: {plmc_dir}")
    logger.info(f"  ✓ PLMC directory: {plmc_dir.name}")

    # Load sequences
    logger.info("\nLoading sequences...")
    if csv_file is not None:
        if not Path(csv_file).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        test_df = pd.read_csv(csv_file)

        # Check if sequence column exists
        if seq_col not in test_df.columns:
            available_cols = ', '.join(test_df.columns)
            raise ValueError(
                f"Column '{seq_col}' not found in CSV file. "
                f"Available columns: {available_cols}"
            )

        logger.info(f"  Source: {Path(csv_file).name}")
        logger.info(f"  Sequence column: '{seq_col}'")

        # Rename to 'seq' for internal processing
        if seq_col != 'seq':
            test_df = test_df.rename(columns={seq_col: 'seq'})
            logger.info(f"  Renamed '{seq_col}' → 'seq' for processing")

        has_fitness = 'log_fitness' in test_df.columns
        if has_fitness:
            logger.info(f"  Mode: Evaluation (fitness labels provided)")
        else:
            logger.info(f"  Mode: Prediction only")
    elif sequences is not None:
        test_df = pd.DataFrame({'seq': sequences})
        logger.info(f"  Source: Sequence list ({len(sequences)} sequences)")
        logger.info(f"  Mode: Prediction only")
    else:
        raise ValueError("Either csv_file or sequences must be provided")

    logger.info(f'  Total sequences: {len(test_df)}')

    # Load predictor
    logger.info("\nLoading model...")
    # Use combined EV+Onehot predictor
    predictor_cls = [EVPredictor, OnehotRidgePredictor]
    predictor_names = ['ev', 'onehot']
    predictor = JointPredictor(str(model_path), predictor_cls, predictor_name=predictor_names)
    predictor.load_model()
    logger.info(f"  ✓ Model loaded from {model_file.name}")

    # Make predictions
    logger.info("\nMaking predictions...")
    test_pred = predictor.predict(test_df.seq.values)
    test_df['pred_fitness'] = test_pred
    logger.info(f"  ✓ Predictions completed for {len(test_df)} sequences")

    # Show prediction statistics
    logger.info(f"\nPrediction statistics:")
    logger.info(f"  Mean: {test_pred.mean():.4f}")
    logger.info(f"  Std:  {test_pred.std():.4f}")
    logger.info(f"  Min:  {test_pred.min():.4f}")
    logger.info(f"  Max:  {test_pred.max():.4f}")

    # Compute metrics if log_fitness column exists
    metrics_msg = ""
    if 'log_fitness' in test_df.columns:
        logger.info("\nEvaluation metrics:")
        metric_fns = {'spearman': spearman, 'auroc': auroc}
        np.set_printoptions(precision=3)
        test_ret = []
        for m, mfn in metric_fns.items():
            value = mfn(test_pred, test_df['log_fitness'].to_numpy())
            logger.info(f"  {m.capitalize()}: {value:.4f}")
            test_ret.append(f"{m}: {value:.3f}")
        metrics_msg = ", ".join(test_ret)

    # Prepare output path
    if out_prefix is None:
        out_prefix = f"predictions_{timestamp}"

    logger.info(f"\nSaving predictions...")
    output_file = OUTPUT_DIR / f"{out_prefix}.csv"
    test_df.to_csv(output_file, index=False)
    output_size_kb = output_file.stat().st_size / 1024
    logger.info(f"  ✓ Saved to: {output_file}")
    logger.info(f"  File size: {output_size_kb:.1f} KB")

    message = f"Predictions completed for {len(test_df)} sequences"
    if metrics_msg:
        message += f" ({metrics_msg})"

    logger.info("\n" + "="*70)
    logger.info("Prediction Complete!")
    logger.info(f"  {message}")
    logger.info(f"  Output: {output_file.name}")
    logger.info("="*70)

    return {
        "message": message,
        "reference": "https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data/blob/master/src/evaluate.py",
        "artifacts": [
            {
                "description": "Fitness predictions",
                "path": str(output_file.resolve())
            }
        ]
    }
