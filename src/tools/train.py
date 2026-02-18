"""
EV+Onehot Model Training Pipeline for protein fitness prediction.

This MCP Server provides 1 tool:
1. ev_onehot_train_fitness_predictor: Train and evaluate combined EV+Onehot predictor model

All tools extracted from `combining-evolutionary-and-assay-labelled-data/src/evaluate.py`.
"""

# Standard imports
from typing import Annotated
import pandas as pd
import numpy as np
from pathlib import Path
import os
import shutil
from fastmcp import FastMCP
from datetime import datetime
import sys
from loguru import logger

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_INPUT_DIR = PROJECT_ROOT / "tmp" / "inputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp" / "outputs"

INPUT_DIR = Path(os.environ.get("TRAIN_INPUT_DIR", DEFAULT_INPUT_DIR))
OUTPUT_DIR = Path(os.environ.get("TRAIN_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp for unique outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Add repo path for imports
REPO_PATH = PROJECT_ROOT / "repo" / "ev_onehot"
sys.path.insert(0, str(REPO_PATH))

# Import predictor modules
from predictor import JointPredictor, OnehotRidgePredictor, EVPredictor
from util import is_valid_seq, spearman

# MCP server instance
train_mcp = FastMCP(name="train")


@train_mcp.tool
def ev_onehot_train_fitness_predictor(
    data_dir: Annotated[str | None, "Directory path containing required files: data.csv (with 'seq' and 'log_fitness' columns), wt.fasta (wild-type sequence), and plmc/ folder (EVmutation model parameters)"] = None,
    train_data_path: Annotated[str | None, "Path to training data CSV file with 'seq' and 'log_fitness' columns. If not provided, uses data_dir/data.csv"] = None,
    test_data_path: Annotated[str | None, "Path to external test data CSV file with 'seq' and 'log_fitness' columns. If provided, trains on all data_dir data and evaluates on this test set"] = None,
    cross_val: Annotated[bool, "Whether to perform 5-fold cross-validation followed by training final model on all data. If True, performs 5-fold CV for evaluation then trains final model. If False, performs train-test split"] = True,
    test_size: Annotated[float, "Fraction of data to use for testing in train-test split (ignored if cross_val=True or test_data_path is provided)"] = 0.2,
    seed: Annotated[int, "Random seed for reproducible train-test splits and cross-validation"] = 6,
    ignore_gaps: Annotated[bool, "Whether to ignore gap characters in sequences. If False, filters out sequences with invalid characters"] = False,
    out_prefix: Annotated[str | None, "Output file prefix for results"] = None,
) -> dict:
    """
    Train and evaluate combined EV+Onehot predictor for protein fitness prediction.
    Input is directory with training data CSV, wild-type FASTA, and EVmutation parameters, and output is trained model and evaluation metrics.
    """
    logger.info("="*70)
    logger.info("Starting EV+Onehot Fitness Predictor Training")
    logger.info("="*70)

    # Input validation
    if data_dir is None:
        raise ValueError("Directory path must be provided")

    logger.info(f"Data directory: {data_dir}")
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Check for required files
    logger.info("Checking required files...")
    required_files = []
    if train_data_path is None:
        default_data_csv = data_dir_path / "data.csv"
        if not default_data_csv.exists():
            raise FileNotFoundError(f"Required file not found: {default_data_csv}")
        train_data_path = str(default_data_csv)
        required_files.append(("Training data", str(default_data_csv)))
    else:
        if not Path(train_data_path).exists():
            raise FileNotFoundError(f"Training data file not found: {train_data_path}")
        required_files.append(("Training data", train_data_path))

    wt_fasta = data_dir_path / "wt.fasta"
    if not wt_fasta.exists():
        raise FileNotFoundError(f"Required file not found: {wt_fasta}")
    required_files.append(("Wild-type FASTA", str(wt_fasta)))

    plmc_dir = data_dir_path / "plmc"
    if not plmc_dir.exists():
        raise FileNotFoundError(f"Required directory not found: {plmc_dir}")
    required_files.append(("EVmutation parameters", str(plmc_dir)))

    # Check test data path if provided
    if test_data_path is not None:
        if not Path(test_data_path).exists():
            raise FileNotFoundError(f"Test data file not found: {test_data_path}")
        required_files.append(("Test data", test_data_path))

    for name, path in required_files:
        logger.info(f"  ✓ {name}: {Path(path).name}")

    metrics_path = data_dir_path / "metrics_summary.csv"

    # Load training dataset
    logger.info(f"\nLoading training data from {Path(train_data_path).name}...")
    data_df = pd.read_csv(train_data_path)
    logger.info(f"  Loaded {len(data_df)} sequences")

    # Validate sequences if not ignoring gaps
    if not ignore_gaps:
        is_valid = data_df['seq'].apply(is_valid_seq)
        n_invalid = (~is_valid).sum()
        if n_invalid > 0:
            logger.warning(f"  Filtering out {n_invalid} invalid sequences")
        data_df = data_df[is_valid]
        logger.info(f"  Valid sequences: {len(data_df)}")

    # Helper function: train predictor
    def train_predictor_internal(data_dir_str, train_df, reg_coef):
        # Use combined EV+Onehot predictor
        logger.info(f"\nInitializing EV+Onehot predictor...")
        logger.info(f"  Training samples: {len(train_df)}")
        logger.info(f"  Regularization: {reg_coef}")

        predictor_cls = [EVPredictor, OnehotRidgePredictor]
        predictor_name = ['ev', 'onehot']
        predictor = JointPredictor(data_dir_str, predictor_cls, predictor_name, reg_coef=reg_coef)

        logger.info(f"Training model...")
        if reg_coef == 'CV':
            logger.info(f"  Performing 5-fold cross-validation for hyperparameter tuning...")
        predictor.train(train_df.seq.values, train_df.log_fitness.values)

        logger.info(f"Saving model...")
        predictor.save_model()
        logger.info(f"  ✓ Model saved to {data_dir_str}/ridge_model.joblib")
        return predictor

    # Helper function: evaluate on test set
    def evaluate_test_set(predictor, test_df):
        logger.info(f"\nEvaluating on test set ({len(test_df)} sequences)...")
        test_df_copy = test_df.copy()
        test_df_copy['pred_fitness'] = predictor.predict(test_df_copy.seq.values)
        correlation = spearman(test_df_copy['pred_fitness'].to_numpy(), test_df_copy['log_fitness'].to_numpy())
        logger.info(f"  Spearman correlation: {correlation:.4f}")
        return test_df_copy, correlation

    results = []
    artifacts = []

    # Determine regularization coefficient
    reg_coef = 'CV' if len(data_df) >= 5 else 1.0

    if cross_val:
        # 5-fold cross validation
        logger.info("#==== Running 5-Fold Cross-Validation ====#")
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        correlations = []

        for fold, (train_index, test_index) in enumerate(kf.split(data_df)):
            logger.info(f"\nFold {fold+1}/5:")
            train = data_df.iloc[train_index]
            test = data_df.iloc[test_index].copy()

            # Train predictor
            predictor = train_predictor_internal(data_dir, train, reg_coef)

            # Evaluate on test fold
            _, correlation = evaluate_test_set(predictor, test)
            correlations.append(correlation)

            results.append({
                'stage': 'cross_validation',
                'fold': fold + 1,
                'n_train': len(train),
                'n_test': len(test),
                'spearman_correlation': correlation
            })

        # Calculate average and std
        avg_corr = np.mean(correlations)
        std_corr = np.std(correlations)

        # Add CV summary statistics to results
        results.append({
            'stage': 'cv_summary',
            'fold': 'mean',
            'n_train': int(np.mean([r['n_train'] for r in results if r['stage'] == 'cross_validation'])),
            'n_test': int(np.mean([r['n_test'] for r in results if r['stage'] == 'cross_validation'])),
            'spearman_correlation': avg_corr
        })

        results.append({
            'stage': 'cv_summary',
            'fold': 'std',
            'n_train': 0,
            'n_test': 0,
            'spearman_correlation': std_corr
        })
        logger.info(f"Cross-Validation Results:")
        logger.info(f"  Mean Spearman: {avg_corr:.4f} ± {std_corr:.4f}")
        logger.info(f"  Individual folds: {[f'{c:.4f}' for c in correlations]}")
        logger.info("#==== Cross-Validation Results Complete ====#")

        # Train final model on all data
        logger.info("#==== Training Final Model on Full Dataset ====#")
        logger.info(f"Training on {len(data_df)} samples (full dataset)")

        # Train on all data
        final_predictor = train_predictor_internal(data_dir, data_df, reg_coef)
        logger.info("#==== Final Model Training Complete ====#")

        # Add final model to artifacts
        final_model_path = data_dir_path / "ridge_model.joblib"
        if final_model_path.exists():
            artifacts.append({
                "description": "Final model (trained on all data)",
                "path": str(final_model_path.resolve())
            })
            logger.info(f"\n  ✓ Final model saved to {final_model_path}")

        # Generate and save predictions on all data for visualization
        logger.info("Generating predictions on full dataset for visualization...")
        all_predictions = final_predictor.predict(data_df.seq.values)
        all_observed = data_df.log_fitness.values

        # Save predictions
        np.save(data_dir_path / "ev_onehot_predictions.npy", all_predictions)
        np.save(data_dir_path / "ev_onehot_observed.npy", all_observed)
        logger.info(f"  ✓ Predictions saved to ev_onehot_predictions.npy and ev_onehot_observed.npy")

        artifacts.append({
            "description": "Predictions on all data",
            "path": str((data_dir_path / "ev_onehot_predictions.npy").resolve())
        })

        _, correlation = evaluate_test_set(final_predictor, data_df)
        results.append({
            'stage': 'final_model',
            'fold': 'all',
            'n_train': len(data_df),
            'n_test': 0,
            'spearman_correlation': correlation
        })

        # Save full summary with CV results and final model info
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(metrics_path, index=False)
        artifacts.append({
            "description": "Training summary (CV + final model)",
            "path": str(metrics_path.resolve())
        })

        message = f"5-fold CV: {avg_corr:.3f} ± {std_corr:.3f}, Final model trained on {len(data_df)} samples"

    elif test_data_path is not None:
        # External test set
        logger.info("#==== Training with External Test Set ====#")
        train = data_df
        logger.info(f"\nLoading external test data from {Path(test_data_path).name}...")
        test = pd.read_csv(test_data_path)
        logger.info(f"  Test samples: {len(test)}")

        # Train predictor
        predictor = train_predictor_internal(data_dir, train, reg_coef)

        # Evaluate on external test set
        _, correlation = evaluate_test_set(predictor, test)

        # Save results
        results.append({
            'n_train': len(train),
            'n_test': len(test),
            'spearman_correlation': correlation
        })

        # Save summary
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(metrics_path, index=False)
        artifacts.append({
            "description": "Evaluation summary",
            "path": str(metrics_path.resolve())
        })

        message = f"Test evaluation completed: Spearman {correlation:.3f}"

    else:
        # Train-test split
        logger.info(f"#==== Training with {int(test_size*100)}% Test Split ====#")
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(data_df, test_size=test_size, random_state=seed)
        logger.info(f"\nData split:")
        logger.info(f"  Training samples: {len(train)}")
        logger.info(f"  Test samples: {len(test)}")

        # Train predictor
        predictor = train_predictor_internal(data_dir, train, reg_coef)

        # Evaluate on test set
        _, correlation = evaluate_test_set(predictor, test)

        results.append({
            'n_train': len(train),
            'n_test': len(test),
            'spearman_correlation': correlation
        })

        # Save summary
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(metrics_path, index=False)
        artifacts.append({
            "description": "Evaluation summary",
            "path": str(metrics_path.resolve())
        })

        message = f"Train-test split completed: Spearman {correlation:.3f}"

    logger.info("### Training Complete!")
    logger.info(f"  {message}")
    logger.info(f"  Generated {len(artifacts)} artifact(s)")
    
    return {
        "message": message,
        "artifacts": artifacts
    }
