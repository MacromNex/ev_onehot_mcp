"""
Protein Fitness Predictors: EV, One-Hot, and Joint Models.

This MCP Server provides 3 tools for training protein fitness prediction models:
1. train_onehot_predictor: Train one-hot encoding + Ridge regression model
2. train_ev_predictor: Train evolutionary model using EVmutation
3. train_joint_predictor: Train joint model combining multiple predictors

All tools extracted from combining-evolutionary-and-assay-labelled-data repository.
"""

# Standard imports
from typing import Annotated, Literal, Any
import pandas as pd
import numpy as np
from pathlib import Path
import os
from fastmcp import FastMCP
from datetime import datetime
import joblib
from loguru import logger
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import Ridge
from Bio import SeqIO

# Import utility functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "repo" / "ev_onehot"))
from util import spearman, seqs_to_onehot, seq2effect
from couplings_model import CouplingsModel

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_INPUT_DIR = PROJECT_ROOT / "tmp" / "inputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp" / "outputs"

INPUT_DIR = Path(os.environ.get("PREDICTOR_INPUT_DIR", DEFAULT_INPUT_DIR))
OUTPUT_DIR = Path(os.environ.get("PREDICTOR_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp for unique outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# MCP server instance
predictor_mcp = FastMCP(name="predictor")

# Constants from tutorial
REG_COEF_LIST = [1e-1, 1e0, 1e1, 1e2, 1e3]


def read_fasta(filename, return_ids=False):
    """Read sequences from FASTA file."""
    records = SeqIO.parse(filename, 'fasta')
    seqs = list()
    ids = list()
    for record in records:
        seqs.append(str(record.seq))
        ids.append(str(record.id))
    if return_ids:
        return seqs, ids
    else:
        return seqs


class BaseRegressionPredictor:
    """Base class for regression-based fitness predictors."""

    def __init__(self, data_path, reg_coef=None, linear_model_cls=Ridge):
        self.data_path = data_path
        self.reg_coef = reg_coef
        self.linear_model_cls = linear_model_cls
        self.model = None

    def seq2feat(self, seqs):
        raise NotImplementedError

    def train(self, train_seqs, train_labels):
        X = self.seq2feat(train_seqs)
        if self.reg_coef is None or self.reg_coef == 'CV':
            best_rc, best_score = None, -np.inf
            for rc in REG_COEF_LIST:
                model = self.linear_model_cls(alpha=rc)
                score = cross_val_score(model, X, train_labels, cv=5,
                                        scoring=make_scorer(spearman)).mean()
                if score > best_score:
                    best_rc = rc
                    best_score = score
            self.reg_coef = best_rc
        self.model = self.linear_model_cls(alpha=self.reg_coef)
        self.model.fit(X, train_labels)

    def predict(self, predict_seqs):
        if self.model is None:
            return np.random.randn(len(predict_seqs))
        X = self.seq2feat(predict_seqs)
        return self.model.predict(X)

    def save_model(self):
        save_path = os.path.join(self.data_path, 'ridge_model.joblib')
        joblib.dump(self.model, save_path)
        logger.info(f"Ridge regression model saved to path {save_path}")

    def load_model(self):
        model_path = os.path.join(self.data_path, 'ridge_model.joblib')
        self.model = joblib.load(model_path)


class OnehotRidgePredictor(BaseRegressionPredictor):
    """Simple one hot encoding + ridge regression."""

    def __init__(self, data_path, reg_coef=1.0):
        super(OnehotRidgePredictor, self).__init__(data_path, reg_coef, Ridge)

    def seq2feat(self, seqs):
        return seqs_to_onehot(seqs)


class EVPredictor(BaseRegressionPredictor):
    """plmc mutation effect prediction."""

    def __init__(self, data_path, reg_coef=1e-8, ignore_gaps=False):
        super(EVPredictor, self).__init__(data_path, reg_coef=reg_coef)
        self.ignore_gaps = ignore_gaps
        self.couplings_model_path = os.path.join(data_path, 'plmc/uniref100.model_params')
        self.couplings_model = CouplingsModel(self.couplings_model_path)
        wtseqs, wtids = read_fasta(os.path.join(data_path, 'wt.fasta'), return_ids=True)
        if '/' in wtids[0]:
            self.offset = int(wtids[0].split('/')[-1].split('-')[0])
        else:
            self.offset = 1
        expected_wt = wtseqs[0]

        for pf, pm in self.couplings_model.index_map.items():
            if expected_wt[pf-self.offset] != self.couplings_model.target_seq[pm]:
                logger.debug(f'WT and model target seq mismatch at {pf}')

    def seq2score(self, seqs):
        return seq2effect(seqs, self.couplings_model, self.offset, ignore_gaps=self.ignore_gaps)

    def seq2feat(self, seqs):
        return self.seq2score(seqs)[:, None]

    def predict_unsupervised(self, seqs):
        return self.seq2score(seqs)


class JointPredictor(BaseRegressionPredictor):
    """Combining regression predictors by training jointly."""

    def __init__(self, data_path, predictor_classes, predictor_name, reg_coef='CV'):
        super(JointPredictor, self).__init__(data_path, reg_coef)
        self.predictors = []
        for c, name in zip(predictor_classes, predictor_name):
            self.predictors.append(c(data_path, ))

    def seq2feat(self, seqs):
        # To apply different regularziation coefficients we scale the features by a multiplier in Ridge regression
        features = [p.seq2feat(seqs) * np.sqrt(1.0 / p.reg_coef) for p in self.predictors]
        return np.concatenate(features, axis=1)


@predictor_mcp.tool
def train_onehot_predictor(
    train_sequences_path: Annotated[str | None, "Path to training sequences file (one sequence per line)"] = None,
    train_fitness_path: Annotated[str | None, "Path to training fitness values file (one value per line)"] = None,
    test_sequences_path: Annotated[str | None, "Path to test sequences file (one sequence per line)"] = None,
    reg_coef: Annotated[float, "Regularization coefficient for Ridge regression"] = 1.0,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Train one-hot encoding + Ridge regression model for protein fitness prediction.
    Input is training sequences and fitness values, output is trained model and predictions on test sequences.
    """
    # Input validation
    if train_sequences_path is None:
        raise ValueError("Path to training sequences file must be provided")
    if train_fitness_path is None:
        raise ValueError("Path to training fitness values file must be provided")
    if test_sequences_path is None:
        raise ValueError("Path to test sequences file must be provided")

    train_seq_file = Path(train_sequences_path)
    train_fit_file = Path(train_fitness_path)
    test_seq_file = Path(test_sequences_path)

    if not train_seq_file.exists():
        raise FileNotFoundError(f"Training sequences file not found: {train_sequences_path}")
    if not train_fit_file.exists():
        raise FileNotFoundError(f"Training fitness file not found: {train_fitness_path}")
    if not test_seq_file.exists():
        raise FileNotFoundError(f"Test sequences file not found: {test_sequences_path}")

    # Load data
    with open(train_seq_file, 'r') as f:
        train_seqs = [line.strip() for line in f if line.strip()]

    train_fitness = np.loadtxt(train_fit_file)

    with open(test_seq_file, 'r') as f:
        test_seqs = [line.strip() for line in f if line.strip()]

    # Create predictor
    output_dir = OUTPUT_DIR / (out_prefix or f"onehot_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = OnehotRidgePredictor(data_path=str(output_dir), reg_coef=reg_coef)

    # Train model
    predictor.train(train_seqs, train_fitness)

    # Make predictions
    predictions = predictor.predict(test_seqs)

    # Save model
    predictor.save_model()
    model_path = output_dir / 'ridge_model.joblib'

    # Save predictions
    pred_path = output_dir / 'predictions.csv'
    pred_df = pd.DataFrame({
        'sequence': test_seqs,
        'predicted_fitness': predictions
    })
    pred_df.to_csv(pred_path, index=False)

    # Save training info
    info_path = output_dir / 'training_info.csv'
    info_df = pd.DataFrame({
        'parameter': ['n_training_samples', 'n_test_samples', 'reg_coef', 'feature_dim'],
        'value': [len(train_seqs), len(test_seqs), predictor.reg_coef, predictor.seq2feat(train_seqs[:1]).shape[1]]
    })
    info_df.to_csv(info_path, index=False)

    # Plot feature importance
    import matplotlib.pyplot as plt
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300

    fig, ax = plt.subplots(figsize=(10, 4))
    coeffs = predictor.model.coef_
    ax.bar(range(len(coeffs)), coeffs, alpha=0.7, color='steelblue')
    ax.set_xlabel('Feature Index', fontsize=12)
    ax.set_ylabel('Ridge Coefficient', fontsize=12)
    ax.set_title('OnehotRidgePredictor Feature Importance', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    fig_path = output_dir / 'feature_importance.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        "message": f"Trained one-hot Ridge predictor with {len(train_seqs)} training samples",
        "reference": "https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data/blob/master/src/predictors/ev_predictors.py",
        "artifacts": [
            {
                "description": "Trained Ridge regression model",
                "path": str(model_path.resolve())
            },
            {
                "description": "Predictions on test sequences",
                "path": str(pred_path.resolve())
            },
            {
                "description": "Training information",
                "path": str(info_path.resolve())
            },
            {
                "description": "Feature importance visualization",
                "path": str(fig_path.resolve())
            }
        ]
    }


@predictor_mcp.tool
def train_ev_predictor(
    data_dir: Annotated[str | None, "Path to directory containing plmc model and wt.fasta files"] = None,
    train_sequences_path: Annotated[str | None, "Path to training sequences file (one sequence per line)"] = None,
    train_fitness_path: Annotated[str | None, "Path to training fitness values file (one value per line)"] = None,
    test_sequences_path: Annotated[str | None, "Path to test sequences file (one sequence per line)"] = None,
    reg_coef: Annotated[float, "Regularization coefficient for Ridge regression"] = 1e-8,
    ignore_gaps: Annotated[bool, "Whether to ignore gaps in sequences"] = False,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Train evolutionary model (EVmutation) for protein fitness prediction using plmc.
    Input is training sequences, fitness values, and EVmutation model files, output is trained model and predictions.
    """
    # Input validation
    if data_dir is None:
        raise ValueError("Path to data directory must be provided")
    if train_sequences_path is None:
        raise ValueError("Path to training sequences file must be provided")
    if train_fitness_path is None:
        raise ValueError("Path to training fitness values file must be provided")
    if test_sequences_path is None:
        raise ValueError("Path to test sequences file must be provided")

    data_path = Path(data_dir)
    train_seq_file = Path(train_sequences_path)
    train_fit_file = Path(train_fitness_path)
    test_seq_file = Path(test_sequences_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not train_seq_file.exists():
        raise FileNotFoundError(f"Training sequences file not found: {train_sequences_path}")
    if not train_fit_file.exists():
        raise FileNotFoundError(f"Training fitness file not found: {train_fitness_path}")
    if not test_seq_file.exists():
        raise FileNotFoundError(f"Test sequences file not found: {test_sequences_path}")

    # Check required files
    plmc_model = data_path / 'plmc' / 'uniref100.model_params'
    wt_fasta = data_path / 'wt.fasta'

    if not plmc_model.exists():
        raise FileNotFoundError(f"PLMC model not found: {plmc_model}")
    if not wt_fasta.exists():
        raise FileNotFoundError(f"Wild-type FASTA not found: {wt_fasta}")

    # Load data
    with open(train_seq_file, 'r') as f:
        train_seqs = [line.strip() for line in f if line.strip()]

    train_fitness = np.loadtxt(train_fit_file)

    with open(test_seq_file, 'r') as f:
        test_seqs = [line.strip() for line in f if line.strip()]

    # Create predictor
    output_dir = OUTPUT_DIR / (out_prefix or f"ev_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = EVPredictor(data_path=str(data_path), reg_coef=reg_coef, ignore_gaps=ignore_gaps)

    # Train model
    predictor.train(train_seqs, train_fitness)

    # Make predictions
    predictions = predictor.predict(test_seqs)

    # Get unsupervised predictions
    unsupervised_pred = predictor.predict_unsupervised(test_seqs)

    # Save model
    predictor.data_path = str(output_dir)
    predictor.save_model()
    model_path = output_dir / 'ridge_model.joblib'

    # Save predictions
    pred_path = output_dir / 'predictions.csv'
    pred_df = pd.DataFrame({
        'sequence': test_seqs,
        'predicted_fitness': predictions,
        'ev_score': unsupervised_pred
    })
    pred_df.to_csv(pred_path, index=False)

    # Save training info
    info_path = output_dir / 'training_info.csv'
    info_df = pd.DataFrame({
        'parameter': ['n_training_samples', 'n_test_samples', 'reg_coef', 'ignore_gaps', 'offset'],
        'value': [len(train_seqs), len(test_seqs), predictor.reg_coef, ignore_gaps, predictor.offset]
    })
    info_df.to_csv(info_path, index=False)

    return {
        "message": f"Trained EV predictor with {len(train_seqs)} training samples",
        "reference": "https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data/blob/master/src/predictors/ev_predictors.py",
        "artifacts": [
            {
                "description": "Trained Ridge regression model",
                "path": str(model_path.resolve())
            },
            {
                "description": "Predictions on test sequences",
                "path": str(pred_path.resolve())
            },
            {
                "description": "Training information",
                "path": str(info_path.resolve())
            }
        ]
    }


@predictor_mcp.tool
def train_joint_predictor(
    data_dir: Annotated[str | None, "Path to directory containing plmc model and wt.fasta files"] = None,
    train_sequences_path: Annotated[str | None, "Path to training sequences file (one sequence per line)"] = None,
    train_fitness_path: Annotated[str | None, "Path to training fitness values file (one value per line)"] = None,
    test_sequences_path: Annotated[str | None, "Path to test sequences file (one sequence per line)"] = None,
    predictor_types: Annotated[list[Literal["onehot", "ev"]], "List of predictor types to combine"] = ["onehot", "ev"],
    reg_coef: Annotated[str | float, "Regularization coefficient ('CV' for cross-validation or float value)"] = "CV",
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Train joint model combining multiple predictors (one-hot and EV) for protein fitness prediction.
    Input is training sequences, fitness values, and model files, output is trained joint model and predictions.
    """
    # Input validation
    if data_dir is None:
        raise ValueError("Path to data directory must be provided")
    if train_sequences_path is None:
        raise ValueError("Path to training sequences file must be provided")
    if train_fitness_path is None:
        raise ValueError("Path to training fitness values file must be provided")
    if test_sequences_path is None:
        raise ValueError("Path to test sequences file must be provided")

    data_path = Path(data_dir)
    train_seq_file = Path(train_sequences_path)
    train_fit_file = Path(train_fitness_path)
    test_seq_file = Path(test_sequences_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not train_seq_file.exists():
        raise FileNotFoundError(f"Training sequences file not found: {train_sequences_path}")
    if not train_fit_file.exists():
        raise FileNotFoundError(f"Training fitness file not found: {train_fitness_path}")
    if not test_seq_file.exists():
        raise FileNotFoundError(f"Test sequences file not found: {test_sequences_path}")

    # Check required files for EV predictor if needed
    if "ev" in predictor_types:
        plmc_model = data_path / 'plmc' / 'uniref100.model_params'
        wt_fasta = data_path / 'wt.fasta'

        if not plmc_model.exists():
            raise FileNotFoundError(f"PLMC model not found: {plmc_model}")
        if not wt_fasta.exists():
            raise FileNotFoundError(f"Wild-type FASTA not found: {wt_fasta}")

    # Load data
    with open(train_seq_file, 'r') as f:
        train_seqs = [line.strip() for line in f if line.strip()]

    train_fitness = np.loadtxt(train_fit_file)

    with open(test_seq_file, 'r') as f:
        test_seqs = [line.strip() for line in f if line.strip()]

    # Map predictor types to classes
    predictor_map = {
        "onehot": OnehotRidgePredictor,
        "ev": EVPredictor
    }

    predictor_classes = [predictor_map[pt] for pt in predictor_types]

    # Create output directory
    output_dir = OUTPUT_DIR / (out_prefix or f"joint_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create joint predictor
    predictor = JointPredictor(
        data_path=str(data_path),
        predictor_classes=predictor_classes,
        predictor_name=predictor_types,
        reg_coef=reg_coef
    )

    # Train model
    predictor.train(train_seqs, train_fitness)

    # Make predictions
    predictions = predictor.predict(test_seqs)

    # Save model
    predictor.data_path = str(output_dir)
    predictor.save_model()
    model_path = output_dir / 'ridge_model.joblib'

    # Save predictions
    pred_path = output_dir / 'predictions.csv'
    pred_df = pd.DataFrame({
        'sequence': test_seqs,
        'predicted_fitness': predictions
    })
    pred_df.to_csv(pred_path, index=False)

    # Save training info
    info_path = output_dir / 'training_info.csv'
    info_df = pd.DataFrame({
        'parameter': ['n_training_samples', 'n_test_samples', 'predictor_types', 'reg_coef', 'feature_dim'],
        'value': [len(train_seqs), len(test_seqs), ','.join(predictor_types), predictor.reg_coef,
                 predictor.seq2feat(train_seqs[:1]).shape[1]]
    })
    info_df.to_csv(info_path, index=False)

    return {
        "message": f"Trained joint predictor combining {len(predictor_types)} models with {len(train_seqs)} training samples",
        "reference": "https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data/blob/master/src/predictors/ev_predictors.py",
        "artifacts": [
            {
                "description": "Trained joint Ridge regression model",
                "path": str(model_path.resolve())
            },
            {
                "description": "Predictions on test sequences",
                "path": str(pred_path.resolve())
            },
            {
                "description": "Training information",
                "path": str(info_path.resolve())
            }
        ]
    }
