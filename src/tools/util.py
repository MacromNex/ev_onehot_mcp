"""
Protein sequence processing and fitness prediction utility functions.

This module provides utility functions for:
- Amino acid sequence encoding and validation
- One-hot encoding for protein sequences
- Mutation parsing and wild-type sequence reconstruction
- Fitness prediction evaluation metrics (Spearman, NDCG, AUROC)

All utilities extracted from the combining-evolutionary-and-assay-labelled-data repository.
These functions are designed to be imported and used by other analysis tools.

Reference: https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data/blob/master/src/utils/metric_utils.py
"""

# Standard imports
from typing import Annotated, Literal, Any
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, ndcg_score
import matplotlib.pyplot as plt
from pathlib import Path
import os
from fastmcp import FastMCP
from datetime import datetime

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_INPUT_DIR = PROJECT_ROOT / "tmp" / "inputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp" / "outputs"

INPUT_DIR = Path(os.environ.get("UTIL_INPUT_DIR", DEFAULT_INPUT_DIR))
OUTPUT_DIR = Path(os.environ.get("UTIL_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp for unique outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# MCP server instance
util_mcp = FastMCP(name="util")

# Amino acid encoding mappings
aa_to_int = {
    'M': 1, 'R': 2, 'H': 3, 'K': 4, 'D': 5, 'E': 6, 'S': 7, 'T': 8, 'N': 9, 'Q': 10,
    'C': 11, 'U': 12, 'G': 13, 'P': 14, 'A': 15, 'V': 16, 'I': 17, 'F': 18, 'Y': 19,
    'W': 20, 'L': 21, 'O': 22,  # Pyrrolysine
    'X': 23,  # Unknown
    'Z': 23,  # Glutamic acid or Glutamine
    'B': 23,  # Asparagine or aspartic acid
    'J': 23,  # Leucine or isoleucine
    'start': 24, 'stop': 25, '-': 26
}
int_to_aa = {value: key for key, value in aa_to_int.items()}


def is_valid_seq(seq, max_len=2000):
    """
    True if seq is valid for the babbler, False otherwise.
    """
    l = len(seq)
    valid_aas = "MRHKDESTNQCUGPAVIFYWLO"
    if (l < max_len) and set(seq) <= set(valid_aas):
        return True
    else:
        return False


def aa_seq_to_int(s):
    """
    Return the int sequence as a list for a given string of amino acids
    """
    return [24] + [aa_to_int[a] for a in s] + [25]


def spearman(y_pred, y_true):
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        return 0.0
    return spearmanr(y_pred, y_true).correlation


def ndcg(y_pred, y_true):
    # Make values positive for sklearn's ndcg_score requirement
    y_true_pos = y_true - y_true.min() + 1.0
    y_pred_pos = y_pred - y_pred.min() + 1.0
    return ndcg_score(y_true_pos.reshape(1, -1), y_pred_pos.reshape(1, -1))


def auroc(y_pred, y_true, y_cutoff=1.0):
    y_true_bin = (y_true >= y_cutoff)
    return roc_auc_score(y_true_bin, y_pred, average='micro')


def format_seq(seq, stop=False):
    """
    Takes an amino acid sequence, returns a list of integers in the codex of the babbler.
    Here, the default is to strip the stop symbol (stop=False) which would have
    otherwise been added to the end of the sequence. If you are trying to generate
    a rep, do not include the stop. It is probably best to ignore the stop if you are
    co-tuning the babbler and a top model as well.
    """
    if stop:
        int_seq = aa_seq_to_int(seq.strip())
    else:
        int_seq = aa_seq_to_int(seq.strip())[:-1]
    return int_seq


def format_batch_seqs(seqs):
    maxlen = -1
    for s in seqs:
        if len(s) > maxlen:
            maxlen = len(s)
    formatted = []
    for seq in seqs:
        pad_len = maxlen - len(seq)
        padded = np.pad(format_seq(seq), (0, pad_len), 'constant', constant_values=0)
        formatted.append(padded)
    return np.stack(formatted)


def seqs_to_onehot(seqs):
    seqs = format_batch_seqs(seqs)
    X = np.zeros((seqs.shape[0], seqs.shape[1]*24), dtype=int)
    for i in range(seqs.shape[1]):
        for j in range(24):
            X[:, i*24+j] = (seqs[:, i] == j)
    return X


def get_wt_seq(mutation_descriptions):
    wt_len = 0
    for m in mutation_descriptions:
        if m == 'WT':
            continue
        if int(m[1:-1]) > wt_len:
            wt_len = int(m[1:-1])
    wt = ['?' for _ in range(wt_len)]
    for m in mutation_descriptions:
        if m == 'WT':
            continue
        idx, wt_char = int(m[1:-1])-1, m[0]   # 1-index to 0-index
        if wt[idx] == '?':
            wt[idx] = wt_char
        else:
            assert wt[idx] == wt_char
    return ''.join(wt), wt_len


def seq2mutation(seq, model, return_str=False, ignore_gaps=False, sep=":", offset=1):
    mutations = []
    for pf, pm in model.index_map.items():
        if seq[pf-offset] != model.target_seq[pm]:
            if ignore_gaps and (
                    seq[pf-offset] == '-' or seq[pf-offset] not in model.alphabet):
                continue
            mutations.append((pf, model.target_seq[pm], seq[pf-offset]))
    if return_str:
        return sep.join([m[1] + str(m[0]) + m[2] for m in mutations])
    return mutations


def seq2effect(seqs, model, offset=1, ignore_gaps=False):
    effects = np.zeros(len(seqs))
    for i in range(len(seqs)):
        mutations = seq2mutation(seqs[i], model, ignore_gaps=ignore_gaps, offset=offset)
        dE, _, _ = model.delta_hamiltonian(mutations)
        effects[i] = dE
    return effects


def mutant2seq(mut, wt, offset):
    if mut.upper() == 'WT':
        return wt
    chars = list(wt)
    mut = mut.replace(':', ',')
    mut = mut.replace(';', ',')
    for m in mut.split(','):
        idx = int(m[1:-1])-offset
        assert wt[idx] == m[0]
        chars[idx] = m[-1]
    return ''.join(chars)


@util_mcp.tool
def util_evaluate_predictions(
    predictions_path: Annotated[str | None, "Path to predictions file (CSV/TSV). Must contain 'predicted' and 'true' columns with numeric fitness values."] = None,
    metrics: Annotated[list[str], "List of metrics to compute"] = ["spearman", "ndcg", "auroc"],
    auroc_cutoff: Annotated[float, "Fitness cutoff for AUROC binary classification"] = 1.0,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Evaluate fitness prediction performance using multiple metrics.
    Input is a predictions file with true and predicted fitness values, output is metrics table and visualization plots.
    """
    # Input validation
    if predictions_path is None:
        raise ValueError("Path to predictions file must be provided")

    predictions_file = Path(predictions_path)
    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    # Load predictions
    import pandas as pd
    if predictions_path.endswith('.csv'):
        data = pd.read_csv(predictions_path)
    else:
        data = pd.read_csv(predictions_path, sep='\t')

    if 'predicted' not in data.columns or 'true' not in data.columns:
        raise ValueError("Predictions file must contain 'predicted' and 'true' columns")

    y_pred = data['predicted'].values
    y_true = data['true'].values

    # Calculate metrics
    results = {}
    if 'spearman' in metrics:
        results['spearman'] = spearman(y_pred, y_true)
    if 'ndcg' in metrics:
        results['ndcg'] = ndcg(y_pred, y_true)
    if 'auroc' in metrics:
        results['auroc'] = auroc(y_pred, y_true, y_cutoff=auroc_cutoff)

    # Save metrics
    if out_prefix is None:
        out_prefix = f"evaluation_{timestamp}"

    metrics_file = OUTPUT_DIR / f"{out_prefix}_metrics.csv"
    metrics_df = pd.DataFrame([results])
    metrics_df.to_csv(metrics_file, index=False)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Correlation visualization
    axes[0].scatter(y_true, y_pred, alpha=0.6, s=50)
    axes[0].plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()],
                 'r--', linewidth=2, label='Perfect prediction')
    axes[0].set_xlabel('True Fitness', fontsize=12)
    axes[0].set_ylabel('Predicted Fitness', fontsize=12)
    if 'spearman' in results:
        axes[0].set_title(f'Spearman r = {results["spearman"]:.3f}', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Prediction error distribution
    errors = y_pred - y_true
    axes[1].hist(errors, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[1].set_xlabel('Prediction Error', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Prediction Error Distribution', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    viz_file = OUTPUT_DIR / f"{out_prefix}_visualization.png"
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    plt.close()

    # Return results
    message = f"Evaluation completed: {', '.join([f'{k}={v:.4f}' for k, v in results.items()])}"

    return {
        "message": message[:120],
        "reference": "https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data/blob/master/src/utils/metric_utils.py",
        "artifacts": [
            {
                "description": "Evaluation metrics",
                "path": str(metrics_file.resolve())
            },
            {
                "description": "Metrics visualization",
                "path": str(viz_file.resolve())
            }
        ]
    }


@util_mcp.tool
def util_encode_sequences(
    sequences_path: Annotated[str | None, "Path to sequences file. Text file with one protein sequence per line, or CSV with 'sequence' column."] = None,
    encoding_type: Annotated[Literal["integer", "onehot"], "Encoding type: 'integer' for integer encoding, 'onehot' for one-hot encoding"] = "onehot",
    include_stop: Annotated[bool, "Include stop codon in integer encoding"] = False,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Encode protein sequences to integer or one-hot representation.
    Input is a file with protein sequences, output is encoded sequences in CSV format.
    """
    # Input validation
    if sequences_path is None:
        raise ValueError("Path to sequences file must be provided")

    sequences_file = Path(sequences_path)
    if not sequences_file.exists():
        raise FileNotFoundError(f"Sequences file not found: {sequences_path}")

    # Load sequences
    import pandas as pd
    if sequences_path.endswith('.csv'):
        data = pd.read_csv(sequences_path)
        if 'sequence' not in data.columns:
            raise ValueError("CSV file must contain 'sequence' column")
        sequences = data['sequence'].tolist()
    else:
        with open(sequences_path, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]

    # Validate sequences
    valid_sequences = []
    invalid_count = 0
    for seq in sequences:
        if is_valid_seq(seq):
            valid_sequences.append(seq)
        else:
            invalid_count += 1

    if invalid_count > 0:
        print(f"Warning: {invalid_count} invalid sequences skipped")

    if len(valid_sequences) == 0:
        raise ValueError("No valid sequences found in input file")

    # Encode sequences
    if encoding_type == "integer":
        encoded = []
        for seq in valid_sequences:
            int_seq = format_seq(seq, stop=include_stop)
            encoded.append(int_seq)

        # Save as CSV (each row is one sequence)
        if out_prefix is None:
            out_prefix = f"encoded_integer_{timestamp}"

        output_file = OUTPUT_DIR / f"{out_prefix}.csv"

        # Pad to same length
        max_len = max(len(e) for e in encoded)
        padded = [e + [0] * (max_len - len(e)) for e in encoded]

        df = pd.DataFrame(padded)
        df.to_csv(output_file, index=False, header=False)

    elif encoding_type == "onehot":
        encoded = seqs_to_onehot(valid_sequences)

        if out_prefix is None:
            out_prefix = f"encoded_onehot_{timestamp}"

        output_file = OUTPUT_DIR / f"{out_prefix}.csv"
        df = pd.DataFrame(encoded)
        df.to_csv(output_file, index=False, header=False)

    message = f"Encoded {len(valid_sequences)} sequences using {encoding_type} encoding"

    return {
        "message": message[:120],
        "reference": "https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data/blob/master/src/utils/metric_utils.py",
        "artifacts": [
            {
                "description": f"Encoded sequences ({encoding_type})",
                "path": str(output_file.resolve())
            }
        ]
    }
