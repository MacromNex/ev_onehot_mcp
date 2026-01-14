# EV+OneHot MCP Server

Protein fitness prediction combining evolutionary coupling (EV) features and one-hot encoding with Ridge regression.

## Overview

This MCP server provides two main tools for protein fitness prediction:

1. **ev_onehot_train_fitness_predictor**: Train a Ridge regression model on protein sequences
2. **ev_onehot_predict_fitness**: Predict fitness for new protein sequences using a trained model

## Installation

```bash
# Create and activate virtual environment
mamba env create -p ./env python=3.10 pip -y
mamba activate ./env

# Install dependencies
pip install -r requirements.txt
pip install --ignore-installed fastmcp
```

## Local usage

### 1. Training a Model
```shell
# 5-fold cross validation
python repo/ev_onehot/train.py example/ --cross_val

# Train test split training
python repo/ev_onehot/train.py example/  -s 1
```

### 2. Making Predictions

```shell
python repo/ev_onehot/pred.py example --seq_path example/data.csv
```
## MCP usage
### Install `ev+onehot` mcp
```shell
fastmcp install claude-code tool-mcps/ev_onehot_mcp/src/server.py --python tool-mcps/ev_onehot_mcp/env/bin/python
```

## Call MCP

```markdown
I have created a plmc model for subtilisin BPN' in directory @examples/case2.1_subtilisin/plmc. Can you help build a ev+onehot model using ev_onehot_mcp and create it to @examples/case2.1_subtilisin/ directory. The wild-type sequence is @examples/case2.1_subtilisin/wt.fasta, and the dataset is @examples/case2.1_subtilisin/data.csv.

Please convert the relative path to absolution path before calling the MCP servers.
```

## API Reference

### ev_onehot_train_fitness_predictor

Train a protein fitness prediction model.

**Parameters:**
- `data_dir` (str): Directory containing `data.csv` with training data
- `train_data_path` (str, optional): Custom path to training CSV
- `test_data_path` (str, optional): Path to external test set
- `cross_val` (bool): Use 5-fold cross-validation (default: False)
- `test_size` (float): Fraction for test split (default: 0.2)
- `seed` (int): Random seed (default: 6)
- `out_prefix` (str, optional): Output file prefix

**Returns:**
- `message`: Training summary with metrics
- `artifacts`: List of generated files (model, predictions, summaries)

**Example:**
```python
result = ev_onehot_train_fitness_predictor(
    data_dir='example',
    cross_val=True,  # 5-fold CV
    seed=42
)
# Result: {
#   'message': '5-fold CV completed: 0.440 ± 0.012',
#   'artifacts': [...]
# }
```

### ev_onehot_predict_fitness

Predict fitness for protein sequences using a trained model.

**Parameters:**
- `model_dir` (str): Directory containing `ridge_model.joblib` and `plmc/`
- `csv_file` (str, optional): CSV file with sequence column
- `sequences` (list, optional): List of protein sequences (alternative to csv_file)
- `seq_col` (str): Name of sequence column in CSV (default: 'seq')
- `out_prefix` (str, optional): Output file prefix

**Returns:**
- `message`: Prediction summary (with metrics if log_fitness provided)
- `artifacts`: List of generated prediction files

**Examples:**
```python
# From CSV with default 'seq' column
result = ev_onehot_predict_fitness(
    model_dir='example',
    csv_file='my_sequences.csv'
)

# From CSV with custom column name
result = ev_onehot_predict_fitness(
    model_dir='example',
    csv_file='my_data.csv',
    seq_col='protein_sequence'  # Use 'protein_sequence' instead of 'seq'
)

# From list
result = ev_onehot_predict_fitness(
    model_dir='example',
    sequences=['MISLIAALAVDRVIGM...', 'MISLVAALAVDRVIGM...']
)
```

## Data Format

### Training Data (data.csv)

Required columns:
- `seq`: Protein sequence (single-letter amino acid codes)
- `log_fitness`: Log-transformed fitness values

Optional columns:
- Any additional metadata (will be preserved in outputs)

Example:
```csv
seq,log_fitness
MISLIAALAVDRVIGM...,0.128
MISLVAALAVDRVIGM...,0.126
```

## Output Files

### Training Outputs

- `ridge_model.joblib`: Trained sklearn Ridge regression model
- `*_test_results.csv`: Predictions on test set with true vs predicted fitness
- `*_summary.csv`: Performance metrics summary
- `*_fold{N}_results.csv`: Individual fold results (if cross_val=True)

### Prediction Outputs

- `predictions_*.csv`: Input sequences with predicted fitness values

Example output:
```csv
seq,pred_fitness
MISLIAALAVDRVIGM...,-0.892
MISLVAALAVDRVIGM...,-0.898
```

## Technical Details

### Model Architecture

- **Feature Encoding**: Combined evolutionary (EV) features from PLMC model + one-hot encoding of amino acid sequences
- **Algorithm**: Ridge regression with L2 regularization
- **Hyperparameter Tuning**: 5-fold cross-validation over [0.1, 1, 10, 100, 1000]
- **Performance**: Improved Spearman correlation using joint EV+OneHot predictor

### Implementation Notes

- Sequences with invalid amino acids are filtered out during training
- Maximum sequence length: 2000 amino acids (configurable)
- Valid amino acids: MRHKDESTNQCUGPAVIFYWLO
- Model files are saved using joblib for efficient storage
- All operations are logged using loguru for transparency and debugging

## Requirements

The implementation uses the combined EV+OneHot predictor, which requires:
1. Multiple sequence alignment (MSA) in A2M format
2. PLMC model trained on that MSA (located in `plmc/` directory)
3. Wild-type sequence in FASTA format (`wt.fasta`)
4. The PLMC model must be compatible with the wild-type sequence

**Important:** The system will raise an error if the PLMC model is not compatible with the wild-type sequence. This validation ensures that evolutionary features are correctly aligned with the protein sequences in your dataset.

## Citation

This implementation is based on the work from:
```
@article{hsu2022,
  title={Learning protein fitness models from evolutionary and assay-labeled data},
  author={Hsu, Chloe and Nisonoff, Hunter and Fannjiang, Clara and Listgarten, Jennifer},
  journal={Nature Biotechnology},
  year={2022}
}
```

Original repository: https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data

## License

This project follows the license of the original repository.
