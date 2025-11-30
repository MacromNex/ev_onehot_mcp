import os
import argparse
import numpy as np
import pandas as pd
from loguru import logger
from predictor import JointPredictor, EVPredictor, OnehotRidgePredictor
from util import spearman, ndcg, auroc


def load_predictor(data_path, predictor_name='ev+onehot'):
    logger.info(f'Predictor {predictor_name} -----')
    predictor_cls = [EVPredictor, OnehotRidgePredictor]
    predictor = JointPredictor(data_path, predictor_cls, predictor_name)

    predictor.load_model()
    return predictor


def main(args):
    logger.info(f'Data path {args.data_path} -----')

    # Load dataset
    seq_path = args.seq_path if args.seq_path is not None else f'{args.data_path}/data_test.csv'
    if os.path.exists(seq_path) and args.seq_path is not None:
        logger.info(f'Predicting sequences from {seq_path} -----')
        seq_df = pd.read_csv(seq_path)
    else:
        raise FileNotFoundError(f'Sequence file {seq_path} not found.')
    
    logger.info(f'Number of samples: {len(seq_df)}')

    predictor = load_predictor(args.data_path)
    test_pred = predictor.predict(seq_df[args.seq_col].values)

    if 'log_fitness' in seq_df.columns:
        metric_fns = {'spearman': spearman,}
        test_ret = [f"{m}: {mfn(test_pred, seq_df['log_fitness'].to_numpy())}" for m, mfn in metric_fns.items()]
        logger.info(", ".join(test_ret))
    
    seq_df['pred_fitness'] = test_pred
    seq_df.to_csv(f'{seq_path}_pred.csv')


def parse_args():
    parser = argparse.ArgumentParser(description='Train ev+onehot model:'
                                                 'python ev_onehot_train.py <dataset_name>')
    parser.add_argument('data_path', type=str,
                        help='Dataset path, including following files: \n'
                             'data_test.csv,    with  `seq` and `log_fitness` columns; \n'
                             'ridge_model.pkl,  pretrained linear regression model; \n'
                             'wt.fasta,         containing wild-type sequences;\n'
                             'plmc/,            folder contain EVmutation model parameters.\n')
    parser.add_argument('--seq_path', dest='seq_path', type=str, default=None,
                        help='If provided, a csv file containing sequences to predict.')
    parser.add_argument('--seq_col', dest='seq_col', type=str, default='seq',
                        help='If provided, the column name in the csv file containing sequences to predict.')
    parser.add_argument('--ignore_gaps', dest='ignore_gaps', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
