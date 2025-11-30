import os
import argparse
import numpy as np
import pandas as pd
from loguru import logger
from predictor import JointPredictor, EVPredictor, OnehotRidgePredictor
from util import is_valid_seq, spearman


def train_predictor(data_dir, train, save_path, predictor_name='ev+onehot', reg_coef='CV'):
    predictor_cls = [EVPredictor, OnehotRidgePredictor]
    predictor = JointPredictor(data_dir, predictor_cls, predictor_name, reg_coef=reg_coef)

    predictor.train(train.seq.values, train.log_fitness.values)
    predictor.save_model()
    return predictor


def train_test_eval(data_df, train, test, data_dir):
    reg_coef = 'CV' if len(data_df) >= 5 else 1.0
    
    # logger.info(f'Number of training samples: {len(train)}, testing samples: {len(test)}')
    save_path = os.path.join(data_dir, 'ridge_model.joblib')
    predictor = train_predictor(data_dir, train, save_path, reg_coef=reg_coef)

    test['pred_fitness'] = predictor.predict(test.seq.values)
    correlation = spearman(test['pred_fitness'].to_numpy(), test['log_fitness'].to_numpy())
    print(f'Spearman correlation on test set: {correlation:.3f}')
    return correlation


def main(args):
    logger.info(f'Data path {args.data_dir} -----')
    train_data_path = args.train_data_path if args.train_data_path is not None else f'{args.data_dir}/data.csv'

    # Load dataset
    data_df = pd.read_csv(train_data_path)
    if not args.ignore_gaps:  # Necessary to run EVPredictor
        # logger.info("Is checking invalid!")
        is_valid = data_df['seq'].apply(is_valid_seq)
        data_df = data_df[is_valid]


    if args.cross_val:
        # 5-fold cross validation
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        correlations = []
        for fold, (train_index, test_index) in enumerate(kf.split(data_df)):
            logger.info(f'Cross validation fold {fold+1}')
            train = data_df.iloc[train_index]
            test = data_df.iloc[test_index].copy()
            corr = train_test_eval(data_df, train, test, args.data_dir)
            correlations.append(corr)
        avg_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        print(f'Average Spearman correlation over 5 folds: {avg_corr:.3f} ± {std_corr:.3f}')

    elif args.test_data_path is not None:
        train = data_df
        test = pd.read_csv(args.test_data_path)
        train_test_eval(data_df, train, test, args.data_dir)
        
    else:
        # conventional train-test split with the specified ratio
        from sklearn.model_selection import train_test_split
        logger.info(f'Performing train-test split with seed {args.seed}')
        train, test = train_test_split(data_df, test_size=args.test_size, random_state=args.seed)
        train_test_eval(data_df, train, test, args.data_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='Train ev+onehot model:'
                                                 'python ev_onehot_train.py <dataset_name>')
    parser.add_argument('data_dir', type=str,
                        help='Dataset path, including following files: \n'
                             'data.csv,     with  `seq` and `log_fitness` columns; \n'
                             'wt.fasta,     containing wild-type sequences;\n'
                             'plmc/,        folder contain EVmutation model parameters.\n')
    parser.add_argument('--train_data_path', type=str, default=None, help='specify the train data csv file, default is data.csv')
    parser.add_argument('--test_data_path', type=str, default=None, help='specify the test data csv file, default is the train data')
    parser.add_argument('--ignore_gaps', dest='ignore_gaps', action='store_true')
    parser.add_argument('-cv', '--cross_val', dest='cross_val', action='store_true', help='Whether to perform 5-fold cross validation, default False')
    parser.add_argument('-s', '--seed', type=int, default=6, help='Seed for random split')
    parser.add_argument('--test_size', type=float, default=0.2, help='Train-test split ratio, default 0.2 for testing')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
