import os
import joblib
import numpy as np
from loguru import logger
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import Ridge
from util import spearman, seqs_to_onehot, seq2effect
from couplings_model import CouplingsModel
from Bio import SeqIO


REG_COEF_LIST = [1e-1, 1e0, 1e1, 1e2, 1e3]

def read_fasta(filename, return_ids=False):
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
            # print(f'Cross validated reg coef {best_rc}')
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
        # logger.info(f"Ridge regression model loaded from path {model_path}")


class JointPredictor(BaseRegressionPredictor):
    """Combining regression predictors by training jointly."""

    def __init__(self, data_path, predictor_classes, predictor_name, reg_coef='CV'):
        super(JointPredictor, self).__init__(data_path, reg_coef)
        self.predictors = []
        for c, name in zip(predictor_classes, predictor_name):
            self.predictors.append(c(data_path))

    def seq2feat(self, seqs):
        # To apply different regularziation coefficients we scale the features by a multiplier in Ridge regression
        features = [p.seq2feat(seqs) * np.sqrt(1.0 / p.reg_coef) for p in self.predictors]
        return np.concatenate(features, axis=1)


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
        # logger.info(f"EVmutation model loaded from path: {self.couplings_model_path}")
        self.couplings_model = CouplingsModel(self.couplings_model_path)
        wtseqs, wtids = read_fasta(os.path.join(data_path, 'wt.fasta'), return_ids=True)
        if '/' in wtids[0]:
            self.offset = int(wtids[0].split('/')[-1].split('-')[0])
        else:
            self.offset = 1
        expected_wt = wtseqs[0]

        # Validate that PLMC model is compatible with wild-type sequence
        mismatches = []
        for pf, pm in self.couplings_model.index_map.items():
            if expected_wt[pf-self.offset] != self.couplings_model.target_seq[pm]:
                mismatches.append(f'position {pf}: expected={expected_wt[pf-self.offset]}, model={self.couplings_model.target_seq[pm]}')

        if mismatches:
            error_msg = f"PLMC model is not compatible with wild-type sequence. Mismatches found:\n" + "\n".join(mismatches[:10])
            if len(mismatches) > 10:
                error_msg += f"\n... and {len(mismatches) - 10} more mismatches"
            raise ValueError(error_msg)

    def seq2score(self, seqs):
        return seq2effect(seqs, self.couplings_model, self.offset, ignore_gaps=self.ignore_gaps)

    def seq2feat(self, seqs):
        return self.seq2score(seqs)[:, None]

    def predict_unsupervised(self, seqs):
        return self.seq2score(seqs)
