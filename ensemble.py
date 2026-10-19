"""
Stacked ensemble. v1 averaged NB + LSTM; NB's probabilities are near 0/1 so it dominated.
Here a tiny logistic regression learns how much to trust each model (on logit scale,
fit on a validation split none of the models trained on).
"""
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class Stack:
    """Logistic regression over a subset of model columns (models with negative weight are dropped)."""
    def __init__(self, cols, lr):
        self.cols, self.lr = cols, lr
        self.weights = np.zeros(3)
        self.weights[cols] = lr.coef_[0]
    def predict_proba(self, X):
        return self.lr.predict_proba(X[:, self.cols])


def fit(nb_p, lstm_p, lr_p, y):
    X = np.column_stack([_logit(nb_p), _logit(lstm_p), _logit(lr_p)])
    cols = [0, 1, 2]
    while True:
        lr = LogisticRegression(C=1.0).fit(X[:, cols], y)
        neg = [c for c, w in zip(cols, lr.coef_[0]) if w < 0]
        if not neg or len(cols) == 1:
            return Stack(cols, lr)
        cols.remove(min(neg, key=lambda c: lr.coef_[0][cols.index(c)]))


def predict_many(meta, nb_p, lstm_p, lr_p):
    X = np.column_stack([_logit(nb_p), _logit(lstm_p), _logit(lr_p)])
    return meta.predict_proba(X)[:, 1]


def predict(meta, nb_p, lstm_p, lr_p):
    return float(predict_many(meta, [nb_p], [lstm_p], [lr_p])[0])


def save(meta, path="ensemble.pkl"):
    with open(path, "wb") as f: pickle.dump(meta, f)

def load(path="ensemble.pkl"):
    with open(path, "rb") as f: return pickle.load(f)
