import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tsaug import TimeWarp, Quantize, Drift, AddNoise

from utils import set_random_seed

set_random_seed(444)


with open('./pickles/ca_w500_s100.pickle', 'rb') as handle:
    d = pickle.load(handle)
    X = d['ca']
    Xs = d['speed']
    y = d['target']
    gr = d['groups']

X = np.moveaxis(np.array([X, Xs]), 0, 1)


def get_sklearn_predictions(test_animal_id, X, y, aug=False):

    test_animal_ids = [test_animal_id]
    ctrl_ids = [0, 1, 2, 3, 4, 5, 6]
    train_animal_ids = [gr_idx for gr_idx in set(gr) if gr_idx not in ctrl_ids]

    if test_animal_ids[0] not in ctrl_ids:
        train_animal_ids = [
            gr_idx
            for gr_idx in set(gr)
            if gr_idx not in (ctrl_ids[0], test_animal_ids[0])
        ]
    else:
        add_train_id = train_animal_ids[0]
        train_animal_ids = [
            gr_idx
            for gr_idx in set(gr)
            if gr_idx not in (test_animal_ids[0], add_train_id)
        ]

    test_index = [idx for idx in range(X.shape[0]) if gr[idx] in test_animal_ids]
    train_index = [idx for idx in range(X.shape[0]) if gr[idx] in train_animal_ids]

    print(set(gr[train_index]), set(gr[test_index]))

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    if aug:
        augs = [
            Quantize(n_levels=[30, 40, 50]),
            Drift(max_drift=(0.1, 0.5)),
            TimeWarp(),
            AddNoise(scale=0.01),
        ]
        for aug in augs:
            X_aug = aug.augment(X_train)
            X_train = np.concatenate([X_train, X_aug], axis=0)
            y_train = np.concatenate([y_train, y_train], axis=0)

    model = RandomForestClassifier(
        n_estimators=1000, max_depth=5, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    print(accuracy_score(model.predict(X_test), y_test), y_test.mean())
    pd.DataFrame(model.predict_proba(X_test)[:, 1]).to_csv(
        f'./csv/sklearn_rf_probs_{test_animal_id}.csv'
    )
    return np.mean(y_test), np.mean(model.predict_proba(X_test)[:, 1])


def infer_model_preds(X, y, thr=0.5):
    scores = []
    for test_id in range(0, 14):
        animal_score = []
        for _ in range(1):
            target, pred = get_sklearn_predictions(test_id, X, y)
            animal_score.append(pred)
        scores.append(np.mean(animal_score))
        print(scores)
    return scores


if __name__ == '__main__':
    acc = infer_model_preds(X, y)
