import pickle
import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils import TsfreshVectorizeTransform
from utils import TsfreshFeaturePreprocessorPipeline

from sklearn.metrics import accuracy_score, roc_auc_score

from tsaug import TimeWarp, Quantize, Drift, AddNoise


def random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value)  # cpu vars
    random.seed(seed_value)  # Python


random_seed(444)

with open('./ca_w500_s100.pickle', 'rb') as handle:
    d = pickle.load(handle)
    X = d['ca']
    Xs = d['speed']
    y = d['target']
    gr = d['groups']

X = np.moveaxis(np.array([X, Xs]), 0, 1)


def get_tsfresh_predictions(test_animal_id, X, y, aug=False):

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

    vectorizer = TsfreshVectorizeTransform(feature_set=None)
    X_train_ca = vectorizer.transform(X_train[:, 0, :])
    X_test_ca = vectorizer.transform(X_test[:, 0, :])
    X_train_spd = vectorizer.transform(X_train[:, 1, :])
    X_test_spd = vectorizer.transform(X_test[:, 1, :])

    preprocessing = TsfreshFeaturePreprocessorPipeline(
        do_scaling=True,
        remove_low_variance=True,
    ).construct_pipeline()

    preprocessing.fit(X_train_ca)
    X_train_ca = preprocessing.transform(X_train_ca)
    X_test_ca = preprocessing.transform(X_test_ca)

    preprocessing.fit(X_train_spd)
    X_train_spd = preprocessing.transform(X_train_spd)
    X_test_spd = preprocessing.transform(X_test_spd)

    X_train = np.concatenate([X_train_ca, X_train_spd], axis=1)
    X_test = np.concatenate([X_test_ca, X_test_spd], axis=1)

    print(X_train.shape, X_test.shape, y_train.mean(), y_test.mean())

    model = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=0.01,
        random_state=0,
    )
    model.fit(X_train, y_train)

    print(accuracy_score(model.predict(X_test), y_test), y_test.mean())
    pd.DataFrame(model.predict_proba(X_test)[:, 1]).to_csv(
        f'./csv/tsfresh_logit_probs_{test_animal_id}.csv'
    )
    return np.mean(y_test), np.mean(model.predict_proba(X_test)[:, 1])


def infer_model_preds(X, y, thr=0.5):
    scores = []
    for test_id in range(0, 14):
        animal_score = []
        for _ in range(1):
            target, pred = get_tsfresh_predictions(test_id, X, y)
            animal_score.append(pred)
        scores.append(np.mean(animal_score))
        print(scores)
    return scores


if __name__ == '__main__':
    acc = infer_model_preds(X, y)
