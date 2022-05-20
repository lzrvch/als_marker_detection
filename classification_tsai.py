import tsai
from tsai.all import *

import pickle
import numpy as np
import pandas as pd

from utils import set_random_seed, suppress_stdout
from tsaug import TimeWarp, AddNoise


expt_tag = 'FCN_lr1-3_const_50epochs_SGD'

set_random_seed(444)

with open('./pickles/ca_w500_s100.pickle', 'rb') as handle:
    d = pickle.load(handle)
    X = d['ca']
    Xs = d['speed']
    y = d['target']
    gr = d['groups']

X = np.moveaxis(np.array([X, Xs]), 0, 1)


def get_dnn_predictions(test_animal_id, X, y, model_cls, trial, aug=False):

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
            TimeWarp(),
            AddNoise(scale=0.01),
        ]
        for aug in augs:
            X_aug = aug.augment(X_train)
            X_train = np.concatenate([X_train, X_aug], axis=0)
            y_train = np.concatenate([y_train, y_train], axis=0)

    print(X_train.shape, X_test.shape, y_train.mean(), y_test.mean())

    X, y, splits = combine_split_data([X_train, X_test], [y_train, y_test])
    tfms = [None, [Categorize()]]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)

    dls = TSDataLoaders.from_dsets(
        dsets.train,
        dsets.valid,
        bs=[512, 1024],
        batch_tfms=[TSStandardize()],
        num_workers=4,
    )

    try:
        model = model_cls['cls'](
            dls.vars, dls.c, **{'seq_len': X.shape[2]}, **model_cls['kwargs']
        )
    except:
        model = model_cls['cls'](dls.vars, dls.c, **model_cls['kwargs'])
    learn = Learner(dls, model, metrics=accuracy, opt_func=SGD)

    learn.fit(n_epoch=50, lr=1e-3)
    # learn.fit_flat_cos(25, lr=1e-7)
    # learn.fit_one_cycle(50, lr_max=5e-3)
    # learn.fit_sgdr(3, 10, lr_max=1e-4)

    pd.DataFrame(learn.recorder.values).to_csv(
        f'./csv/{expt_tag}_metrics_{test_animal_id}_{trial}.csv'
    )

    valid_probas, valid_targets, valid_preds = learn.get_preds(
        dl=dls.valid, with_decoded=True
    )
    with open(f'./csv/{expt_tag}_preds_{test_animal_id}_{trial}.pkl', 'wb') as outfile:
        pickle.dump(
            {'probas': np.array(valid_probas), 'preds': np.array(valid_preds)}, outfile
        )

    return np.mean(np.array(valid_targets)), np.mean(np.array(valid_preds))


def infer_model_preds(model_cls, X, y, thr=0.5, n_trials=7):
    scores = []
    for test_id in range(0, 14):
        animal_score = []
        for trial in range(n_trials):
            with suppress_stdout():
                target, pred = get_dnn_predictions(test_id, X, y, model_cls, trial)
            animal_score.append(pred)
        scores.append(np.mean(animal_score))
        print(scores)
    return scores


model_zoo = [
    (XceptionTimePlus, {}),
    (FCNPlus, {}),
]

for model_cls, kwargs in model_zoo:
    acc = infer_model_preds({'cls': model_cls, 'kwargs': kwargs}, X, y)
    print(str(model_cls), acc)
