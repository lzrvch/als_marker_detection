import os
import sys
import random
import psutil
import numpy as np
import pandas as pd
import torch

from contextlib import contextmanager
from functools import partial

from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
import tsfresh.utilities.dataframe_functions as tsfresh_utils

from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline


CPU_COUNT = psutil.cpu_count(logical=True)


class NoFitMixin:
    def fit(self, X, y=None):
        return self


class DFTransform(TransformerMixin, NoFitMixin):
    def __init__(self, func, copy=False, **kwargs):
        self.func = func
        self.copy = copy
        self.kwargs = kwargs

    def set_params(self, **params):
        for key, value in params.items():
            self.kwargs[key] = value

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        return self.func(X_, **self.kwargs)


class DFStandardScaler(TransformerMixin, NoFitMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X, y)
        return self

    def transform(self, X):
        X[X.columns] = self.scaler.transform(X[X.columns])
        return X


class DFLowVarianceRemoval(TransformerMixin, NoFitMixin):
    def __init__(self, variance_threshold=0.2):
        self.high_variance_features = None
        self.variance_threshold = variance_threshold
        self.safety_eps = 1e-9

    def fit(self, X, y=None):
        self.high_variance_features = X.loc[
            :, (X.std() / (self.safety_eps + X.mean())).abs() > self.variance_threshold
        ].columns.values
        return self

    def transform(self, X):
        return X.loc[:, self.high_variance_features]


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def simple_undersampling(X, y, subsample_size=None, pandas=True):
    # assuming binary target
    dominant_class_label = int(y.mean() > 0.5)
    X = pd.DataFrame(X) if not pandas else X
    num_samples = (y != dominant_class_label).sum()
    dominant_indices = np.random.choice(
        X.shape[0] - num_samples, num_samples, replace=False
    )
    X_undersampled = pd.concat(
        [
            X.iloc[np.where(y != dominant_class_label)[0], :],
            X.iloc[np.where(y == dominant_class_label)[0], :].iloc[dominant_indices, :],
        ]
    )
    y_undersampled = np.array(
        [int(not dominant_class_label)] * num_samples
        + [dominant_class_label] * num_samples
    )
    if subsample_size is not None:
        sample_indices = np.random.choice(
            X_undersampled.shape[0],
            int(subsample_size * X_undersampled.shape[0]),
            replace=False,
        )
        X_undersampled, y_undersampled = (
            X_undersampled.iloc[sample_indices, :],
            y_undersampled[sample_indices],
        )
    return X_undersampled, y_undersampled


def set_random_seed(seed_value):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


class TsfreshVectorizeTransform(TransformerMixin, NoFitMixin):
    def __init__(self, to_file=None, feature_set=None, n_jobs=CPU_COUNT, verbose=True):
        self.to_file = to_file
        self.feature_set = feature_set
        self.n_jobs = n_jobs
        self.verbose = verbose

    @staticmethod
    def transform_to_tsfresh_format(X):
        df = pd.DataFrame(columns=['id', 'time', 'value'], dtype=float)
        for index in range(X.shape[1]):
            tmp = pd.DataFrame(X[:, index], columns=['value'])
            tmp['id'] = list(range(X.shape[0]))
            tmp['time'] = [index] * X.shape[0]
            df = pd.concat([df, tmp], ignore_index=True, sort=False)
        return df

    @staticmethod
    def get_feature_dict(feature_set=None):
        full_feature_dict = ComprehensiveFCParameters()
        simple_baseline_features = {
            key: None
            for key in [
                'abs_energy',
                'mean',
                'median',
                'minimum',
                'maximum',
                'standard_deviation',
            ]
        }
        distribution_features_dict = distribution_features_tsfresh_dict()
        temporal_feature_dict = {
            key: full_feature_dict[key]
            for key in set(full_feature_dict) - set(distribution_features_dict)
        }
        no_entropy_features_dict = {
            key: value
            for key, value in full_feature_dict.items()
            if 'entropy' not in key
        }
        feature_dict = {
            'simple_baseline': simple_baseline_features,
            'distribution_features': distribution_features_dict,
            'temporal_features': temporal_feature_dict,
            'no_entropy': no_entropy_features_dict,
        }
        return feature_dict.get(feature_set, full_feature_dict)

    def transform(self, X):
        tsfresh_df = self.transform_to_tsfresh_format(X)
        ts_feature_dict = self.get_feature_dict(self.feature_set)
        X_feats = extract_features(
            tsfresh_df,
            default_fc_parameters=ts_feature_dict,
            column_id='id',
            column_sort='time',
            disable_progressbar=np.logical_not(self.verbose),
            n_jobs=self.n_jobs,
        )
        return X_feats


def distribution_features_tsfresh_dict():
    ratios_beyond_r_sigma_rvalues = [1, 1.5, 2, 2.5, 3, 5, 6, 7, 10]

    feature_dict = {
        'symmetry_looking': [{'r': value} for value in np.arange(0.05, 1.0, 0.05)],
        'standard_deviation': None,
        'kurtosis': None,
        'variance_larger_than_standard_deviation': None,
        'ratio_beyond_r_sigma': [
            {'r': value} for value in ratios_beyond_r_sigma_rvalues
        ],
        'count_below_mean': None,
        'maximum': None,
        'variance': None,
        'abs_energy': None,
        'mean': None,
        'skewness': None,
        'length': None,
        'large_standard_deviation': [
            {'r': value} for value in np.arange(0.05, 1.0, 0.05)
        ],
        'count_above_mean': None,
        'minimum': None,
        'sum_values': None,
        'quantile': [{'q': value} for value in np.arange(0.1, 1.0, 0.1)],
        'ratio_value_number_to_time_series_length': None,
        'median': None,
    }

    return feature_dict


class TsfreshFeaturePreprocessorPipeline:
    def __init__(
        self,
        impute=True,
        do_scaling=True,
        remove_low_variance=True,
        keep_features_list=None,
    ):
        self.impute = impute
        self.do_scaling = do_scaling
        self.remove_low_variance = remove_low_variance
        self.keep_features_list = keep_features_list

    @staticmethod
    def _tsfresh_imputation(X):
        tsfresh_utils.impute(X)
        return X

    @staticmethod
    def _select_features(X, feature_list=None):
        feature_list = X.columns.value if feature_list is None else feature_list
        return X.loc[:, feature_list]

    def construct_pipeline(self):
        chained_transformers = []
        if self.keep_features_list is not None:
            chained_transformers.append(
                (
                    'select_features',
                    DFTransform(
                        partial(
                            self._select_features, feature_list=self.keep_features_list
                        )
                    ),
                )
            )
        if self.impute:
            chained_transformers.append(
                ('imputation', DFTransform(self._tsfresh_imputation))
            )
        if self.remove_low_variance:
            chained_transformers.append(('low_var_removal', DFLowVarianceRemoval()))
        if self.do_scaling:
            chained_transformers.append(('standard_scaling', DFStandardScaler()))
        return Pipeline(chained_transformers)
