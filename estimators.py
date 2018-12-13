from utils import Namespace

from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

import numpy as np
import os


class CatBoostWrapper(CatBoostClassifier):
    _param_names = ['iterations']
    is_lattice = False

    def __init__(self, **params):
        super().__init__(random_seed=117, verbose=False, **params)

    def get_params(self, deep=True):
        params = super().get_params(False)
        return {
            name: params[name] for name in self._param_names
        }

    def set_params(self, **params):
        super().set_params(random_seed=117, verbose=False, **params)

    @classmethod
    def get_name(self):
        return 'catboost'

    @classmethod
    def get_param_names(cls):
        return cls._param_names


class RandomForestWrapper:
    _param_names = ['n_estimators']
    is_lattice = False

    def __init__(self, **params):
        self._params = params
        self._classifier = RandomForestClassifier(random_state=117, **params)

    def get_params(self, deep=True):
        return self._params

    def fit(self, X, y):
        self._classifier.fit(X, y)

    def predict(self, X):
        return self._classifier.predict(X)

    def set_params(self, **params):
        self._params = params
        self._classifier.set_params(random_state=117, **params)

    @classmethod
    def get_name(self):
        return 'RF'

    @classmethod
    def get_param_names(cls):
        return cls._param_names


class BasicLatticeEstimator:
    _param_names = []
    _POSITIVE = 'positives'
    _NEGATIVE = 'negatives'
    is_lattice = True

    def __init__(self, **params):
        self.set_params(**params)

    def fit(self, X, y):
        self._data = {}
        self._calculated_stroke = {}
        self._data[self._POSITIVE] = [features for features, label in zip(X, y) if label == 1]
        self._data[self._NEGATIVE] = [features for features, label in zip(X, y) if label == 0]
        self._calculated_stroke[self._POSITIVE] = {}
        self._calculated_stroke[self._NEGATIVE] = {}

    def predict(self, X):
        return [self._predict_on_features(features) for features in X]

    def _predict_on_features(self, features):
        raise NotImplementedError()

    def get_params(self, deep=True):
        return {name: getattr(self, '_' + name) for name in self._param_names}

    def set_params(self, **params):
        for name in self._param_names:
            setattr(self, '_' + name, params[name])

    def _calculate_stroke_part(self, feature_indexes, context_name):
        stroke_part = self._calculated_stroke[context_name].get(feature_indexes)
        context = self._data[context_name]
        if stroke_part is not None:
            return stroke_part
        cnt = 0
        for features in context:
            for ind in feature_indexes:
                if features[ind] == 0:
                    break
            else:
                cnt += 1
        stroke_part = cnt / len(context)
        self._calculated_stroke[context_name][feature_indexes] = stroke_part
        return stroke_part

    def calculate_support(self, feature_indexes, base_context_name=None):
        if base_context_name is None:
            base_context_name = self._POSITIVE
        return self._calculate_stroke_part(feature_indexes, base_context_name)

    def calculate_confidence(self, feature_indexes, base_context_name=None):
        if base_context_name is None:
            base_context_name = self._POSITIVE
        if base_context_name == self._POSITIVE:
            stroke_context_name = self._NEGATIVE
        elif base_context_name == self._NEGATIVE:
            stroke_context_name = self._POSITIVE
        else:
            raise ValueError('Unknown context names: ' + base_context_name)
        return self._calculate_stroke_part(feature_indexes, stroke_context_name)

    @property
    def _positives(self):
        return self._data[self._POSITIVE]

    @property
    def _negatives(self):
        return self._data[self._NEGATIVE]

    def _get_stats(
            self, features, calc_support=False, calc_conf=False,
            base_context_name=None):
        assert calc_conf or calc_support
        if base_context_name is None:
            base_context_name = self._POSITIVE
        base_context = self._data[base_context_name]

        confidences = np.zeros(len(base_context))
        supports = np.zeros(len(base_context))
        for ind, base_features in enumerate(base_context):
            interseption = build_interseption_indexes(base_features, features)
            if calc_support:
                supports[ind] = self.calculate_support(interseption, base_context_name)
            if calc_conf:
                confidences[ind] = self.calculate_confidence(interseption, base_context_name)

        return supports, confidences

    def _calc_mean_stats(self, features, calc_support=False, calc_conf=False):
        assert calc_conf or calc_support
        stat_arrs = self._get_stats(features, calc_support, calc_conf)
        return [
            np.mean(stat_arr)
            for stat_arr, need_calc in zip(stat_arrs, [calc_support, calc_conf])
            if need_calc
        ]

    @classmethod
    def get_name(cls):
        return cls._name

    @classmethod
    def get_param_names(cls):
        return cls._param_names


def build_interseption_indexes(feature_values_a, feature_values_b):
    return tuple(
        ind
        for ind, (value_a, value_b) in enumerate(zip(feature_values_a, feature_values_b))
        if value_a == 1 and value_b == 1
    )


class SimpleEstimator(BasicLatticeEstimator):
    _name = 'Simple'
    _param_names = []

    def _predict_on_features(self, features):
        for positive in self._positives:
            interseption = build_interseption_indexes(positive, features)
            confidence = self.calculate_confidence(interseption)
            if confidence == 0:
                return 1
        return 0


class SupportMoreThanConfidenceEstimator(BasicLatticeEstimator):
    _name = 'SupportMoreThanConfidence'
    _param_names = []

    def _predict_on_features(self, features):
        mean_support, mean_confidence = self._calc_mean_stats(features, True, True)

        if mean_support > mean_confidence:
            return 1
        else:
            return 0


class MinSupportEstimator(BasicLatticeEstimator):
    _name = 'MinSupport'
    _param_names = ['min_support']

    def _predict_on_features(self, features):
        mean_support = self._calc_mean_stats(features, calc_support=True)[0]

        if mean_support > self._min_support:
            return 1
        else:
            return 0


class ReverseMinSupportEstimator(BasicLatticeEstimator):
    _name = 'ReverseMinSupport'
    _param_names = ['min_support']

    def _predict_on_features(self, features):
        mean_support = self._calc_mean_stats(features, calc_support=True)[0]

        if mean_support > self._min_support:
            return 0
        else:
            return 1


class MinSupportMaxConfidenceEstimator(BasicLatticeEstimator):
    _name = 'MinSupportMaxConfidence'
    _param_names = ['min_support', 'max_confidence']

    def _predict_on_features(self, features):
        mean_support, mean_confidence = self._calc_mean_stats(features, True, True)

        if mean_support > self._min_support and mean_confidence < self._max_confidence:
            return 1
        else:
            return 0


class MinMeanAndMinSupportMaxConfidenceEstimator(BasicLatticeEstimator):
    _name = 'MinMeanAndMinSupportMaxConfidence'
    _param_names = ['min_mean_support', 'min_min_support', 'max_confidence']

    def _predict_on_features(self, features):
        sups, confs = self._get_stats(features, True, True)
        if (
                np.mean(sups) > self._min_mean_support and
                np.min(sups) > self._min_min_support and
                np.mean(confs) < self._max_confidence
                ):
            return 1
        else:
            return 0


class MinSupportMaxMaxConfidenceEstimator(BasicLatticeEstimator):
    _name = 'MinSupportMaxMaxConfidence'
    _param_names = ['min_support', 'max_confidence']

    def _predict_on_features(self, features):
        sups, confs = self._get_stats(features, True, True)
        if (np.mean(sups) > self._min_support and np.max(confs) < self._max_confidence):
            return 1
        else:
            return 0


class MinSupportMaxConfidenceQuantileEstimator(BasicLatticeEstimator):
    _name = 'MinSupportMaxConfidenceQuantile'
    _param_names = ['min_support', 'max_confidence', 'confidence_quantile']

    def __init__(self, **params):
        super().__init__(**params)
        assert self._confidence_quantile >= 0 and self._confidence_quantile <= 1, (
            'quantile should be in [0, 1] segment'
        )

    def _predict_on_features(self, features):
        sups, confs = self._get_stats(features, True, True)
        quantile = np.quantile(confs, self._confidence_quantile)

        if (np.mean(sups) > self._min_support and quantile < self._max_confidence):
            return 1
        else:
            return 0


class MinMedianSupportEstimator(BasicLatticeEstimator):
    _name = 'MinMedianSupport'
    _param_names = ['min_support']

    def _predict_on_features(self, features):
        supports = self._get_stats(features, calc_support=True)[0]
        half = int(len(supports) / 2)

        if np.partition(supports, half)[half] > self._min_support:
            return 1
        else:
            return 0


class MinMinSupportEstimator(BasicLatticeEstimator):
    _name = 'MinMinSupport'
    _param_names = ['min_support']

    def _predict_on_features(self, features):
        supports = self._get_stats(features, calc_support=True)[0]
        half = int(len(supports) / 2)

        if np.min(supports) > self._min_support:
            return 1
        else:
            return 0


class PositiveSupportVsNegativeSupportEstimator(BasicLatticeEstimator):
    _name = 'PositiveSupportVsNegativeSupport'

    def _predict_on_features(self, features):
        mean_positive_support = np.mean(self._get_stats(
            features, calc_support=True, base_context_name=self._POSITIVE
        )[0])
        mean_negative_support = np.mean(self._get_stats(
            features, calc_support=True, base_context_name=self._NEGATIVE
        )[0])
        if mean_positive_support > mean_negative_support:
            return 0
        else:
            return 1


class PositiveConfVsNegativeConfEstimator(BasicLatticeEstimator):
    _name = 'PositiveConfVsNegativeConf'

    def _predict_on_features(self, features):
        mean_positive_conf = np.mean(self._get_stats(
            features, calc_conf=True, base_context_name=self._POSITIVE
        )[1])
        mean_negative_conf = np.mean(self._get_stats(
            features, calc_support=True, base_context_name=self._NEGATIVE
        )[1])
        if mean_positive_conf < mean_negative_conf:
            return 1
        else:
            return 0


class ConfAndSupNegVsPosEstimator(BasicLatticeEstimator):
    _name = 'ConfAndSupNegVsPos'

    def _predict_on_features(self, features):
        mean_positive_support, mean_positive_conf = np.mean(self._get_stats(
            features, calc_conf=True, base_context_name=self._POSITIVE
        ), axis=1)
        mean_negative_support, mean_negative_conf = np.mean(self._get_stats(
            features, calc_support=True, base_context_name=self._NEGATIVE
        ), axis=1)
        if (
                mean_positive_conf < mean_negative_conf and
                mean_positive_support > mean_negative_support
                ):
            return 1
        else:
            return 0


class BasicRawDataEstimator(BasicLatticeEstimator):
    _PRINT_THRESHOLDS = os.getenv('PRINT_THRESHOLDS', '0') == '1'

    @property
    def _positives(self):
        if not isinstance(self._data[self._POSITIVE], np.ndarray):
            self._data[self._POSITIVE] = np.array(self._data[self._POSITIVE])
        return self._data[self._POSITIVE]

    @property
    def _negatives(self):
        if not isinstance(self._data[self._NEGATIVE], np.ndarray):
            self._data[self._NEGATIVE] = np.array(self._data[self._NEGATIVE])
        return self._data[self._NEGATIVE]


class FeatureQuantileEstimator(BasicRawDataEstimator):
    _name = 'FeatureQuantile'
    _param_names = ['quantile', 'enough_passed']

    def fit(self, X, y):
        super().fit(X, y)
        self._thresholds = np.quantile(self._positives, self._quantile, axis=0)
        if self._PRINT_THRESHOLDS:
            print('thresholds:')
            print(self._thresholds)

    def _predict_on_features(self, features):
        passed = 0
        for feature, threshold in zip(features, self._thresholds):
            if feature > threshold:
                passed += 1
        if passed >= self._enough_passed:
            return 1
        else:
            return 0


class TwoSideFeatureQuantileEstimator(BasicRawDataEstimator):
    _name = 'TwoSideFeatureQuantile'
    _param_names = [
        'positive_quantile', 'min_positive_passed',
        'max_negative_passed', 'negative_quantile',
    ]

    def fit(self, X, y):
        super().fit(X, y)
        self._positive_thresholds = np.quantile(self._positives, self._positive_quantile, axis=0)
        self._negative_thresholds = np.quantile(self._negatives, self._negative_quantile, axis=0)

        if self._PRINT_THRESHOLDS:
            print('positive_thresholds:')
            print(self._positive_thresholds)
            print('negative thresholds:')
            print(self._negative_thresholds)

    def _predict_on_features(self, features):
        positive_passed = 0
        negative_passed = 0
        for feature, positive_threshold, negative_threshold in zip(
                features, self._positive_thresholds, self._negative_thresholds):
            if feature > positive_threshold:
                positive_passed += 1
            if feature < negative_threshold:
                negative_passed += 1

        positive_condition = (
            positive_passed >= self._min_positive_passed and
            negative_passed <= self._max_negative_passed
        )
        if positive_condition:
            return 1
        else:
            return 0


class ComparePositiveAndNegativeFeatureQuantileEstimator(BasicRawDataEstimator):
    _name = 'ComparePositiveAndNegativeFeatureQuantile'
    _param_names = ['positive_quantile', 'negative_quantile']

    def fit(self, X, y):
        super().fit(X, y)
        self._positive_thresholds = np.quantile(self._positives, self._positive_quantile, axis=0)
        self._negative_thresholds = np.quantile(self._negatives, self._negative_quantile, axis=0)

        if self._PRINT_THRESHOLDS:
            print('positive_thresholds:')
            print(self._positive_thresholds)
            print('negative thresholds:')
            print(self._negative_thresholds)

    def _predict_on_features(self, features):
        positive_passed = 0
        negative_passed = 0
        for feature, positive_threshold, negative_threshold in zip(
                features, self._positive_thresholds, self._negative_thresholds):
            if feature > positive_threshold:
                positive_passed += 1
            if feature < negative_threshold:
                negative_passed += 1

        if positive_passed > negative_passed:
            return 1
        else:
            return 0


class Estimators(Namespace):
    _name_to_value = {
        estimator_cls.get_name(): estimator_cls
        for estimator_cls in [
            CatBoostWrapper, RandomForestWrapper, SimpleEstimator,
            SupportMoreThanConfidenceEstimator, MinSupportEstimator, ReverseMinSupportEstimator,
            MinSupportMaxConfidenceEstimator, MinMeanAndMinSupportMaxConfidenceEstimator,
            MinMedianSupportEstimator, MinMinSupportEstimator,
            MinSupportMaxMaxConfidenceEstimator, MinSupportMaxConfidenceQuantileEstimator,
            PositiveSupportVsNegativeSupportEstimator, ConfAndSupNegVsPosEstimator,
            FeatureQuantileEstimator, TwoSideFeatureQuantileEstimator,
            ComparePositiveAndNegativeFeatureQuantileEstimator,
        ]
    }
