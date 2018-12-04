from utils import Namespace

from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


class CatBoostWrapper(CatBoostClassifier):
    _param_names = ['iterations']

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

    def calculate_support(self, feature_indexes):
        return self._calculate_stroke_part(feature_indexes, self._POSITIVE)

    def calculate_confidence(self, feature_indexes):
        return self._calculate_stroke_part(feature_indexes, self._NEGATIVE)

    @property
    def _positives(self):
        return self._data[self._POSITIVE]

    @property
    def _negatives(self):
        return self._data[self._NEGATIVE]

    def _calc_mean_stats(self, features, calc_support=False, calc_conf=False):
        assert calc_conf or calc_support
        total_support = 0
        total_confidence = 0
        for positive in self._positives:
            interseption = build_interseption_indexes(positive, features)
            if calc_conf:
                confidence = self.calculate_confidence(interseption)
                total_confidence += confidence
            if calc_support:
                support = self.calculate_support(interseption)
                total_support += support
        mean_support = total_support / len(self._positives)
        mean_confidence = total_confidence / len(self._positives)
        result = []
        if calc_support:
            result.append(mean_support)
        if calc_conf:
            result.append(mean_confidence)
        return result

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


class Estimators(Namespace):
    _name_to_value = {
        estimator_cls.get_name(): estimator_cls
        for estimator_cls in [
            CatBoostWrapper, RandomForestWrapper, SimpleEstimator,
            SupportMoreThanConfidenceEstimator, MinSupportEstimator, ReverseMinSupportEstimator,
            MinSupportMaxConfidenceEstimator,
        ]
    }
