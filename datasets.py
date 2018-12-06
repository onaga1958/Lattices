from utils import Namespace

import pandas as pd
import numpy as np

from os.path import isfile


class Dataset:
    def prepare_dataset(self):
        raise NotImplementedError()

    @classmethod
    def get_prepared_path(cls):
        return 'prepared_' + cls._name + '.csv'

    @classmethod
    def get_raw_path(cls):
        return cls._name + '.csv'

    def _read_data(self, path):
        return pd.read_csv(path, header=None).values

    def get_X_y(self):
        self.prepare_dataset()
        data = self._read_data(self.get_prepared_path())
        return data[:, 1:], data[:, 0]

    @classmethod
    def get_name(cls):
        return cls._name


class BigBreastCanser(Dataset):
    _name = 'big_breast_cancer'
    _values = range(1, 11)

    def _make_one_hot(self, value):
        return [1 if one_hot_value == value else 0 for one_hot_value in self._values]

    def _make_greater_than_value(self, value):
        return [1 if one_hot_value >= value else 0 for one_hot_value in self._values]

    def _make_less_than_value(self, value):
        return [1 if one_hot_value <= value else 0 for one_hot_value in self._values]

    @classmethod
    def get_raw_path(cls):
        return BigBreastCanser._name + '.csv'

    def _prepare_factor(self, value, ind):
        raise NotImplementedError()

    def get_raw_X_y(self):
        data = self._read_data(self.get_raw_path())
        y = (data[:, -1] == 4).astype(np.int)
        X_raw = data[:, 1:-1].astype(np.int)
        return X_raw, y

    def prepare_dataset(self):
        X_raw, y = self.get_raw_X_y()

        with open(self.get_prepared_path(), 'w') as f_out:
            for answer, factors in zip(y, X_raw):
                f_out.write(str(answer))
                for ind, value in enumerate(factors):
                    prepeared_features = self._prepare_factor(ind, value)
                    f_out.write(',' + ','.join(map(str, prepeared_features)))
                f_out.write('\n')


class BigBreastCanserOneHot(BigBreastCanser):
    _name = 'big_breast_cancer_one_hot'

    def _prepare_factor(self, ind, value):
        return self._make_one_hot(value)


class BigBreastCanserLinearThickness(BigBreastCanser):
    _name = 'big_breast_cancer_linear_thickness'

    def _prepare_factor(self, ind, value):
        if ind == 0:
            return self._make_greater_than_value(value)
        else:
            return self._make_one_hot(value)


class BigBreastCanserAllLinear(BigBreastCanser):
    _name = 'big_breast_cancer_all_linear'

    def _prepare_factor(self, ind, value):
        return self._make_greater_than_value(value)


class BigBreastCanserCustom(BigBreastCanser):
    _name = 'big_breast_cancer_custom'

    def _prepare_factor(self, ind, value):
        if ind == 0:
            features = self._make_greater_than_value(value)
            features[3] = 1 if value == 4 else 0
            features[6] = 1 if value == 7 else 0
            return features
        if ind == 1:
            features = self._make_greater_than_value(value)[:4]
            features.append(1 if value >= 5 else 0)
            features.append(1 if value >= 8 else 0)
            return features
        if ind == 2:
            features = self._make_greater_than_value(value)[:3]
            features.append(1 if value == 4 else 0)
            features.append(1 if value >= 5 else 0)
            features.append(1 if value >= 7 else 0)
            return features
        if ind == 3:
            return self._make_greater_than_value(value)[:7]
        if ind == 4:
            return self._make_greater_than_value(value)[:6]
        if ind == 5:
            features = self._make_greater_than_value(value)[:6]
            features.append(1 if value == 10 else 0)
            return features
        if ind == 6:
            features = self._make_greater_than_value(value)[:9]
            features[2] = 1 if value == 3 else 0
            features[4] = 1 if value == 5 else 0
            features[7] = 1 if value == 8 else 0
            return features
        if ind == 7:
            features = self._make_greater_than_value(value)[:4]
            features.append(1 if value >= 8 else 0)
            features.append(1 if value == 10 else 0)
            return features
        if ind == 8:
            features = self._make_greater_than_value(value)[:3]
            features.append(1 if value == 10 else 0)
            return features


class SmallBreasCanser(Dataset):
    _name = 'small_breast_cancer'

    def prepare_dataset(self):
        if isfile(self.prepared_path):
            return

        data = self._read_data(self.raw_path)
        y = (data[:, 0] == 'recurrence-events').astype(np.int)
        X_raw = data[:, 1:]
        factor_sets = [sorted(set(feature_values)) for feature_values in X_raw.T]
        facotrs_dicts = [
            {feature_value: ind for ind, feature_value in enumerate(feature_values)}
            for feature_values in factor_sets
        ]

        with open(self.prepared_path, 'w') as f_out:
            for answer, factors in zip(y, X_raw):
                f_out.write(str(answer))
                for value, factor_dict in zip(factors, facotrs_dicts):
                    if len(factor_dict) == 2:
                        one_hot_encoding = [factor_dict[value]]
                    else:
                        one_hot_encoding = [0 for value_var in factor_dict]
                        one_hot_encoding[factor_dict[value]] = 1
                    f_out.write(',' + ','.join(map(str, one_hot_encoding)))
                f_out.write('\n')


class DatasetPreparations(Namespace):
    _name_to_value = {
        'one_hot': BigBreastCanserOneHot(),
        'linear_thickness': BigBreastCanserLinearThickness(),
        'all_linear': BigBreastCanserAllLinear(),
        'custom': BigBreastCanserCustom(),
        'small': SmallBreasCanser(),
    }
