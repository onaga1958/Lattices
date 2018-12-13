from estimators import Estimators

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from functools import reduce
from collections import defaultdict


class Result:
    _delimiter = '\t'

    def __init__(self, name, params, metrics):
        self._name = name
        self._params = params
        self._metrics = metrics

    @property
    def accuracy(self):
        return self._metrics['accuracy']

    @property
    def name(self):
        return self._name

    def get_param(self, param_name):
        return self._params[param_name]

    @classmethod
    def from_str(cls, string):
        name, params, metrics = string.strip().split(cls._delimiter)
        return cls(name, json.loads(params), json.loads(metrics))

    def to_str(self):
        return self._delimiter.join([
            self._name, json.dumps(self._params, sort_keys=True),
            json.dumps(self._metrics, sort_keys=True),
        ])


class ResultsManager:
    _result_path_prefix = '_result_'

    def __init__(self, dataset):
        self._result_path = self._result_path_prefix + dataset.get_name()
        self._results = defaultdict(list)
        if os.path.isfile(self._result_path):
            with open(self._result_path, 'r+') as f:
                for line in f.readlines():
                    result = Result.from_str(line)
                    self._results[result.name].append(result)

    def add_result(self, estimator, metrics):
        result = Result(estimator.get_name(), estimator.get_params(), metrics)
        print(result.to_str())
        self._results[result.name].append(result)

    def dump(self):
        print('dumping results...')
        with open(self._result_path + '_tmp', 'w') as f:
            for result_str in sorted(result.to_str() for result in self._get_results()):
                f.write(result_str + '\n')
        os.rename(self._result_path + '_tmp', self._result_path)

    def _get_results(self, estimator_name=None):
        if estimator_name is None:
            return reduce(lambda x, y: x + y, self._results.values(), [])
        else:
            return self._results[estimator_name]

    def best_accuracy(self, args):
        candidates = self._get_results(args.estimator)
        if args.only_lattice:
            candidates = [
                cand for cand in candidates if Estimators.get_value(cand.name).is_lattice
            ]
        sorted_indexes = np.argsort([cand.accuracy for cand in candidates])
        if args.unique_names:
            printed = set()
            for ind in reversed(sorted_indexes):
                cand = candidates[ind]
                if cand.name not in printed:
                    print(cand.to_str())
                    printed.add(cand.name)
                    if len(printed) == args.top_number:
                        return
        else:
            for ind in sorted_indexes[-args.top_number:]:
                print(candidates[ind].to_str())

    @staticmethod
    def _build_info(values):
        sorted_values = sorted(set(values))
        return sorted_values, {
            value: ind
            for ind, value in enumerate(sorted_values)
        }

    def build_accuracy_plot(self, args):
        name = args.estimator
        param_names = Estimators.get_value(name).get_param_names()
        params_to_vary = [name for name in param_names if name not in args.fixed_parameters]
        if len(params_to_vary) not in [1, 2]:
            print(
                'Can build plot only if number of not fixed params is equal to 1 or 2.\n'
                'Param names: ' + str(param_names) + '\nNot fixed params: ' + str(params_to_vary)
            )
            return

        candidates = [
            result
            for result in self._get_results(name)
            if all(
                result.get_param(name) == fixed_value
                for name, fixed_value in args.fixed_parameters.items()
            ) and result.accuracy >= args.min_accuracy
        ]
        if len(candidates) == 0:
            print('There are no suitable candidates')
            return

        if len(params_to_vary) == 1:
            candidates = sorted(candidates, key=lambda result: result.get_param(params_to_vary[0]))
            accuracies = [cand.accuracy for cand in candidates]
            param_values = [cand.get_param(params_to_vary[0]) for cand in candidates]
            plt.plot(param_values, accuracies)
        if len(params_to_vary) == 2:
            sorted_first, first_ind_dict = self._build_info(
                [cand.get_param(params_to_vary[0]) for cand in candidates]
            )
            sorted_second, second_ind_dict = self._build_info(
                [cand.get_param(params_to_vary[1]) for cand in candidates]
            )
            table = np.array([[np.nan for _ in sorted_first] for _ in sorted_second])
            # mask = [[True for _ in sorted_second] for _ in sorted_first]
            for cand in candidates:
                first_ind = first_ind_dict[cand.get_param(params_to_vary[0])]
                second_ind = second_ind_dict[cand.get_param(params_to_vary[1])]
                table[second_ind][first_ind] = cand.accuracy
                # mask[second_ind][first_ind] = False
            sns.heatmap(
                table, xticklabels=sorted_first, yticklabels=sorted_second,  # mask=mask,
            )
            plt.ylabel(params_to_vary[1])
        plt.xlabel(params_to_vary[0])
        plt.title(name)
        plt.show()
