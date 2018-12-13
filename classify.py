#!/usr/local/bin/python3.6
from result_manager import ResultsManager
from estimators import Estimators
from datasets import DatasetPreparations

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import cross_validate, KFold

import numpy as np

import argparse
import json
import shlex

from itertools import product


class ArgumentParserError(Exception):
    pass


class ThrowingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ArgumentParserError(message)


def parse_args():
    parser = ThrowingArgumentParser()
    parser.add_argument(
        '-d', '--dataset', choices=DatasetPreparations.get_all_names(),
    )
    mode = parser.add_subparsers()

    runner = mode.add_parser('run')
    runner.add_argument(
        '-e', '--estimator', required=True, choices=Estimators.get_all_names(),
    )
    runner.add_argument('-p', '--params', default={}, type=json.loads)
    runner.set_defaults(func=runner_main)

    fit_all = mode.add_parser('fit_all')
    fit_all.add_argument('-e', '--estimator', required=True, choices=Estimators.get_all_names())
    fit_all.add_argument('-p', '--params', default={}, type=json.loads)
    fit_all.set_defaults(func=fit_all_main)

    analyze = mode.add_parser('analyze')
    analyze.add_argument('-e', '--estimator', choices=Estimators.get_all_names())
    analyze.set_defaults(func=analyze_main)
    analyze_mode = analyze.add_subparsers()

    best = analyze_mode.add_parser('best')
    best.add_argument(
        '-e', '--estimator', choices=Estimators.get_all_names(),
    )
    best.add_argument('-n', '--top_number', default=1, type=int)
    best.add_argument('-l', '--only_lattice', action='store_true')
    best.add_argument('--unique_names', action='store_true')
    best.set_defaults(analyze_func=ResultsManager.best_accuracy)

    graph = analyze_mode.add_parser('graph')
    graph.add_argument(
        '-e', '--estimator', required=True, choices=Estimators.get_all_names(),
    )
    graph.add_argument(
        '-f', '--fixed_parameters', default={}, type=json.loads
    )
    graph.add_argument(
        '-m', '--min_accuracy', default=0, type=float
    )
    graph.set_defaults(analyze_func=ResultsManager.build_accuracy_plot)

    return parser.parse_args()


def runner_main(args, dataset, results_manager):
    X, y = dataset.get_X_y()
    estimator_cls = Estimators.get_value(args.estimator)
    param_names_to_variate = [
        name
        for name in estimator_cls.get_param_names()
        if name in args.params
    ]
    for param_values in product(*[args.params[name] for name in param_names_to_variate]):
        param_dict = {
            name: value
            for name, value in zip(param_names_to_variate, param_values)
        }
        print('begin param set: ' + str(param_dict))
        estimator = estimator_cls(**param_dict)
        result = cross_validate(
            estimator, X, y,
            scoring=['accuracy', 'f1', 'precision', 'recall'],
            cv=KFold(n_splits=5, shuffle=True, random_state=117),
        )
        metrics = {
            metric_name[5:]: np.mean(metric_value)
            for metric_name, metric_value in result.items()
            if metric_name.startswith('test_')
        }
        results_manager.add_result(estimator, metrics)
        results_manager.dump()
    print('done')
    return False


def fit_all_main(args, dataset, results_manager):
    X, y = dataset.get_X_y()
    estimator = Estimators.get_value(args.estimator)(**args.params)
    estimator.fit(X, y)


def analyze_main(args, dataset, results_manager):
    args.analyze_func(results_manager, args)
    return False


def main():
    args = parse_args()
    dataset = DatasetPreparations.get_value(args.dataset)
    results_manager = ResultsManager(dataset)
    args.func(args, dataset, results_manager)


if __name__ == '__main__':
    main()
