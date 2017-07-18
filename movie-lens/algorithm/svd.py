from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pprint
import numpy as np
from timeit import default_timer
from collections import defaultdict
from surprise import SVDpp, GridSearch, Dataset, accuracy, dump


class Recommender:
    def __init__(self, param_grid, bsl_options, sim_options, perf_measure='rmse'):
        self.param_grid = param_grid
        self.bsl_options = bsl_options
        self.sim_options = sim_options
        self.perf_measure = perf_measure
        self.data = self.load_data()

    def recommend(self, uids, n_items=5, verbose=False):
        data = self.data
        trained_model = os.path.expanduser('./svd')

        start = default_timer()
        try:
            _, algo = dump.load(trained_model)
        except FileNotFoundError:
            # Perform random sampling on the raw ratings
            raw_ratings = data.raw_ratings
            np.random.shuffle(raw_ratings)
            threshold = int(.8 * len(raw_ratings))
            trainset_raw_ratings = raw_ratings[:threshold]
            test_raw_ratings = raw_ratings[threshold:]

            # Assign new ratings to the original data
            data.raw_ratings = trainset_raw_ratings

            # Perform Grid Search
            if self.perf_measure not in ['rmse', 'fcp']:
                raise ValueError('Invalid accuracy measurement provided.')

            if verbose:
                print('Performing Grid Search...')

            data.split(n_folds=3)
            grid_search = GridSearch(SVDpp, param_grid=self.param_grid, measures=['RMSE', 'FCP'], verbose=True)
            grid_search.evaluate(data)
            algo = grid_search.best_estimator[self.perf_measure]
            algo.sim_options = self.sim_options
            algo.bsl_options = self.bsl_options
            algo.verbose = verbose

            if verbose:
                print('Grid Search completed..')
                pp = pprint.PrettyPrinter()
                pp.pprint(vars(algo))

            # Retrain on the whole train set
            print('Training using trainset...')
            trainset = data.build_full_trainset()
            algo.train(trainset)
            algo.verbose = verbose
            dump.dump(trained_model, algo=algo)

        # Generate top-N recommendations
        data = self.data
        trainset = data.build_full_trainset()
        testset = trainset.build_anti_testset()
        predictions = algo.test(testset)
        accuracy.rmse(predictions, verbose=True)
        predictions = self.limit_predictions(predictions, n_items)
        uids = list(uids)
        results = dict()
        for uid in uids:
            results[str(uid)] = self.get_top_items(uid, predictions)

        if verbose:
            duration = default_timer() - start
            print('+' * 40)
            print('Time elapsed:', duration)
            print('+' * 40)

        return results

    @staticmethod
    def get_top_items(uid, predictions):
        if not uid:
            raise ValueError('Invalid user ID provided!')
        try:
            pred_uid = predictions[str(uid)]
            return pred_uid
        except KeyError:
            print('Cannot find the given user!')

    @staticmethod
    def load_data():
        data = Dataset.load_builtin('ml-100k')
        return data

    @staticmethod
    def limit_predictions(predictions, n):
        results = defaultdict(list)
        for uid, iid, _, est, _ in predictions:
            results[uid].append((iid, est))

        for uid, ratings in results.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            results[uid] = ratings[:n]

        return results
