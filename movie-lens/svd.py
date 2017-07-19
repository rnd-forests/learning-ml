from __future__ import absolute_import, division, print_function, \
    unicode_literals

import os
import math
import pprint
import datetime
import numpy as np
from timeit import default_timer
from collections import defaultdict
from surprise import SVD, SVDpp, KNNBasic, KNNBaseline, \
    GridSearch, Dataset, accuracy, dump


class Recommender:
    def __init__(self, algorithm, param_grid, bsl_options, sim_options,
                 rating_scale=(1, 5), perf_measure='rmse', n_folds=3, dump_model=True):
        self.algorithm = algorithm
        self.param_grid = param_grid
        self.bsl_options = bsl_options
        self.sim_options = sim_options
        self.rating_scale = rating_scale
        self.perf_measure = perf_measure
        self.n_folds = n_folds
        self.dump_model = dump_model
        self.data = self.load_data()

    def recommend(self, uids, n_items=10, verbose=False):
        data = self.data
        trained_model = os.path.expanduser('./svd')

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
                raise ValueError('Invalid accuracy measurement provided...')

            if verbose:
                print('Performing Grid Search...')

            data.split(n_folds=self.n_folds)
            grid_search = GridSearch(self.algorithm, param_grid=self.param_grid,
                                     measures=[self.perf_measure], verbose=True)
            grid_search.evaluate(data)
            algo = grid_search.best_estimator[self.perf_measure]
            algo.sim_options = self.sim_options
            algo.bsl_options = self.bsl_options
            algo.verbose = verbose

            if verbose:
                print('Grid Search completed...')
                pp = pprint.PrettyPrinter()
                pp.pprint(vars(algo))

            # Retrain on the whole train set
            print('Training using trainset...')
            trainset = data.build_full_trainset()
            algo.train(trainset)
            algo.verbose = verbose
            if self.dump_model:
                dump.dump(trained_model, algo=algo)

            if verbose:
                # Test on the testset
                print('Evaluating using testset...')
                testset = data.construct_testset(test_raw_ratings)
                predictions = algo.test(testset)
                accuracy.rmse(predictions, verbose=True)

        # Generate top-N recommendations
        start = default_timer()
        data = self.data
        trainset = data.build_full_trainset()
        testset = trainset.build_anti_testset()
        predictions = algo.test(testset)
        accuracy.rmse(predictions, verbose=True)
        predictions = self.limit_predictions(predictions, n_items)
        uids = list(uids)
        results = dict()
        for uid in uids:
            results[str(uid)] = self.get_top_predictions(uid, predictions)

        if verbose:
            duration = default_timer() - start
            duration = datetime.timedelta(seconds=math.ceil(duration))
            print('+' * 40)
            print('Time elapsed:', duration)
            print('+' * 40)

        return results

    @staticmethod
    def get_top_predictions(uid, predictions):
        if not uid:
            raise ValueError('Invalid user ID provided...')
        try:
            pred_uid = predictions[str(uid)]
            return pred_uid
        except KeyError:
            print('Cannot find the given user...')

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


if __name__ == "__main__":
    # Matrix factorization - SVD using Stochastic Gradient Descent
    bsl_options = {'method': 'sgd'}
    param_grid = {'n_factors': [20, 50], 'lr_all': [0.0003, 0.0007]}
    recommender = Recommender(algorithm=SVD,
                              param_grid=param_grid,
                              bsl_options=bsl_options,
                              sim_options={},
                              perf_measure='rmse',
                              dump_model=False)


    # Matrix factorization - SVD++ using Alternating Least Squares
    # bsl_options = {'method': 'als'}
    # param_grid = {'n_epochs': [10, 20], 'reg_all': [0.02, 0.04]}
    # recommender = Recommender(algorithm=SVDpp,
    #                           param_grid=param_grid,
    #                           bsl_options=bsl_options,
    #                           sim_options={},
    #                           perf_measure='rmse')


    # Neighborhood-based collaborative filtering (kNN-basic)
    # param_grid = {'k': [20, 40, 60]}
    # sim_options = {'name': 'pearson_baseline', 'user_based': True}
    # recommender = Recommender(algorithm=KNNBasic,
    #                           param_grid=param_grid,
    #                           bsl_options={},
    #                           sim_options=sim_options,
    #                           perf_measure='rmse')


    # Neighborhood-based collaborative filtering (kNN-baseline)
    # param_grid = {'k': [20, 40, 60]}
    # bsl_options = {'method': 'sgd', 'learning_rate': 0.0007}
    # sim_options = {'name': 'pearson_baseline', 'user_based': True}
    # recommender = Recommender(algorithm=KNNBaseline,
    #                           param_grid=param_grid,
    #                           bsl_options=bsl_options,
    #                           sim_options=sim_options,
    #                           perf_measure='rmse')


    uids = [1, 2]
    recommendations = recommender.recommend(uids=uids, verbose=True)

    pp = pprint.PrettyPrinter()
    pp.pprint(recommendations)
