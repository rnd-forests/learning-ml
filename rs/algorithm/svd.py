from __future__ import absolute_import, division, print_function

import os
import numpy as np
import pandas as pd
from timeit import default_timer
from collections import defaultdict
from surprise import SVDpp, GridSearch, Dataset, Reader, accuracy, dump


class Recommender:
    def __init__(self, rating_file_path, item_file_path, rating_scale,
                 param_grid, bsl_options, sim_options, perf_measure='rmse'):
        self.rating_data_path = rating_file_path
        self.item_data_path = item_file_path
        self.rating_scale = rating_scale
        self.param_grid = param_grid
        self.bsl_options = bsl_options
        self.sim_options = sim_options
        self.perf_measure = perf_measure
        self.data = self.load_data()

    def recommend(self, uid, n_items=5, verbose=False):
        data = self.data
        trained_model = os.path.expanduser('./svd')

        # The raw ratings
        raw_ratings = data.raw_ratings

        # Shuffle the ratings
        np.random.shuffle(raw_ratings)

        # Split the ratings into train_raw_ratings (80%) and test_raw_ratings (20%)
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
            print(vars(algo))

        start = default_timer()

        # try:
        #     _, algo = dump.load(file_name)
        # except FileNotFoundError:
        #     algo = self.find_best_estimator(verbose)
        #     algo.verbose = verbose
        #     algo.train(train_set)
        #     dump.dump(file_name, algo=algo)

        # Retrain on the whole train set
        print('Training on the trainset (80% of ratings)...')
        trainset = data.build_full_trainset()
        algo.train(trainset)

        # Test on the testset
        print('Testing on the testset (20% of ratings)...')
        testset = data.construct_testset(test_raw_ratings)
        predictions = algo.test(testset)
        accuracy.rmse(predictions, verbose=True)

        # Test on an-ti testset
        print('Making predictions...')
        data = self.data
        trainset = data.build_full_trainset()
        testset = trainset.build_anti_testset()
        predictions = algo.test(testset)
        accuracy.rmse(predictions, verbose=True)

        if verbose:
            duration = default_timer() - start
            print('+' * 40)
            print('Time elapsed:', duration)
            print('+' * 40)

        print('+' * 40)

        predictions = self.limit_predictions(predictions, n_items)

        if uid:
            try:
                pred_uid = predictions[str(uid)]
                if verbose:
                    items = pd.read_csv(self.item_data_path, index_col=None, usecols=('id', 'title'))
                    for _, value in enumerate(pred_uid):
                        item = items.loc[items['id'] == int(value[0])]
                        title = item['title'].values[0] if not item['title'].empty else '<No item found>'
                        print('iid -> {:<5} | est -> {:<18} | {}'.format(value[0], value[1], title))
                return pred_uid
            except KeyError:
                print('Cannot find the given user!')

        if verbose:
            print(predictions)

        return predictions

    def load_data(self):
        rating_scale = self.rating_scale
        file_path = os.path.expanduser(self.rating_data_path)
        if not os.path.exists(file_path):
            raise RuntimeError('Cannot find the given dataset!')
        reader = Reader(line_format='user item rating', sep=',', rating_scale=rating_scale, skip_lines=1)
        data = Dataset.load_from_file(file_path=file_path, reader=reader)

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
