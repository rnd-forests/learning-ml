from __future__ import absolute_import, division, print_function

import os
import pandas as pd
from timeit import default_timer
from collections import defaultdict
from surprise import SVD, GridSearch, Dataset, Reader, accuracy, dump


class Recommender:
    def __init__(self, rating_scale, rating_data_path, item_data_path,
                 param_grid, bsl_options, sim_options, perf_measure='rmse'):
        self.rating_scale = rating_scale
        self.rating_data_path = rating_data_path
        self.item_data_path = item_data_path
        self.param_grid = param_grid
        self.bsl_options = bsl_options
        self.sim_options = sim_options
        self.perf_measure = perf_measure
        self.data = self.load_data()

    def find_estimator(self, verbose=False):
        if self.perf_measure not in ['rmse', 'fcp']:
            raise ValueError('Invalid accuracy measurement provided.')
        grid_search = self.perform_grid_search()

        algo = grid_search.best_estimator[self.perf_measure]
        algo.sim_options = self.sim_options
        algo.bsl_options = self.bsl_options

        if verbose:
            print(vars(algo))

        return algo

    def recommend(self, uid, n_items=5, verbose=False):
        data = self.data
        file_name = os.path.expanduser('./svd')
        train_set = data.build_full_trainset()

        start = default_timer()

        try:
            _, algo = dump.load(file_name)
        except FileNotFoundError:
            algo = self.find_estimator(verbose)
            algo.train(train_set)
            dump.dump(file_name, algo=algo)

        test_set = train_set.build_anti_testset()
        predictions = algo.test(test_set)

        if verbose:
            duration = default_timer() - start
            print('+' * 40)
            print('Time elapsed:', duration)
            print('+' * 40)

        accuracy.rmse(predictions, verbose=True)
        accuracy.mae(predictions, verbose=True)
        # accuracy.fcp(predictions, verbose=True)
        print('+' * 40)

        predictions = self.limit_predictions(predictions, n_items)

        if uid:
            try:
                pred_uid = predictions[str(uid)]
                if verbose:
                    items = pd.read_csv(self.item_data_path, index_col=None, usecols=('id', 'title'))
                    for _, value in enumerate(pred_uid):
                        item = items.loc[items['id'] == int(value[0])]
                        title = item['title'].values[0] if not item['title'].empty else '**No item found!**'
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

    def perform_grid_search(self):
        data = self.data
        data.split(n_folds=5)
        grid_search = GridSearch(SVD, param_grid=self.param_grid, measures=['RMSE', 'FCP'], verbose=False)
        grid_search.evaluate(data)

        return grid_search

    @staticmethod
    def limit_predictions(predictions, n):
        results = defaultdict(list)
        for uid, iid, _, est, _ in predictions:
            results[uid].append((iid, est))

        for uid, ratings in results.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            results[uid] = ratings[:n]

        return results
