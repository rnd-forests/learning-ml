from algorithm import Recommender

import pprint

if __name__ == "__main__":
    # bsl_options = {'method': 'sgd'}
    # param_grid = {'n_epochs': [20, 30], 'n_factors': [20, 50], 'lr_all': [0.0003, 0.0007]}

    bsl_options = {'method': 'als'}
    param_grid = {'n_epochs': [10, 20], 'reg_all': [0.02, 0.04]}
    sim_options = {'name': 'pearson_baseline', 'user_based': False}

    recommender = Recommender(param_grid=param_grid,
                              bsl_options=bsl_options,
                              sim_options=sim_options,
                              perf_measure='rmse')

    uids = [1, 2, 3, 4, 5]
    results = recommender.recommend(uids=uids, n_items=10, verbose=True)

    pp = pprint.PrettyPrinter()
    pp.pprint(results)
