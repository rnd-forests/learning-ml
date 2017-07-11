from algorithm import Recommender

import pprint

if __name__ == "__main__":
    items = './data/posts.csv'
    ratings = './data/votes.csv'

    scale = (-1, 1)
    bsl_options = {'method': 'sgd'}
    param_grid = {'n_epochs': [20,30], 'n_factors': [20, 50], 'lr_all': [0.0003, 0.0007]}
    sim_options = {'name': 'pearson_baseline', 'user_based': False}

    recommender = Recommender(rating_file_path=ratings,
                              item_file_path=items,
                              rating_scale=scale,
                              param_grid=param_grid,
                              bsl_options=bsl_options,
                              sim_options=sim_options,
                              perf_measure='rmse')

    uids = [1087]
    results = recommender.recommend(uids=uids, n_items=25, verbose=False)

    pp = pprint.PrettyPrinter()
    pp.pprint(results)
