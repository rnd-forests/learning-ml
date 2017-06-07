from algorithm import Recommender

if __name__ == "__main__":
    item_data_path = './data/posts.csv'
    rating_data_path = './data/votes.csv'

    param_grid = {'n_epochs': [20, 50],
                  'lr_all': [0.001, 0.003, 0.005],
                  'reg_all': [0.2, 0.6]
                  }

    bsl_options = {'method': 'sgd'}

    sim_options = {'name': 'pearson_baseline',
                   'user_based': False}

    recommender = Recommender(rating_scale=(-1, 1), rating_data_path=rating_data_path,
                              item_data_path=item_data_path, param_grid=param_grid,
                              bsl_options=bsl_options, sim_options=sim_options, perf_measure='fcp')

    uuid = 1087
    recommender.recommend(uid=uuid, n_items=10, verbose=True)
