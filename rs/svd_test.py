from algorithm import Recommender

if __name__ == "__main__":
    items = './data/posts.csv'
    ratings = './data/votes.csv'

    bsl_options = {'method': 'sgd'}
    param_grid = {'lr_all': [0.003, 0.007]}
    sim_options = {'name': 'pearson_baseline', 'user_based': False}

    recommender = Recommender(rating_file_path=ratings,
                              item_file_path=items,
                              rating_scale=(-1, 1), param_grid=param_grid,
                              bsl_options=bsl_options, sim_options=sim_options, perf_measure='fcp')

    uuid = 1087
    recommender.recommend(uid=uuid, n_items=25, verbose=True)
