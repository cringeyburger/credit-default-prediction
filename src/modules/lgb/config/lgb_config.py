import pandas as pd

# Hyperparameter choices from: https://github.com/microsoft/LightGBM/issues/695#issuecomment-315591634


def load_config(data_path):
    id_name = "account_number"
    label_name = "bad_flag"
    seed = 42
    train = pd.read_parquet(data_path)
    scale_pos_weight = int(
        (len(train[label_name]) - sum(train[label_name])) / sum(train[label_name])
    )

    return {
        "lgb_params": {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "boosting": "gbdt",
            "max_depth": 6,
            "num_leaves": 32,
            "learning_rate": 0.01,
            "bagging_freq": 5,
            "bagging_fraction": 0.8,
            "feature_fraction": 0.6,
            "min_data_in_leaf": 150,
            "max_bin": 255,
            "min_data_in_bin": 50,
            "tree_learner": "serial",
            "boost_from_average": "false",
            "lambda_l1": 10,
            "lambda_l2": 20,
            "min_gain_to_split": 0.5,
            "num_threads": -1,
            "verbosity": -1,
            "scale_pos_weight": scale_pos_weight,
        },
        "feature_name": [
            col for col in train.columns if col not in [id_name, label_name]
        ],
        "rounds": 6000,
        "early_stopping_rounds": 300,
        "verbose_eval": 50,
        "folds": 5,
        "seed": seed,
        "id_name": id_name,
        "label_name": label_name,
    }


# Trial 49 finished with value: -0.8129502418747538 and parameters: {'num_leaves': 1198, 'min_child_weight': 14.72663719822306, 'subsample': 0.7001268841940887, 'bagging_fraction': 0.5476762093799651, 'feature_fraction': 0.74346803974528, 'min_data_in_leaf': 139}. Best is trial 41 with value: -0.8278527840661074.
# Best Parameters from Hyperparameter Tuning: {'num_leaves': 9, 'min_child_weight': 0.04534070609429863, 'subsample': 0.525848315011058, 'bagging_fraction': 0.6424344055510417, 'feature_fraction': 0.6570021865305363, 'min_data_in_leaf': 147}

# "lgb_params": {
#             "objective": "binary",
#             "metric": ["binary_logloss", "auc"],
#             "boosting": "gbdt",
#             "max_depth": 6,
#             "num_leaves": 32,
#             "learning_rate": 0.01,
#             "bagging_freq": 5,
#             "bagging_fraction": 0.8,
#             "feature_fraction": 0.6,
#             "min_data_in_leaf": 150,
#             "max_bin": 255,
#             "min_data_in_bin": 50,
#             "tree_learner": "serial",
#             "boost_from_average": "false",
#             "lambda_l1": 10,
#             "lambda_l2": 20,
#             "min_gain_to_split": 0.5,
#             "num_threads": -1,
#             "verbosity": -1,
#             "scale_pos_weight": scale_pos_weight,
#         },
#         "feature_name": [
#             col for col in train.columns if col not in [id_name, label_name]
#         ],
#         "rounds": 6000,
#         "early_stopping_rounds": 300,
#         "verbose_eval": 50,
#         "folds": 5,
#         "seed": seed,
#         "id_name": id_name,
#         "label_name": label_name,
