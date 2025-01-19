import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import optuna


# Hyperparameter choices from: https://github.com/microsoft/LightGBM/issues/695#issuecomment-315591634
def objective(trial, train_df, feature_name, label_name, config):
    num_leaves = trial.suggest_int("num_leaves", 7, 4095)
    min_child_weight = trial.suggest_float("min_child_weight", 0.01, 15.26)
    subsample = trial.suggest_float("subsample", 0.4, 1.0)
    bagging_fraction = trial.suggest_float("bagging_fraction", 0.5, 1.0)
    feature_fraction = trial.suggest_float("feature_fraction", 0.4, 0.8)
    min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 20, 200)

    fixed_params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_threads": config["lgb_params"]["num_threads"],
        "bagging_freq": config["lgb_params"]["bagging_freq"],
        "boost_from_average": False,
        "max_bin": config["lgb_params"]["max_bin"],
        "min_data_in_bin": config["lgb_params"]["min_data_in_bin"],
    }

    lgb_params = {
        **fixed_params,
        "max_depth": 63,
        "num_leaves": num_leaves,
        "min_child_weight": min_child_weight,
        "subsample": subsample,
        "bagging_fraction": bagging_fraction,
        "feature_fraction": feature_fraction,
        "min_data_in_leaf": min_data_in_leaf,
        "scale_pos_weight": 69,
        "n_estimators": 9999999,
    }

    skf = StratifiedKFold(
        n_splits=config["folds"], shuffle=True, random_state=config["seed"]
    )

    auc_scores = []

    for train_idx, val_idx in skf.split(train_df[feature_name], train_df[label_name]):
        train_data = lgb.Dataset(
            train_df.iloc[train_idx][feature_name],
            label=train_df.iloc[train_idx][label_name],
        )
        val_data = lgb.Dataset(
            train_df.iloc[val_idx][feature_name],
            label=train_df.iloc[val_idx][label_name],
        )

        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=config["early_stopping_rounds"])
            ],
        )

        preds = model.predict(train_df.iloc[val_idx][feature_name])
        auc_scores.append(roc_auc_score(train_df.iloc[val_idx][label_name], preds))

    return -np.mean(auc_scores)


def tune_hyperparameters(train_df, feature_name, label_name, config):
    study = optuna.create_study(direction="minimize")

    # Optimize the objective function
    study.optimize(
        lambda trial: objective(trial, train_df, feature_name, label_name, config),
        n_trials=50,
    )

    return study.best_params
