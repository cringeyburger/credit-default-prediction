import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import sys
import os
import pandas as pd
import warnings
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from lgb_base import calculate_metrics, log_result

warnings.simplefilter("ignore")

auc_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
specificity_scores = []


def train_lgb_model(train, config, output_path, run_id):
    output_path = output_path + run_id + "/"
    os.makedirs(output_path, exist_ok=True)

    log = open(output_path + "/train.log", "w", buffering=1)
    log.write(str(config) + "\n")

    features = config["feature_name"]
    params = config["lgb_params"]
    verbose = config["verbose_eval"]
    early_stopping_rounds = config["early_stopping_rounds"]
    folds = config["folds"]
    seed = config["seed"]

    id_name = config["id_name"]
    label_name = config["label_name"]

    oof = train[[id_name]].copy()
    oof[label_name] = 0

    all_valid_metric, feature_importance = [], []

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    for fold, (train_index, val_index) in enumerate(skf.split(train, train[label_name])):
        evals_result_dic = {}

        train_data = lgb.Dataset(
            train.loc[train_index, features],
            label=train.loc[train_index, label_name],
        )
        val_data = lgb.Dataset(
            train.loc[val_index, features],
            label=train.loc[val_index, label_name],
        )

        model = lgb.train(
            params,
            train_set=train_data,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.record_evaluation(evals_result_dic),
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(verbose),
            ],
        )

        model.save_model(output_path + f"/fold{fold}.ckpt")

        valid_preds = model.predict(
            train.loc[val_index, features], num_iteration=model.best_iteration
        )
        oof.loc[val_index, label_name] = valid_preds

        metric_values = calculate_metrics(train.loc[val_index, label_name], valid_preds)

        all_valid_metric.append(metric_values)
        auc_scores.append(metric_values["AUC"])
        precision_scores.append(metric_values["Precision"])
        recall_scores.append(metric_values["Recall"])
        f1_scores.append(metric_values["F1-Score"])
        specificity_scores.append(metric_values["Specificity"])

        log_result(log, f"- fold{fold} valid metric: {all_valid_metric[-1]}\n")

        importance_gain = model.feature_importance(importance_type="gain")
        importance_split = model.feature_importance(importance_type="split")
        feature_name = model.feature_name()

        feature_importance.append(
            pd.DataFrame(
                {
                    "feature_name": feature_name,
                    "importance_gain": importance_gain,
                    "importance_split": importance_split,
                }
            )
        )

    feature_importance_df = pd.concat(feature_importance)

    feature_importance_df.groupby(["feature_name"]).mean().reset_index().sort_values(
        by=["importance_gain"], ascending=False
    ).to_csv(output_path + "/feature_importance.csv", index=False)

    mean_valid_metric = {
        "AUC": np.mean(auc_scores),
        "Precision": np.mean(precision_scores),
        "Recall": np.mean(recall_scores),
        "F1-Score": np.mean(f1_scores),
        "Specificity": np.mean(specificity_scores),
    }

    global_valid_metric = calculate_metrics(
        train[label_name].values, oof[label_name].values
    )

    log_result(
        log,
        f"all valid mean metric: {mean_valid_metric}, global valid metric: {global_valid_metric}",
    )

    oof.to_csv(output_path + "/oof.csv", index=False)

    log.close()

    return oof, mean_valid_metric, global_valid_metric, output_path
