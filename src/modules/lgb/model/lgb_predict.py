import os
import sys
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from lgb_base import log_result


def predict_lgb_model_metrics(test, config, output_path):
    features = config["feature_name"]
    id_name = config["id_name"]
    folds = config["folds"]

    sub = test[[id_name]].copy()
    sub["prediction"] = 0

    log = open(output_path + "/train.log", "a", buffering=1)

    for fold in range(folds):
        model_file_path = output_path + f"/fold{fold}.ckpt"
        model = lgb.Booster(model_file=model_file_path)
        test_preds = model.predict(test[features], num_iteration=model.best_iteration)
        sub["prediction"] += test_preds / folds

    sub[[id_name, "prediction"]].to_csv(output_path + "/submission.csv", index=False)

    y_true = test["bad_flag"]

    precision, recall, thresholds = precision_recall_curve(y_true, sub["prediction"])
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    if len(thresholds) > 0:
        best_threshold = thresholds[np.argmax(f1_scores)]
        log_result(log, "Using best threshold evaluation")
    else:
        best_threshold = 0.5
        log_result(log, "Using 0.5 evaluation")

    y_pred_binary = (sub["prediction"] > best_threshold).astype(int)

    auc = roc_auc_score(y_true, sub["prediction"])
    if np.sum(y_pred_binary) == 0:
        precision, recall, f1 = 0.0, 0.0, 0.0
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        specificity = tn / (tn + fp)
    else:
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        specificity = tn / (tn + fp)

    metrics = {
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Specificity": specificity,
    }

    log_result(log, f"Evaluation Metrics: {metrics}")
    log.close()
