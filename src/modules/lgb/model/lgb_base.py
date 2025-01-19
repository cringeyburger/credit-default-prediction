from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import os
import numpy as np


def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    auc = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, (y_pred > 0.5).astype(int))
    recall = recall_score(y_true, (y_pred > 0.5).astype(int))
    f1 = f1_score(y_true, (y_pred > 0.5).astype(int))
    tn, fp, fn, tp = confusion_matrix(y_true, (y_pred > 0.5).astype(int)).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Specificity": specificity,
    }


def log_result(logFile, text, isPrint=True):
    if isPrint:
        print(text)
    logFile.write(text)
    logFile.write("\n")


def create_folders(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)
