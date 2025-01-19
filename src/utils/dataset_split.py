"""Module for splitting dataset and cleaning"""

from sklearn.model_selection import train_test_split

import pandas as pd


def split_train_test(
    data_path,
    target_column,
    save_path,
    train_ratio=0.9,
    random_state=42,
):
    """
    Splits the dataset into training and test sets, and saves them as parquet files.
    """
    assert 0 < train_ratio < 1, "Train ratio must be between 0 and 1."

    data = pd.read_parquet(data_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=random_state, stratify=y
    )

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_data.to_parquet(f"{save_path}/train.parquet", index=False)
    test_data.to_parquet(f"{save_path}/test.parquet", index=False)

    print("***Data successfully split and saved as parquet files***")
