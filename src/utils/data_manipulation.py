"""Module for data manipulation functions"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import PowerTransformer


def make_parquet_copy(csv_data_path, parquet_output_path):
    """Make Parquet copies of the CSV datafiles"""
    data = pd.read_csv(csv_data_path)
    data.to_parquet(parquet_output_path, index=False)
    print(f"Parquet file saved at: {parquet_output_path}")


def nan_to_zero(data_path):
    """
    Reads a parquet dataset, fills all NaN values with mean of column, and saves the updated dataset.
    """
    data = pd.read_parquet(data_path)
    # data = data.apply(lambda col: (col.fillna(col.median())))
    data = data.fillna(0)
    data.to_parquet(data_path, index=False)
    print(f"***Dataset with NaN values filled saved to {data_path}***")


def denoise(data_path):
    """Denoise the data"""
    data = pd.read_parquet(data_path)
    for col in tqdm(data.columns):
        if col not in ["account_number", "bad_flag"]:
            data[col] = np.floor(data[col] * 1000) / 1000
    data.to_parquet(data_path, index=False)


def clip_outliers(data_path, lower_percentile=0.01, upper_percentile=0.99):
    data = pd.read_parquet(data_path)
    numeric_cols = data.select_dtypes(
        include=["float64", "float32", "int64", "int32"]
    ).columns

    numeric_cols = [col for col in numeric_cols if col != "account_number"]

    for col in numeric_cols:
        lower_bound = data[col].quantile(lower_percentile)
        upper_bound = data[col].quantile(upper_percentile)

        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)

    data.to_parquet(data_path, index=False)
    print(f"***Clipped outliers in {len(numeric_cols)} numeric columns***")


def drop_features_with_zero_sum(data_path):
    """Drops features from dev dataset if their total sum is 0 or NaN"""
    data = pd.read_parquet(data_path)
    column_sums = data.sum(axis=0, skipna=True)
    drop_columns = column_sums[(column_sums == 0) | (column_sums.isna())].index.tolist()
    data = data.drop(columns=drop_columns)
    unique_counts = data.nunique(axis=1)
    data = data[unique_counts > 1]
    data.to_parquet(data_path, index=False)
    print(drop_columns)
    print("***Dropped zero-sum and single-value features***")


def create_aggregate_features(data, feature_groups):
    for group_name, group_cols in feature_groups.items():
        data[f"{group_name}_mean"] = data[group_cols].mean(axis=1)
        data[f"{group_name}_sum"] = data[group_cols].sum(axis=1)
        data[f"{group_name}_count"] = (data[group_cols] > 0).sum(axis=1)
        data[f"{group_name}_max"] = data[group_cols].max(axis=1)
        data[f"{group_name}_min"] = data[group_cols].min(axis=1)
    return data


def create_ratios(data, feature_groups):
    data["onus_to_transaction_ratio"] = (
        data[feature_groups["onus"]].sum(axis=1)
        / data[feature_groups["transaction"]].sum(axis=1).replace(0, np.nan)
    ).fillna(0)

    data["bureau_to_bureau_enquiry_ratio"] = (
        data[feature_groups["bureau"]].sum(axis=1)
        / data[feature_groups["bureau_enquiry"]].sum(axis=1).replace(0, np.nan)
    ).fillna(0)

    return data


def create_features(path):
    """
    Reads a dataset from a parquet file, creates summary, interaction, statistical,
    and advanced features, and saves the updated dataset.
    """

    data = pd.read_parquet(path)
    feature_groups = {
        "onus": [col for col in data.columns if col.startswith("onus_attribute")],
        "transaction": [
            col for col in data.columns if col.startswith("transaction_attribute")
        ],
        "bureau_enquiry": [
            col for col in data.columns if col.startswith("bureau_enquiry")
        ],
        "bureau": [col for col in data.columns if col.startswith("bureau_attribute")],
    }

    data["onus_transaction_interaction"] = data[feature_groups["onus"]].sum(
        axis=1
    ) * data[feature_groups["transaction"]].sum(axis=1)

    data["transaction_skewness"] = data[feature_groups["transaction"]].skew(axis=1)
    data["transaction_kurtosis"] = data[feature_groups["transaction"]].kurtosis(axis=1)
    data["onus_variance"] = data[feature_groups["onus"]].var(axis=1)

    data["total_transactions"] = data[feature_groups["transaction"]].sum(axis=1)
    data["avg_transaction_value"] = data[feature_groups["transaction"]].mean(axis=1)
    data["active_transactions_count"] = (data[feature_groups["transaction"]] > 0).sum(
        axis=1
    )
    data["active_bureau_enquiries_count"] = (
        data[feature_groups["bureau_enquiry"]] > 0
    ).sum(axis=1)

    data["inquiry_to_transaction_ratio"] = (
        data[feature_groups["bureau_enquiry"]].sum(axis=1)
        / data[feature_groups["transaction"]].sum(axis=1).replace(0, np.nan)
    ).fillna(0)

    data["onus_to_bureau_ratio"] = (
        data[feature_groups["onus"]].sum(axis=1)
        / data[feature_groups["bureau_enquiry"]].sum(axis=1).replace(0, np.nan)
    ).fillna(0)

    data["normalized_transaction_sum"] = (
        data[feature_groups["transaction"]].sum(axis=1)
        / data[feature_groups["transaction"]].max(axis=1).replace(0, np.nan)
    ).fillna(0)

    data["bureau_to_onus_ratio"] = (
        data[feature_groups["bureau"]].sum(axis=1)
        / data[feature_groups["onus"]].sum(axis=1).replace(0, np.nan)
    ).fillna(0)

    data["non_zero_onus_count"] = (data[feature_groups["onus"]] > 0).sum(axis=1)

    threshold = 2 * data[feature_groups["transaction"]].mean().mean()
    data["high_transaction_count"] = (
        data[feature_groups["transaction"]] > threshold
    ).sum(axis=1)

    data.to_parquet(path, index=False)
    print("***Features created and dataset saved successfully***")


def power_transform_skewed_features(data_path, skew_threshold=0.75):
    """
    Normalize non-negative transactions and onus features using power transformation
    """

    data = pd.read_parquet(data_path)

    numeric_cols = data.select_dtypes(
        include=["float64", "float32", "int64", "int32"]
    ).columns
    numeric_cols = [col for col in numeric_cols if col != "account_number"]

    filtered_cols = [
        col for col in numeric_cols if col.startswith(("transactions_", "onus_"))
    ]

    skewed_features = data[filtered_cols].apply(lambda x: x.skew()).abs()
    skewed_cols = skewed_features[skewed_features > skew_threshold].index

    print(f"***Applying power transformation to {len(skewed_cols)} skewed features***")

    pt = PowerTransformer(method="yeo-johnson", standardize=True)

    for col in skewed_cols:
        data[col] = pt.fit_transform(data[[col]])
    data.to_parquet(data_path, index=False)

    print(f"***Power transformation applied and saved to {data_path}***")


def normalize_dataset(train_path, test_path):
    """
    Normalized datasets with missing values
    """
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    scaler = MinMaxScaler()
    scaler.fit(train)

    train_normalized = scaler.transform(train)
    test_normalized = scaler.transform(test)

    pd.DataFrame(train_normalized, columns=train.columns).to_parquet(
        train_path, index=False
    )
    pd.DataFrame(test_normalized, columns=test.columns).to_parquet(
        test_path, index=False
    )

    print("***Normalized datasets***")


def resample_training_data(data_path, target_column):
    """
    Load a training dataset from a Parquet file, perform SMOTE-Tomek resampling,
    and save the resampled dataset as a Parquet file.
    """
    data = pd.read_parquet(data_path)
    print(f"Loaded training data with shape: {data.shape}")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    if y.nunique() != 2:
        raise ValueError(
            "Target column must contain exactly two unique values for binary classification."
        )

    if y.dtype not in ["int64", "int32", "category"]:
        print("Converting continuous target to binary classes")
        y = y.astype("int")

    smote_tomek = SMOTETomek(random_state=42)
    print("Performing SMOTE-Tomek resampling...")
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

    resampled_data = pd.concat(
        [
            pd.DataFrame(X_resampled, columns=X.columns),
            pd.DataFrame(y_resampled, columns=[target_column]),
        ],
        axis=1,
    )
    resampled_data["account_number"] = range(1, len(resampled_data) + 1)

    print(f"Resampled data shape: {resampled_data.shape}")

    resampled_data.to_parquet(data_path, index=False)
    print(f"***Resampled data saved to: {data_path}***")

def remove_uncommon(data_path, test_path):
    data = pd.read_parquet(data_path)
    test = pd.read_parquet(test_path)

    data_cols = set(data.columns)
    test_cols = set(test.columns)

    data_only_cols = data_cols - test_cols
    test_only_cols = test_cols - data_cols

    data.drop(columns=data_only_cols, inplace=True, errors="ignore")
    test.drop(columns=test_only_cols, inplace=True, errors="ignore")

    # Save the modified DataFrames back to their respective paths
    data.to_parquet(data_path)
    test.to_parquet(test_path)
