"""Module for processing the data for training"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "../../src")
sys.path.append(src_path)

from utils import data_manipulation, dataset_split

required_dirs = [
    "data/raw",
    "data/preprocessed",
    "data/processed",
]

for directory in required_dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)

created_dirs = [directory for directory in required_dirs if os.path.exists(directory)]
print(f"***Created/Verified directories: {created_dirs}***")

# Define paths
current_dir = os.getcwd()
RAW_DATA_PATH = os.path.join(current_dir, "data/raw/dev_data.csv")
RAW_VAL_PATH = os.path.join(current_dir, "data/raw/val_data.csv")
DATA_PATH = os.path.join(current_dir, "data/preprocessed/dev_data.parquet")
VAL_PATH = os.path.join(current_dir, "data/preprocessed/val_data.parquet")
PREPROCESSED_PATH = os.path.join(current_dir, "data/preprocessed/")
TRAIN_PATH = os.path.join(current_dir, "data/processed/train.parquet")
DEVSET_PATH = os.path.join(current_dir, "data/processed/devset.parquet")
TEST_PATH = os.path.join(current_dir, "data/processed/test.parquet")
PROCESSED_PATH = os.path.join(current_dir, "data/processed/")
TARGET_COLUMN = "bad_flag"

# # Create Parquet files
# data_manipulation.make_parquet_copy(RAW_DATA_PATH, DATA_PATH)
# data_manipulation.make_parquet_copy(RAW_VAL_PATH, VAL_PATH)

# # Replace NaN with zero
# data_manipulation.nan_to_zero(DATA_PATH)
# data_manipulation.nan_to_zero(VAL_PATH)

# # Denoise the datasets
# data_manipulation.denoise(DATA_PATH)
# data_manipulation.denoise(VAL_PATH)
# print("***Denoised both datasets***")

# # Remove zero-sum features
# data_manipulation.drop_features_with_zero_sum(DATA_PATH)
# data_manipulation.drop_features_with_zero_sum(VAL_PATH)

# # Split
# dataset_split.split_train_test(DATA_PATH, TARGET_COLUMN, PROCESSED_PATH)

# # Clip outliers
# data_manipulation.clip_outliers(TRAIN_PATH)
# data_manipulation.clip_outliers(TEST_PATH)
# data_manipulation.clip_outliers(VAL_PATH)

# # Create summary features
# data_manipulation.create_features(TRAIN_PATH)
# data_manipulation.create_features(TEST_PATH)
# data_manipulation.create_features(VAL_PATH)

# # Remove zero-sum features
# data_manipulation.drop_features_with_zero_sum(TRAIN_PATH)
# data_manipulation.drop_features_with_zero_sum(TEST_PATH)
# data_manipulation.drop_features_with_zero_sum(VAL_PATH)

# # Apply power transformation
# data_manipulation.power_transform_skewed_features(TRAIN_PATH)
# data_manipulation.power_transform_skewed_features(TEST_PATH)
# data_manipulation.power_transform_skewed_features(VAL_PATH)

# # Remove zero-sum features
# data_manipulation.drop_features_with_zero_sum(TRAIN_PATH)
# data_manipulation.drop_features_with_zero_sum(TEST_PATH)
# data_manipulation.drop_features_with_zero_sum(VAL_PATH)

# remove featuers whihc arent in other datasets
data_manipulation.remove_uncommon(TRAIN_PATH, TEST_PATH)