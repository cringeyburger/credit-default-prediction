import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.decomposition import TruncatedSVD, PCA

def knn_imputation(data_path):
    data = pd.read_parquet(data_path)
    target_column = "bad_flag"
    features = data.drop(columns=[target_column])
    target = data[target_column]

    knn_imputer = KNNImputer(n_neighbors=5)
    print("***Starting KNN Imputer***")
    features_knn_imputed = knn_imputer.fit_transform(features)
    df_knn_imputed = pd.DataFrame(features_knn_imputed, columns=features.columns)
    df_knn_imputed[target_column] = target

    df_knn_imputed.to_parquet(data_path, index=False)
    print("***Finished KNN Imputer***")

def mice_imputation(data_path):
    data = pd.read_parquet(data_path)
    target_column = "bad_flag"
    features = data.drop(columns=[target_column])
    target = data[target_column]

    mice_imputer = IterativeImputer(random_state=42)
    print("***Starting MICE Imputer***")
    features_mice_imputed = mice_imputer.fit_transform(features)
    df_mice_imputed = pd.DataFrame(features_mice_imputed, columns=features.columns)
    df_mice_imputed[target_column] = target

    df_mice_imputed.to_parquet(data_path, index=False)
    print("***Finished MICE Imputer***")

def svd_imputation(data_path):
    data = pd.read_parquet(data_path)
    target_column = "bad_flag"
    features = data.drop(columns=[target_column, "account_number"])
    target = data[target_column]
    account_num = data["account_number"]

    # Simple mean imputation before applying TruncatedSVD
    mean_imputer = SimpleImputer(strategy='mean')
    features_filled = mean_imputer.fit_transform(features)

    # Apply SVD (dimensionality reduction as a rough imputation strategy)
    print("***Starting SVD Imputer***")
    n_components = min(features.shape[1] - 1, 10)  # Adjust as needed
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    features_svd_imputed = svd.fit_transform(features_filled)

    # Build a DataFrame from the reduced features
    df_svd_imputed = pd.DataFrame(
        features_svd_imputed,
        columns=[f"SVD_{i}" for i in range(features_svd_imputed.shape[1])]
    )
    df_svd_imputed[target_column] = target.values
    df_svd_imputed["account_number"] = account_num.values

    df_svd_imputed.to_parquet(data_path, index=False)
    print("***Finished SVD Imputer***")

def bpca_imputation(data_path):
    data = pd.read_parquet(data_path)
    target_column = "bad_flag"
    features = data.drop(columns=[target_column])
    target = data[target_column]

    # Simple mean imputation before applying PCA
    mean_imputer = SimpleImputer(strategy='mean')
    features_filled = mean_imputer.fit_transform(features)

    # Use PCA to approximate Bayesian PCA
    print("***Starting Bayesian PCA Imputer***")
    n_components = min(features.shape[1] - 1, 10)  # Adjust as needed
    pca = PCA(n_components=n_components, svd_solver='full', random_state=42)
    features_bpca_imputed = pca.fit_transform(features_filled)

    # Build a DataFrame from the reduced features
    df_bpca_imputed = pd.DataFrame(
        features_bpca_imputed,
        columns=[f"BPCA_{i}" for i in range(features_bpca_imputed.shape[1])]
    )
    df_bpca_imputed[target_column] = target.values

    df_bpca_imputed.to_parquet(data_path, index=False)
    print("***Finished Bayesian PCA Imputer***")
