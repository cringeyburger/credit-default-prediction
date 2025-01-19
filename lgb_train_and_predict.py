
import pandas as pd
import os
import warnings


from src.modules.lgb.model.lgb_train import train_lgb_model
from src.modules.lgb.model.lgb_predict import predict_lgb_model_metrics
from src.modules.lgb.model.lgb_base import create_folders
from src.modules.lgb.config.lgb_config import load_config
from src.modules.lgb.config.lgb_htune import tune_hyperparameters

warnings.simplefilter("ignore")
current_dir = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(current_dir, "data/processed/train.parquet")
TEST_PATH = os.path.join(current_dir, "data/processed/test.parquet")
YAML_PATH = os.path.join(current_dir, "src/modules/lgb/config/lgb_configs.yaml")
OUTPUT_PATH = "./output/"

create_folders([OUTPUT_PATH])

config = load_config(TRAIN_PATH)
train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)


# print("**Starting Hyperparameter Tuning**")
# best_params = tune_hyperparameters(
#     train_df=train,
#     feature_name=config["feature_name"],
#     label_name=config["label_name"],
#     config=config,
# )
# print(f"Best Parameters from Hyperparameter Tuning: {best_params}")


RUN_ID = "2_all_data"

oof_predictions, mean_valid_metric, global_valid_metric, output_path = train_lgb_model(
    train, config, output_path=OUTPUT_PATH, run_id=RUN_ID
)
# OUTPUT_PATH = OUTPUT_PATH + RUN_ID + "/"

predict_lgb_model_metrics(test, config, output_path)
print("**Prediction completed**")
