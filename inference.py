import json
import os
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from strategy import prices_to_strategy


DEFAULT_DATA_DIR = Path("/saisdata/34")
DEFAULT_RESULT_DIR = Path("/saisresult")
DEFAULT_MODEL_PATH = Path("/app/model/lgb_model.txt")
DEFAULT_METADATA_PATH = Path("/app/model/metadata.json")

TARGET_COL = "A"
FEATURE_COLS = [
    "系统负荷预测值",
    "风光总加预测值",
    "联络线预测值",
    "风电预测值",
    "光伏预测值",
    "水电预测值",
    "非市场化机组预测值",
]
TIME_FEATURE_COLS = ["hour", "minute", "dayofweek", "month"]
ALL_FEATURES = FEATURE_COLS + TIME_FEATURE_COLS


def parse_times(series):
    try:
        return pd.to_datetime(series)
    except ValueError:
        return pd.to_datetime(series, format="mixed")


def add_time_features(df):
    df = df.copy()
    df["times"] = parse_times(df["times"])
    df["hour"] = df["times"].dt.hour
    df["minute"] = df["times"].dt.minute
    df["dayofweek"] = df["times"].dt.dayofweek
    df["month"] = df["times"].dt.month
    return df


def load_metadata(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_features(df, metadata):
    missing = [col for col in ["times"] + FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in test feature file: {missing}")

    df = add_time_features(df)
    medians = metadata.get("feature_medians", {})
    for col in ALL_FEATURES:
        if df[col].isna().any() and col in medians:
            df[col] = df[col].fillna(medians[col])

    return df


def run_inference(data_dir, result_dir, model_path, metadata_path):
    test_path = data_dir / "test_in_feature_ori.csv"
    output_path = result_dir / "output.csv"

    if not test_path.exists():
        raise FileNotFoundError(f"Cannot find test feature file: {test_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Cannot find LightGBM model file: {model_path}")

    result_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(metadata_path)
    model = lgb.Booster(model_file=str(model_path))

    df_test = pd.read_csv(test_path)
    df_test = prepare_features(df_test, metadata)
    df_test = df_test.sort_values("times").reset_index(drop=True)

    pred = model.predict(df_test[ALL_FEATURES], num_iteration=model.best_iteration)
    price_df = pd.DataFrame({"times": df_test["times"], "实时价格": pred})
    strategy_df = prices_to_strategy(price_df)
    strategy_df.to_csv(output_path, index=False)

    print(f"Saved {output_path} with shape={strategy_df.shape}")


def main():
    data_dir = Path(os.environ.get("SAIS_DATA_DIR", DEFAULT_DATA_DIR))
    result_dir = Path(os.environ.get("SAIS_RESULT_DIR", DEFAULT_RESULT_DIR))
    model_path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    metadata_path = Path(os.environ.get("MODEL_METADATA_PATH", DEFAULT_METADATA_PATH))
    run_inference(data_dir, result_dir, model_path, metadata_path)


if __name__ == "__main__":
    main()
