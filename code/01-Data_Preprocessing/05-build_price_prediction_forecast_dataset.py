"""Build a 15-minute forecast-feature price-prediction dataset.

The input boundary CSV already contains one row every 15 minutes. This script
keeps only forecast-value boundary columns, converts the standardized hourly
price target into the same 15-minute format, then merges both datasets on time.

The output is intended for forecast-feature baseline models:

    time, boundary forecast features..., price
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path("raw_data/targets/preduction.csv")
DEFAULT_PRICE_PATH = Path("output/raw_price_standardized/price.csv")
DEFAULT_OUTPUT_PATH = Path("output/clean_datasets/price_prediction_forecast_dataset.csv")
DEFAULT_TIME_COLUMN = "times"
DEFAULT_PRICE_TIME_COLUMN = "time"
DEFAULT_PRICE_COLUMN_PREFIX = "A"
DEFAULT_START_TIME = "2025-01-02 00:00:00"
DEFAULT_END_TIME = "2025-12-31 23:45:00"
EXPECTED_MINUTES = (0, 15, 30, 45)
FORECAST_SUFFIX = "_Forecast_Value"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
PRICE_OUTPUT_COLUMN = "price"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a 15-minute modeling dataset from forecast boundary values "
            "and the standardized price target."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input 15-minute boundary prediction CSV. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--price",
        type=Path,
        default=DEFAULT_PRICE_PATH,
        help=f"Standardized hourly-wide price CSV. Default: {DEFAULT_PRICE_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output merged 15-minute modeling CSV. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--time-column",
        default=DEFAULT_TIME_COLUMN,
        help=f"Timestamp column in the boundary input. Default: {DEFAULT_TIME_COLUMN}",
    )
    parser.add_argument(
        "--price-time-column",
        default=DEFAULT_PRICE_TIME_COLUMN,
        help=f"Timestamp column in the price input. Default: {DEFAULT_PRICE_TIME_COLUMN}",
    )
    parser.add_argument(
        "--price-column-prefix",
        default=DEFAULT_PRICE_COLUMN_PREFIX,
        help=f"Prefix for hourly-wide price columns. Default: {DEFAULT_PRICE_COLUMN_PREFIX}",
    )
    parser.add_argument(
        "--start-time",
        default=DEFAULT_START_TIME,
        help=f"Expected first 15-minute timestamp. Default: {DEFAULT_START_TIME}",
    )
    parser.add_argument(
        "--end-time",
        default=DEFAULT_END_TIME,
        help=f"Expected last 15-minute timestamp. Default: {DEFAULT_END_TIME}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output CSV if it already exists.",
    )
    return parser.parse_args()


def validate_15min_time_index(
    df: pd.DataFrame,
    time_column: str,
    start_time: str,
    end_time: str,
    dataset_name: str,
) -> None:
    times = pd.DatetimeIndex(df[time_column])
    expected_times = pd.date_range(
        pd.Timestamp(start_time),
        pd.Timestamp(end_time),
        freq="15min",
    )

    if times.has_duplicates:
        duplicated = times[times.duplicated()].unique()[:5]
        raise ValueError(f"{dataset_name} contains duplicate timestamps: {list(duplicated)}")

    if not times.is_monotonic_increasing:
        raise ValueError(f"{dataset_name} timestamps are not sorted in increasing order.")

    if not times.equals(expected_times):
        missing = expected_times.difference(times)
        extra = times.difference(expected_times)
        details = []
        if len(missing):
            details.append(
                f"missing {len(missing)} timestamps, first few: "
                f"{[timestamp.strftime(TIME_FORMAT) for timestamp in missing[:5]]}"
            )
        if len(extra):
            details.append(
                f"extra {len(extra)} timestamps, first few: "
                f"{[timestamp.strftime(TIME_FORMAT) for timestamp in extra[:5]]}"
            )
        raise ValueError(
            f"{dataset_name} timestamps do not match the expected 15-minute range: "
            + "; ".join(details)
        )

    invalid_seconds = df[
        (df[time_column].dt.second != 0)
        | (df[time_column].dt.microsecond != 0)
        | (df[time_column].dt.nanosecond != 0)
    ]
    if not invalid_seconds.empty:
        raise ValueError(f"{dataset_name} timestamps must be aligned to exact minute boundaries.")

    minute_values = set(df[time_column].dt.minute.unique())
    expected_minute_values = set(EXPECTED_MINUTES)
    if minute_values != expected_minute_values:
        raise ValueError(
            f"{dataset_name} should contain minute values {sorted(expected_minute_values)}, "
            f"found {sorted(minute_values)}."
        )


def read_forecast_features(input_path: Path, time_column: str) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Prediction input CSV does not exist: {input_path}")

    df = pd.read_csv(input_path)
    if time_column not in df.columns:
        raise KeyError(
            f"Time column {time_column!r} was not found. Available columns: {df.columns.tolist()}"
        )

    forecast_columns = [column for column in df.columns if column.endswith(FORECAST_SUFFIX)]
    if not forecast_columns:
        candidate_columns = [column for column in df.columns if column != time_column]
        raise ValueError(
            f"No forecast-value columns ending in {FORECAST_SUFFIX!r} were found. "
            f"Available non-time columns: {candidate_columns}"
        )

    output = df[[time_column, *forecast_columns]].copy()
    output[time_column] = pd.to_datetime(output[time_column], errors="raise")
    output[time_column] = output[time_column].astype("datetime64[ns]")

    for column in forecast_columns:
        output[column] = pd.to_numeric(output[column], errors="raise")

    output = output.sort_values(time_column).reset_index(drop=True)
    output = output.rename(
        columns={
            time_column: "time",
            **{column: column.removesuffix(FORECAST_SUFFIX) for column in forecast_columns},
        }
    )
    return output


def read_price_csv(price_path: Path, time_column: str, price_column_prefix: str) -> pd.DataFrame:
    if not price_path.exists():
        raise FileNotFoundError(f"Price CSV does not exist: {price_path}")

    df = pd.read_csv(price_path)
    price_columns = [f"{price_column_prefix}_{minute}min" for minute in EXPECTED_MINUTES]
    missing_columns = [
        column
        for column in [time_column, *price_columns]
        if column not in df.columns
    ]
    if missing_columns:
        raise KeyError(
            f"Missing required price columns {missing_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )

    output = df[[time_column, *price_columns]].copy()
    output[time_column] = pd.to_datetime(output[time_column], errors="raise")
    output[time_column] = output[time_column].astype("datetime64[ns]")

    for column in price_columns:
        output[column] = pd.to_numeric(output[column], errors="raise")

    return output.sort_values(time_column).reset_index(drop=True)


def reshape_price_to_15min(
    price: pd.DataFrame,
    time_column: str,
    price_column_prefix: str,
) -> pd.DataFrame:
    pieces = []
    for minute in EXPECTED_MINUTES:
        source_column = f"{price_column_prefix}_{minute}min"
        piece = price[[time_column, source_column]].copy()
        piece["time"] = piece[time_column] + pd.to_timedelta(minute, unit="min")
        piece = piece.rename(columns={source_column: PRICE_OUTPUT_COLUMN})
        pieces.append(piece[["time", PRICE_OUTPUT_COLUMN]])

    price_long = pd.concat(pieces, ignore_index=True)
    price_long = price_long.sort_values("time").reset_index(drop=True)
    price_long["time"] = price_long["time"].astype("datetime64[ns]")
    return price_long


def merge_features_and_price(features: pd.DataFrame, price_long: pd.DataFrame) -> pd.DataFrame:
    feature_times = pd.DatetimeIndex(features["time"])
    price_times = pd.DatetimeIndex(price_long["time"])
    missing_price_times = feature_times.difference(price_times)
    missing_feature_times = price_times.difference(feature_times)

    if len(missing_price_times) or len(missing_feature_times):
        details = []
        if len(missing_price_times):
            details.append(
                f"{len(missing_price_times)} feature timestamps have no price, first few: "
                f"{[timestamp.strftime(TIME_FORMAT) for timestamp in missing_price_times[:5]]}"
            )
        if len(missing_feature_times):
            details.append(
                f"{len(missing_feature_times)} price timestamps have no features, first few: "
                f"{[timestamp.strftime(TIME_FORMAT) for timestamp in missing_feature_times[:5]]}"
            )
        raise ValueError("Feature and price timestamps do not align: " + "; ".join(details))

    merged = features.merge(price_long, on="time", how="inner", validate="one_to_one")
    if merged.shape[0] != features.shape[0]:
        raise ValueError(
            f"Merged dataset has {merged.shape[0]} rows, expected {features.shape[0]} rows."
        )
    return merged


def write_output(df: pd.DataFrame, output_path: Path, overwrite: bool) -> Path:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = df.copy()
    output["time"] = output["time"].dt.strftime(TIME_FORMAT)
    output.to_csv(output_path, index=False)
    return output_path


def build_price_prediction_forecast_dataset(
    input_path: Path,
    price_path: Path,
    output_path: Path,
    time_column: str,
    price_time_column: str,
    price_column_prefix: str,
    start_time: str,
    end_time: str,
    overwrite: bool,
) -> Path:
    features = read_forecast_features(input_path, time_column)
    validate_15min_time_index(features, "time", start_time, end_time, "Forecast features")

    price = read_price_csv(price_path, price_time_column, price_column_prefix)
    price_long = reshape_price_to_15min(price, price_time_column, price_column_prefix)
    validate_15min_time_index(price_long, "time", start_time, end_time, "Price target")

    merged = merge_features_and_price(features, price_long)
    written_path = write_output(merged, output_path, overwrite)

    feature_columns = [column for column in merged.columns if column not in {"time", PRICE_OUTPUT_COLUMN}]
    print(f"Input forecast features: {input_path}")
    print(f"Price target: {price_path}")
    print(f"Wrote forecast price-prediction dataset: {written_path}")
    print(f"Rows: {merged.shape[0]}")
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Target column: {PRICE_OUTPUT_COLUMN}")
    return written_path


def main() -> None:
    args = parse_args()
    build_price_prediction_forecast_dataset(
        input_path=args.input,
        price_path=args.price,
        output_path=args.output,
        time_column=args.time_column,
        price_time_column=args.price_time_column,
        price_column_prefix=args.price_column_prefix,
        start_time=args.start_time,
        end_time=args.end_time,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
