"""Build a standardized hourly price target dataset from 15-minute prices.

The input CSV contains one price row every 15 minutes, except for a small
number of missing timestamps. This script removes 2025-01-01, fills the missing
15-minute values, then reshapes prices into one row per hour with columns for
0, 15, 30, and 45 minutes.

Small gaps are filled by linear interpolation between the nearest observed
prices. Longer gaps are filled by averaging same-time prices from neighboring
days, preserving the daily price pattern better than a flat interpolation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path("raw_data/targets/price.csv")
DEFAULT_OUTPUT_DIR = Path("output/raw_price_standardized")
DEFAULT_TIME_COLUMN = "times"
DEFAULT_PRICE_COLUMN = "A"
DEFAULT_DROP_DATE = "2025-01-01"
DEFAULT_START_TIME = "2025-01-02 00:00:00"
DEFAULT_END_TIME = "2025-12-31 23:45:00"
DEFAULT_MAX_LOCAL_GAP = 2
EXPECTED_MINUTES = (0, 15, 30, 45)
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
PRICE_OUTPUT_NAME = "price.csv"
MASK_OUTPUT_NAME = "price_imputation_mask.csv"
REPORT_OUTPUT_NAME = "price_imputation_report.csv"


@dataclass(frozen=True)
class MissingRun:
    start: pd.Timestamp
    end: pd.Timestamp
    timestamps: pd.DatetimeIndex
    method: str

    @property
    def missing_points(self) -> int:
        return len(self.timestamps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a standardized hourly price CSV from 15-minute price data, "
            "including imputation for missing timestamps."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input 15-minute price CSV. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output CSVs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--time-column",
        default=DEFAULT_TIME_COLUMN,
        help=f"Timestamp column in the input CSV. Default: {DEFAULT_TIME_COLUMN}",
    )
    parser.add_argument(
        "--price-column",
        default=DEFAULT_PRICE_COLUMN,
        help=f"Price column in the input CSV. Default: {DEFAULT_PRICE_COLUMN}",
    )
    parser.add_argument(
        "--drop-date",
        default=DEFAULT_DROP_DATE,
        help=f"Date to remove before reshaping. Default: {DEFAULT_DROP_DATE}",
    )
    parser.add_argument(
        "--start-time",
        default=DEFAULT_START_TIME,
        help=f"Expected first 15-minute timestamp after filtering. Default: {DEFAULT_START_TIME}",
    )
    parser.add_argument(
        "--end-time",
        default=DEFAULT_END_TIME,
        help=f"Expected last 15-minute timestamp after filtering. Default: {DEFAULT_END_TIME}",
    )
    parser.add_argument(
        "--max-local-gap",
        type=int,
        default=DEFAULT_MAX_LOCAL_GAP,
        help=(
            "Maximum consecutive missing 15-minute slots to fill with local "
            f"linear interpolation. Default: {DEFAULT_MAX_LOCAL_GAP}"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing price output CSVs.",
    )
    return parser.parse_args()


def read_price_csv(input_path: Path, time_column: str, price_column: str) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Price input CSV does not exist: {input_path}")

    df = pd.read_csv(input_path)
    missing_columns = [
        column for column in (time_column, price_column) if column not in df.columns
    ]
    if missing_columns:
        raise KeyError(
            f"Missing required columns {missing_columns}. Available columns: {df.columns.tolist()}"
        )

    df = df[[time_column, price_column]].copy()
    df[time_column] = pd.to_datetime(df[time_column], errors="raise")
    df[time_column] = df[time_column].astype("datetime64[ns]")
    df[price_column] = pd.to_numeric(df[price_column], errors="raise")
    df = df.sort_values(time_column).reset_index(drop=True)
    return df


def filter_price_dates(
    df: pd.DataFrame,
    time_column: str,
    drop_date: str,
) -> pd.DataFrame:
    drop_date_value = pd.Timestamp(drop_date).date()
    filtered = df[df[time_column].dt.date != drop_date_value].copy()
    return filtered.reset_index(drop=True)


def validate_price_timestamps(df: pd.DataFrame, time_column: str) -> None:
    times = pd.DatetimeIndex(df[time_column])

    if times.has_duplicates:
        duplicated = times[times.duplicated()].unique()[:5]
        raise ValueError(f"Input contains duplicate timestamps: {list(duplicated)}")

    if not times.is_monotonic_increasing:
        raise ValueError("Input timestamps are not sorted in increasing order.")

    invalid_seconds = df[
        (df[time_column].dt.second != 0)
        | (df[time_column].dt.microsecond != 0)
        | (df[time_column].dt.nanosecond != 0)
    ]
    if not invalid_seconds.empty:
        raise ValueError("All input timestamps must be aligned to exact minute boundaries.")

    minute_values = set(df[time_column].dt.minute.unique())
    expected_minute_values = set(EXPECTED_MINUTES)
    if not minute_values.issubset(expected_minute_values):
        raise ValueError(
            f"Expected minute values to be a subset of {sorted(expected_minute_values)}, "
            f"found {sorted(minute_values)}."
        )


def build_complete_price_series(
    df: pd.DataFrame,
    time_column: str,
    price_column: str,
    start_time: str,
    end_time: str,
) -> pd.Series:
    expected_times = pd.date_range(
        pd.Timestamp(start_time),
        pd.Timestamp(end_time),
        freq="15min",
    )
    observed_times = pd.DatetimeIndex(df[time_column])
    extra_times = observed_times.difference(expected_times)
    if len(extra_times):
        raise ValueError(
            "Filtered price data contains timestamps outside the expected range. "
            f"First extra timestamps: {[timestamp.strftime(TIME_FORMAT) for timestamp in extra_times[:5]]}"
        )

    series = df.set_index(time_column)[price_column].reindex(expected_times)
    series.index.name = "time"
    return series


def find_missing_runs(series: pd.Series) -> list[pd.DatetimeIndex]:
    missing_times = series.index[series.isna()]
    if len(missing_times) == 0:
        return []

    runs: list[pd.DatetimeIndex] = []
    current_run = [missing_times[0]]
    expected_step = pd.Timedelta(minutes=15)

    for timestamp in missing_times[1:]:
        if timestamp - current_run[-1] == expected_step:
            current_run.append(timestamp)
        else:
            runs.append(pd.DatetimeIndex(current_run))
            current_run = [timestamp]

    runs.append(pd.DatetimeIndex(current_run))
    return runs


def fill_local_gap(
    filled: pd.Series,
    missing_run: pd.DatetimeIndex,
) -> bool:
    before_time = missing_run[0] - pd.Timedelta(minutes=15)
    after_time = missing_run[-1] + pd.Timedelta(minutes=15)

    if before_time not in filled.index or after_time not in filled.index:
        return False
    if pd.isna(filled.loc[before_time]) or pd.isna(filled.loc[after_time]):
        return False

    before_value = filled.loc[before_time]
    after_value = filled.loc[after_time]
    steps = len(missing_run) + 1

    for offset, timestamp in enumerate(missing_run, start=1):
        fraction = offset / steps
        filled.loc[timestamp] = before_value + (after_value - before_value) * fraction

    return True


def fill_same_time_neighbor_days(
    filled: pd.Series,
    original: pd.Series,
    missing_run: pd.DatetimeIndex,
) -> bool:
    for timestamp in missing_run:
        candidates = []
        for offset in (pd.Timedelta(days=-1), pd.Timedelta(days=1)):
            neighbor_time = timestamp + offset
            if neighbor_time in original.index and pd.notna(original.loc[neighbor_time]):
                candidates.append(original.loc[neighbor_time])

        if not candidates:
            return False

        filled.loc[timestamp] = sum(candidates) / len(candidates)

    return True


def impute_missing_prices(
    series: pd.Series,
    max_local_gap: int,
) -> tuple[pd.Series, pd.Series, list[MissingRun]]:
    if max_local_gap < 1:
        raise ValueError("--max-local-gap must be at least 1.")

    original = series.copy()
    filled = series.copy()
    imputation_mask = original.isna().astype(int)
    imputation_runs: list[MissingRun] = []

    for missing_run in find_missing_runs(original):
        if len(missing_run) <= max_local_gap:
            if not fill_local_gap(filled, missing_run):
                raise ValueError(
                    "Could not locally interpolate missing price run "
                    f"{missing_run[0]} to {missing_run[-1]}."
                )
            method = "linear_neighbor_interpolation"
        else:
            if not fill_same_time_neighbor_days(filled, original, missing_run):
                raise ValueError(
                    "Could not fill missing price run with same-time neighboring days "
                    f"{missing_run[0]} to {missing_run[-1]}."
                )
            method = "same_time_neighbor_days"

        imputation_runs.append(
            MissingRun(
                start=missing_run[0],
                end=missing_run[-1],
                timestamps=missing_run,
                method=method,
            )
        )

    if filled.isna().any():
        unresolved = filled.index[filled.isna()][:5]
        raise ValueError(
            "Price series still contains missing values after imputation. "
            f"First unresolved timestamps: {list(unresolved)}"
        )

    return filled, imputation_mask, imputation_runs


def reshape_hourly(series: pd.Series, value_column: str) -> pd.DataFrame:
    working = series.rename(value_column).reset_index()
    working["hour"] = working["time"].dt.floor("h")
    working["minute"] = working["time"].dt.minute

    hourly = working.pivot(
        index="hour",
        columns="minute",
        values=value_column,
    ).reindex(columns=EXPECTED_MINUTES)
    hourly.columns = [f"{value_column}_{minute}min" for minute in EXPECTED_MINUTES]
    hourly = hourly.reset_index().rename(columns={"hour": "time"})
    hourly["time"] = hourly["time"].astype("datetime64[ns]")
    return hourly


def validate_hourly_output(
    hourly: pd.DataFrame,
    start_time: str,
    end_time: str,
    label: str,
) -> None:
    expected_times = pd.date_range(
        pd.Timestamp(start_time).floor("h"),
        pd.Timestamp(end_time).floor("h"),
        freq="h",
    )
    times = pd.DatetimeIndex(hourly["time"])

    if hourly.shape[0] != len(expected_times):
        raise ValueError(
            f"{label} should have {len(expected_times)} hourly rows, "
            f"found {hourly.shape[0]}."
        )
    if not times.equals(expected_times):
        raise ValueError(
            f"{label} hourly time column does not match expected range "
            f"{expected_times[0]} to {expected_times[-1]}."
        )
    if hourly.isna().any().any():
        raise ValueError(f"{label} contains missing values after hourly reshaping.")


def build_imputation_report(imputation_runs: list[MissingRun]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "missing_block_start": run.start.strftime(TIME_FORMAT),
                "missing_block_end": run.end.strftime(TIME_FORMAT),
                "missing_points": run.missing_points,
                "method": run.method,
            }
            for run in imputation_runs
        ]
    )


def write_csv_outputs(
    price_hourly: pd.DataFrame,
    mask_hourly: pd.DataFrame,
    imputation_report: pd.DataFrame,
    price_column: str,
    output_dir: Path,
    overwrite: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_dir / PRICE_OUTPUT_NAME,
        output_dir / MASK_OUTPUT_NAME,
        output_dir / REPORT_OUTPUT_NAME,
    ]

    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not overwrite:
        raise FileExistsError(
            f"Output already exists: {existing_paths}. Pass --overwrite to replace it."
        )

    price_to_write = price_hourly.copy()
    price_to_write["time"] = price_to_write["time"].dt.strftime(TIME_FORMAT)

    mask_to_write = mask_hourly.copy()
    mask_to_write["time"] = mask_to_write["time"].dt.strftime(TIME_FORMAT)
    mask_columns = [column for column in mask_to_write.columns if column != "time"]
    mask_to_write[mask_columns] = mask_to_write[mask_columns].astype(int)
    mask_to_write = mask_to_write.rename(
        columns={
            f"{price_column}_{minute}min": f"{price_column}_{minute}min_imputed"
            for minute in EXPECTED_MINUTES
        }
    )

    price_to_write.to_csv(output_paths[0], index=False)
    mask_to_write.to_csv(output_paths[1], index=False)
    imputation_report.to_csv(output_paths[2], index=False)
    return output_paths


def build_raw_price_dataset(
    input_path: Path,
    output_dir: Path,
    time_column: str,
    price_column: str,
    drop_date: str,
    start_time: str,
    end_time: str,
    max_local_gap: int,
    overwrite: bool,
) -> list[Path]:
    df = read_price_csv(input_path, time_column, price_column)
    df = filter_price_dates(df, time_column, drop_date)
    validate_price_timestamps(df, time_column)

    series = build_complete_price_series(df, time_column, price_column, start_time, end_time)
    filled, imputation_mask, imputation_runs = impute_missing_prices(series, max_local_gap)

    price_hourly = reshape_hourly(filled, price_column)
    mask_hourly = reshape_hourly(imputation_mask, price_column)
    validate_hourly_output(price_hourly, start_time, end_time, "price")
    validate_hourly_output(mask_hourly, start_time, end_time, "price imputation mask")

    imputation_report = build_imputation_report(imputation_runs)
    output_paths = write_csv_outputs(
        price_hourly,
        mask_hourly,
        imputation_report,
        price_column,
        output_dir,
        overwrite,
    )

    print(f"Wrote standardized price outputs to: {output_dir}")
    print(f"Hourly rows: {price_hourly.shape[0]}")
    print(f"Imputed 15-minute values: {int(imputation_mask.sum())}")
    print(f"Missing blocks filled: {len(imputation_runs)}")
    print(f"Time format on disk: {TIME_FORMAT}")
    for path in output_paths:
        print(f"- {path}")

    return output_paths


def main() -> None:
    args = parse_args()
    build_raw_price_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        time_column=args.time_column,
        price_column=args.price_column,
        drop_date=args.drop_date,
        start_time=args.start_time,
        end_time=args.end_time,
        max_local_gap=args.max_local_gap,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
