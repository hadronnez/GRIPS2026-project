"""Build standardized raw boundary datasets from 15-minute boundary data.

The input CSV contains one row every 15 minutes. This script reshapes it into
one row per hour and writes one CSV per boundary variable. Each output file
contains the hourly time key plus that variable's actual and forecast values at
0, 15, 30, and 45 minutes.

CSV stores timestamps as text, so outputs use the stable format
YYYY-MM-DD HH:MM:SS. Read them later with parse_dates=["time"], then cast with
.astype("datetime64[ns]") when you need the exact same dtype as the NWP time
coordinate.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path("raw_data/targets/preduction.csv")
DEFAULT_OUTPUT_DIR = Path("output/raw_boundary_standardized")
DEFAULT_TIME_COLUMN = "times"
DEFAULT_START_TIME = "2025-01-02 00:00:00"
DEFAULT_END_TIME = "2025-12-31 23:45:00"
EXPECTED_MINUTES = (0, 15, 30, 45)
ACTUAL_SUFFIX = "_Actual_Value"
FORECAST_SUFFIX = "_Forecast_Value"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class BoundaryVariable:
    name: str
    actual_column: str
    forecast_column: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build standardized hourly boundary CSV files from 15-minute actual "
            "and forecast boundary data."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input 15-minute boundary CSV. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for per-variable output CSVs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--time-column",
        default=DEFAULT_TIME_COLUMN,
        help=f"Timestamp column in the input CSV. Default: {DEFAULT_TIME_COLUMN}",
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
        help="Overwrite existing per-variable CSV outputs.",
    )
    return parser.parse_args()


def read_boundary_csv(input_path: Path, time_column: str) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Boundary input CSV does not exist: {input_path}")

    df = pd.read_csv(input_path)
    if time_column not in df.columns:
        raise KeyError(
            f"Time column {time_column!r} was not found. Available columns: {df.columns.tolist()}"
        )

    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column], errors="raise")
    df[time_column] = df[time_column].astype("datetime64[ns]")
    df = df.sort_values(time_column).reset_index(drop=True)
    return df


def discover_boundary_variables(df: pd.DataFrame, time_column: str) -> list[BoundaryVariable]:
    actual_columns = sorted(column for column in df.columns if column.endswith(ACTUAL_SUFFIX))
    variables: list[BoundaryVariable] = []

    for actual_column in actual_columns:
        name = actual_column.removesuffix(ACTUAL_SUFFIX)
        forecast_column = f"{name}{FORECAST_SUFFIX}"
        if forecast_column not in df.columns:
            raise ValueError(
                f"Found actual column {actual_column!r}, but missing matching "
                f"forecast column {forecast_column!r}."
            )
        variables.append(
            BoundaryVariable(
                name=name,
                actual_column=actual_column,
                forecast_column=forecast_column,
            )
        )

    unused_forecast_columns = [
        column
        for column in df.columns
        if column.endswith(FORECAST_SUFFIX)
        and f"{column.removesuffix(FORECAST_SUFFIX)}{ACTUAL_SUFFIX}" not in df.columns
    ]
    if unused_forecast_columns:
        raise ValueError(
            "Found forecast columns without matching actual columns: "
            f"{unused_forecast_columns}"
        )

    if not variables:
        candidate_columns = [column for column in df.columns if column != time_column]
        raise ValueError(
            "No boundary variable pairs were discovered. Expected columns ending "
            f"in {ACTUAL_SUFFIX!r} and {FORECAST_SUFFIX!r}. Found: {candidate_columns}"
        )

    return variables


def validate_15min_time_index(
    df: pd.DataFrame,
    time_column: str,
    start_time: str,
    end_time: str,
) -> None:
    times = pd.DatetimeIndex(df[time_column])
    expected_times = pd.date_range(
        pd.Timestamp(start_time),
        pd.Timestamp(end_time),
        freq="15min",
    )

    if times.has_duplicates:
        duplicated = times[times.duplicated()].unique()[:5]
        raise ValueError(f"Input contains duplicate timestamps: {list(duplicated)}")

    if not times.is_monotonic_increasing:
        raise ValueError("Input timestamps are not sorted in increasing order.")

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
        if not details:
            details.append(
                f"expected {expected_times[0]} to {expected_times[-1]}, "
                f"found {times[0]} to {times[-1]}"
            )
        raise ValueError("Input timestamps do not match the expected 15-minute range: " + "; ".join(details))

    invalid_seconds = df[
        (df[time_column].dt.second != 0)
        | (df[time_column].dt.microsecond != 0)
        | (df[time_column].dt.nanosecond != 0)
    ]
    if not invalid_seconds.empty:
        raise ValueError("All input timestamps must be aligned to exact minute boundaries.")

    minute_values = set(df[time_column].dt.minute.unique())
    expected_minute_values = set(EXPECTED_MINUTES)
    if minute_values != expected_minute_values:
        raise ValueError(
            f"Expected minute values {sorted(expected_minute_values)}, "
            f"found {sorted(minute_values)}."
        )


def safe_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    safe = safe.strip("._")
    if not safe:
        raise ValueError(f"Could not create a safe filename for variable name {name!r}.")
    return f"{safe}.csv"


def reshape_variable_to_hourly(
    df: pd.DataFrame,
    time_column: str,
    variable: BoundaryVariable,
) -> pd.DataFrame:
    working = df[[time_column, variable.actual_column, variable.forecast_column]].copy()
    working["time"] = working[time_column].dt.floor("h")
    working["minute"] = working[time_column].dt.minute

    actual = working.pivot(
        index="time",
        columns="minute",
        values=variable.actual_column,
    ).reindex(columns=EXPECTED_MINUTES)
    forecast = working.pivot(
        index="time",
        columns="minute",
        values=variable.forecast_column,
    ).reindex(columns=EXPECTED_MINUTES)

    if list(actual.columns) != list(EXPECTED_MINUTES) or list(forecast.columns) != list(EXPECTED_MINUTES):
        raise ValueError(
            f"{variable.name} did not reshape into the expected 0/15/30/45-minute columns."
        )

    actual.columns = [f"{variable.actual_column}_{minute}min" for minute in EXPECTED_MINUTES]
    forecast.columns = [f"{variable.forecast_column}_{minute}min" for minute in EXPECTED_MINUTES]

    hourly = pd.concat([actual, forecast], axis=1).reset_index()
    hourly["time"] = hourly["time"].astype("datetime64[ns]")
    return hourly


def validate_hourly_output(
    hourly: pd.DataFrame,
    variable: BoundaryVariable,
    start_time: str,
    end_time: str,
) -> None:
    expected_times = pd.date_range(
        pd.Timestamp(start_time).floor("h"),
        pd.Timestamp(end_time).floor("h"),
        freq="h",
    )
    times = pd.DatetimeIndex(hourly["time"])

    if hourly.shape[0] != len(expected_times):
        raise ValueError(
            f"{variable.name} should have {len(expected_times)} hourly rows, "
            f"found {hourly.shape[0]}."
        )
    if not times.equals(expected_times):
        raise ValueError(
            f"{variable.name} hourly time column does not match expected range "
            f"{expected_times[0]} to {expected_times[-1]}."
        )
    if hourly["time"].dtype != "datetime64[ns]":
        raise TypeError(
            f"{variable.name} time column should be datetime64[ns], found {hourly['time'].dtype}."
        )


def write_variable_csv(
    hourly: pd.DataFrame,
    output_dir: Path,
    variable: BoundaryVariable,
    overwrite: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / safe_filename(variable.name)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

    hourly_to_write = hourly.copy()
    hourly_to_write["time"] = hourly_to_write["time"].dt.strftime(TIME_FORMAT)
    hourly_to_write.to_csv(output_path, index=False)
    return output_path


def build_raw_boundary_datasets(
    input_path: Path,
    output_dir: Path,
    time_column: str,
    start_time: str,
    end_time: str,
    overwrite: bool,
) -> list[Path]:
    df = read_boundary_csv(input_path, time_column)
    variables = discover_boundary_variables(df, time_column)
    validate_15min_time_index(df, time_column, start_time, end_time)

    output_paths: list[Path] = []
    for variable in variables:
        hourly = reshape_variable_to_hourly(df, time_column, variable)
        validate_hourly_output(hourly, variable, start_time, end_time)
        output_paths.append(write_variable_csv(hourly, output_dir, variable, overwrite))

    print(f"Wrote {len(output_paths)} standardized boundary CSV files to: {output_dir}")
    print(f"Hourly rows per file: {len(pd.date_range(pd.Timestamp(start_time), pd.Timestamp(end_time).floor('h'), freq='h'))}")
    print(f"Time format on disk: {TIME_FORMAT}")
    print('Reload time as datetime64[ns] with: df["time"] = pd.to_datetime(df["time"]).astype("datetime64[ns]")')
    for path in output_paths:
        print(f"- {path}")

    return output_paths


def main() -> None:
    args = parse_args()
    build_raw_boundary_datasets(
        input_path=args.input,
        output_dir=args.output_dir,
        time_column=args.time_column,
        start_time=args.start_time,
        end_time=args.end_time,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
