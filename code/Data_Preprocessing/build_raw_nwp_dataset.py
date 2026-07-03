"""Build the standardized raw NWP xarray dataset.

Each source NetCDF file is a daily forecast. The date in the filename is the
issue date, and lead_time 0..23 maps to the next day's 00:00..23:00 target
hours. The output keeps the weather cube intact as:

    cube(time, channel, lat, lon)

The time coordinate is stored as numpy datetime64[ns], which works smoothly as
a merge/join key with pandas datetime64[ns] boundary data later.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_INPUT_DIR = Path("raw_data/meteodata")
DEFAULT_OUTPUT_PATH = Path("output/raw_nwp_standardized.nc")
DEFAULT_START_ISSUE_DATE = "2025-01-01"
DEFAULT_END_ISSUE_DATE = "2025-12-30"
EXPECTED_LEAD_HOURS = 24

DIMENSION_ALIASES = {
    "time": {"time", "issue_time", "forecast_time", "init_time", "valid_time"},
    "lead_time": {"lead_time", "lead", "step", "forecast_hour", "fhour", "hour"},
    "channel": {"channel", "channels", "variable", "variables", "var", "feature", "features"},
    "lat": {"lat", "latitude", "y"},
    "lon": {"lon", "longitude", "x"},
}


@dataclass(frozen=True)
class NwpFile:
    path: Path
    issue_date: pd.Timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a standardized raw NWP dataset from daily NetCDF forecast files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing daily .nc files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path ending in .nc or .zarr. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--start-issue-date",
        default=DEFAULT_START_ISSUE_DATE,
        help=f"First issue date to include. Default: {DEFAULT_START_ISSUE_DATE}",
    )
    parser.add_argument(
        "--end-issue-date",
        default=DEFAULT_END_ISSUE_DATE,
        help=(
            "Last issue date to include. Default excludes 2025-12-31 because that "
            f"file forecasts 2026-01-01. Default: {DEFAULT_END_ISSUE_DATE}"
        ),
    )
    parser.add_argument(
        "--data-variable",
        default=None,
        help="Name of the NetCDF data variable to use. If omitted, a single data variable is discovered.",
    )
    parser.add_argument(
        "--output-format",
        choices=("auto", "netcdf", "zarr"),
        default="auto",
        help="Output format. 'auto' uses the output suffix. Default: auto",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file or Zarr directory.",
    )
    return parser.parse_args()


def parse_issue_date_from_filename(path: Path) -> pd.Timestamp:
    match = re.fullmatch(r"(\d{8})", path.stem)
    if not match:
        raise ValueError(
            f"Cannot parse issue date from {path.name!r}. Expected filename like 20250101.nc."
        )
    return pd.to_datetime(match.group(1), format="%Y%m%d")


def discover_nwp_files(
    input_dir: Path,
    start_issue_date: pd.Timestamp,
    end_issue_date: pd.Timestamp,
) -> list[NwpFile]:
    if not input_dir.exists():
        raise FileNotFoundError(f"NWP input directory does not exist: {input_dir}")

    discovered: list[NwpFile] = []
    for path in sorted(input_dir.glob("*.nc")):
        issue_date = parse_issue_date_from_filename(path)
        if start_issue_date <= issue_date <= end_issue_date:
            discovered.append(NwpFile(path=path, issue_date=issue_date))

    expected_dates = pd.date_range(start_issue_date, end_issue_date, freq="D")
    found_dates = {item.issue_date.normalize() for item in discovered}
    missing_dates = [date for date in expected_dates if date not in found_dates]
    if missing_dates:
        missing = ", ".join(date.strftime("%Y-%m-%d") for date in missing_dates[:10])
        suffix = " ..." if len(missing_dates) > 10 else ""
        raise FileNotFoundError(
            f"Missing {len(missing_dates)} required NWP files between "
            f"{start_issue_date.date()} and {end_issue_date.date()}: {missing}{suffix}"
        )

    return discovered


def canonical_dimension_name(name: str) -> str | None:
    normalized = name.lower()
    for canonical_name, aliases in DIMENSION_ALIASES.items():
        if normalized in aliases:
            return canonical_name
    return None


def infer_dimension_renames(ds: xr.Dataset, data_var: str) -> dict[str, str]:
    renames: dict[str, str] = {}
    used_canonical_names: set[str] = set()

    for dim in ds[data_var].dims:
        canonical_name = canonical_dimension_name(dim)
        if canonical_name is None:
            continue
        if canonical_name in used_canonical_names:
            raise ValueError(
                f"Multiple dimensions in variable {data_var!r} map to {canonical_name!r}. "
                f"Dimensions found: {ds[data_var].dims}"
            )
        used_canonical_names.add(canonical_name)
        if dim != canonical_name:
            renames[dim] = canonical_name

    return renames


def choose_data_variable(ds: xr.Dataset, requested: str | None) -> str:
    if requested is not None:
        if requested not in ds.data_vars:
            raise KeyError(
                f"Requested data variable {requested!r} was not found. "
                f"Available variables: {list(ds.data_vars)}"
            )
        return requested

    data_vars = list(ds.data_vars)
    if len(data_vars) != 1:
        raise ValueError(
            "Could not infer the NWP data variable because the file contains "
            f"{len(data_vars)} data variables: {data_vars}. Pass --data-variable."
        )
    return data_vars[0]


def open_standardized_file(path: Path, requested_data_var: str | None) -> xr.DataArray:
    ds = xr.open_dataset(path, chunks={})
    data_var = choose_data_variable(ds, requested_data_var)
    renames = infer_dimension_renames(ds, data_var)
    if renames:
        ds = ds.rename(renames)

    required_dims = {"lead_time", "channel", "lat", "lon"}
    missing_dims = required_dims.difference(ds[data_var].dims)
    if missing_dims:
        raise ValueError(
            f"{path.name} is missing required dimensions {sorted(missing_dims)} "
            f"after standardization. Found dimensions: {ds[data_var].dims}"
        )

    if "time" in ds[data_var].dims:
        if ds.sizes["time"] != 1:
            raise ValueError(
                f"{path.name} should contain exactly one issue-time slice, "
                f"but dimension 'time' has length {ds.sizes['time']}."
            )
        data = ds[data_var].isel(time=0, drop=True)
    else:
        data = ds[data_var]

    data = data.transpose("lead_time", "channel", "lat", "lon")

    if data.sizes["lead_time"] != EXPECTED_LEAD_HOURS:
        raise ValueError(
            f"{path.name} should contain {EXPECTED_LEAD_HOURS} lead times, "
            f"but found {data.sizes['lead_time']}."
        )

    expected_leads = np.arange(EXPECTED_LEAD_HOURS)
    actual_leads = np.asarray(data["lead_time"].values)
    if not np.array_equal(actual_leads, expected_leads):
        raise ValueError(
            f"{path.name} has unexpected lead_time values. Expected "
            f"{expected_leads.tolist()}, found {actual_leads.tolist()}."
        )

    return data


def build_daily_dataset(nwp_file: NwpFile, requested_data_var: str | None) -> xr.Dataset:
    data = open_standardized_file(nwp_file.path, requested_data_var)
    lead_times = data["lead_time"].values.astype("int64")
    target_times = (
        nwp_file.issue_date.normalize()
        + pd.Timedelta(days=1)
        + pd.to_timedelta(lead_times, unit="h")
    )

    daily = xr.Dataset(
        data_vars={
            "cube": (
                ("time", "channel", "lat", "lon"),
                data.data,
                {
                    "description": (
                        "Raw NWP forecast cube. The time coordinate is the target "
                        "valid hour derived from filename issue_date + 1 day + lead_time."
                    )
                },
            )
        },
        coords={
            "time": ("time", target_times.to_numpy(dtype="datetime64[ns]")),
            "issue_date": (
                "time",
                np.repeat(
                    nwp_file.issue_date.normalize().to_datetime64().astype("datetime64[ns]"),
                    EXPECTED_LEAD_HOURS,
                ),
            ),
            "lead_time": ("time", lead_times.astype("int16")),
            "channel": data["channel"].values,
            "lat": data["lat"].values,
            "lon": data["lon"].values,
        },
        attrs={
            "time_alignment_rule": "time = filename issue_date + 1 day + lead_time hours",
        },
    )

    return daily


def validate_combined_dataset(ds: xr.Dataset, files: Iterable[NwpFile]) -> None:
    files = list(files)
    expected_rows = len(files) * EXPECTED_LEAD_HOURS
    if ds.sizes["time"] != expected_rows:
        raise ValueError(f"Expected {expected_rows} hourly rows, found {ds.sizes['time']}.")

    times = pd.DatetimeIndex(ds["time"].values)
    if times.dtype != "datetime64[ns]":
        raise TypeError(f"time coordinate should be datetime64[ns], found {times.dtype}.")
    if times.has_duplicates:
        duplicated = times[times.duplicated()].unique()[:5]
        raise ValueError(f"time coordinate contains duplicate timestamps: {list(duplicated)}")
    if not times.is_monotonic_increasing:
        raise ValueError("time coordinate is not strictly increasing.")

    expected_times = pd.date_range(
        files[0].issue_date.normalize() + pd.Timedelta(days=1),
        files[-1].issue_date.normalize() + pd.Timedelta(days=1, hours=23),
        freq="h",
    )
    if not times.equals(expected_times):
        raise ValueError(
            "time coordinate does not match the expected continuous hourly target range. "
            f"Expected {expected_times[0]} to {expected_times[-1]}, "
            f"found {times[0]} to {times[-1]}."
        )


def resolve_output_format(output_path: Path, output_format: str) -> str:
    if output_format != "auto":
        return output_format
    if output_path.suffix == ".zarr":
        return "zarr"
    return "netcdf"


def ensure_can_write(output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)


def write_dataset(ds: xr.Dataset, output_path: Path, output_format: str, overwrite: bool) -> None:
    ensure_can_write(output_path, overwrite)

    if output_format == "zarr":
        if importlib.util.find_spec("zarr") is None:
            raise ImportError(
                "Zarr output was requested, but the zarr package is not installed. "
                "Use --output-format netcdf or install zarr."
            )
        mode = "w" if overwrite else "w-"
        ds.to_zarr(output_path, mode=mode)
        return

    if output_path.suffix == ".zarr":
        raise ValueError("NetCDF output cannot be written to a .zarr path.")

    encoding = {
        "cube": {
            "zlib": True,
            "complevel": 4,
            "chunksizes": (
                min(EXPECTED_LEAD_HOURS, ds.sizes["time"]),
                ds.sizes["channel"],
                ds.sizes["lat"],
                ds.sizes["lon"],
            ),
        }
    }
    ds.to_netcdf(output_path, engine="netcdf4", encoding=encoding)


def build_raw_nwp_dataset(
    input_dir: Path,
    output_path: Path,
    start_issue_date: str,
    end_issue_date: str,
    data_variable: str | None,
    output_format: str,
    overwrite: bool,
) -> xr.Dataset:
    start = pd.to_datetime(start_issue_date).normalize()
    end = pd.to_datetime(end_issue_date).normalize()
    if start > end:
        raise ValueError(f"start issue date {start.date()} is after end issue date {end.date()}.")

    files = discover_nwp_files(input_dir, start, end)
    daily_datasets = [build_daily_dataset(nwp_file, data_variable) for nwp_file in files]
    combined = xr.concat(daily_datasets, dim="time", combine_attrs="drop").sortby("time")
    combined = combined.chunk({"time": EXPECTED_LEAD_HOURS})
    combined.attrs.update(
        {
            "description": "Standardized raw NWP dataset for model-preparation workflows.",
            "source_directory": str(input_dir),
            "source_file_pattern": "YYYYMMDD.nc",
            "source_file_count": len(files),
            "included_issue_dates": f"{start.date()} to {end.date()}",
            "excluded_issue_dates": (
                "2025-12-31 is excluded by default because it forecasts 2026-01-01, "
                "which is unmatched by the 2025 boundary dataset."
            ),
            "time_dtype": "datetime64[ns]",
        }
    )

    validate_combined_dataset(combined, files)
    resolved_format = resolve_output_format(output_path, output_format)
    write_dataset(combined, output_path, resolved_format, overwrite)

    print(f"Wrote standardized NWP dataset: {output_path}")
    print(f"Output format: {resolved_format}")
    print(f"Rows/time steps: {combined.sizes['time']}")
    print(f"Time range: {combined['time'].values[0]} to {combined['time'].values[-1]}")
    print(f"Cube shape: {combined['cube'].shape}")
    print(f"time dtype: {combined['time'].dtype}")

    return combined


def main() -> None:
    args = parse_args()
    build_raw_nwp_dataset(
        input_dir=args.input_dir,
        output_path=args.output,
        start_issue_date=args.start_issue_date,
        end_issue_date=args.end_issue_date,
        data_variable=args.data_variable,
        output_format=args.output_format,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
