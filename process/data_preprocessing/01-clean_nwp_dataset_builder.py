"""Build the standardized clean NWP xarray dataset.

Each source NetCDF file is a daily forecast. The date in the filename is the
issue date, and lead_time 0..23 maps to the next day's 00:00..23:00 target
hours. The output structure is:

    cube(time, channel, lat, lon)

where time is a coordinate indexing across all forecast hours, and lead_time
is not stored (it's folded into the time coordinate). Each cube at time t
has shape (channel, lat, lon).

Time is stored as datetime64[ns] throughout.
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

try:
    import dask
    import dask.array as da
except ImportError:
    dask = None


DEFAULT_INPUT_DIR = Path("input/meteodata")
DEFAULT_OUTPUT_PATH = Path("output/data_preprocessing/clean_datasets/clean_nwp_dataset.nc")
DEFAULT_START_ISSUE_DATE = "2025-01-01"
DEFAULT_END_ISSUE_DATE = "2025-12-30"
EXPECTED_LEAD_HOURS = 24

DIMENSION_ALIASES = {
    "time": {"time"},
    "lead_time": {"lead_time"},
    "channel": {"channel"},
    "lat": {"lat"},
    "lon": {"lon"},
}

# Encoding for datetime coordinates—preserves datetime64[ns]
DATETIME_ENCODING = {
    "units": "nanoseconds since 1970-01-01",
    "dtype": "int64",
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
    # Use chunks to enable lazy loading with dask (avoid loading entire file into RAM)
    # Empty dict {} means auto-chunk; or specify explicit chunk sizes
    ds = xr.open_dataset(path, chunks="auto")
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
    """Build daily dataset with lead_times expanded into time dimension.
    
    Each lead_time becomes a separate time index. The cube has no lead_time
    dimension; instead, each time slice indexes a (channel, lat, lon) cube.
    
    Time coordinate is stored as datetime64[ns].
    """
    data = open_standardized_file(nwp_file.path, requested_data_var)
    # data shape: (lead_time=24, channel, lat, lon)
    
    lead_times = data["lead_time"].values.astype("int64")
    target_times = (
        nwp_file.issue_date.normalize()
        + pd.Timedelta(days=1)
        + pd.to_timedelta(lead_times, unit="h")
    )

    # Reorder so lead_time becomes the first (time) dimension
    # Keep as DataArray (do NOT call .values) to preserve dask chunks
    cubes_data = data.transpose("lead_time", "channel", "lat", "lon")
    
    # Ensure time is datetime64[ns]
    target_times_ns = target_times.to_numpy(dtype="datetime64[ns]")
    issue_date_ns = np.repeat(
        nwp_file.issue_date.normalize().to_datetime64().astype("datetime64[ns]"),
        EXPECTED_LEAD_HOURS,
    )
    
    daily = xr.Dataset(
        data_vars={
            "cube": (
                ("time", "channel", "lat", "lon"),
                cubes_data.data,  # Use .data to get the underlying dask/numpy array
                {
                    "description": (
                        "Raw NWP forecast cube indexed by valid time. "
                        "Each cube has shape (channel, lat, lon). "
                        "Time coordinate is the target valid hour derived from "
                        "filename issue_date + 1 day + original lead_time."
                    )
                },
            )
        },
        coords={
            "time": ("time", target_times_ns),
            "issue_date": ("time", issue_date_ns),
            "channel": data["channel"].values,
            "lat": data["lat"].values,
            "lon": data["lon"].values,
        },
        attrs={
            "time_alignment_rule": "time = filename issue_date + 1 day + original lead_time hours",
            "note": "lead_time dimension removed; folded into time coordinate",
        },
    )

    return daily


def validate_combined_dataset(ds: xr.Dataset, files: Iterable[NwpFile]) -> None:
    """Validate using only the time coordinate—never touches cube data."""
    files = list(files)
    expected_rows = len(files) * EXPECTED_LEAD_HOURS
    if ds.sizes["time"] != expected_rows:
        raise ValueError(f"Expected {expected_rows} hourly rows, found {ds.sizes['time']}.")

    # .values on a coordinate is safe—coordinates are always small and already in RAM
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


def write_dataset_incremental(
    daily_datasets: list[xr.Dataset],
    output_path: Path,
    output_format: str,
    overwrite: bool,
) -> None:
    """Write daily datasets one at a time to avoid allocating the full array in RAM.

    For NetCDF: writes day 0 to create the file, then appends days 1..N.
    For Zarr: uses region writes to append along the time axis.
    Max RAM usage is one day's data at a time (~30 MB for 24 × 7 × 104 × 225 float64).
    
    Time coordinates are preserved as datetime64[ns] via explicit encoding.
    """
    ensure_can_write(output_path, overwrite)

    if output_format == "zarr":
        if importlib.util.find_spec("zarr") is None:
            raise ImportError(
                "Zarr output was requested but zarr is not installed. "
                "Use --output-format netcdf or install zarr."
            )
        
        # Encoding for datetime coordinates to preserve datetime64[ns]
        encoding = {
            "time": DATETIME_ENCODING,
            "issue_date": DATETIME_ENCODING,
        }
        
        mode = "w" if overwrite else "w-"
        for i, daily in enumerate(daily_datasets):
            if i == 0:
                daily.to_zarr(output_path, mode=mode, encoding=encoding)
            else:
                daily.to_zarr(output_path, append_dim="time", encoding=encoding)
        return

    if output_path.suffix == ".zarr":
        raise ValueError("NetCDF output cannot be written to a .zarr path.")

    encoding = {
        "cube": {
            "zlib": True,
            "complevel": 4,
            "chunksizes": (
                EXPECTED_LEAD_HOURS,                  # one full day per chunk
                daily_datasets[0].sizes["channel"],
                daily_datasets[0].sizes["lat"],
                daily_datasets[0].sizes["lon"],
            ),
        },
        "time": DATETIME_ENCODING,
        "issue_date": DATETIME_ENCODING,
    }

    # Write day 0 to create the file with correct schema, dimensions, and encoding
    daily_datasets[0].compute().to_netcdf(
        output_path,
        engine="netcdf4",
        encoding=encoding,
        unlimited_dims=["time"],
    )

    import netCDF4 as nc
    with nc.Dataset(output_path, "a") as ncf:
        for daily in daily_datasets[1:]:
            daily_computed = daily.compute()
            t0 = ncf.dimensions["time"].size  # index to start appending at
            n  = daily_computed.sizes["time"] # always EXPECTED_LEAD_HOURS

            # Coordinates: convert datetime64[ns] to int64 nanoseconds since epoch for netCDF4
            time_values = daily_computed["time"].values.astype("datetime64[ns]").astype("int64")
            issue_date_values = daily_computed["issue_date"].values.astype("datetime64[ns]").astype("int64")

            ncf["time"][t0 : t0 + n] = time_values
            ncf["issue_date"][t0 : t0 + n] = issue_date_values

            # Cube data: shape (n, channel, lat, lon)
            ncf["cube"][t0 : t0 + n, :, :, :] = daily_computed["cube"].values

            del daily_computed  # free RAM before next iteration


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

    # Build a lightweight combined dataset from coordinates only (no cube data loaded)
    # This is used purely for validation—the time coordinate is tiny
    combined_coords = xr.concat(
        [ds[["issue_date"]].assign_coords(time=ds["time"]) for ds in daily_datasets],
        dim="time",
        combine_attrs="drop",
    ).sortby("time")
    combined_coords.attrs.update(
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
            "cube_structure": "cube(time, channel, lat, lon) — each time indexes a (channel, lat, lon) array",
        }
    )

    # Validate coordinates before writing (coordinates are small, no cube data loaded)
    validate_combined_dataset(combined_coords, files)

    resolved_format = resolve_output_format(output_path, output_format)

    # Write one day at a time—max RAM usage is ~30 MB per iteration
    print(f"Writing {len(daily_datasets)} daily files incrementally...")
    write_dataset_incremental(daily_datasets, output_path, resolved_format, overwrite)

    first_ds = daily_datasets[0]
    last_ds = daily_datasets[-1]
    total_times = len(files) * EXPECTED_LEAD_HOURS
    cube_shape = (total_times, first_ds.sizes["channel"], first_ds.sizes["lat"], first_ds.sizes["lon"])

    print(f"\nWrote standardized NWP dataset: {output_path}")
    print(f"Output format: {resolved_format}")
    print(f"Rows/time steps: {total_times}")
    print(f"Time range: {first_ds['time'].values[0]} to {last_ds['time'].values[-1]}")
    print(f"Cube shape: {cube_shape}")
    print(f"time dtype: datetime64[ns]")

    # Return a lazy view of the written file with explicit datetime64[ns] enforcement
    opened = xr.open_dataset(output_path, chunks={"time": EXPECTED_LEAD_HOURS})
    
    # Verify and enforce datetime64[ns] on reload
    if opened["time"].dtype != "datetime64[ns]":
        opened["time"].values = opened["time"].values.astype("datetime64[ns]")
    if opened["issue_date"].dtype != "datetime64[ns]":
        opened["issue_date"].values = opened["issue_date"].values.astype("datetime64[ns]")
    
    return opened


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
