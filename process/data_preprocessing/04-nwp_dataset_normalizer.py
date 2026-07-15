"""Normalize the standardized NWP cube into a single merged train+test NetCDF dataset.

The input dataset is expected to contain:

    cube(time, channel, lat, lon)

with channel labels:

    ghi, sp, t2m, tcc, tp, u100, v100

For each split (train/test), normalization parameters are calculated only
from that split's own data. Channels with a mean-map step are centered
first, then normalized with one scalar min and one scalar max over the
whole split/channel grid. The two normalized splits are then concatenated
along time into a single output dataset, with a `split` coordinate marking
which split each time step came from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dask
import numpy as np
import xarray as xr


DEFAULT_INPUT_PATH = Path("output/data_preprocessing/clean_datasets/clean_nwp_dataset.nc")
DEFAULT_OUTPUT_DIR = Path("output/data_preprocessing/normalized_datasets/")

SPLITS = {
    "train": ("2025-01-02", "2025-10-31T23:00:00"),
    "test": ("2025-11-01", "2025-12-31T23:00:00"),
}

EXPECTED_CHANNEL_ORDER = ["ghi", "sp", "t2m", "tcc", "tp", "u100", "v100"]
MEAN_SUBTRACT_CHANNELS = ["sp", "t2m", "u100", "v100"]
MINMAX_ONLY_CHANNELS = ["ghi", "tp"]
UNCHANGED_CHANNELS = ["tcc"]
NORMALIZED_CHANNELS = ["ghi", "tp", "u100", "v100"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize raw_nwp_standardized.nc into a single merged NetCDF file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input standardized NWP NetCDF. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the normalized output. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing normalized output file.",
    )
    return parser.parse_args()


def ensure_expected_channels(ds: xr.Dataset) -> None:
    actual = [str(value) for value in ds["channel"].values.tolist()]
    if actual != EXPECTED_CHANNEL_ORDER:
        raise ValueError(
            "Unexpected channel coordinate. Expected "
            f"{EXPECTED_CHANNEL_ORDER}, found {actual}."
        )


def ensure_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Pass --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)


def scalar_minmax(data: xr.DataArray) -> tuple[float, float]:
    min_value, max_value = dask.compute(data.min(skipna=True), data.max(skipna=True))
    min_float = float(min_value.values)
    max_float = float(max_value.values)
    if not np.isfinite(min_float) or not np.isfinite(max_float):
        raise ValueError("Min/max calculation produced a non-finite value.")
    return min_float, max_float


def minmax_normalize(data: xr.DataArray, min_value: float, max_value: float) -> xr.DataArray:
    denominator = max_value - min_value
    if np.isclose(denominator, 0.0):
        return xr.zeros_like(data)
    return (data - min_value) / denominator


def normalize_channel(split_cube: xr.DataArray, channel_name: str) -> xr.DataArray:
    source = split_cube.sel(channel=channel_name)

    if channel_name in UNCHANGED_CHANNELS:
        return source

    if channel_name in MEAN_SUBTRACT_CHANNELS:
        mean_map = source.mean(dim="time", skipna=True).compute()
        data_for_minmax = source - mean_map
    elif channel_name in MINMAX_ONLY_CHANNELS:
        data_for_minmax = source
    else:
        raise ValueError(f"No normalization rule configured for channel {channel_name!r}.")

    min_value, max_value = scalar_minmax(data_for_minmax)
    return minmax_normalize(data_for_minmax, min_value, max_value)


def write_netcdf(ds: xr.Dataset, path: Path, overwrite: bool) -> None:
    ensure_can_write(path, overwrite)
    encoding = {
        "cube": {
            "zlib": True,
            "complevel": 4,
            "chunksizes": (
                min(24, ds.sizes["time"]),
                ds.sizes["channel"],
                ds.sizes["lat"],
                ds.sizes["lon"],
            ),
        }
    }
    ds.to_netcdf(path, engine="netcdf4", encoding=encoding)


def normalize_split(
    source_ds: xr.Dataset,
    split_name: str,
    start_time: str,
    end_time: str,
) -> xr.Dataset:
    """Normalize a single split and return its dataset (not yet written to disk)."""
    split_ds = source_ds.sel(time=slice(start_time, end_time))
    if split_ds.sizes["time"] == 0:
        raise ValueError(f"Split {split_name!r} selected no time steps.")

    normalized_channels = []
    for channel_name in EXPECTED_CHANNEL_ORDER:
        print(f"[{split_name}] processing channel {channel_name}", flush=True)
        normalized = normalize_channel(split_ds["cube"], channel_name)
        normalized_channels.append(normalized.expand_dims(channel=[channel_name]))

    normalized_cube = xr.concat(normalized_channels, dim="channel").transpose(
        "time", "channel", "lat", "lon"
    )

    normalized_ds = xr.Dataset(
        data_vars={
            "cube": (
                normalized_cube.dims,
                normalized_cube.data,
                {
                    "description": "Normalized NWP cube.",
                },
            )
        },
        coords={
            "time": split_ds["time"],
            "issue_date": split_ds["issue_date"],
            "channel": normalized_cube["channel"],
            "lat": split_ds["lat"],
            "lon": split_ds["lon"],
            # Tag each time step with the split it came from so the merged
            # dataset can still be filtered back out (ds.sel(time=ds.split == "train")).
            "split": ("time", np.full(split_ds.sizes["time"], split_name, dtype=object)),
        },
    )

    return normalized_ds


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input NetCDF does not exist: {args.input}")

    source_ds = xr.open_dataset(args.input, chunks={})
    ensure_expected_channels(source_ds)

    split_datasets: list[xr.Dataset] = []
    for split_name, (start_time, end_time) in SPLITS.items():
        normalized_ds = normalize_split(
            source_ds=source_ds,
            split_name=split_name,
            start_time=start_time,
            end_time=end_time,
        )
        split_datasets.append(normalized_ds)

    # Merge the normalized splits into a single dataset, concatenated along time.
    # Each time step keeps a `split` coordinate so train/test rows can still be
    # recovered later, e.g. merged_ds.sel(time=merged_ds.split == "train").
    print("merging train/test splits", flush=True)
    merged_ds = xr.concat(split_datasets, dim="time").sortby("time")
    merged_ds.attrs.update(
        {
            **source_ds.attrs,
            "description": "Normalized NWP dataset (train+test merged).",
            "source_dataset": str(args.input),
            "normalization_note": (
                "ghi/tp use split-specific scalar min-max over all time/lat/lon; "
                "sp/t2m/u100/v100 subtract split-specific time mean maps first, "
                "then use scalar min-max over all time/lat/lon; tcc is unchanged. "
                "Normalization parameters were computed independently per split. "
                "The `split` coordinate marks which split each time step belongs to."
            ),
        }
    )

    output_path = args.output_dir / "normalized_nwp_dataset.nc"
    print(f"writing merged dataset to {output_path}", flush=True)
    write_netcdf(merged_ds, output_path, args.overwrite)
    source_ds.close()

    print(f"Wrote merged normalized output to {output_path}", flush=True)


if __name__ == "__main__":
    main()