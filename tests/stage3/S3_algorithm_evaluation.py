"""Evaluate Stage 2 forecasts by realized Stage 3 dispatch profit.

For each target day, this script:

1. Uses the model-predicted price curve to choose the Stage 3 schedule.
2. Scores that schedule with the competition profit definition:

       profit = sum(P_t * E_t), t = 0..95

   where P_t is the actual price and E_t is -1000 for charging, +1000 for
   discharging, and 0 for no operation.
3. Compares the realized profit against the oracle schedule chosen from actual
   prices for the same day.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tests.stage3.S3_algorithm import (  # noqa: E402
    CHARGE_POWER,
    DISCHARGE_POWER,
    SLOTS_PER_DAY,
    optimize_one_day,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/Stage_2/dispatch_profit_evaluation"


@dataclass(frozen=True)
class PredictionInput:
    label: str
    path: Path


def parse_prediction_input(value: str) -> PredictionInput:
    if "=" in value:
        label, path_value = value.split("=", 1)
        label = label.strip()
        path = Path(path_value.strip())
    else:
        path = Path(value.strip())
        label = path.parent.name or path.stem

    if not label:
        raise ValueError(f"Could not parse a model label from {value!r}.")
    return PredictionInput(label=label, path=path)


def read_predictions(path: Path, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    df = pd.read_csv(path, dtype={"target_date": str})
    required_columns = {"split", "target_date", "slot_index", "price", "prediction"}
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f"{path} is missing required columns: {sorted(missing)}")

    selected = df[df["split"] == split].copy()
    if selected.empty:
        available = sorted(df["split"].dropna().unique().tolist())
        raise ValueError(
            f"{path} has no rows for split {split!r}. Available splits: {available}"
        )

    selected["slot_index"] = pd.to_numeric(selected["slot_index"], errors="raise").astype(int)
    selected["price"] = pd.to_numeric(selected["price"], errors="raise")
    selected["prediction"] = pd.to_numeric(selected["prediction"], errors="raise")
    selected = selected.sort_values(["target_date", "slot_index"]).reset_index(drop=True)
    return selected


def validate_daily_group(target_date: str, group: pd.DataFrame) -> None:
    if group.shape[0] != SLOTS_PER_DAY:
        raise ValueError(
            f"{target_date} has {group.shape[0]} rows; expected {SLOTS_PER_DAY}."
        )

    slot_index = group["slot_index"].to_numpy()
    expected_slot_index = np.arange(SLOTS_PER_DAY)
    if not np.array_equal(slot_index, expected_slot_index):
        raise ValueError(f"{target_date} does not have slot_index 0..95 in order.")


def profit_from_schedule(prices: np.ndarray, energy: np.ndarray) -> float:
    return float(np.sum(prices * energy))


def evaluate_one_model(
    model: PredictionInput,
    split: str,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    df = read_predictions(model.path, split)
    daily_rows: list[dict[str, float | int | str]] = []

    for target_date, group in df.groupby("target_date", sort=True):
        validate_daily_group(str(target_date), group)

        actual_prices = group["price"].to_numpy(dtype=float)
        predicted_prices = group["prediction"].to_numpy(dtype=float)

        (
            predicted_energy,
            predicted_charge_start,
            predicted_discharge_start,
            predicted_price_spread,
        ) = optimize_one_day(predicted_prices)
        (
            oracle_energy,
            oracle_charge_start,
            oracle_discharge_start,
            oracle_price_spread,
        ) = optimize_one_day(actual_prices)

        realized_profit = profit_from_schedule(actual_prices, predicted_energy)
        planned_profit = profit_from_schedule(predicted_prices, predicted_energy)
        oracle_profit = profit_from_schedule(actual_prices, oracle_energy)
        regret = oracle_profit - realized_profit

        daily_rows.append(
            {
                "model": model.label,
                "split": split,
                "target_date": str(target_date),
                "realized_profit": realized_profit,
                "planned_profit": planned_profit,
                "oracle_profit": oracle_profit,
                "regret": regret,
                "profit_ratio": np.nan
                if np.isclose(oracle_profit, 0.0)
                else realized_profit / oracle_profit,
                "predicted_charge_start": -1
                if predicted_charge_start is None
                else int(predicted_charge_start),
                "predicted_discharge_start": -1
                if predicted_discharge_start is None
                else int(predicted_discharge_start),
                "oracle_charge_start": -1
                if oracle_charge_start is None
                else int(oracle_charge_start),
                "oracle_discharge_start": -1
                if oracle_discharge_start is None
                else int(oracle_discharge_start),
                "predicted_price_spread": float(predicted_price_spread),
                "oracle_price_spread": float(oracle_price_spread),
                "selected_cycle": int(predicted_charge_start is not None),
                "oracle_selected_cycle": int(oracle_charge_start is not None),
            }
        )

    daily = pd.DataFrame(daily_rows)
    summary = summarize_daily_results(model.label, split, daily)
    return summary, daily_rows


def summarize_daily_results(
    model_label: str,
    split: str,
    daily: pd.DataFrame,
) -> dict[str, float | int | str]:
    total_profit = float(daily["realized_profit"].sum())
    oracle_total_profit = float(daily["oracle_profit"].sum())
    total_regret = float(daily["regret"].sum())
    planned_total_profit = float(daily["planned_profit"].sum())

    return {
        "model": model_label,
        "split": split,
        "days": int(daily.shape[0]),
        "selected_cycles": int(daily["selected_cycle"].sum()),
        "oracle_selected_cycles": int(daily["oracle_selected_cycle"].sum()),
        "total_profit": total_profit,
        "avg_daily_profit": float(daily["realized_profit"].mean()),
        "planned_total_profit": planned_total_profit,
        "avg_planned_daily_profit": float(daily["planned_profit"].mean()),
        "oracle_total_profit": oracle_total_profit,
        "avg_oracle_daily_profit": float(daily["oracle_profit"].mean()),
        "total_regret": total_regret,
        "avg_daily_regret": float(daily["regret"].mean()),
        "profit_capture_ratio": np.nan
        if np.isclose(oracle_total_profit, 0.0)
        else total_profit / oracle_total_profit,
        "avg_daily_profit_ratio": float(daily["profit_ratio"].mean()),
    }


def write_outputs(
    summaries: list[dict[str, float | int | str]],
    daily_rows: list[dict[str, float | int | str]],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    daily_path = output_dir / "daily_profit.csv"

    if not summaries:
        raise ValueError("No summaries to write.")
    if not daily_rows:
        raise ValueError("No daily rows to write.")

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    with daily_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(daily_rows[0].keys()))
        writer.writeheader()
        writer.writerows(daily_rows)

    return summary_path, daily_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 2 predictions with Stage 3 dispatch profit."
    )
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help=(
            "Prediction file to evaluate. Use label=path or just path. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Prediction split to evaluate. Default: val.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_inputs = [parse_prediction_input(value) for value in args.prediction]

    summaries = []
    daily_rows = []
    for prediction_input in prediction_inputs:
        summary, model_daily_rows = evaluate_one_model(prediction_input, args.split)
        summaries.append(summary)
        daily_rows.extend(model_daily_rows)

    summary_path, daily_path = write_outputs(summaries, daily_rows, args.output_dir)

    print("Dispatch profit summary:")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"Summary written to: {summary_path}")
    print(f"Daily profit written to: {daily_path}")
    print(f"Charge energy value: {CHARGE_POWER}")
    print(f"Discharge energy value: {DISCHARGE_POWER}")


if __name__ == "__main__":
    main()
