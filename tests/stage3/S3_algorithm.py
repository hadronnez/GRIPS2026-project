import pandas as pd
import numpy as np
from pathlib import Path

ROLLING_WINDOW_LENGTH = 8
SLOTS_PER_DAY = 96
CHARGE_START_MIN = 0
CHARGE_START_MAX = 80
DISCHARGE_START_MIN = ROLLING_WINDOW_LENGTH
DISCHARGE_START_MAX = 88

CHARGE_POWER = -1000.0
DISCHARGE_POWER = 1000.0

INPUT_PATH = Path(
    "output/raw_price_standardized/price_long.csv"
)
OUTPUT_PATH = Path(
    "output/battery_optimized/price_long_with_power.csv"
)

def optimize_one_day(prices):
    """
    Find the best charge/discharge cycle for one 96-slot day.
    Charging:
        8 consecutive entries of -1000.0
    Discharging:
        8 consecutive entries of +1000.0
    The discharge window must start after the charging window finishes.
    If there is no positive price advantage, the returned power array
    contains only zeros.
    """
    prices = np.asarray(prices, dtype=float)

    # Error detection
    if len(prices) != SLOTS_PER_DAY:
        raise ValueError(
            f"Expected {SLOTS_PER_DAY} prices for one day, "
            f"but received {len(prices)}."
        )

    if not np.isfinite(prices).all():
        raise ValueError(
            "The price data contains missing or infinite values."
        )

    # rolling_sums[start] is the sum of:
    # prices[start : start + 8]
    #
    # For 96 values, there are 89 valid window starts:
    # 0, 1, ..., 88.
    rolling_sums = np.array(
        [
            prices[
                start : start + ROLLING_WINDOW_LENGTH
            ].sum()
            for start in range(DISCHARGE_START_MAX + 1)
        ],
        dtype=float,
    )

    # Starting at zero means that a cycle is selected only when
    # discharge_sum - charge_sum is positive.
    best_difference = 0.0
    best_charge_start = None
    best_discharge_start = None

    for charge_start in range(
        CHARGE_START_MIN,
        CHARGE_START_MAX + 1,
    ):
        first_allowed_discharge = max(
            DISCHARGE_START_MIN,
            charge_start + ROLLING_WINDOW_LENGTH,
        )

        for discharge_start in range(
            first_allowed_discharge,
            DISCHARGE_START_MAX + 1,
        ):
            difference = (
                rolling_sums[discharge_start]
                - rolling_sums[charge_start]
            )

            if difference > best_difference:
                best_difference = float(difference)
                best_charge_start = charge_start
                best_discharge_start = discharge_start

    # Initially, the battery does nothing during the entire day.
    power = np.zeros(SLOTS_PER_DAY, dtype=float)

    # A cycle is inserted only if a positive difference was found.
    if best_charge_start is not None:
        power[best_charge_start : (best_charge_start + ROLLING_WINDOW_LENGTH)] = CHARGE_POWER
        power[best_discharge_start : (best_discharge_start + ROLLING_WINDOW_LENGTH)] = DISCHARGE_POWER

    return (
        power,
        best_charge_start,
        best_discharge_start,
        best_difference,
    )

def optimize_all_days(prices):
    prices = np.asarray(prices, dtype=float)

    if len(prices) == 0:
        return np.array([], dtype=float)

    if len(prices) % SLOTS_PER_DAY != 0:
        raise ValueError(
            f"The input contains {len(prices)} rows. "
            f"The number of rows must be divisible by "
            f"{SLOTS_PER_DAY}."
        )

    # One output value for every input row.
    power = np.zeros(len(prices), dtype=float)
    number_of_days = len(prices) // SLOTS_PER_DAY

    for day_number in range(number_of_days):
        start = day_number * SLOTS_PER_DAY
        end = start + SLOTS_PER_DAY

        daily_prices = prices[start:end]

        (
            daily_power,
            charge_start,
            discharge_start,
            difference,
        ) = optimize_one_day(daily_prices)

        if len(daily_power) != SLOTS_PER_DAY:
            raise ValueError(
                f"Day {day_number + 1}: optimize_one_day() "
                f"returned {len(daily_power)} values instead of "
                f"{SLOTS_PER_DAY}."
            )

        # Put this day's result into the corresponding positions.
        power[start:end] = daily_power

        if charge_start is None:
            print(
                f"Day {day_number + 1}: no cycle selected."
            )
        else:
            print(
                f"Day {day_number + 1}: "
                f"charge start = {charge_start}, "
                f"discharge start = {discharge_start}, "
                f"difference = {difference}"
            )

    return power

def main():
    data = pd.read_csv(INPUT_PATH)

    required_columns = {"time", "price"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise KeyError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    prices = pd.to_numeric(
        data["price"],
        errors="raise",
    ).to_numpy(dtype=float)

    power = optimize_all_days(prices)

    # Copy the input rows in exactly their existing order.
    output = data.copy()

    # power[i] is assigned to output row i.
    output["power"] = power

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Output written to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()