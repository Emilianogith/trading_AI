# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:49:46 2024

@author: rcpal
"""

import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import MetaTrader5 as mt5
import os



# ============================================================
# MT5 CONNECTION
# ============================================================

if not mt5.initialize(
    login=1514907582,
    server="FTMO-Demo",
    password="5*@rh2Z*q?7lC"
):
    print("initialize() failed, error code =", mt5.last_error())
    quit()




# ============================================================
# PARAMETERS
# ============================================================

SYMBOL = "AUDCHF"
NUMBER_OF_BARS = 1000

# Present candle must open at least 0.70%
# beyond the support/resistance closing price
BREAKOUT_PERCENT = 0.007

# Candle -1 must open at least 0.65%
# away from the maximum resistance high
# or minimum support low
CANDLE_MINUS_1_DISTANCE = 0.0065

# Maximum allowed retracement of candle -1
RETRACEMENT_PERCENT = 0.75

timezone = pytz.timezone("Etc/UTC")

utc_from = datetime(
    2026,
    8,
    21,
    tzinfo=timezone
)


# ============================================================
# DOWNLOAD WEEKLY DATA
# ============================================================

rates = mt5.copy_rates_from(
    SYMBOL,
    mt5.TIMEFRAME_W1,
    utc_from,
    NUMBER_OF_BARS
)

if rates is None:
    print(
        "Could not retrieve data:",
        mt5.last_error()
    )

    mt5.shutdown()
    quit()


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(rates)

df["time"] = pd.to_datetime(
    df["time"],
    unit="s",
    utc=True
)

df = df.sort_values(
    "time"
).reset_index(
    drop=True
)


# ============================================================
# MEAN PRICE OF EACH CANDLE
# ============================================================
#
# Mean price = (Open + Close) / 2
#
# Used for:
#
#   - average_11_20
#   - average_21_30
#

df["mean_price"] = (
    df["open"] + df["close"]
) / 2


# ============================================================
# CREATE DATASET
# ============================================================

dataset = []

target_upside = []

target_downside = []

dates = []


# ============================================================
# MAIN LOOP
# ============================================================
#
# We need:
#
# -30 ... -1
#
# because:
#
# Input:
#   -10 ... -1
#
# Additional information:
#   -11 ... -20
#   -21 ... -30
#
# Target:
#   present candle
#   +1
#   +2
#
# ============================================================

for i in range(52, len(df) - 2):


    # ========================================================
    # INPUT FEATURES
    # ========================================================

    row = []


    # --------------------------------------------------------
    # PREVIOUS 10 WEEKLY CANDLES
    #
    # -10 ... -1
    #
    # Each candle:
    #
    # Open
    # High
    # Low
    # Close
    # --------------------------------------------------------

    for j in range(i - 52, i):

        row.extend([
            df.loc[j, "open"],
            df.loc[j, "high"],
            df.loc[j, "low"],
            df.loc[j, "close"]
        ])


    # --------------------------------------------------------
    # AVERAGE MEAN PRICE -11 TO -20
    # --------------------------------------------------------

    average_11_20 = df.loc[
        i - 20:i - 11,
        "mean_price"
    ].mean()

    


    # --------------------------------------------------------
    # AVERAGE MEAN PRICE -21 TO -30
    # --------------------------------------------------------

    average_21_30 = df.loc[
        i - 30:i - 21,
        "mean_price"
    ].mean()

    


    # ========================================================
    # COMMON TARGET DATA
    # ========================================================

    present_open = df.loc[
        i,
        "open"
    ]
    # ========================================================
    # AVERAGE -11 TO -20 CONDITION
    # ========================================================
    
    # UPSIDE:
    # Average price of candles -11 to -20
    # must not be higher than today's opening price.
    
    upside_average_condition = (
        average_11_20 <= present_open
    )
    
    
    # DOWNSIDE:
    # Average price of candles -11 to -20
    # must not be lower than today's opening price.
    
    downside_average_condition = (
        average_11_20 >= present_open
    )


    # --------------------------------------------------------
    # CANDLES -2 TO -10
    #
    # These candles are candidates for creating
    # support/resistance.
    #
    # Candle -1 is excluded.
    # --------------------------------------------------------

    previous_2_to_10 = df.iloc[
        i - 10:i - 1
    ]


    # --------------------------------------------------------
    # CANDLE -1
    # --------------------------------------------------------

    candle_minus_1_open = df.loc[
        i - 1,
        "open"
    ]

    candle_minus_1_close = df.loc[
        i - 1,
        "close"
    ]


    # --------------------------------------------------------
    # PRESENT + NEXT 2 CANDLES
    # --------------------------------------------------------

    three_candles = df.iloc[
        i:i + 3
    ]


    # ========================================================
    # ======================= UPSIDE =========================
    # ========================================================

    # --------------------------------------------------------
    # FIND VALID RESISTANCE CANDLES
    #
    # A red candle is a valid resistance candle ONLY when
    # the candle immediately before it is green.
    #
    # Valid pattern:
    #
    #       GREEN -> RED
    #                    ^
    #                    |
    #              resistance
    #
    # Candidate candles are -2 through -10.
    # --------------------------------------------------------

    red_candles_list = []


    for j in range(
        i - 10,
        i - 1
    ):

        current_candle = df.loc[
            j
        ]

        previous_candle = df.loc[
            j - 1
        ]


        # ----------------------------------------------------
        # Current candle must be RED
        #
        # Close < Open
        # ----------------------------------------------------

        current_is_red = (
            current_candle["close"]
            < current_candle["open"]
        )


        # ----------------------------------------------------
        # Previous candle must be GREEN
        #
        # Close > Open
        # ----------------------------------------------------

        previous_is_green = (
            previous_candle["close"]
            > previous_candle["open"]
        )


        # ----------------------------------------------------
        # GREEN -> RED
        # ----------------------------------------------------

        if (
            current_is_red
            and previous_is_green
        ):

            red_candles_list.append(
                j
            )


    # Create DataFrame containing only valid
    # resistance candles

    red_candles = df.loc[
        red_candles_list
    ]


    # --------------------------------------------------------
    # CHECK IF VALID RESISTANCE EXISTS
    # --------------------------------------------------------

    if len(red_candles) > 0:


        # ----------------------------------------------------
        # RESISTANCE
        #
        # Highest CLOSE among valid red candles.
        # ----------------------------------------------------

        highest_resistance_close = (
            red_candles["close"].max()
        )


        # ----------------------------------------------------
        # MAXIMUM HIGH
        #
        # Highest HIGH among valid red candles.
        #
        # This is used for the candle -1 opening condition.
        # ----------------------------------------------------

        highest_red_high = (
            red_candles["high"].max()
        )


        # ----------------------------------------------------
        # PRESENT CANDLE OPENING CONDITION
        #
        # Present candle must open at least +0.70%
        # above the resistance close.
        # ----------------------------------------------------

        upside_opening_condition = (
            present_open
            >=
            highest_resistance_close
            * (1 + BREAKOUT_PERCENT)
        )


        # ----------------------------------------------------
        # CANDLE -1 OPENING CONDITION
        #
        # Candle -1 must open at least 0.65%
        # BELOW the maximum high of the valid
        # resistance candles.
        # ----------------------------------------------------

        upside_minus_1_open_condition = (
            candle_minus_1_open
            <=
            highest_red_high
            * (1 - CANDLE_MINUS_1_DISTANCE)
        )


        # ----------------------------------------------------
        # 75% INPUT RANGE CONDITION
        #
        # The highest high of the valid resistance candles
        # must not already be above 75% of the distance
        # between resistance and today's opening price.
        # ----------------------------------------------------

        upside_distance = (
            present_open
            - highest_resistance_close
        )


        upside_75_level = (
            highest_resistance_close
            +
            upside_distance
            * RETRACEMENT_PERCENT
        )


        upside_input_condition = (
            highest_red_high
            <=
            upside_75_level
        )


    else:

        # No valid alternating red candle exists.

        highest_resistance_close = np.nan

        highest_red_high = np.nan

        upside_75_level = np.nan

        upside_opening_condition = False

        upside_minus_1_open_condition = False

        upside_input_condition = False


    # --------------------------------------------------------
    # CANDLE -1 MUST BE BULLISH
    # --------------------------------------------------------

    upside_movement = (
        candle_minus_1_close
        -
        candle_minus_1_open
    )


    upside_candle_condition = (
        upside_movement > 0
    )


    # --------------------------------------------------------
    # MAXIMUM 75% RETRACEMENT
    #
    # Candle -1 is the upside movement.
    #
    # Allowed retracement:
    #
    # 75% of candle -1 movement.
    # --------------------------------------------------------

    upside_minimum_price = (
        candle_minus_1_close
        -
        upside_movement
        * RETRACEMENT_PERCENT
    )


    # --------------------------------------------------------
    # Present candle + next 2 candles
    #
    # Their LOW cannot go below the allowed
    # retracement level.
    # --------------------------------------------------------

    upside_retracement_condition = (
        three_candles["low"]
        >=
        upside_minimum_price
    ).all()


    # --------------------------------------------------------
    # FINAL UPSIDE TARGET
    # --------------------------------------------------------

    if (
        upside_opening_condition
        and upside_minus_1_open_condition
        and upside_candle_condition
        and upside_input_condition
        and upside_retracement_condition
        and upside_average_condition
    ):

        target_up = 1

    else:

        target_up = 0


    # ========================================================
    # ====================== DOWNSIDE ========================
    # ========================================================

    # --------------------------------------------------------
    # FIND VALID SUPPORT CANDLES
    #
    # A green candle is a valid support candle ONLY when
    # the candle immediately before it is red.
    #
    # Valid pattern:
    #
    #       RED -> GREEN
    #                    ^
    #                    |
    #                  support
    #
    # Candidate candles are -2 through -10.
    # --------------------------------------------------------

    green_candles_list = []


    for j in range(
        i - 10,
        i - 1
    ):

        current_candle = df.loc[
            j
        ]

        previous_candle = df.loc[
            j - 1
        ]


        # ----------------------------------------------------
        # Current candle must be GREEN
        #
        # Close > Open
        # ----------------------------------------------------

        current_is_green = (
            current_candle["close"]
            > current_candle["open"]
        )


        # ----------------------------------------------------
        # Previous candle must be RED
        #
        # Close < Open
        # ----------------------------------------------------

        previous_is_red = (
            previous_candle["close"]
            < previous_candle["open"]
        )


        # ----------------------------------------------------
        # RED -> GREEN
        # ----------------------------------------------------

        if (
            current_is_green
            and previous_is_red
        ):

            green_candles_list.append(
                j
            )


    # Create DataFrame containing only valid
    # support candles

    green_candles = df.loc[
        green_candles_list
    ]


    # --------------------------------------------------------
    # CHECK IF VALID SUPPORT EXISTS
    # --------------------------------------------------------

    if len(green_candles) > 0:


        # ----------------------------------------------------
        # SUPPORT
        #
        # Lowest CLOSE among valid green candles.
        # ----------------------------------------------------

        lowest_support_close = (
            green_candles["close"].min()
        )


        # ----------------------------------------------------
        # MINIMUM LOW
        #
        # Lowest LOW among valid green candles.
        #
        # This is used for the candle -1 opening condition.
        # ----------------------------------------------------

        lowest_green_low = (
            green_candles["low"].min()
        )


        # ----------------------------------------------------
        # PRESENT CANDLE OPENING CONDITION
        #
        # Present candle must open at least -0.70%
        # below the support close.
        # ----------------------------------------------------

        downside_opening_condition = (
            present_open
            <=
            lowest_support_close
            * (1 - BREAKOUT_PERCENT)
        )


        # ----------------------------------------------------
        # CANDLE -1 OPENING CONDITION
        #
        # Candle -1 must open at least 0.65%
        # ABOVE the minimum low of the valid
        # support candles.
        # ----------------------------------------------------

        downside_minus_1_open_condition = (
            candle_minus_1_open
            >=
            lowest_green_low
            * (1 + CANDLE_MINUS_1_DISTANCE)
        )


        # ----------------------------------------------------
        # 75% INPUT RANGE CONDITION
        #
        # The lowest low of the valid support candles
        # must not already be below 75% of the distance
        # between support and today's opening price.
        # ----------------------------------------------------

        downside_distance = (
            lowest_support_close
            -
            present_open
        )


        downside_75_level = (
            lowest_support_close
            -
            downside_distance
            * RETRACEMENT_PERCENT
        )


        downside_input_condition = (
            lowest_green_low
            >=
            downside_75_level
        )


    else:

        # No valid alternating green candle exists.

        lowest_support_close = np.nan

        lowest_green_low = np.nan

        downside_75_level = np.nan

        downside_opening_condition = False

        downside_minus_1_open_condition = False

        downside_input_condition = False


    # --------------------------------------------------------
    # CANDLE -1 MUST BE BEARISH
    # --------------------------------------------------------

    downside_movement = (
        candle_minus_1_open
        -
        candle_minus_1_close
    )


    downside_candle_condition = (
        downside_movement > 0
    )


    # --------------------------------------------------------
    # MAXIMUM 75% RETRACEMENT
    #
    # Candle -1 is the downside movement.
    # --------------------------------------------------------

    downside_maximum_price = (
        candle_minus_1_close
        +
        downside_movement
        * RETRACEMENT_PERCENT
    )


    # --------------------------------------------------------
    # Present candle + next 2 candles
    #
    # Their HIGH cannot go above the allowed
    # retracement level.
    # --------------------------------------------------------

    downside_retracement_condition = (
        three_candles["high"]
        <=
        downside_maximum_price
    ).all()


    # --------------------------------------------------------
    # FINAL DOWNSIDE TARGET
    # --------------------------------------------------------

    if (
        downside_opening_condition
        and downside_minus_1_open_condition
        and downside_candle_condition
        and downside_input_condition
        and downside_retracement_condition
        and downside_average_condition
    ):

        target_down = 1

    else:

        target_down = 0


    # ========================================================
    # SAVE
    # ========================================================

    dataset.append(
        row
    )

    target_upside.append(
        target_up
    )

    target_downside.append(
        target_down
    )

    dates.append(
        df.loc[i, "time"]
    )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

dataset = np.array(
    dataset
)

target_upside = np.array(
    target_upside
)

target_downside = np.array(
    target_downside
)


# ============================================================
# ADD TARGETS
# ============================================================

dataset_with_targets = np.column_stack(
    (
        dataset,
        target_upside,
        target_downside
    )
)


# ============================================================
# COLUMN NAMES
# ============================================================

columns = []


for n in range(52, 0, -1):

    columns.extend([
        f"open_{n}",
        f"high_{n}",
        f"low_{n}",
        f"close_{n}"
    ])


columns.extend([
    
    "target_upside",
    "target_downside"
])

# ============================================================
# MANUAL TARGET CONTROL BY CANDLE DATE
# ============================================================

FORCE_UPSIDE_ZERO = [
    # "2025-03-10",
]

FORCE_DOWNSIDE_ZERO = [
    # "2025-04-14",
]

FORCE_BOTH_ZERO = [
    # "2025-05-19",
]

# Normalize dates for matching
dates_normalized = pd.to_datetime(dates, utc=True).strftime("%Y-%m-%d")

for i, candle_date in enumerate(dates_normalized):
    if candle_date in FORCE_UPSIDE_ZERO:
        dataset_with_targets[i][-2] = 0

    if candle_date in FORCE_DOWNSIDE_ZERO:
        dataset_with_targets[i][-1] = 0

    if candle_date in FORCE_BOTH_ZERO:
        dataset_with_targets[i][-2] = 0
        dataset_with_targets[i][-1] = 0
        
# ============================================================
# CREATE FINAL DATAFRAME
# ============================================================

final_dataset = pd.DataFrame(
    dataset_with_targets,
    columns=columns
)


# ============================================================
# ADD PRESENT CANDLE DATE
# ============================================================

final_dataset.insert(
    0,
    "present_candle_date",
    dates
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("DATASET RESULTS")
print("========================================")

print("\nDataset shape:")
print(
    final_dataset.shape
)


print("\nNumber of input features:")
print(
    42
)


print("\nNumber of upside targets = 1:")
print(
    (
        final_dataset["target_upside"]
        == 1
    ).sum()
)


print("\nNumber of downside targets = 1:")
print(
    (
        final_dataset["target_downside"]
        == 1
    ).sum()
)


print("\nNumber of rows with both targets = 1:")
print(
    (
        (final_dataset["target_upside"] == 1)
        &
        (final_dataset["target_downside"] == 1)
    ).sum()
)


# ============================================================
# DISPLAY UPSIDE TARGETS
# ============================================================

print("\n========================================")
print("UPSIDE TARGETS")
print("========================================")

print(
    final_dataset[
        final_dataset["target_upside"] == 1
    ][
        [
            "present_candle_date",
            "target_upside"
        ]
    ]
)


# ============================================================
# DISPLAY DOWNSIDE TARGETS
# ============================================================

print("\n========================================")
print("DOWNSIDE TARGETS")
print("========================================")

print(
    final_dataset[
        final_dataset["target_downside"] == 1
    ][
        [
            "present_candle_date",
            "target_downside"
        ]
    ]
)


# ============================================================
# SAVE DATASET
# ============================================================

final_dataset.to_csv(
    f"{SYMBOL}_weekly_dataset.csv",
    index=False
)
filename = f"{SYMBOL}_weekly_dataset.csv"

final_dataset.to_csv(
    filename,
    index=False
)

print("Dataset salvato in:")
print(os.path.abspath(filename))

print("\nDataset saved as:")
print(
    f"{SYMBOL}_weekly_dataset.csv"
)


# ============================================================
# CLOSE MT5
# ============================================================

mt5.shutdown()