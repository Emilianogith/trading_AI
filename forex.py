# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:49:46 2024

@author: rcpal
"""
import pandas as pd
import pytz
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import MetaTrader5 as mt5
 
# establish MetaTrader 5 connection to a specified trading account
if not mt5.initialize(login=1510239402, server="FTMO-Demo",password="dms6H!td"):
    print("initialize() failed, error code =",mt5.last_error())
    quit()
    
def discard_columns(matrix):
    # Check if the matrix has enough columns
    if len(matrix[0]) < 6:
        raise ValueError("The matrix must have at least 6 columns")
    
    # Discard the first 3 and the last 3 columns
    modified_matrix = [row[1:-3] for row in matrix]
    
    return modified_matrix
def create_row_matrix(matrix):
    # Ensure the matrix has at least 10 rows
    if len(matrix) < 10:
        raise ValueError("The matrix must have at least 10 rows")
    if len(matrix[0]) != 4:
        raise ValueError("The matrix must have exactly 4 columns")

    # Create the new matrix
    new_matrix = []
    for i in range(9, len(matrix)):
        new_row = []
        for j in range(i-9, i):
            new_row.extend(matrix[j])
          # Divide the new row by the first column value of the current row
        first_column_value = matrix[i][0]
        if first_column_value == 0:
            raise ValueError(f"Division by zero encountered in row {i}")

        new_row = [((value / first_column_value)*100)-100 for value in new_row]
        new_matrix.append(new_row)

    return new_matrix
# set time zone to UTC
timezone = pytz.timezone("Etc/UTC")
# create 'datetime' object in UTC time zone to avoid the implementation of a local time zone offset
utc_from = datetime(2023, 7, 24, tzinfo=timezone)
# get 10 EURUSD D1 bars starting from 01.10.2020 in UTC time zone
rates = mt5.copy_rates_from("AUDCHF", mt5.TIMEFRAME_D1, utc_from, 100)
rates2 = mt5.copy_rates_from("AUDNZD", mt5.TIMEFRAME_D1, utc_from, 100)
rates3 = mt5.copy_rates_from("EURGBP", mt5.TIMEFRAME_D1, utc_from, 100)

# create DataFrame out of the obtained data
rates_frame = pd.DataFrame(rates)
# convert time in seconds into the datetime format
rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
# Convert DataFrame to NumPy matrix
matrix = rates_frame.to_numpy()
dataset=pd.DataFrame(discard_columns(matrix))
datasetused= dataset.to_numpy()
#target vector
# Initialize the vector with 100 zeros
target = [0] * 91
# Set specific positions to 1, riga -9 per target
positions = [25,57,60,81]
for pos in positions:
    target[pos] = 1
target1= np.array(target)   
#Adding the vector as a new column
dataset36eurusd=create_row_matrix(datasetused)
dataset36targeteurusd = np.column_stack((dataset36eurusd, target1))

#######GBPJPY

rates1 = mt5.copy_rates_from("NZDUSD", mt5.TIMEFRAME_D1, utc_from, 100)
# create DataFrame out of the obtained data
rates_framenzdchf = pd.DataFrame(rates1)
# convert time in seconds into the datetime format
rates_framenzdchf['time']=pd.to_datetime(rates_framenzdchf['time'], unit='s')
# Convert DataFrame to NumPy matrix
matrixnzdchf = rates_framenzdchf.to_numpy()
dataset=pd.DataFrame(discard_columns(matrixnzdchf))
datasetusednzdchf= dataset.to_numpy()
dataset36nzdchf=create_row_matrix(datasetusednzdchf)

#target vector
# Initialize the vector with 100 zeros
targetnzdchf = [0] * 91
# Set specific positions to 1
positionsnzdchf = [23, 31, 78]
for pos in positionsnzdchf:
    targetnzdchf[pos] = 1
target1nzdchf= np.array(targetnzdchf)  
dataset36targetnzdchf = np.column_stack((dataset36nzdchf, target1nzdchf))
# display data
print("\nDisplay dataframe with data")
print(rates_frame)  

#%%

import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import MetaTrader5 as mt5


# ============================================================
# MT5 CONNECTION
# ============================================================

if not mt5.initialize(
    login=1514371841,
    server="FTMO-Demo",
    password="@2$Sz9lq*?"
):
    print("initialize() failed, error code =", mt5.last_error())
    quit()


# ============================================================
# PARAMETERS
# ============================================================

SYMBOL = "AUDCHF"
NUMBER_OF_BARS = 200

timezone = pytz.timezone("Etc/UTC")

utc_from = datetime(
    2026, 8, 21,
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
    print("Could not retrieve data:", mt5.last_error())
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

df = df.sort_values("time").reset_index(drop=True)


# ============================================================
# MEAN PRICE OF EACH CANDLE
# ============================================================
#
# Mean price = (Open + Close) / 2
#
# This is used only for the two additional input features.
#

df["mean_price"] = (
    df["open"] + df["close"]
) / 2


# ============================================================
# CREATE INPUT DATASET + TARGET
# ============================================================

dataset = []
targets = []


# We need:
#
# Input:
#   -10 to -1  = 10 weekly candles
#
# Additional information:
#   -11 to -20
#   -21 to -30
#
# Target:
#   Present candle
#   +1 candle
#   +2 candle
#
# Therefore we need at least 30 candles before
# the present candle.

for i in range(30, len(df) - 2):

    # ========================================================
    # INPUT FEATURES
    # ========================================================

    row = []


    # --------------------------------------------------------
    # PREVIOUS 10 WEEKLY CANDLES
    #
    # -10, -9, -8, ..., -1
    #
    # Each candle:
    # Open, High, Low, Close
    # --------------------------------------------------------

    for j in range(i - 10, i):

        row.extend([
            df.loc[j, "open"],
            df.loc[j, "high"],
            df.loc[j, "low"],
            df.loc[j, "close"]
        ])


    # --------------------------------------------------------
    # FEATURE 41
    #
    # Average mean-price of candles -11 to -20
    # --------------------------------------------------------

    average_11_20 = df.loc[
        i - 20:i - 11,
        "mean_price"
    ].mean()

    row.append(average_11_20)


    # --------------------------------------------------------
    # FEATURE 42
    #
    # Average mean-price of candles -21 to -30
    # --------------------------------------------------------

    average_21_30 = df.loc[
        i - 30:i - 21,
        "mean_price"
    ].mean()

    row.append(average_21_30)


    # ========================================================
    # TARGET
    # ========================================================

    # --------------------------------------------------------
    # ========================================================
    # 1. HIGHEST CLOSE BETWEEN -2 AND -10
    # ========================================================
    
    previous_2_to_10 = df.iloc[i - 10:i - 1]
    
    highest_close = previous_2_to_10["close"].max()
    
    
    # ========================================================
    # 2. TODAY'S OPEN MUST BE +0.79%
    #    ABOVE THAT HIGHEST CLOSE
    # ========================================================
    
    present_open = df.loc[i, "open"]
    
    opening_condition = (
        present_open >= highest_close * 1.007
    )
    
    
    # ========================================================
    # 3. CANDLE -1 MUST BE AN UPSIDE CANDLE
    # ========================================================
    
    candle_minus_1_open = df.loc[i - 1, "open"]
    candle_minus_1_close = df.loc[i - 1, "close"]
    
    upside_movement = (
        candle_minus_1_close - candle_minus_1_open
    )
    
    upside_condition = upside_movement > 0
    
    
    # ========================================================
    # 4. HIGHEST HIGH OF INPUT CANDLES MUST NOT BE ABOVE
    #    75% OF THE DISTANCE TO TODAY'S OPEN
    # ========================================================
    
    highest_input_high = df.iloc[i - 10:i]["high"].max()
    
    distance_to_open = (
        present_open - highest_close
    )
    
    level_75 = (
        highest_close + distance_to_open * 0.75
    )
    
    input_range_condition = (
        highest_input_high <= level_75
    )
    
    
    # ========================================================
    # 5. MAXIMUM 75% RETRACEMENT OF CANDLE -1
    # ========================================================
    
    minimum_allowed_price = (
        candle_minus_1_close
        - upside_movement * 0.75
    )
    
    three_candles = df.iloc[i:i + 3]
    
    retracement_condition = (
        three_candles["low"] >= minimum_allowed_price
    ).all()
    
    
    # ========================================================
    # FINAL TARGET
    # ========================================================
    
    if (
        opening_condition
        and upside_condition
        and input_range_condition
        and retracement_condition
    ):
        target = 1
    else:
        target = 0

    # --------------------------------------------------------
    # Check:
    #
    # Present candle
    # +1 weekly candle
    # +2 weekly candle
    #
    # Their LOW cannot go below the 75% retracement level.
    # --------------------------------------------------------

    three_candles = df.iloc[i:i + 3]

    retracement_condition = (
        three_candles["low"] >= minimum_allowed_price
    ).all()


    # --------------------------------------------------------
    # FINAL TARGET
    # --------------------------------------------------------

    if (
        opening_condition
        and upside_condition
        and retracement_condition
    ):
        target = 1
    else:
        target = 0


    # ========================================================
    # SAVE INPUT + TARGET
    # ========================================================

    dataset.append(row)
    targets.append(target)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

dataset = np.array(dataset)
targets = np.array(targets)


# ============================================================
# ADD TARGET AS LAST COLUMN
# ============================================================

dataset_with_target = np.column_stack(
    (dataset, targets)
)


# ============================================================
# COLUMN NAMES
# ============================================================

columns = []

for n in range(10, 0, -1):

    columns.extend([
        f"open_{n}",
        f"high_{n}",
        f"low_{n}",
        f"close_{n}"
    ])

columns.extend([
    "average_11_20",
    "average_21_30",
    "target"
])


# ============================================================
# CREATE FINAL DATAFRAME
# ============================================================

final_dataset = pd.DataFrame(
    dataset_with_target,
    columns=columns
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nDataset shape:")
print(final_dataset.shape)

print("\nNumber of input features:")
print(len(columns) - 1)

print("\nNumber of target = 1:")
print((final_dataset["target"] == 1).sum())

print("\nNumber of target = 0:")
print((final_dataset["target"] == 0).sum())

print("\nFinal dataset:")
print(final_dataset)


# ============================================================
# OPTIONAL: SAVE DATASET
# ============================================================

final_dataset.to_csv(
    f"{SYMBOL}_weekly_dataset.csv",
    index=False
)


# ============================================================
# CLOSE MT5
# ============================================================

mt5.shutdown()

#%%

import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import MetaTrader5 as mt5


# ============================================================
# MT5 CONNECTION
# ============================================================

if not mt5.initialize(
    login=1514371841,
    server="FTMO-Demo",
    password="@2$Sz9lq*?"
):
    print("initialize() failed, error code =", mt5.last_error())
    quit()


# ============================================================
# PARAMETERS
# ============================================================

SYMBOL = "AUDCHF"
NUMBER_OF_BARS = 200

timezone = pytz.timezone("Etc/UTC")

utc_from = datetime(
    2026, 8, 21,
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
    print("Could not retrieve data:", mt5.last_error())
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

df = df.sort_values("time").reset_index(drop=True)


# ============================================================
# MEAN PRICE
# ============================================================

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


# We need:
#
# -30 to -1  -> input information
# present     -> target event
# +1
# +2         -> target confirmation
#
# Therefore:

for i in range(30, len(df) - 2):

    # ========================================================
    # INPUT FEATURES
    # ========================================================

    row = []


    # --------------------------------------------------------
    # 10 PREVIOUS WEEKLY CANDLES
    #
    # -10 ... -1
    # --------------------------------------------------------

    for j in range(i - 10, i):

        row.extend([
            df.loc[j, "open"],
            df.loc[j, "high"],
            df.loc[j, "low"],
            df.loc[j, "close"]
        ])


    # --------------------------------------------------------
    # AVERAGE PRICE -11 TO -20
    # --------------------------------------------------------

    average_11_20 = df.loc[
        i - 20:i - 11,
        "mean_price"
    ].mean()

    row.append(average_11_20)


    # --------------------------------------------------------
    # AVERAGE PRICE -21 TO -30
    # --------------------------------------------------------

    average_21_30 = df.loc[
        i - 30:i - 21,
        "mean_price"
    ].mean()

    row.append(average_21_30)


   # ========================================================
    # COMMON DATA
    # ========================================================
    
    present_open = df.loc[i, "open"]
    
    # Candles -2 through -10
    previous_2_to_10 = df.iloc[i - 10:i - 1]
    
    # Candle -1
    candle_minus_1_open = df.loc[i - 1, "open"]
    candle_minus_1_close = df.loc[i - 1, "close"]
    
    # Present + next 2 candles
    three_candles = df.iloc[i:i + 3]
    
    
    # ========================================================
    # ======================= UPSIDE =========================
    # ========================================================
    
    # Candles -2 through -10
    previous_2_to_10 = df.iloc[i - 10:i - 1]
    
    # Highest closing price
    highest_close = previous_2_to_10["close"].max()
    
    # Today's opening price
    present_open = df.loc[i, "open"]
    
    
    # --------------------------------------------------------
    # 1. TODAY'S OPEN MUST BE +0.7%
    # --------------------------------------------------------
    
    upside_opening_condition = (
        present_open >= highest_close * 1.007
    )
    
    
    # --------------------------------------------------------
    # 2. CANDLE -1 MUST BE BULLISH
    # --------------------------------------------------------
    
    candle_minus_1_open = df.loc[i - 1, "open"]
    candle_minus_1_close = df.loc[i - 1, "close"]
    
    upside_movement = (
        candle_minus_1_close - candle_minus_1_open
    )
    
    upside_candle_condition = (
        upside_movement > 0
    )
    
    
    # --------------------------------------------------------
    # 3. HIGHEST HIGH FROM -2 TO -10
    # --------------------------------------------------------
    
    highest_input_high = previous_2_to_10["high"].max()
    
    distance_to_open = (
        present_open - highest_close
    )
    
    level_75 = (
        highest_close + distance_to_open * 0.75
    )
    
    upside_input_condition = (
        highest_input_high <= level_75
    )
    
    
    # --------------------------------------------------------
    # 4. MAXIMUM 75% RETRACEMENT
    # --------------------------------------------------------
    
    minimum_allowed_price = (
        candle_minus_1_close
        - upside_movement * 0.75
    )
    
    three_candles = df.iloc[i:i + 3]
    
    upside_retracement_condition = (
        three_candles["low"] >= minimum_allowed_price
    ).all()
    
    
    # --------------------------------------------------------
    # FINAL UPSIDE TARGET
    # --------------------------------------------------------
    
    if (
        upside_opening_condition
        and upside_candle_condition
        and upside_input_condition
        and upside_retracement_condition
    ):
        target_up = 1
    else:
        target_up = 0
    
    
   # ========================================================
    # ====================== DOWNSIDE =========================
    # ========================================================
    
    # Lowest closing price between -2 and -10
    lowest_close = previous_2_to_10["close"].min()
    
    
    # --------------------------------------------------------
    # 1. TODAY'S OPEN MUST BE -0.7%
    # --------------------------------------------------------
    
    downside_opening_condition = (
        present_open <= lowest_close * 0.993
    )
    
    
    # --------------------------------------------------------
    # 2. CANDLE -1 MUST BE BEARISH
    # --------------------------------------------------------
    
    downside_movement = (
        candle_minus_1_open - candle_minus_1_close
    )
    
    downside_candle_condition = (
        downside_movement > 0
    )
    
    
    # --------------------------------------------------------
    # 3. LOWEST LOW FROM -2 TO -10
    # --------------------------------------------------------
    
    lowest_input_low = previous_2_to_10["low"].min()
    
    distance_to_open = (
        lowest_close - present_open
    )
    
    level_75_downside = (
        lowest_close - distance_to_open * 0.75
    )
    
    downside_input_condition = (
        lowest_input_low >= level_75_downside
    )
    
    
    # --------------------------------------------------------
    # 4. MAXIMUM 75% RETRACEMENT
    # --------------------------------------------------------
    
    maximum_allowed_price = (
        candle_minus_1_close
        + downside_movement * 0.75
    )
    
    downside_retracement_condition = (
        three_candles["high"] <= maximum_allowed_price
    ).all()
    
    
    # --------------------------------------------------------
    # FINAL DOWNSIDE TARGET
    # --------------------------------------------------------
    
    if (
        downside_opening_condition
        and downside_candle_condition
        and downside_input_condition
        and downside_retracement_condition
    ):
        target_down = 1
    else:
        target_down = 0

    # ========================================================
    # SAVE
    # ========================================================

    dataset.append(row)

    target_upside.append(target_up)
    target_downside.append(target_down)

    dates.append(df.loc[i, "time"])


# ============================================================
# CONVERT TO NUMPY
# ============================================================

dataset = np.array(dataset)

target_upside = np.array(target_upside)

target_downside = np.array(target_downside)


# ============================================================
# ADD TARGETS
# ============================================================

dataset_with_target = np.column_stack(
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

for n in range(10, 0, -1):

    columns.extend([
        f"open_{n}",
        f"high_{n}",
        f"low_{n}",
        f"close_{n}"
    ])


columns.extend([
    "average_11_20",
    "average_21_30",
    "target_upside",
    "target_downside"
])


# ============================================================
# FINAL DATAFRAME
# ============================================================

final_dataset = pd.DataFrame(
    dataset_with_target,
    columns=columns
)


# Add present candle date as first column

final_dataset.insert(
    0,
    "present_candle_date",
    dates
)


# ============================================================
# DISPLAY
# ============================================================

print("\nDataset shape:")
print(final_dataset.shape)

print("\nNumber of input features:")
print(42)

print("\nUpside targets:")
print(
    (final_dataset["target_upside"] == 1).sum()
)

print("\nDownside targets:")
print(
    (final_dataset["target_downside"] == 1).sum()
)

print("\nFinal dataset:")
print(final_dataset)


# ============================================================
# SAVE
# ============================================================

final_dataset.to_csv(
    f"{SYMBOL}_weekly_dataset.csv",
    index=False
)


# ============================================================
# CLOSE MT5
# ============================================================

mt5.shutdown()

#%%

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

for i in range(30, len(df) - 2):


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

    for j in range(i - 10, i):

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

    row.append(
        average_11_20
    )


    # --------------------------------------------------------
    # AVERAGE MEAN PRICE -21 TO -30
    # --------------------------------------------------------

    average_21_30 = df.loc[
        i - 30:i - 21,
        "mean_price"
    ].mean()

    row.append(
        average_21_30
    )


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


for n in range(10, 0, -1):

    columns.extend([
        f"open_{n}",
        f"high_{n}",
        f"low_{n}",
        f"close_{n}"
    ])


columns.extend([
    "average_11_20",
    "average_21_30",
    "target_upside",
    "target_downside"
])


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