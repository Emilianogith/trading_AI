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
if not mt5.initialize(login=1510032364, server="FTMO-Demo",password="?!3it@XE@D19NB"):
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
utc_from = datetime(2022, 5, 30, tzinfo=timezone)
# get 10 EURUSD D1 bars starting from 01.10.2020 in UTC time zone
rates = mt5.copy_rates_from("NZDJPY", mt5.TIMEFRAME_D1, utc_from, 130)
rates2 = mt5.copy_rates_from("AUDNZD", mt5.TIMEFRAME_D1, utc_from, 100)
rates3 = mt5.copy_rates_from("EURCHF", mt5.TIMEFRAME_D1, utc_from, 100)

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
target = [0] * 121
# Set specific positions to 1, riga -9 per target
positions = [58, 106,120]
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
