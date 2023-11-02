"""
Project 2: Walmart Store Sales Forcasting

CS 598 Practical Statistical Learning

2023-11-13

UIUC Fall 2023

**Authors**
* Ryan Fogle
    - rsfogle2@illinois.edu
    - UIN: 652628818
* Sean Enright
    - seanre2@illinois.edu
    - UIN: 661791377
"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import statsmodels.api as sm
import patsy

def myeval():
    file_path = 'Proj2_Data/test_with_label.csv'
    test_with_label = pd.read_csv(file_path)
    num_folds = 10
    wae = []

    for i in range(num_folds):
        file_path = f'Proj2_Data/fold_{i+1}/test.csv'
        test = pd.read_csv(file_path)
        test = test.drop(columns=['IsHoliday']).merge(test_with_label, on=['Date', 'Store', 'Dept'])

        file_path = f'Proj2_Data/fold_{i+1}/mypred.csv'
        test_pred = pd.read_csv(file_path)

        # Left join with the test data
        new_test = test_pred.merge(test, on=['Date', 'Store', 'Dept'], how='left')

        # Compute the Weighted Absolute Error
        actuals = new_test['Weekly_Sales']
        preds = new_test['Weekly_Pred']
        weights = new_test['IsHoliday_x'].apply(lambda x: 5 if x else 1)
        wae.append(sum(weights * abs(actuals - preds)) / sum(weights))

    return wae

num_folds = 10

# Approach I
for i in range(1, num_folds + 1):
    # Reading train data
    file_path = f'Proj2_Data/fold_{i}/train.csv'
    train = pd.read_csv(file_path)

    # Reading test data
    file_path = f'Proj2_Data/fold_{i}/test.csv'
    test = pd.read_csv(file_path)

    most_recent_date = train['Date'].max()

    # Filter and select necessary columns
    tmp_train = train[train['Date'] == most_recent_date].copy()
    tmp_train.rename(columns={'Weekly_Sales': 'Weekly_Pred'}, inplace=True)
    tmp_train = tmp_train.drop(columns=['Date', 'IsHoliday'])

    # Left join with the test data
    test_pred = test.merge(tmp_train, on=['Dept', 'Store'], how='left')

    # Fill NaN values with 0 for the Weekly_Pred column
    test_pred['Weekly_Pred'].fillna(0, inplace=True)

    # Write the output to CSV
    file_path = f'Proj2_Data/fold_{i}/mypred.csv'
    test_pred.to_csv(file_path, index=False)

print("Approach I")
wae = myeval()
for value in wae:
    print(f"\t{value:.3f}")
print(f"\tAverage {sum(wae) / len(wae):.3f}\n")

# Approach II
for i in range(1, num_folds + 1):
    # Reading train data
    file_path = f'Proj2_Data/fold_{i}/train.csv'
    train = pd.read_csv(file_path)

    # Reading test data
    file_path = f'Proj2_Data/fold_{i}/test.csv'
    test = pd.read_csv(file_path)

    # Define start and end dates based on test data
    start_last_year = pd.to_datetime(test['Date'].min()) - timedelta(days=375)
    end_last_year = pd.to_datetime(test['Date'].max()) - timedelta(days=350)

    # Filter train data based on the defined dates and compute 'Wk' column
    tmp_train = train[(train['Date'] > str(start_last_year)) 
                      & (train['Date'] < str(end_last_year))].copy()
    tmp_train['Date'] = pd.to_datetime(tmp_train['Date'])  
    tmp_train['Wk'] = tmp_train['Date'].dt.isocalendar().week
    tmp_train.rename(columns={'Weekly_Sales': 'Weekly_Pred'}, inplace=True)
    tmp_train.drop(columns=['Date', 'IsHoliday'], inplace=True)

    # Compute 'Wk' column for test data
    test['Date'] = pd.to_datetime(test['Date'])
    test['Wk'] = test['Date'].dt.isocalendar().week

    # Left join with the tmp_train data
    test_pred = test.merge(tmp_train, on=['Dept', 'Store', 'Wk'], how='left').drop(columns=['Wk'])

    # Fill NaN values with 0 for the Weekly_Pred column
    test_pred['Weekly_Pred'].fillna(0, inplace=True)

    # Save the output to CSV
    file_path = f'Proj2_Data/fold_{i}/mypred.csv'
    test_pred.to_csv(file_path, index=False)

print("Approach II")
wae = myeval()
for value in wae:
    print(f"\t{value:.3f}")
print(f"\tAverage {sum(wae) / len(wae):.3f}\n")
