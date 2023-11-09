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
        try:
            weights = new_test['IsHoliday_x'].apply(lambda x: 5 if x else 1)
        except KeyError: 
            weights = new_test['IsHoliday'].apply(lambda x: 5 if x else 1)
        wae.append(sum(weights * abs(actuals - preds)) / sum(weights))

    return wae

def preprocess(data):
    tmp = pd.to_datetime(data['Date'])
    data['Wk'] = tmp.dt.isocalendar().week
    data['Yr'] = tmp.dt.year
    data['Wk'] = pd.Categorical(data['Wk'], categories=[i for i in range(1, 53)])  # 52 weeks 
    #    data['IsHoliday'] = data['IsHoliday'].apply(int)
    return data

# Approach III (optimized)
num_folds = 10

for fold in range(1, num_folds + 1):
    print(f"Starting fold {fold}")
    # Reading train data
    file_path = f'Proj2_Data/fold_{fold}/train.csv'
    train = pd.read_csv(file_path)

    # Reading test data
    file_path = f'Proj2_Data/fold_{fold}/test.csv'
    test = pd.read_csv(file_path)

    # pre-allocate a pd to store the predictions
    test_pred = pd.DataFrame()

    train_pairs = train[['Store', 'Dept']].drop_duplicates(ignore_index=True)
    test_pairs = test[['Store', 'Dept']].drop_duplicates(ignore_index=True)
    unique_pairs = pd.merge(train_pairs, test_pairs, how = 'inner', on =['Store', 'Dept'])

    train_split = unique_pairs.merge(train, on=['Store', 'Dept'], how='left')
    train_split = preprocess(train_split)
    y, X = patsy.dmatrices('Weekly_Sales ~ Weekly_Sales + Store + Dept + Yr  + Wk', 
                        data = train_split, 
                        return_type='dataframe')
    train_split = dict(tuple(X.groupby(['Store', 'Dept'])))


    test_split = unique_pairs.merge(test, on=['Store', 'Dept'], how='left')
    test_split = preprocess(test_split)
    y, X = patsy.dmatrices('Yr ~ Store + Dept + Yr  + Wk', 
                        data = test_split, 
                        return_type='dataframe')
    X['Date'] = test_split['Date']
    test_split = dict(tuple(X.groupby(['Store', 'Dept'])))

    keys = list(train_split)

    for key in keys:
        X_train = train_split[key]
        X_test = test_split[key]
    
        Y = X_train['Weekly_Sales']
        X_train = X_train.drop(['Weekly_Sales','Store', 'Dept'], axis=1)
        
        cols_to_drop = X_train.columns[(X_train == 0).all()]
        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)
    
        cols_to_drop = []
        for i in range(len(X_train.columns) - 1, 1, -1):  # Start from the last column and move backward
            col_name = X_train.columns[i]
            # Extract the current column and all previous columns
            tmp_Y = X_train.iloc[:, i].values
            tmp_X = X_train.iloc[:, :i].values

            coefficients, residuals, rank, s = np.linalg.lstsq(tmp_X, tmp_Y, rcond=None)
            if np.sum(residuals) < 1e-10:
                    cols_to_drop.append(col_name)
                
        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)

        model = sm.OLS(Y, X_train).fit()
        mycoef = model.params.fillna(0)
        
        tmp_pred = X_test[['Store', 'Dept', 'Date']]
        X_test = X_test.drop(['Store', 'Dept', 'Date'], axis=1)
        
        tmp_pred['Weekly_Pred'] = np.dot(X_test, mycoef)
        test_pred = pd.concat([test_pred, tmp_pred], ignore_index=True)
        
    test_pred['Weekly_Pred'].fillna(0, inplace=True)
    
    # Save the output to CSV
    file_path = f'Proj2_Data/fold_{fold}/mypred.csv'
    test_pred.to_csv(file_path, index=False)
    
wae = myeval()
for value in wae:
    print(f"\t{value:.3f}")
print(f"{sum(wae) / len(wae):.3f}")