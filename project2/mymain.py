"""
# Project 2: Walmart Store Sales Forcasting

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

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataLoader:
    """Loads, cleans and makes predictions for Weekly Sales data"""

    RESPONSE_VAR = "Weekly_Sales"
    DATE_COL     = "Date"
    HOLIDAY_COL  = "IsHoliday"

    DTYPE = {
        "Store"       : "object",
        "Dept"        : "object",
        "Date"        : "object",
        "IsHoliday"   : "bool",
        "Weekly_Sales": "float64"
    }

    def __init__(self) -> None:
        self.pipeline = self._make_regression_pipeline()

    def load_data(self, stem=Path.cwd()):
        """Read and parse the train and test data files from the working directory.

        Args:
            stem (Path.PurePath, optional): Path to train and test files. Defaults to Path.cwd().

        Returns:
            (pd.DataFrame, pd.DataFrame): train and test dataframes
        """
        path_train = stem / "train.csv"
        path_test = stem / "test.csv"

        train = pd.read_csv(path_train, dtype=self.DTYPE)
        test = pd.read_csv(path_test, dtype=self.DTYPE)
        return train, test

    def load_fold_data(self, fold=1):
        """Load train and test data for a given fold. Only used in testing.

        Args:
            fold (int, optional): Fold number. Defaults to 1.

        Returns:
            (pd.DataFrame, pd.DataFrame, pd.Series): train and test dataframes, test_y column
        """
        stem = Path.cwd() / "Proj2_Data" / f"fold_{fold}"
        train, test = self.load_data(stem)

        test_set = pd.read_csv(Path.cwd() / "Proj2_Data" / "test_with_label.csv",
                               dtype=self.DTYPE)
        test_y = test.merge(test_set, on=["Dept", "Store", "Date"],
                            how="left")[self.RESPONSE_VAR]
        return train, test, test_y
    
    def predict(self, train, test_X):
        """Predict weekly sales for a given test set, given training data.
           
        For each department, the sales data is smoothed via PCA and its Date column is split into
        "Year" and "Week" columns.

        Then, for each store within that department, the training data is processed via the
        regression pipeline, which processes categorical and numeric columns separately, then
        aggregates them before building a ridge regression model with the data. Weekly sales
        predictions are then made with the model.

        Finally, the weekly sales data are collected for each store and department and used to
        generate a prediction dataframe.

        Args:
            train (pd.DataFrame): Training data
            test_X (pd.DataFrame): Test data predictors

        Returns:
            pd.Dataframe: Predicted weekly sales
        """
        pred_y = np.zeros(test_X.shape[0])
        for dept in dict(tuple(train.groupby("Dept"))):
            # Create dept masks
            train_mask = (train["Dept"] == dept).values
            test_mask = test_X["Dept"] == dept
            train_dept = train[train_mask].drop(columns=["Dept"])
            test_X_dept = test_X[test_mask].drop(columns=["Dept"])

            # Perform PCA on training data
            train_dept = self._apply_pca(train_dept)

            # Clean and split
            train_X_dept, train_y_dept, test_X_dept = self._clean_data(train_dept, test_X_dept)

            # Create regression model for each store within a given department
            y_pred_store = np.zeros(test_X_dept.shape[0])
            for store in dict(tuple(train_X_dept.groupby("Store"))):
                # Create store masks
                train_mask_store = (train_X_dept["Store"] == store).values
                test_mask_store = (test_X_dept["Store"] == store)

                train_X_store = train_X_dept[train_mask_store].drop(columns=["Store"])
                train_y_store = train_y_dept[train_mask_store]
                test_X_store = test_X_dept[test_mask_store].drop(columns=["Store"])
                
                # Create Model
                model = self.pipeline.fit(train_X_store, train_y_store)
                
                # Make Predictions
                if test_X_store.shape[0] > 0:
                    y_pred_store[test_mask_store] = model.predict(test_X_store)
            
            pred_y[test_mask] = y_pred_store
        
        # Create prediction file
        test_X["Weekly_Pred"] = pred_y
        test_X["Weekly_Pred"].fillna(0, inplace=True)
        return test_X 

    def _clean_data(self, train, test):
        """Converts "Date" column into "Year" and "Week" column for train and test sets
           and splits test into X and y.

        Args:
            train (pd.DataFrame): Train set for a department
            test (pd.DataFrame): Test set for a department

        Returns:
            (pd.DataFrame, pd.DataFrame, pd.DataFrame): Cleaned train_X, train_y and test sets
        """
        # Split "Date" column into "Year" and "Week" columns
        train["Year"] = self._date_to_calendar(train["Date"], 0)
        train["Week"] = self._date_to_calendar(train["Date"], 1)
        train.drop(columns=["Date"], inplace=True)
        test["Year"] = self._date_to_calendar(test["Date"], 0)
        test["Week"] = self._date_to_calendar(test["Date"], 1)
        test.drop(columns=["Date"], inplace=True)

        # Split processed data into train_X and train_Y
        train_X = train.drop(columns=["Weekly_Sales"])
        train_y = train["Weekly_Sales"]
        return train_X, train_y, test
    
    @staticmethod
    def _make_regression_pipeline():
        """Create pipeline with column preprocessor and ridge regression."""

        # Process column by datatype
        categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
        numerical_preprocessor = StandardScaler(with_std=False)

        # Select columns by datatype
        categorical_columns_selector = selector(dtype_include=object)
        numerical_columns_selector = selector(dtype_exclude=object)

        # Create column transformer based on datatype
        preprocessor = ColumnTransformer([
            ("one-hot-encoder", categorical_preprocessor, categorical_columns_selector),
            ("scaler", numerical_preprocessor, numerical_columns_selector)
        ], remainder="passthrough")

        # Create pipeline with column transformations, followed by ridge regression
        return make_pipeline(preprocessor, Ridge(alpha=0.15))
    
    @staticmethod
    def _apply_pca(train, d=8):
        """Apply Principal Component Analysis (PCA) to a matrix to remove noise.

        Args:
            train (pd.DataFrame): Train data for a given department
            d (int, optional): Maximum number of principal components to keep. Defaults to 8.

        Returns:
            pd.DataFrame: PCA-smoothed train data for a given department
        """

        # Reshape training data to prepare for PCA, using weekly sales for a given store and date
        train_pivot = train.pivot(index="Date",
                                  columns="Store",
                                  values="Weekly_Sales").reset_index()
        
        # The reshaped grid may have missing values, which are set to zero.
        train_pivot.fillna(0, inplace=True)
        X = train_pivot.drop(columns=["Date"]).values

        # Perform Principal Component Analysis (PCA) on X to remove noise
        pca = PCA(n_components=min(X.shape[1], d))
        pca.fit((X - X.mean(axis=0)))
        X_tilde = pca.inverse_transform(pca.transform(X - X.mean(axis=0))) + X.mean(axis=0)
        
        # Reconstruct training dataframe from PCA-processed data
        pca_pivot = pd.DataFrame(X_tilde, columns=train_pivot.columns[1:])
        pca_pivot["Date"] = train_pivot["Date"]
        return pca_pivot.melt(id_vars="Date", var_name="Store", value_name="Weekly_Sales")
    
    @staticmethod
    def _date_to_calendar(column, idx):
        """Split "Date" column into "Week" or "Year" column.

        Args:
            column (pd.Series): Date column
            idx (int): Destination column. 0: year, 1: week

        Returns:
            pd.Series: Week or Year column
        """
        # 0: year, 1: week
        column = column.apply(lambda x: datetime.strptime(x, "%Y-%m-%d").isocalendar()[idx])
        column = column.astype("object") if idx == 1 else column.astype("float")
        return column


def myeval():
    file_path = "Proj2_Data/test_with_label.csv"
    test_with_label = pd.read_csv(file_path)
    num_folds = 10
    wae = []

    for i in range(num_folds):
        file_path = f"Proj2_Data/fold_{i + 1}/test.csv"
        test = pd.read_csv(file_path)
        test = test.drop(columns=["IsHoliday"]).merge(test_with_label, on=["Date", "Store", "Dept"])

        file_path = f"Proj2_Data/fold_{i + 1}/mypred.csv"
        test_pred = pd.read_csv(file_path)

        # Left join with the test data
        new_test = test_pred.merge(test, on=["Date", "Store", "Dept"], how="left")

        # Compute the Weighted Absolute Error
        actuals = new_test["Weekly_Sales"]
        preds = new_test["Weekly_Pred"]
        weights = new_test["IsHoliday_x"].apply(lambda x: 5 if x else 1)
        wae.append(sum(weights * abs(actuals - preds)) / sum(weights))
    return wae


if __name__ == "__main__":
    is_submission = True # True to submit for grading. False for testing.
    dl = DataLoader()

    if is_submission: # Submit for grading and run in test environment
        train, test_X = dl.load_data()
        pred = dl.predict(train, test_X)
        # Save the output to CSV
        file_path = f"mypred.csv"
        pred.to_csv(file_path, index=False)

    else: # Internal testing before submission
        # import tqdm here because not available in submission environment
        from tqdm import tqdm

        start = datetime.now()
        num_folds = 10
        loop_time = np.zeros(num_folds)
        wmae = np.zeros(num_folds)
        
        for fold in tqdm(range(num_folds)):
            loop_start = datetime.now()            
            # Data loading and cleaning
            train, test_X, test_y = dl.load_fold_data(fold + 1)
            pred = dl.predict(train, test_X)
            # Save the output to CSV
            file_path = f"Proj2_Data/fold_{fold + 1}/mypred.csv"
            pred.to_csv(file_path, index=False)
            # Evaluate WAE and computation time
            weights = test_X["IsHoliday"].apply(lambda x: 5 if x else 1).values
            wae = np.sum(weights * np.abs(pred["Weekly_Pred"] - test_y)) / np.sum(weights)
            wmae[fold] = wae
            loop_time[fold] = (datetime.now() - loop_start).total_seconds()

        # Summarize performance    
        df = pd.DataFrame(data={
            "Fold" : np.arange(1, num_folds + 1),
            "WMAE" : wmae,
            "Execution Time" : loop_time}
        ).set_index("Fold")
        print(df.to_markdown()) # requires tabulate package
        print(f"Mean WAE: {df['WMAE'].mean()}")
        print(f"\nTotal Time: {(datetime.now() - start).total_seconds():.1f} s\n")