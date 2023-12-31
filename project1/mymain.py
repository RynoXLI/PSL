"""
# Project 1: Predict the Housing Prices in Ames

CS 598 Practical Statistical Learning

2023-10-16

UIUC Fall 2023

**Authors**
* Ryan Fogle
    - rsfogle2@illinois.edu
    - UIN: 652628818
* Sean Enright
    - seanre2@illinois.edu
    - UIN: 661791377
"""

#######################################################
#### STEP 0: Load the necessary Python packages
#######################################################
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from feature_engine.outliers import Winsorizer
from datetime import datetime

# Set random seed to the last four digits of our UINs
np.random.seed(8818 + 1377 + 1)

#######################################################
#### STEP 0.5: Define functions & classes for project
#######################################################

class DataLoader:
    RESPONSE_VAR = "Sale_Price"
    CATEGORICALS = [
        "Year_Built", "Year_Remod_Add", "Garage_Yr_Blt",
        "Mo_Sold", "Year_Sold"
    ]
    DROP_COLS = [
    "Street", # high imbalance
    "Utilities", # high imbalance
    "Condition_2", # high imbalance
    "Roof_Matl", # high imbalance
    "Heating", # high imbalance
    "Pool_QC", # high imbalance
    "Misc_Feature", # Mostly missing
    "Low_Qual_Fin_SF", # High amount of zeros
    "Pool_Area", # high amount of zeros
    "BsmtFin_SF_2", # high amount of zeros
    "Three_season_porch", # high amount of zeros
    "Screen_Porch", # high amount of zeros
    "Misc_Val", # high amount of zeros
    "Mas_Vnr_Type", # Mostly missing
    ]

    # Winsorized columns. Provided by professor
    WIN_COLS = [
        "Lot_Frontage", 
        "Lot_Area",
        "Mas_Vnr_Area",
        "Bsmt_Unf_SF", 
        "Total_Bsmt_SF", 
        "Second_Flr_SF", 
        'First_Flr_SF', 
        "Gr_Liv_Area", 
        "Garage_Area", 
        "Wood_Deck_SF", 
        "Open_Porch_SF", 
        "Enclosed_Porch", 
        ]

    def __init__(self):
        # Some columns appear numeric, but are actually categorical.
        # This maps them appropriately.
        self.dtype_dict = {}
        for col in self.CATEGORICALS:
            self.dtype_dict[col] = "O" # O: Object

    def _clean_data(self, stem):
        """Parse the training and test data files, drop necessary columns, and
        identify response column(s)."""
        path_train = stem / "train.csv"
        path_test = stem / "test.csv"

        df_train = pd.read_csv(path_train, index_col=0, dtype=self.dtype_dict)
        # Separate predictors from response
        train_y = np.log(df_train[self.RESPONSE_VAR])
        train_X = df_train.drop(columns=self.RESPONSE_VAR)
        train_X = train_X.drop(columns=self.DROP_COLS)

        test_X = pd.read_csv(path_test, index_col=0, dtype=self.dtype_dict)
        test_X = test_X.drop(columns=self.DROP_COLS)
        return train_X, train_y, test_X

    def get_prediction_data(self):
        """Read and parse training and test data for evaluation."""
        return self._clean_data(stem=Path.cwd())

    def get_fold_data(self, fold=1):
        """Read and parse test and training data for a single fold."""
        # Parse three data files of a fold
        stem = Path.cwd() / "project1" / "proj1" / f"fold{fold}"
        train_X, train_y, test_X = self._clean_data(stem=stem)

        path_test_y = stem / "test_y.csv"
        test_y = pd.read_csv(path_test_y, index_col=0)
        test_y = np.log(test_y)
        return train_X, train_y, test_X, test_y

    def make_regression_preprocessor(self, train_X):
        """Create preprocessor for linear regression model pipeline."""
        # Select columns by datatype
        numerical_columns_selector = selector(dtype_exclude=object)
        categorical_columns_selector = selector(dtype_include=object)
        # Process column by datatype
        categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
        numerical_preprocessor = StandardScaler()
        # Process predictors
        categorical_columns = categorical_columns_selector(train_X)
        numerical_columns = numerical_columns_selector(train_X)
        # Use ColumnTransformer to split, process, and then concatenate columns
        preprocessor = ColumnTransformer([
            ("one-hot-encoder", categorical_preprocessor, categorical_columns),
            ("standard_scaler", numerical_preprocessor, numerical_columns),
            ('winsorizer', Winsorizer(), self.WIN_COLS)
        ])
        return preprocessor
    
    def make_tree_preprocessor(self, train_X):
        """Create preprocessor for tree model pipeline."""
        # Select columns by datatype
        categorical_columns_selector = selector(dtype_include=object)
        # Process column by datatype
        categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
        # Process predictors
        categorical_columns = categorical_columns_selector(train_X)
        # Use ColumnTransformer to split, process, and then concatenate columns
        preprocessor = ColumnTransformer([
            ("one-hot-encoder", categorical_preprocessor, categorical_columns),
        ], remainder='passthrough')
        return preprocessor

def predict_regression(train_X, train_y, preprocessor):
    """Make predictions using a linear regression model."""
    model_regression = make_pipeline(preprocessor,
                                     ElasticNet(alpha=0.001,
                                                l1_ratio=0.1,
                                                max_iter=10000))
    model_regression.fit(train_X, train_y)
    return model_regression.predict(test_X)

def predict_tree(train_X, train_y, preprocessor):
    """Make predictions using a tree model."""
    model_tree = make_pipeline(preprocessor,
                               LGBMRegressor(n_estimators=20000, 
                                             learning_rate=0.01,
                                             max_depth=2, 
                                             subsample=0.8, 
                                             reg_alpha=0.01, 
                                             reg_lambda=0.01,
                                             verbose=-1)
                                             )
    model_tree.fit(train_X, train_y)
    return model_tree.predict(test_X)

def summarize_rmse(rmse_array, desc):
    """Display RMSE summaries for the ten folds."""
    start_idx = (0, len(rmse_array) // 2)
    first_half = rmse_array[0:start_idx[1]]
    second_half = rmse_array[start_idx[1]:]
    titles = ("First Half", "Second Half")
    print(f"===============\n{desc}\n===============\n")
    for title, half, idx in zip(titles, (first_half, second_half), start_idx):
        print(f"{title}\n---------------\n{half}\n"
              f"Range: ({min(half):.4f}, {max(half):.4f})\n"
              f" Mean: {np.mean(half):.4f}\n"
              f"Worst: Fold {np.argmax(half) + idx + 1}\n")

def write_prediction(pred, test_X, filename="mysubmission1.txt"):
    """Output a file containing predictions for a given test partition."""
    pred = np.round(np.exp(pred), 1)
    df_out = pd.DataFrame({
        "PID": test_X.index,
        "Sale_Price": pred
    })
    csv_delimiter = ",  "
    np.savetxt(filename, df_out, delimiter=csv_delimiter,
                header=csv_delimiter.join(df_out.columns.values),
                fmt=["%i", "%s"], comments='', encoding=None)
        

#######################################################
#### STEP 0.75: Define main loop
#######################################################

if __name__ == "__main__":
    is_submission = True # True to submit for grading. False for testing.
    dl = DataLoader()

    if is_submission: # Run in testing environment

        #######################################################
        #### STEP 1: Preprocess the training data, then fit the two models
        #######################################################
        train_X, train_y, test_X = dl.get_prediction_data()

        preprocessor = dl.make_regression_preprocessor(train_X)
        tree_preprocessor = dl.make_tree_preprocessor(train_X)

        pred_regression = predict_regression(train_X, train_y, preprocessor)
        pred_tree = predict_tree(train_X, train_y, tree_preprocessor)

        #######################################################
        #### STEP 2: Preprocess the training data, then fit the two models
        #######################################################
        write_prediction(pred_regression, test_X, "mysubmission1.txt")
        write_prediction(pred_tree, test_X, "mysubmission2.txt")

    else: # Run in development environment
        start = datetime.now()
        # library not available in test environment
        from tqdm import tqdm
        num_folds = 10
        rmse_regression = np.zeros(num_folds)
        rmse_tree = np.zeros(num_folds)
        loop_time = np.zeros(num_folds)
        for fold in tqdm(range(num_folds)):
            loop_start = datetime.now()
            # Data loading and cleaning
            train_X, train_y, test_X, test_y = dl.get_fold_data(fold=fold+1)
            preprocessor = dl.make_regression_preprocessor(train_X)
            tree_preprocessor = dl.make_tree_preprocessor(train_X)

            pred_regression = predict_regression(train_X, train_y, preprocessor)
            pred_tree = predict_tree(train_X, train_y, tree_preprocessor)
            rmse_regression[fold] = mean_squared_error(test_y,
                                                    pred_regression,
                                                    squared=False)
            rmse_tree[fold] = mean_squared_error(test_y,
                                                 pred_tree,
                                                 squared=False)
            loop_time[fold] = (datetime.now() - loop_start).total_seconds()
        
        # Output Results, you'll need to install tabulate
        df = pd.DataFrame(np.array([np.arange(1, 11).tolist(), 
                                    rmse_regression.tolist(), 
                                    rmse_tree.tolist(), 
                                    loop_time.tolist()]).T.tolist(), 
                          columns=['Fold', 'Regression RMSE', 'Tree RMSE', 'Run Time'])
        try:
            df.to_markdown('dev-results.md', index=False)
        except ImportError:
            print(df)

        summarize_rmse(rmse_regression, "Regression")
        summarize_rmse(rmse_tree, "Tree")

        # Print total time taken
        print('Total Time (s):', (datetime.now() - start).total_seconds())