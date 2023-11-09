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

### Library Restrictions (for Reference, delete later)

To Download data: wget https://liangfgithub.github.io/Data/proj2.zip && unzip proj2.zip

pandas, scipy, numpy
sklearn
datetime
dateutil
prophet
patsy (for model matrix), statsmodels
xgboost
warnings

"""

from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge, SGDRegressor, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA, TruncatedSVD

class DataLoader:
    RESPONSE_VAR = "Weekly_Sales"
    DATE_COL = "Date"
    HOLIDAY_COL = "IsHoliday"
    DROP_COLS = [DATE_COL]

    DTYPE = {
        "Store": "object",
        "Dept": "object",
        "Date": "object",
        "IsHoliday": "bool",
        "Weekly_Sales": "float64"
    }

    def __init__(self) -> None:
        pass

    def get_prediction_data(self):
        """Read and parse training and test data for evaluation."""
        return self._clean_data(stem=Path.cwd())

    def get_fold_data(self, fold=1):
        """Read and parse test and training data for a single fold."""
        # Parse three data files of a fold
        stem = Path.cwd() / "project2" / "Proj2_Data" / f"fold_{fold}"
        train_X, train_y, test_X = self._clean_data(stem=stem, train=True)

        test_set = pd.read_csv(Path.cwd() / "project2" / "Proj2_Data" / "test_with_label.csv", dtype=self.DTYPE)
        test_set['Week'] = test_set[self.DATE_COL].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').isocalendar()[1]).astype('object')
        test_set['Year'] = test_set[self.DATE_COL].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').isocalendar()[0]).astype('float')

        test_y = test_X.merge(test_set, on=['Dept', 'Store', 'Date'], how="left")[self.RESPONSE_VAR]
        test_X = test_X.drop(columns=self.DROP_COLS)

        return train_X, train_y, test_X, test_y
    
    def _clean_data(self, stem, train=False):
        """Parse the training and test data files, drop necessary columns, and
        identify response column(s)."""
        path_train = stem / "train.csv"
        path_test = stem / "test.csv"

        df_train = pd.read_csv(path_train, dtype=self.DTYPE)
        df_train['Week'] = df_train[self.DATE_COL].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').isocalendar()[1]).astype('object')
        df_train['Year'] = df_train[self.DATE_COL].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').isocalendar()[0]).astype('float')
        # Separate predictors from response
        train_y = df_train[self.RESPONSE_VAR]

        train_X = df_train.drop(columns=self.RESPONSE_VAR).drop(columns=self.DROP_COLS)

        test_X = pd.read_csv(path_test, dtype=self.DTYPE)
        test_X['Week'] = test_X[self.DATE_COL].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').isocalendar()[1]).astype('object')
        test_X['Year'] = test_X[self.DATE_COL].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').isocalendar()[0]).astype('float')

        if not train:
            test_X = test_X.drop(columns=self.DROP_COLS)
        return train_X, train_y, test_X
    
    def make_preprocessor(self, train_X):
        """Create preprocessor for linear regression model pipeline."""

        categorical_columns_selector = selector(dtype_include=object)
        numerical_columns_selector = selector(dtype_exclude=object)
        # Process column by datatype
        categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
        numerical_preprocessor = StandardScaler(with_std=False)
        # Process predictors
        categorical_columns = categorical_columns_selector(train_X)
        preprocessor = ColumnTransformer([
            ("one-hot-encoder", categorical_preprocessor, categorical_columns),
            ("scaler", numerical_preprocessor, numerical_columns_selector)
        ], remainder='passthrough')

        return preprocessor
    
    def make_regression(self, train_X, train_y):
        """Make predictions using a linear regression model."""
        model_regression = make_pipeline(self.make_preprocessor(train_X), Ridge(alpha=0.25)
                                        )
        model_regression.fit(train_X, train_y)
        return model_regression
    
    def wmae(self, y_pred, y_test, weights):
        """Return the WMAE"""
        return np.sum(weights * np.abs(y_pred - y_test)) / np.sum(weights)
        # return sum(weights * abs(y_pred - y_test)) / sum(weights)
    

if __name__ == "__main__":
    is_submission = False # True to submit for grading. False for testing.
    dl = DataLoader()

    if is_submission: # Run in testing environment
        train_X, train_y, test_X = dl.get_prediction_data()
    else:
        start = datetime.now()

        # import tqdm here because not available in submission environment
        from tqdm import tqdm

        num_folds = 10
        loop_time = np.zeros(num_folds)
        wmae = np.zeros(num_folds)

        for fold in tqdm(range(num_folds)):
            loop_start = datetime.now()
            # Data loading and cleaning
            train_X, train_y, test_X, test_y = dl.get_fold_data(fold=fold+1)
            pred_y = np.zeros(test_y.shape[0])
            
            # group by department, store
            for dept, store in dict(tuple(train_X.groupby(['Dept', 'Store']))):
            # for dept in dict(tuple(train_X.groupby('Dept'))):
                # create masks
                train_mask = ((train_X['Dept'] == dept) & (train_X['Store'] == store)).values
                test_mask = ((test_X['Dept'] == dept) & (test_X['Store'] == store))
                # train_mask = (train_X['Dept'] == dept).values
                # test_mask = test_X['Dept'] == dept

                # train and test data
                train_X_dept = train_X[train_mask].drop(columns=['Dept', 'Store'])
                train_y_dept = train_y[train_mask]
                test_X_dept = test_X[test_mask].drop(columns=['Dept', 'Store'])

                # Train model
                model = dl.make_regression(train_X_dept, train_y_dept)

                # Run predictions
                if test_X_dept.shape[0] > 0:
                    pred_y[test_mask] = model.predict(test_X_dept)
            
            weights = test_X['IsHoliday'].apply(lambda x: 5 if x else 1).values
            wmae[fold] = dl.wmae(pred_y, test_y, weights)
            print(f'Fold {fold}')
            print(wmae[fold])
            loop_time[fold] = (datetime.now() - loop_start).total_seconds()
            pd.DataFrame(np.array([pred_y.tolist(), test_y.tolist(), weights.tolist()]).T.tolist(), columns=['pred', 'test', 'weights']).to_csv(f'fold{fold}.csv')

        df = pd.DataFrame(np.array([np.arange(1, 11).tolist(),
                  wmae.tolist(),
                  loop_time.tolist()
        ]).T.tolist(), columns=['Fold', 'WMAE', 'Loop Time'])
        print(df.to_markdown(index=False))
        print('Mean WAE: ', df['WMAE'].mean())
        # Print total time taken
        print('Total Time (s):', (datetime.now() - start).total_seconds())
