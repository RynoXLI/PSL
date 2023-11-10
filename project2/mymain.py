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
from sklearn.svm import SVR

class DataLoader:
    RESPONSE_VAR = "Weekly_Sales"
    DATE_COL = "Date"
    HOLIDAY_COL = "IsHoliday"

    DTYPE = {
        "Store": "object",
        "Dept": "object",
        "Date": "object",
        "IsHoliday": "bool",
        "Weekly_Sales": "float64"
    }

    def __init__(self) -> None:
        pass
    
    def load_data(self, stem):
        train, test = self._load_data(stem)

        test_X = test.drop(columns=['Weekly_Sales'])
        test_y = test['Weekly_sales']

        return train, test_X, test_y
    
    def _load_data(self, stem):
        path_train = stem / "train.csv"
        path_test = stem / "test.csv"

        train = pd.read_csv(path_train, dtype=self.DTYPE)
        test = pd.read_csv(path_test, dtype=self.DTYPE)
        return train, test
    
    def load_fold_data(self, fold=1):
        stem = Path.cwd() / "project2" / "Proj2_Data" / f"fold_{fold}"
        train, test = self._load_data(stem)

        test_set = pd.read_csv(Path.cwd() / "project2" / "Proj2_Data" / "test_with_label.csv", dtype=self.DTYPE)
        test_y = test.merge(test_set, on=['Dept', 'Store', 'Date'], how="left")[self.RESPONSE_VAR]

        return train, test, test_y
    
    def train_predict(self, train, test_X, test_y):
        pred_y = np.zeros(test_y.shape[0])
        for dept in dict(tuple(train.groupby('Dept'))):
            # create masks
            train_mask = (train['Dept'] == dept).values
            test_mask = test_X['Dept'] == dept
            
            train_dept = train[train_mask].drop(columns=['Dept'])
            # train_X_dept = train[train_mask].drop(columns=['Dept', 'Weekly_Sales'])
            # train_y_dept = train[train_mask]['Weekly_Sales']

            test_X_dept = test_X[test_mask].drop(columns=['Dept'])

            # Run PCA
            train_X_dept, train_y_dept, test_X_dept = self._clean_data(train_dept, test_X_dept)

            # Run per store
            y_pred_store = np.zeros(test_X_dept.shape[0])
            for store in dict(tuple(train_X_dept.groupby('Store'))):
                train_mask_store = (train_X_dept['Store'] == store).values
                test_mask_store = (test_X_dept['Store'] == store)

                train_X_store = train_X_dept[train_mask_store].drop(columns=['Store'])
                train_y_store = train_y_dept[train_mask_store]
                test_X_store = test_X_dept[test_mask_store].drop(columns=['Store'])
                
                # Create Model
                model = dl.make_regression(train_X_store, train_y_store)
                
                # Run Predictions
                if test_X_store.shape[0] > 0:
                    # print(y_pred_store.shape)
                    # print(test_mask_store.shape)
                    # print(y_pred_store[test_mask_store].shape)
                    # print(test_X_store.shape)
                    y_pred_store[test_mask_store] = model.predict(test_X_store)
            
            pred_y[test_mask] = y_pred_store
        
        weights = test_X['IsHoliday'].apply(lambda x: 5 if x else 1).values
        return self.wmae(pred_y, test_y, weights)
    
    def _clean_data(self, train, test):
        """Parse the training and test data files, drop necessary columns, and
        identify response column(s)."""

        # Run PCA to remove noise
        train_pivot = train.pivot(index='Date', columns='Store', values='Weekly_Sales').reset_index().fillna(0)
        dates = train_pivot['Date']

        X = train_pivot.drop(columns=['Date']).values
        pca = PCA(n_components=min(X.shape[1], 8)).fit((X - X.mean(axis=0)))
        pca_pivot = pd.DataFrame(pca.inverse_transform(pca.transform(X - X.mean(axis=0))) + X.mean(axis=0), columns=train_pivot.columns[1:])
        pca_pivot['Date'] = dates
        train_pca = pca_pivot.melt(id_vars='Date', var_name='Store', value_name='Weekly_Sales')

        train_pca['Week'] = train_pca['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').isocalendar()[1]).astype('object')
        train_pca['Year'] = train_pca['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').isocalendar()[0]).astype('float')
        # train_pca['IsHoliday'] = train['IsHoliday'].astype('object').fillna(False)
        train_pca = train_pca.drop(columns=['Date'])

        test['Week'] = test['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').isocalendar()[1]).astype('object')
        test['Year'] = test['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').isocalendar()[0]).astype('float')
        test = test.drop(columns=['Date'])

        train_X = train_pca.drop(columns=['Weekly_Sales'])
        train_Y = train_pca['Weekly_Sales']

        return train_X, train_Y, test
    
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

            train, test_X, test_Y = dl.load_fold_data(fold+1)
            wae = dl.train_predict(train, test_X, test_Y)
            
            wmae[fold] = wae
            print(f'Fold {fold}')
            print(wmae[fold])
            loop_time[fold] = (datetime.now() - loop_start).total_seconds()

        df = pd.DataFrame(np.array([np.arange(1, 11).tolist(),
                  wmae.tolist(),
                  loop_time.tolist()
        ]).T.tolist(), columns=['Fold', 'WMAE', 'Loop Time'])
        print(df.to_markdown(index=False))
        print('Mean WAE: ', df['WMAE'].mean())
        # Print total time taken
        print('Total Time (s):', (datetime.now() - start).total_seconds())
