from pathlib import Path

import sklearn
import numpy as np
import pandas as pd

from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Full compatibility with Pipelines and ColumnTransformers
#sklearn.set_config(transform_output="pandas")

RESPONSE_VAR = "Sale_Price"
CATEGORICALS = ["Year_Built", "Year_Remod_Add", "Garage_Yr_Blt", "Mo_Sold", "Year_Sold"]

# Some columns appear numeric, but are actually categorical. This maps them appropriately.
dtype_dict = {}
for col in CATEGORICALS:
    dtype_dict[col] = "O"

# Select columns by datatype
numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

# Process column by datatype
categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

def get_train(fold=1):
    path = Path.cwd() / "project1" / "proj1" / f"fold{fold}" / "train.csv"
    df = pd.read_csv(path, index_col=0, dtype=dtype_dict)
    # Separate predictors from response
    resp = np.log(df[RESPONSE_VAR])
    pred = df.drop(columns=RESPONSE_VAR)
    if False: # Check which columns are numeric
        print("Numerical datatypes\n===============")
        for n, d in zip(pred.dtypes.index, pred.dtypes):
            print(f"{n}: {d}")
    # Process predictors
    categorical_columns = categorical_columns_selector(pred)
    numerical_columns = numerical_columns_selector(pred)
    # Use ColumnTransformer to split, process, and then concatenate columns
    preprocessor = ColumnTransformer(
    [
        ("one-hot-encoder", categorical_preprocessor, categorical_columns),
        ("standard_scaler", numerical_preprocessor, numerical_columns),
    ]
    )
    return preprocessor, pred, resp

if __name__ == "__main__":
    preprocessor, pred, resp = get_train(fold=1)
    data_train, data_test, target_train, target_test = train_test_split(
        pred, resp, random_state=42
    )
    model = make_pipeline(preprocessor, LinearRegression())
    _ = model.fit(data_train, target_train)
    rmse = mean_squared_error(target_test, model.predict(data_test), squared=False)
    print(rmse)