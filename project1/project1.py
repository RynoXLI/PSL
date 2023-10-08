from pathlib import Path

import sklearn
import numpy as np
import pandas as pd

# Full compatibility with Pipelines and ColumnTransformers
sklearn.set_config(transform_output="pandas")

RESPONSE_VAR = "Sale_Price"
CONT = ["Lot_Frontage", "Lot_Area", "Year_Built", "Year_Remod_Add",
        "Mas_Vnr_Area", "BsmtFin_SF_1", "BsmtFin_SF_2", "Bsmt_Unf_SF",
        "Total_Bsmt_SF", "First_Flr_SF", "Gr_Liv_Area", "Garage_Area",
        "Wood_Deck_SF","Open_Porch_SF","Enclosed_Porch","Three_season_porch",
        "Screen_Porch","Pool_Area", "Misc_Val", "Longitude", "Latitude"]

fold = 1

def get_train(fold=1):
    path = Path.cwd() / "project1" / "proj1" / f"fold{fold}" / "train.csv"
    df = pd.read_csv(path, index_col=0)
    resp = df[RESPONSE_VAR]
    pred = df[df.columns.drop(RESPONSE_VAR)]
    return pred, resp

if __name__ == "__main__":
    pred, resp = get_train()
    print(resp.head())
    print(pred.head())
    print(pred.columns)
    print(pred.dtypes)