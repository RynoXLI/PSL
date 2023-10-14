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

## Section 1: Technical Details

In this project, we consider the Ames Housing dataset and seek to predict the log-scale housing price based on all available predictors. We use a regression model and a tree model to accomplish this task.

Here we describe our data preprocessing steps and describe the models employed for prediction. Our implementation uses Python.

### Data Preprocessing

The Ames Housing dataset consists of 82 variables. For our prediction task, `Sale_Price` is our response variable, which leaves 81 predictors, which are both numeric and categorical. Our analysis of these variables found that many of these variables could be excluded, and that other variables required transformation before being used to fit our models. We describe our data preprocessing below.

#### Preprocessing Pipeline

An overview of the preprocessing pipeline is described here. Further details follow.

The entry point into the preprocessing pipeline is `Pandas.read_csv`, which parses the training and test partition files. We allow automatic detection of datatypes, which identifies most of the categorical and numerical predictors properly, although some categorical variables are manually specified. This is described in further detail below.

After excluding variables and identifying the remaining categorical variables, the rest of the preprocessing is handled by the `sklearn.pipeline.make_pipeline` API, which allows provides convenient and clear preprocessing functions. First, we split the data into categorical and numerical partitions with `sklearn.compose.make_column_selector`. Then we encode categorical data and transform numerical date with `sklearn.compose.ColumnTransformer`. Finally, these columns are concatenated and output for model fitting.

The preprocessing pipelines diverge at the numerical transformation step, depending on which model is being fit: 
* For the regression model, the numerical variables are transformed.
* For the tree model, we do no further transformation of numerical variables.


#### Excluded Variables

Our analysis of the dataset found that the many variables were either highly imbalanced, consisted of mainly zeros, or were predominantly missing values, and should be excluded in order to improve model performance.

**Highly imbalanced predictors**

 `Street`,  `Utilities`, `Condition_2`, `Roof_Matl`, `Heating`, `Pool_QC`

**Predictors with a high amount of zeros**

`Low_Qual_Fin_SF`, `Pool_Area`, `BsmtFin_SF_2`, `Three_season_porch`, `Screen_Porch`, `Misc_Val`

**Predictors with mostly missing values**

 `Misc_Feature`, `Mas_Vnr_Type`

 After dropping these variables, 67 remaining predictors remain.

#### Categorical Variables

`Pandas.read_csv` identifies all columns with string-formatted data as the "Object" data type. We use `sklearn.compose.make_column_selector` to designate all "Object" columns as categorical variables.

Before performing the conversion, though, we must manually correct the datatype of a handful of variables that are automatically identified as numerical, but are actually categorical. These variables are all time based, i.e., those that reference the year or month of an event. These variables are: `Year_Built`, `Year_Remod_Add`, `Garage_Yr_Blt`, `Mo_Sold`, and `Year_Sold`.

This gives us the full set of categorical variables, which are: `MS_SubClass`, `MS_Zoning`, `Alley`, `Lot_Shape`, `Land_Contour`, `Lot_Config`, `Land_Slope`, `Neighborhood`, `Condition_1`, `Bldg_Type`, `House_Style`, `Overall_Qual`, `Overall_Cond`, `Year_Built`, `Year_Remod_Add`, `Roof_Style`, `Exterior_1st`, `Exterior`, `Foundation`, `Bsmt_Qual`, `Bsmt_Cond`, `Bsmt_Exposure`, `BsmtFin_Type_1`, `BsmtFin_Type_2`, `Heating_QC`, `Central_Air`, `Electrical`, `Kitchen_Qual`, `Functional`, `Fireplace_Qu`, `Garage_Type`, `Garage_Yr_Blt`, `Garage_Finish`, `Garage_Qual`, `Garage_Cond`, `Paved_Drive`, `Fence`, `Mo_Sold`, `Year_Sold`, `Sale_Type`, and `Sale_Condition`.

These variables are encoded with one-hot encoding via the `sklearn.preprocessing.OneHotEncoder` class, and any unseen levels encountered were set to be ignored.

#### Transformations of Numerical Variables

**Scaling of Variables**

We scale all numerical variables with `sklearn.preprocessing.StandardScaler`, which centers each predictor, so that its mean is zero, and scales it so its standard deviation is one.

**Winsorized Variables**

The following predictors are winsorized to reduce the effect of possible outliers. We use the `Winsorizer` class from the `feature_engine` library to accomplish this. We use the default configuration, so values are capped at +3 standard deviations above the mean.

Winsorized predictors:

`Lot_Frontage`, `Lot_Area`, `Mas_Vnr_Area`, `Bsmt_Unf_SF`, `Total_Bsmt_SF`, `Second_Flr_SF`, `First_Flr_SF`, `Gr_Liv_Area`, `Garage_Area`, `Wood_Deck_SF`, `Open_Porch_SF`, `Enclosed_Porch`

### Models

#### Regression Model

#### Tree Model

## Section 2: Performance Metrics

Report the accuracy of your models on the test data (refer to the provided evaluation metric below), the execution time of your code, and details of the computer system you used (e.g., Macbook Pro, 2.53 GHz, 4GB memory or AWS t2.large) for all 10 training/test splits.