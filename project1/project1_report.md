# Project 1: Predict the Housing Prices in Ames

CS 598 Practical Statistical Learning, UIUC Fall 2023
* Ryan Fogle (rsfogle2@, UIN: 652628818)
* Sean Enright (seanre2@, UIN: 661791377)

## Section 1: Technical Details

In this project, we consider the Ames Housing dataset and seek to predict the log-scale housing price based on all available predictors. We use a linear regression model and a tree model to accomplish this task. Here we describe our data preprocessing steps and describe the models employed for prediction. Our implementation uses Python.

### Data Preprocessing

The Ames Housing dataset consists of 82 variables. For our prediction task, `Sale_Price` is our response variable, which leaves 81 predictors, which are both numeric and categorical. Our analysis of these variables found that many of these variables could be excluded, and that other variables required transformation before being used to fit our models. We describe our data preprocessing below.

#### Preprocessing Pipeline

An overview of the preprocessing pipeline is described here. Further details follow.

The entry point into the preprocessing pipeline is `Pandas.read_csv`, which parses the training and test partition files. We allow automatic detection of datatypes, which identifies most of the categorical and numerical predictors properly, but some categorical variables are not identified and need to be manually specified.

After excluding variables and identifying the remaining categorical variables, the rest of the preprocessing is handled by the `sklearn.pipeline.make_pipeline` API. First, we split the data into categorical and numerical partitions with `sklearn.compose.make_column_selector`. Then we encode categorical predictors and transform numerical predictors with `sklearn.compose.ColumnTransformer`. Finally, these columns are concatenated and output for model fitting.

The preprocessing pipelines diverge at the numerical transformation step, depending on which model is being fit: 
* For the linear regression model, the numerical variables are transformed.
* For the tree model, we do no further transformation of numerical variables.

#### Excluded Variables

Our analysis of the dataset found that the many variables were either highly imbalanced, consisted of mainly zeros, or were predominantly missing values, and should be excluded in order to improve model performance. These variables were found by sifting through a pandas profiling report from the Python library: [ydata-profiling](https://docs.profiling.ydata.ai/4.6/)

`Street`,  `Utilities`, `Condition_2`, `Roof_Matl`, `Heating`, `Pool_QC`, `Low_Qual_Fin_SF`, `Pool_Area`, `BsmtFin_SF_2`, `Three_season_porch`, `Screen_Porch`, `Misc_Val`, `Misc_Feature`, `Mas_Vnr_Type`

 After dropping these variables, 67 remaining predictors remain.

#### Categorical Variables

`Pandas.read_csv` identifies all columns with string-formatted data as the "Object" data type. We use `sklearn.compose.make_column_selector` to designate all "Object" columns as categorical variables.

Before performing the conversion, though, we must manually correct the datatype of a handful of variables that are automatically identified as numerical, but are actually categorical. These are: `Year_Built`, `Year_Remod_Add`, `Garage_Yr_Blt`, `Mo_Sold`, and `Year_Sold`.

This gives us the full set of categorical variables: `MS_SubClass`, `MS_Zoning`, `Alley`, `Lot_Shape`, `Land_Contour`, `Lot_Config`, `Land_Slope`, `Neighborhood`, `Condition_1`, `Bldg_Type`, `House_Style`, `Overall_Qual`, `Overall_Cond`, `Year_Built`, `Year_Remod_Add`, `Roof_Style`, `Exterior_1st`, `Exterior`, `Foundation`, `Bsmt_Qual`, `Bsmt_Cond`, `Bsmt_Exposure`, `BsmtFin_Type_1`, `BsmtFin_Type_2`, `Heating_QC`, `Central_Air`, `Electrical`, `Kitchen_Qual`, `Functional`, `Fireplace_Qu`, `Garage_Type`, `Garage_Yr_Blt`, `Garage_Finish`, `Garage_Qual`, `Garage_Cond`, `Paved_Drive`, `Fence`, `Mo_Sold`, `Year_Sold`, `Sale_Type`, and `Sale_Condition`.

These variables are encoded with one-hot encoding via the `sklearn.preprocessing.OneHotEncoder` class, and any unseen levels encountered are set to be ignored.

#### Transformations of Numerical Variables

**Scaling of Variables**

We scale all numerical variables with `sklearn.preprocessing.StandardScaler`, which centers and scales each predictor, so that its mean is zero and its standard deviation is one.

**Winsorized Variables**

The following predictors are winsorized to reduce the effect of possible outliers. We use the `Winsorizer` class from the `feature_engine` library with default configuration, so values are capped at +3 standard deviations above the mean.

`Lot_Frontage`, `Lot_Area`, `Mas_Vnr_Area`, `Bsmt_Unf_SF`, `Total_Bsmt_SF`, `Second_Flr_SF`, `First_Flr_SF`, `Gr_Liv_Area`, `Garage_Area`, `Wood_Deck_SF`, `Open_Porch_SF`, `Enclosed_Porch`

### Models

#### Linear Regression Model

We used sci-kit learn's [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) to fit our linear-based method. We define these parameters: `alpha=0.01`, `l1_ratio=0.1`, `max_iter=10000`. The remaing variables were kept default.

#### Tree Model

For the tree-based model we used the python implementation of [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor). We define these parameters: `n_estimators=1000`,  `learning_rate=0.01`, `max_depth=2`, `subsample=0.8`, `reg_alpha=0.01`, `reg_lambda=0.01`. The rest of the variables were kept default.

## Section 2: Performance Metrics

In our testing our data meets the thresholds given in the report.
### Table of Results

|   Fold |   Regression RMSE |   Tree RMSE |   Run Time |
|-------:|------------------:|------------:|-----------:|
|      1 |          0.120731 |    0.118118 |    15.7136 |
|      2 |          0.116166 |    0.121221 |    15.6464 |
|      3 |          0.112114 |    0.119834 |    16.3107 |
|      4 |          0.11141  |    0.118533 |    15.6715 |
|      5 |          0.109573 |    0.114355 |    15.2552 |
|      6 |          0.132178 |    0.130318 |    15.3701 |
|      7 |          0.132842 |    0.1326   |    15.9748 |
|      8 |          0.126514 |    0.128918 |    16.533  |
|      9 |          0.126624 |    0.13156  |    15.6905 |
|     10 |          0.122893 |    0.124922 |    15.1485 |

### Computer System

For the evaluation of this report, we used a Ryzen 5600X with 32GB of RAM for all 10 training/test splits.

### Execution Time

It took a total time of 2 minutes and 37 seconds to execute the code. The average run time was 15 seconds per fold to preprocess the data, train both models and generate the predictions.

<!-- Report the accuracy of your models on the test data (refer to the provided evaluation metric below), the execution time of your code, and details of the computer system you used (e.g., Macbook Pro, 2.53 GHz, 4GB memory or AWS t2.large) for all 10 training/test splits. -->