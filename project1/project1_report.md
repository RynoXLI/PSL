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

Here we describe our data preprocessing steps and describe the models employed for prediction.

### Data Preprocessing

The Ames Housing dataset consists of 82 variables. For our prediction task, `Sale_Price` is our response variable, which leaves 81 predictors, which are both numeric and categorical. Our analysis of these variables found that many of these variables could be excluded, and that other variables required transformation before being used to fit our models. We describe our data preprocessing below.

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
All of the variables that we considered to be categorical are time based, i.e., those that reference the year or month of an event. These variables were encoded with one-hot encoding, and any new levels encountered were ignored.

The categorical variables we identified are: `Year_Built`, `Year_Remod_Add`, `Garage_Yr_Blt`, `Mo_Sold`, and `Year_Sold`.

#### Transformations of Numerical Variables
For numerical variables, were there any transformations applied?

### Models


#### Regression Model

#### Tree Model

## Section 2: Performance Metrics

Report the accuracy of your models on the test data (refer to the provided evaluation metric below), the execution time of your code, and details of the computer system you used (e.g., Macbook Pro, 2.53 GHz, 4GB memory or AWS t2.large) for all 10 training/test splits.