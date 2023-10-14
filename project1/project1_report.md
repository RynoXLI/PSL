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

Our analysis of the dataset found that the many variables were either highly imbalanced, consisted of mainly zeros, or were predominantly missing values, and should be excluded in order to improve model performance. These variables were found by sifting through a pandas profiling report from the python library: [ydata-profiling](https://docs.profiling.ydata.ai/4.6/)

**Highly imbalanced predictors**

 `Street`,  `Utilities`, `Condition_2`, `Roof_Matl`, `Heating`, `Pool_QC`

**Predictors with a high amount of zeros**

`Low_Qual_Fin_SF`, `Pool_Area`, `BsmtFin_SF_2`, `Three_season_porch`, `Screen_Porch`, `Misc_Val`

**Predictors with mostly missing values**

 `Misc_Feature`, `Mas_Vnr_Type`

 After dropping these variables, 67 remaining predictors remain.


#### Categorical Variables
All of the variables that we considered to be categorical are time based, i.e., those that reference the year or month of an event. These variables were encoded with one-hot encoding, and any new levels encountered were ignored. We used sklearn's one hot encoding implementation.

The categorical variables we identified are: `Year_Built`, `Year_Remod_Add`, `Garage_Yr_Blt`, `Mo_Sold`, and `Year_Sold`.

#### Transformations of Numerical Variables
For numerical variables, our regression pipeline consisted of scaling and standardizing the numeric features, as well as winsorizing. For scaling and standardization we used sklearn's Standard Scaler, and for Winsorization we used [feature-engine](https://feature-engine.trainindata.com/en/1.3.x/quickstart/index.html)'s Winsorizor. 

No numeric transformations were done for the tree-based model.

### Models


#### Regression Model

#### Tree Model

## Section 2: Performance Metrics

### Results

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

### Regression Summary
#### First Half
- **Range**: (0.1096, 0.1207)
- **Mean**: 0.1140
#### Second Half
- **Range**: (0.1229, 0.1328)
- **Mean**: 0.1282

### Tree Summary

#### First Half
- **Range**: (0.1144, 0.1212)
- **Mean**: 0.1184
#### Second Half
- **Range**: (0.1249, 0.1326)
- **Mean**: 0.1297

### Computer System

For the evaluation of this report, we used a Ryzen 5600X w/ 32GB of RAM and (insert here Sean's computer)

### Execution Time

It took a total time of 2 minutes and 37 seconds to execute the code. An average of run time of 15 seconds per fold to preprocess and train the data. 

<!-- Report the accuracy of your models on the test data (refer to the provided evaluation metric below), the execution time of your code, and details of the computer system you used (e.g., Macbook Pro, 2.53 GHz, 4GB memory or AWS t2.large) for all 10 training/test splits. -->