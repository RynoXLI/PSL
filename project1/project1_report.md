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



### Data Preprocessing


#### Excluded Variables

 Which variables did you exclude from the analysis?

#### Categorical Variables

Identify the variables treated as categorical.
How were these variables encoded, were any levels merged, etc?

All of the variables that we considered to be categorical are time based, i.e., those that reference the year or month of an event. These variables were encoded with one-hot encoding, and any new levels encountered were ignored.

The categorical variables we identified are: `Year_Built`, `Year_Remod_Add`, `Garage_Yr_Blt`, `Mo_Sold`, and `Year_Sold`.

#### Transformations of Numerical Variables
For numerical variables, were there any transformations applied?

### Models


#### Regression Model

#### Tree Model

## Section 2: Performance Metrics

Report the accuracy of your models on the test data (refer to the provided evaluation metric below), the execution time of your code, and details of the computer system you used (e.g., Macbook Pro, 2.53 GHz, 4GB memory or AWS t2.large) for all 10 training/test splits.