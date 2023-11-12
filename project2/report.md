# Project 1: Predict the Housing Prices in Ames

CS 598 Practical Statistical Learning, UIUC Fall 2023

* Ryan Fogle (rsfogle2@, UIN: 652628818)
* Sean Enright (seanre2@, UIN: 661791377)

## Section 1: Technical Details

In this project, we consider the historical sales data from 45 Walmart stores spread across different regions to predict the future weekly sales for every department in each store. We  use SVD to "de-noise" the on the Department level, and then use Ridge regression to train a model for each Department and Store pairing.

### Data Preprocessing

The Walmart dataset consists of four features (`Store`, `Dept`, `Date`, `IsHoliday`) and one predictor (`Weekly_Sales`). The first step of our process is spliting the data by each Department and creating a data matrix $X_{n \times m}$ where $n$ is the number of weeks in a year, and $m$ is the number of Stores in a Department (we use `Pandas`' `pivot` method to create the matrix). For the cell entries that have no sales for a given week and store, we fill this cells with 0.

We use `sklearn`'s PCA implementation and select the first eight principle components (for departments that had 8 or less stores we select the first $m$ stores) to create a new matrix $\widetilde{F}_{n \times 8}$. This new matrix is then inverse transformed back to the original data format to create a new "de-noised" design matrix. Then using `pandas`' `melt` method we transform the data back to a data matrix of `Store`, `Weekly_Sales`, and attach the `Date` column back.

Using the `Date` column we create two new columns `Year `(numerical predictor) and `Week `(Categorical predictor). We then drop the `Date` column.

Furthermore, we then group each department's data matrix by Store, to create a new subset data matrix and model for each department and store combination.

Each Department and Store data matrix then runs through a preprocessing pipeline that one-hot encodes the `Week` predictor and removes the mean from the `Year` predictor.

For this project, we ignore the `IsHoliday` column for training. 

### Modeling

Once each Department and Store subset of data runs through the data preprocessing step, we use ridge regression with an L2 coefficient of `0.15` . The L2 coefficient was found empirically to give the best results, it helped keep the model from overfitting. 

Finally predictions are made for each Store and Department data entries from the test dataset. 

## Section 2: Performance Metrics

In our testing our data meets the thresholds given in the report.

### Evaluation Metric

For this project, we consider the following metric: Weighted mean absolute error

$$
WMAE = \frac{1}{\sum{w_i}} \sum^n_{i=1} {w_i | y_i - \hat{y}_i | }
$$

Where 
- $n$ is the number of rows
- $\hat{y}_i$ is the predicted sales
- $y_i$ is the actual sales
- $w_i$ are weights. $ w = 5$ if the week is a holiday week, $1$ otherwise. 

### Table of Results

| Fold |    WMAE | Execution Time |
| ---: | ------: | -------------: |
|    1 | 1772.35 |        20.0322 |
|    2 | 1326.03 |        20.5013 |
|    3 | 1375.05 |        20.7597 |
|    4 | 1458.92 |        21.9577 |
|    5 | 2270.62 |        22.2905 |
|    6 | 1602.64 |        23.5902 |
|    7 | 1667.75 |        24.1015 |
|    8 | 1375.47 |        24.3622 |
|    9 | 1401.77 |        24.9397 |
|   10 | 1399.33 |         25.344 |

**Mean WAE: 1564.99**

### Computer System

For the evaluation of this report, we used a Ryzen 5600X with 32GB of RAM for all 10 training/test splits.

### Execution Time

It took a total time of 3 minutes and 48 seconds to execute the code. The average run time was 23 seconds per fold to preprocess the data, train the model, and generate the predictions.

<!-- Report the accuracy of your models on the test data (refer to the provided evaluation metric below), the execution time of your code, and details of the computer system you used (e.g., Macbook Pro, 2.53 GHz, 4GB memory or AWS t2.large) for all 10 training/test splits. -->
