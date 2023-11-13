# Project 2: Walmart Store Sales Forcasting

CS 598 Practical Statistical Learning, UIUC Fall 2023

* Ryan Fogle (rsfogle2@, UIN: 652628818)
* Sean Enright (seanre2@, UIN: 661791377)

## Section 1: Technical Details

In this project, we consider the historical sales data from 45 Walmart stores spread across different regions to predict the future weekly sales for every department in each store. We use SVD to "de-noise" the weekly sales data at the department level, and then use Ridge regression to train a model for each department and store pairing. Our implementation uses Python.

### Data Preprocessing and Prediction

The Walmart dataset consists of four features, `Store`, `Dept`, `Date`, `IsHoliday`, and one predictor, `Weekly_Sales`.

The first step of our process is spliting the data by each department and creating a data matrix $X_{n \times m}$ where $n$ is the number of weeks with sales data, and $m$ is the number of stores in a department (we use `Pandas`' `pivot` method to create the matrix). Matrix positions with no sales for a given week and store are set to zero.

We use `sklearn`'s PCA implementation to perform Singular Value Decomposition on the data. First we select the initial $d$ principal components ($d=8$ in our implementation) and apply dimensionality reduction on $X$, centering it by its column (`Store`) means, to create matrix $\widetilde{F}_{n \times d}$. For departments with fewer than $d$ stores, we select the first $m$ stores for dimensionality reduction. $\widetilde{F}$ is then inverse transformed back to the original shape of $X$ to create a new de-noised design matrix $\widetilde{X}$, which is then de-centered with the pre-transformation column means. Finally, we restore the appropriate `Store`, `Weekly_Sales` lables using `pandas`' `melt` method, and reattach the `Date` column.

The `Date` column is then replaced by two new columns: `Year` (numerical) and `Week` (categorical).

We then iterate over each store, creating a new subset data matrix for each department and store combination. Stores found in the training set, but absent in the test set are skipped. This subset data matrix is run through a preprocessing pipeline via `sklearn`'s `pipeline` API, which performs these steps:

1) Detect whether column is categorical (`Week`) or numeric (`Year`).
2) Process categorical and numeric columns separately:
    * If categorical, one-hot encode the predictor.
    * If numeric, standardize by removing the mean, but do not scale.
3) Create a model by applying ridge regression to the training data, using regularization strength $\alpha=0.15$. We ignore the `IsHoliday` column for training. 
4) Use the model to predict weekly sales, given the input test data. 

The ridge regression L2 coefficient was found empirically to give the best results, while avoiding overfitting.

## Section 2: Performance Metrics

In our testing our data meets the thresholds given in the report.

### Evaluation Metric

For this project, we consider the following metric: Weighted mean absolute error

$$
WMAE = \frac{1}{\sum{w_i}} \sum^n_{i=1} {w_i | y_i - \hat{y}_i | }
$$

Where $n$ is the number of rows, $\hat{y}_i$ is the predicted sales, $y_i$ is the actual sales, and $w_i$ are weights. $ w = 5$ if the week is a holiday week, $1$ otherwise. 

### Results
Our prediction gave a mean WAE of 1564.99. A table of result data is included in the appendix.

### Computer System

For the evaluation of this report, we used a Ryzen 5600X with 32GB of RAM for all 10 training/test splits.

### Execution Time

It took a total time of 3 minutes and 48 seconds to execute the code. The average run time was 23 seconds per fold to preprocess the data, train the model, and generate the predictions.

<!-- Report the accuracy of your models on the test data (refer to the provided evaluation metric below), the execution time of your code, and details of the computer system you used (e.g., Macbook Pro, 2.53 GHz, 4GB memory or AWS t2.large) for all 10 training/test splits. -->

## Appendix

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
