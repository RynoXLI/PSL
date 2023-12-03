"""
# Project 3: Movie Review Sentiment Analysis

CS 598 Practical Statistical Learning

2023-12-04

UIUC Fall 2023

**Authors**
* Ryan Fogle
    - rsfogle2@illinois.edu
    - UIN: 652628818
* Sean Enright
    - seanre2@illinois.edu
    - UIN: 661791377
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

dtypes_dict = {"review": "string",
               "sentiment": "Int32"}

def predict(vocab, train_x, train_y, test_x):
    """Trains a logistic regression model with limited vocabulary on the
       training set and predicts probability of a positive review with the test
       set.

    Args:
        vocab (list): List of word or phrases in vocabulary
        train_x (pandas.core.series.Series): Training set review text
        train_y (pandas.core.series.Series): Training set class labels
        test_x (pandas.core.series.Series): Test set review text

    Returns:
        np.ndarray: Probabilities of the test set observations being positive
            reviews
    """

    vectorizer = CountVectorizer(
        ngram_range=(1,4),
        token_pattern=r"\b[\w+\|']+\b",
        vocabulary=vocab
    )

    tfidf_transformer = TfidfTransformer(use_idf=True)
    
    Cs = np.logspace(np.log10(4), np.log10(9.5), 20)
    model = LogisticRegressionCV(
        n_jobs=-1,
        Cs=Cs,
        solver="saga",
        penalty="l2",
        scoring="roc_auc",
        random_state=0
    )
    
    # Use reduced vocabulary to build model
    x = vectorizer.transform(train_x)
    x = tfidf_transformer.fit_transform(x)
    model.fit(x, train_y)
    
    # Make predictions with test set
    x = vectorizer.transform(test_x)
    x = tfidf_transformer.fit_transform(x)
    preds =  model.predict_proba(x)
    return preds

def get_data(base_path=Path.cwd()):
    """Retrieve the training and test data, formatted as DataFrames.

    Args:
        base_path (optional): Path of folder with training and test data.
                              Defaults to Path.cwd().

    Returns:
        (train_x, train_y, test_x): Training and test dataframes
    """
    path_train  = base_path / "train.tsv"
    path_test   = base_path / "test.tsv"
    
    train = pd.read_csv(path_train, sep="\t", header=0, dtype=dtypes_dict)
    train_x = train["review"].str.replace("&lt;.*?&gt;", " ", regex=True)
    train_y = train["sentiment"]

    test = pd.read_csv(path_test, sep="\t", header=0,
                       dtype=dtypes_dict, index_col="id")
    test_x = test["review"].str.replace("&lt;.*?&gt;", " ", regex=True)
    return train_x, train_y, test_x

def get_fold_data(fold):
    """Retrieve the training and test data for a given fold.

    Args:
        fold (int): Fold number for training andtest data

    Returns:
        (train_x, train_y, test_x, test_y): Training and test dataframes
    """
    fold_path = Path.cwd() / "proj3_data" / f"split_{fold}"
    train_x, train_y, test_x = get_data(base_path=fold_path)
    path_test_y = fold_path / "test_y.tsv"
    
    test_y = pd.read_csv(path_test_y, sep="\t", header=0, dtype=dtypes_dict)["sentiment"]
    return train_x, train_y, test_x, test_y

def read_vocab(vocab_file="myvocab.txt"):
    """Reads a vocabulary file, returning a list of words or phrases.
       
       Each element of the list corresponds to aline of the vocabulary file.

    Args:
        vocab_file (str, optional): Vocab file path. Defaults to "myvocab.txt".

    Returns:
        list: List of vocabulary words of phrases
    """
    vocab_path = Path.cwd() / vocab_file
    with open(vocab_path) as f:
        vocab_list = f.read().splitlines()
    return vocab_list


if __name__ == "__main__":
    is_submission = True  # True to submit for grading. False for testing.
    vocab = read_vocab()
    #print(f"Vocab size: {len(vocab)}")
    
    # Submit for grading and run in test environment
    if is_submission:
        # Read data and make predictions
        train_x, train_y, test_x = get_data()
        preds = predict(vocab, train_x, train_y, test_x)
        # Write predictions to file
        test_x = test_x.to_frame()
        test_x["prob"] = preds[:, 1]
        test_x.to_csv("mysubmission.csv", sep=",", header=True, columns=["prob"])

    # Internal testing before submission
    else:
        # import tqdm here because not available in submission environment
        from tqdm import tqdm
        num_folds = 5
        auroc = np.empty(num_folds)
        execution_time = np.empty(num_folds)
        for fold in tqdm(range(num_folds)):
            
            start_time = time.time()
            
            train_x, train_y, test_x, test_y = get_fold_data(fold + 1)
            preds = predict(vocab, train_x, train_y, test_x)
            
            auroc[fold] = round(roc_auc_score(test_y, y_score=preds[:, 1]), 4)
            execution_time[fold] = round(time.time() - start_time, 1)
        
        df = pd.DataFrame(data={
            "Fold" : np.arange(1, num_folds + 1),
            "AUROC" : auroc,
            "Execution Time (s)" : execution_time}
        ).set_index("Fold")
        try:
            print(df.to_markdown()) # requires tabulate package
        except ImportError:
            print(df)
        print(f"\nMin AUROC: {df['AUROC'].min()}")
        print(f"\nTotal Time: {df['Execution Time (s)'].sum():.1f} s\n")