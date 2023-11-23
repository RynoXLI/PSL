"""
# Project 3: Movie REview Sentiment Analysis

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
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score

stop_words = [
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "their",
    "they",
    "his",
    "her",
    "she",
    "he",
    "a",
    "an",
    "and",
    "is",
    "was",
    "are",
    "were",
    "him",
    "himself",
    "has",
    "have",
    "it",
    "its",
    "the",
    "us",
]


class DataLoader:
    def __init__(self):
        pass

    def load_fold_data(self, fold=1):
        stem = Path.cwd() / "proj3_data" / f"split_{fold}"

        train = pd.read_csv(stem / "train.tsv", sep="\t")
        train["review"] = train["review"].str.replace("&lt;.*?&gt;", " ", regex=True)

        test_X = pd.read_csv(stem / "test.tsv", sep="\t")
        test_y = pd.read_csv(stem / "test_y.tsv", sep="\t")["sentiment"]
        train_y = train["sentiment"]
        train_X = train.loc[:, train.columns != "sentiment"]

        return train_X, train_y, test_X, test_y

    def fit(self, X, y):
        # Count Vectorizor, tokenization, preprocessing
        print("Starting Count Vectorizor")
        self.cv = CountVectorizer(
            preprocessor=lambda x: x.lower(),
            stop_words=list(stop_words),
            ngram_range=(1, 4),
            min_df=0.001,
            max_df=0.5,
            token_pattern=r"\b[\w+\|']+\b",
        )
        X = self.cv.fit_transform(X["review"])
        print(f"CountVectorizer output X-Shape:{X.shape}")

        # ## T-test strategy
        X = X.toarray()
        pos_X = X[y.values == 1]
        neg_X = X[y.values == 0]
        print("Postive Size", pos_X.shape)
        print("Negative Size", neg_X.shape)

        tstat = (pos_X.mean(axis=0) - neg_X.mean(axis=0)) / np.sqrt(
            pos_X.var(axis=0) / pos_X.shape[1] + neg_X.var(axis=0) / neg_X.shape[1]
        )
        tstat_inds = np.abs(tstat).argsort()
        self.subset_inds = tstat_inds[-2000:]

        X = X[:, self.subset_inds]

        # first tfidf, use lasso to do dimension reduction
        tfidf_1 = TfidfTransformer()
        X_1 = tfidf_1.fit_transform(X)
        print(f"TfidfTransform outpout X-shape: {X.shape}")
        lasso = LogisticRegression(n_jobs=2, C=0.85, solver="saga", penalty="l1")
        lasso.fit(X_1, y)
        print("Number of Features:", (lasso.coef_[0] != 0).sum())
        self.inds = np.where(lasso.coef_[0] != 0)[0]
        X = X[:, self.inds]

        # Preform second transform, run ridge
        self.tfidf = TfidfTransformer()
        X_2 = self.tfidf.fit_transform(X)
        self.model = LogisticRegressionCV(
            n_jobs=-1,
            Cs=np.arange(1.0, 25.0, step=0.5),
            solver="saga",
            penalty="l2",
            scoring="roc_auc",
        )
        # self.model = SVC(kernel='linear', C= 1.0, probability=True)
        self.model.fit(X_2, y)
        print("Best C:", self.model.C_)

    def predict(self, X):
        X = self.cv.transform(X["review"])
        X = self.tfidf.transform(X[:, self.subset_inds][:, self.inds])
        # X = self.tfidf.transform(X)
        return self.model.predict_proba(X)


if __name__ == "__main__":
    is_submission = False  # True to submit for grading. False for testing.
    dl = DataLoader()

    if is_submission:  # Submit for grading and run in test environment
        pass

    else:  # Internal testing before submission
        # import tqdm here because not available in submission environment
        from tqdm import tqdm

        folds = 5
        for fold in tqdm(range(folds)):
            train_X, train_y, test_X, test_y = dl.load_fold_data(fold=fold + 1)

            dl.fit(train_X, train_y)
            pd.DataFrame(dl.cv.vocabulary_.items(), columns=["Vocab", "Count"]).to_csv(
                f"vocab-{fold+1}.csv", index=False
            )
            preds = dl.predict(test_X)
            print(
                f"Fold {fold+1} AUC Score:", roc_auc_score(test_y, y_score=preds[:, 1])
            )
