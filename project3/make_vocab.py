from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

np.random.seed(0)

base_path = Path.cwd() / 'proj3_data' / 'split_1' 
path_train  = base_path / 'train.tsv'
path_test   = base_path / 'test.tsv'
path_test_y = base_path / 'test_y.tsv'

dtypes_dict = {'review': 'string',
               'sentiment': 'Int32'}
train = pd.read_csv(path_train, sep='\t', header=0, dtype=dtypes_dict)
train_x = train['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)
train_y = train['sentiment']

test = pd.read_csv(path_test, sep='\t', header=0, dtype=dtypes_dict)
test_x = test['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)
test_y = pd.read_csv(path_test_y, sep='\t', header=0, dtype=dtypes_dict)['sentiment']

stop_words = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'their',
    'they', 'his', 'her', 'she', 'he', 'a', 'an', 'and', 'is', 'was', 'are', 'were', 'him',
    'himself', 'has', 'have', 'it', 'its', 'the', 'us', 'br'
]

vectorizer = CountVectorizer(
    preprocessor=lambda x: x.lower(), # Convert to lowercase
    stop_words=stop_words,            # Remove stop words
    ngram_range=(1, 4),               # Use 1- to 4-grams
    min_df=0.001,                     # Minimum term frequency
    max_df=0.5,                       # Maximum document frequency
    token_pattern=r"\b[\w+\|']+\b"    # Use word tokenizer
)

tfidf_transformer = TfidfTransformer(use_idf=True)

# Select n-grams by T-test strategy
class T_TestTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocab_size=2000):
        self.vocab_size = vocab_size
        self.subset_inds = []

    def fit(self, x, y):
        # Use two-sample t-test to determine n-grams more likely
        # to be associated with positive or negative reviews
        x = x.toarray()
        mask = y.values.to_numpy() == 1
        pos_x = x[mask]
        neg_x = x[~mask]
        tstat = (pos_x.mean(axis=0) - neg_x.mean(axis=0)) / np.sqrt(
            pos_x.var(axis=0) / pos_x.shape[1] + neg_x.var(axis=0) / neg_x.shape[1]
        )
        self.subset_inds = np.abs(tstat).argsort()[-self.vocab_size:]
        return self

    def transform(self, x, y=None):
        # Select columns corresponing to n-grams likely to be relevant
        return x[:, self.subset_inds]

logistic_regression = LogisticRegression(n_jobs=2, C=0.85, solver="saga", penalty="l1")
    
pipeline_vocab = Pipeline([
    ('vect', vectorizer),
    ('t-score', T_TestTransformer()),
    ('tfidf', tfidf_transformer),
    ('logreg', logistic_regression),
])

# Spare matrix of shape (# train rows, vectorizer embedding size)
start_time = time.time()
pipeline_vocab.fit(train_x, train_y)
print(f"Vocab pipeline fitting: {(time.time() - start_time):.1f} s")
print(f"Training set accuracy: "
      f"{(pipeline_vocab.predict(train['review']) == train['sentiment']).sum() / train.shape[0] * 100:.2f} %\n")

# Vocab selection
vocab_t = pipeline_vocab['vect'].get_feature_names_out()[pipeline_vocab['t-score'].subset_inds]
vocab_lasso = vocab_t[(pipeline_vocab['logreg'].coef_ != 0).reshape(-1)]
print(f"\nVocab size (t-test selection): {len(vocab_t)}")
print(f"  Vocab size (logistic lasso): {len(vocab_lasso)} words")

# Write vocab to file
pd.DataFrame(vocab_t).to_csv('myvocab-t-score.txt', header=False, index=False)
pd.DataFrame(vocab_lasso).to_csv('myvocab-lasso.txt', header=False, index=False)

# ============================
# TEST WITH REDUCED VOCAB
# ============================

vectorizer = CountVectorizer(ngram_range=(1, 4))

tfidf_transformer = TfidfTransformer()#use_idf=True)

logistic_regression = LogisticRegressionCV(
            n_jobs=-1,
            Cs=np.arange(1.0, 25.0, step=0.5),
            solver="saga",
            penalty="l2",
            scoring="roc_auc",
        )

# Use reduced vocabulary to build model
vectorizer.fit(vocab_lasso)
x = vectorizer.transform(train_x)
x = tfidf_transformer.fit_transform(x)
model = logistic_regression.fit(x, train_y)

# Make predictions with test set
x = vectorizer.transform(test_x)
x = tfidf_transformer.fit_transform(x)
preds = model.predict_proba(x)
print(f"Test set AUC Score: {roc_auc_score(test_y, y_score=preds[:, 1]):.4f}")