from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


def get_fold_data(fold):
    base_path = Path.cwd() / 'proj3_data' / f'split_{fold}' 
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
    test_y = pd.read_csv(path_test_y, sep='\t', header=0,
                         dtype=dtypes_dict)['sentiment']
    return train_x, train_y, test_x, test_y

stop_words = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'their', 'they', 'his', 'her', 'she', 'he', 'a', 'an', 'and', 'is',
    'was', 'are', 'were', 'him', 'himself', 'has', 'have', 'it', 'its', 'the',
    'us', 'br'
]

# Data
fold = 2
train_x, train_y, test_x, test_y = get_fold_data(1)
full_x = pd.concat((train_x, test_x), axis=0)
full_y = pd.concat((train_y, test_y), axis=0)

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
        mask = y.values.to_numpy() == 1
        pos_x = x[mask]
        neg_x = x[~mask]
        
        m = pos_x.shape[0]
        n = neg_x.shape[0]
        mean_pos = pos_x.mean(axis=0)
        mean_neg = neg_x.mean(axis=0)
        # Var(X) = E[X^2] - (E[X])^2
        var_pos = pos_x.power(2).mean(axis=0) - np.power(pos_x.mean(axis=0), 2)
        var_neg = neg_x.power(2).mean(axis=0) - np.power(neg_x.mean(axis=0), 2)
        t_stat = (mean_pos - mean_neg) / np.sqrt(var_pos / m + var_neg / n)
        self.subset_inds = np.abs(np.ravel(t_stat)).argsort()[-self.vocab_size:]
        return self

    def transform(self, x, y=None):
        # Select columns corresponing to n-grams likely to be relevant
        return x[:, self.subset_inds]

# C=0.85
Cs = np.logspace(-2, -0.4, 20)
logistic_regression = LogisticRegressionCV(
    n_jobs=-1,
    Cs=Cs,
    solver="saga",
    penalty="l1",
    scoring="roc_auc",
    max_iter=1000,
    random_state=0
)
    
pipeline_vocab = Pipeline([
    ('vectorizer', vectorizer),
    ('t-score', T_TestTransformer()),
    ('tfidf', tfidf_transformer),
    ('logreg', logistic_regression),
])

# Fit with LogisticRegressionCV to find optimal regularization parameter
# that reduces vocab below 1000
pipeline_vocab.fit(full_x, full_y)

# Evaluate regularization parameter values and select the optimal one
max_vocab = 950
vocab_size = np.empty(Cs.shape)
scores = pipeline_vocab['logreg'].scores_[1].mean(axis=0)
for i in range(len(Cs)):
    vocab_size[i] = (pipeline_vocab['logreg'].coefs_paths_[1]
                     .mean(axis=0)[i, :] != 0).sum() - 1
mask = vocab_size < max_vocab
best_c = Cs[mask][np.argmax(scores[mask])]
print((f"Best C value: {best_c},"
       f"\nVocab size: {vocab_size[mask][np.argmax(scores[mask])]},"
       f"\nAUROC: {scores[mask].max():.4f}"))

# Refit data with regularization parameter value found above
pipeline_vocab.steps.pop(3)
pipeline_vocab.steps.append(
    ['logreg', LogisticRegression(n_jobs=-1,
                                  C=best_c,
                                  solver="saga",
                                  penalty="l1",
                                  max_iter=1000)])

pipeline_vocab.fit(full_x, full_y)

# Select vocabulary
t_score_inds = pipeline_vocab['t-score'].subset_inds
vocab_t = pipeline_vocab['vectorizer'].get_feature_names_out()[t_score_inds]
lasso_inds = (pipeline_vocab['logreg'].coef_ != 0).reshape(-1)
vocab = vocab_t[lasso_inds]
print(f"\nVocab size  (t-test selection): {len(vocab_t)}")
print(f"  Vocab size (logistic w/ lasso): {len(vocab)} words\n")

# Write vocab to file
pd.DataFrame(vocab_t).to_csv('myvocab-t-score.txt', header=False, index=False)
pd.DataFrame(vocab).to_csv('myvocab.txt', header=False, index=False)

# ============================
# TEST WITH REDUCED VOCAB
# ============================

make_prediction = True

if make_prediction:
    from predict import predict
    preds = predict(vocab, train_x, train_y, test_x)
    print((f"Fold {fold} AUROC: "
           f"{roc_auc_score(test_y, y_score=preds[:, 1]):.4f}\n"))