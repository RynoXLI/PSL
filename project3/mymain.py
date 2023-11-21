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
import pandas as pd
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import nltk
from scipy.stats import ttest_ind

# download necessary packages for nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

class DataLoader:
    def __init__(self):
        pass

    def load_fold_data(self, fold=1):
        stem = Path.cwd() / "proj3_data" / f"split_{fold}"

        train = pd.read_csv(stem / 'train.tsv', sep='\t')
        test_X = pd.read_csv(stem / 'test.tsv', sep='\t')
        test_y = pd.read_csv(stem / 'test_y.tsv', sep='\t')['sentiment']
        train_y = train['sentiment']
        train_X = train.loc[:, train.columns != "sentiment"]

        return train_X, train_y, test_X, test_y

    def preprocess_text(self, X: pd.DataFrame):
        # Define regex pattern
        p_ws = re.compile(r'\s+')
        p_html = re.compile(r'<[^>]*>')
        m_trans = str.maketrans("", "", string.punctuation) # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
        wnl = WordNetLemmatizer()

        review_tokens = (X['review']
         .apply(lambda x: p_html.sub(' ', x)) # remove html
         .apply(lambda x: p_ws.sub(' ', x)) # Remove extra whitespace
         .str.strip() # Remove starting and ending whitespace
         .str.lower() # Lowercase text
         .apply(lambda x: x.translate(m_trans)) # Remove punctuation
         .apply(lambda x: [wnl.lemmatize(word) for word in word_tokenize(x) if word not in stop_words]) # Tokenize words, remove stop words, lemmatize
         )

        return review_tokens
    
    def fit(self, X, y):
        print('Preprocessing Text...')
        toks = self.preprocess_text(X)
        print('Completed...Starting Count Vectorizor')

        # self.pipe = Pipeline(steps=[
        #     ('cv', CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, ngram_range=(1, 4), max_features=999)),
        #     ('tfidf', TfidfTransformer()),
        #     ('svc', SVC(kernel='rbf'))
        # ])
        # self.pipe.fit(toks, y)

        self.cv = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, ngram_range=(1, 4), max_df=0.5, min_df=0.01)

        # what they did #2
        X = self.cv.fit_transform(toks)
        print(f'CountVectorizer(), X-Shape:{X.shape}')
        self.tfidf = TfidfTransformer()
        X = self.tfidf.fit_transform(X)
        print(f'TfidfTransform(), X-shape: {X.shape}')
        # self.model = SVC(kernel='linear', probability=True)
        # self.model = LogisticRegression(C=0.5, class_weight='balanced')
        self.model = LogisticRegression(n_jobs=2)
        self.model.fit(X, y)
        print('Model Fitted')
    
    def predict(self, X):
        toks = self.preprocess_text(X)
        X = self.cv.transform(toks)
        X = self.tfidf.transform(X)
        return self.model.predict_proba(X)
        

if __name__ == "__main__":
    is_submission = False # True to submit for grading. False for testing.
    dl = DataLoader()

    if is_submission: # Submit for grading and run in test environment
        pass

    else: # Internal testing before submission
        # import tqdm here because not available in submission environment
        from tqdm import tqdm

        folds = 5
        for fold in tqdm(range(folds)):
            train_X, train_y, test_X, test_y = dl.load_fold_data(fold=fold+1)

            dl.fit(train_X, train_y)
            pd.DataFrame(dl.cv.vocabulary_.items(), columns=['Vocab', 'Count']).to_csv(f'vocab-{fold+1}.csv', index=False)
            preds = dl.predict(test_X)
            print(f"Fold {fold+1} AUC Score:", roc_auc_score(test_y, y_score=preds[:,1]))