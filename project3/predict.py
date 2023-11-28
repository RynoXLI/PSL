from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

def predict(vocab, train_x, train_y, test_x):
    np.random.seed(0)

    vectorizer = CountVectorizer(
        ngram_range=(1, 4))

    tfidf_transformer = TfidfTransformer(use_idf=True)

    model = LogisticRegressionCV(
        n_jobs=-1,
        Cs=np.arange(1.0, 13.0, step=0.25),
        solver="saga",
        penalty="l2",
        scoring="roc_auc",
    )
    
    # Use reduced vocabulary to build model
    vectorizer.fit(vocab)
    x = vectorizer.transform(train_x)
    x = tfidf_transformer.fit_transform(x)
    model.fit(x, train_y)
    
    # Make predictions with test set
    x = vectorizer.transform(test_x)
    x = tfidf_transformer.fit_transform(x)
    preds =  model.predict_proba(x)
    return preds

def get_fold_data(fold):
    base_path = Path.cwd() / "proj3_data" / f"split_{fold}" 
    path_train  = base_path / "train.tsv"
    path_test   = base_path / "test.tsv"
    path_test_y = base_path / "test_y.tsv"

    dtypes_dict = {"review": "string",
                   "sentiment": "Int32"}
    train = pd.read_csv(path_train, sep="\t", header=0, dtype=dtypes_dict)
    train_x = train["review"].str.replace("&lt;.*?&gt;", " ", regex=True)
    train_y = train["sentiment"]

    test = pd.read_csv(path_test, sep="\t", header=0, dtype=dtypes_dict)
    test_x = test["review"].str.replace("&lt;.*?&gt;", " ", regex=True)
    test_y = pd.read_csv(path_test_y, sep="\t", header=0, dtype=dtypes_dict)["sentiment"]
    return train_x, train_y, test_x, test_y

def read_vocab(vocab_file="myvocab.txt"):
    vocab_path = Path.cwd() / vocab_file
    with open(vocab_path) as f:
        vocab_list = f.read().splitlines()
    return vocab_list


if __name__ == "__main__":
    is_submission = False  # True to submit for grading. False for testing.
    vocab = read_vocab()
    print(f"Vocab size: {len(vocab)}")
    
    # Submit for grading and run in test environment
    if is_submission:
        pass

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