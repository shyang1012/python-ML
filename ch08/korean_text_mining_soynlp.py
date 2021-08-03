import pandas as pd
import numpy as np
import os
from soynlp.tokenizer import LTokenizer
from soynlp.word import WordExtractor
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.utils.fixes import loguniform
import math


def get_base_dir(path):
    result = os.path.join(os.path.dirname(os.path.abspath(__file__)),path)
    if not os.path.exists(result):
        os.makedirs(result)
    return result


def load_rating_train():
    df = pd.read_csv(get_base_dir('data')+'/ratings_train.txt', delimiter='\t', keep_default_na=False)
    return df


def load_rating_test():
    df = pd.read_csv(get_base_dir('data')+'/ratings_test.txt', delimiter='\t', keep_default_na=False)
    return df


if __name__ == '__main__':
    df_train = load_rating_train()
    df_test = load_rating_test()

    soy_train_path = get_base_dir('data')+'/soy_train.npz'
    soy_test_path = get_base_dir('data')+'/soy_test.npz'

    print(df_train.head())

    X_train = df_train['document'].values
    y_train = df_train['label'].values
    X_test = df_test['document'].values
    y_test = df_test['label'].values

    print(len(X_train), np.bincount(y_train))

    print(len(X_test), np.bincount(y_test))

    lto = LTokenizer()
    print(X_train[4])
    print(lto.tokenize(X_train[4]))

    word_ext = WordExtractor()
    word_ext.train(X_train)
    scores = word_ext.word_scores()
    score_dict = {key: scores[key].cohesion_forward *
              math.exp(scores[key].right_branching_entropy) 
              for key in scores}
    lto = LTokenizer(scores=score_dict)
    print(lto.tokenize(X_train[4]))

    if not os.path.isfile(soy_train_path):
        tfidf = TfidfVectorizer(ngram_range=(1, 2), 
                            min_df=3,
                            max_df=0.9,
                            tokenizer=lto.tokenize, 
                            token_pattern=None)
        tfidf.fit(X_train)
        X_train_soy = tfidf.transform(X_train)
        X_test_soy = tfidf.transform(X_test)

        save_npz(soy_train_path, X_train_soy)
        save_npz(soy_test_path, X_test_soy)
    else:
        X_train_soy = load_npz(soy_train_path)
        X_test_soy = load_npz(soy_test_path)

    sgd = SGDClassifier(loss='log', random_state=1)
    param_dist = {'alpha': loguniform(0.0001, 100.0)}
    rsv_soy = RandomizedSearchCV(estimator=sgd,
                             param_distributions=param_dist,
                             n_iter=50,
                             random_state=1,
                             verbose=1)
    rsv_soy.fit(X_train_soy, y_train)
    print(rsv_soy.best_score_)
    print(rsv_soy.best_params_)
    print(rsv_soy.score(X_test_soy, y_test))