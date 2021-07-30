import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import *
import os
import re
from nltk.stem.porter import *
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import *
from sklearn.linear_model import *
from sklearn.model_selection import *

"""
BoW(Bag-of-Word) 모델 소개
- 텍스트나 단어 같은 범주형 데이터를 머신러닝 알고리즘에 주입하기 위해서는 주입전에 수치형태로 변환 필요
- BoW는 텍스트를 수치 특성 백터로 표현하는 2

BoW 아이디어
 1. 전체 문서에 대해 고유한 토큰(token), 예를 들어 단어로 이루어진 어휘사전(vocabulary) 생성
 2. 특정 문서에 각 단어가 얼마나 자주 등장하는지 헤아려 문서의 특성 벡터 생성
"""

def get_base_dir(path):
    result = os.path.join(os.path.dirname(os.path.abspath(__file__)),path)
    if not os.path.exists(result):
        os.makedirs(result)
    return result

def preprocesser(text):
    text = re.sub('<[^>]*>','', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) +' '.join(emoticons).replace('-', ''))
    return text

def load_movie_review():
    df = pd.read_csv(get_base_dir('data')+'/movie_data.csv', encoding='utf-8')
    df['review'] = df['review'].apply(preprocesser)
    print(df.head(3))
    return df

def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    porter = PorterStemmer()
    return[porter.stem(word) for word in text.split()]


if __name__ == '__main__':
    df = load_movie_review()

    # count = CountVectorizer()

    # docs = df['review'].values

    # bag = count.fit_transform(docs)

    # print(count.vocabulary_)

    # print(bag.toarray())

    X_train = df.loc[:25000, 'review'].values
    y_train = df.loc[:25000, 'sentiment'].values
    X_test = df.loc[25000:, 'review'].values
    y_test = df.loc[25000:, 'sentiment'].values


    tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

    nltk.download('stopwords')
    stop = stopwords.words('english')
    
    param_grid = [
        {
            'vect__ngram_range':[(1,1)]
            ,'vect__stop_words':[stop, None]
            ,'vect__tokenizer':[tokenizer,tokenizer_porter]
            ,'clf__penalty':['l1', 'l2']
            ,'clf__C':[1.0, 10.0, 100.0]
        }
        ,{
            'vect__ngram_range':[(1,1)]
            ,'vect__stop_words':[stop, None]
            ,'vect__tokenizer':[tokenizer,tokenizer_porter]
            ,'clf__penalty':['l1', 'l2']
            ,'vect__use_idf':[False]
            ,'vect__norm':[None]
            ,'clf__C':[1.0, 10.0, 100.0]
        }
        
    ]

    lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0, solver='liblinear'))])

    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                            scoring='accuracy',
                            cv=5,
                            n_jobs=-1)
    gs_lr_tfidf.fit(X_train, y_train)

    print('최적의 매개변수 조합: %s ' % gs_lr_tfidf.best_params_)
    print('CV 정확도: %.3f' % gs_lr_tfidf.best_score_)

    clf = gs_lr_tfidf.best_estimator_
    print('테스트 정확도: %.3f' % clf.score(X_test, y_test))


    