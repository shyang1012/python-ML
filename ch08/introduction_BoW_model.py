import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import *
import os
import re
from nltk.stem.porter import *
import nltk
from nltk.corpus import stopwords

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
    #1. 단어를 특성 백터로 변환
    count= CountVectorizer()
    docs = np.array([
         'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'
    ])
    bag = count.fit_transform(docs)
    print(count.vocabulary_) # n-그램 모델
    print(bag.toarray())
    
    # 2. tf-idf(term frequency-inverse document frequency)를 사용하여 단어 적합성 평가
    # tf-idf는 특성 백터에서 자주 등장하는 단어의 가중치를 낮추는 기법
    # tf-idf는 단어빈도와 역문서빈도의 곱으로 정의됨

    np.set_printoptions(precision=2)

    tfidf = TfidfTransformer(use_idf=True, 
                         norm='l2', 
                         smooth_idf=True)

    #일반적으로 tf-idf를 계산하기 전에 단어 빈도(tf)를 정규화하지만 
    # TfidfTransformer 클래스는 tf-idf를 직접 정규화
    print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

    tf_is = 3
    n_docs = 3
    idf_is = np.log((n_docs+1) / (3+1))



    tfidf_is = tf_is * (idf_is + 1)
    print('tf-idf of term "is" = %.2f' % tfidf_is)

    tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
    raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
    print(raw_tfidf) 

    l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf**2))
    print(l2_tfidf)

    print(df.loc[0,'review'][-50:])
    # print(preprocesser(df.loc[0,'review'][-50:]))

    #불용어 제거 실습
    nltk.download('stopwords')
    stop = stopwords.words('english')
    result=[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
    if w not in stop]
    print(result)

    