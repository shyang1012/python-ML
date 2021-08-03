import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import *
import os
import re
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.pipeline import *
from sklearn.linear_model import *
from sklearn.model_selection import *
import pyprind

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
    # df['review'] = df['review'].apply(preprocesser)
    print(df.head(3))
    return df

stop = stopwords.words('english')

def tokenizer(text):
    text = preprocesser(text)
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # 헤더 넘기기
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

if __name__ =='__main__':
    df = load_movie_review()

    vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,
                         preprocessor=None, 
                         tokenizer=tokenizer)

    clf = SGDClassifier(loss='log', random_state=1)


    doc_stream = stream_docs(path=get_base_dir('data')+'/movie_data.csv') 

    pbar = pyprind.ProgBar(45)

    classes = np.array([0, 1])
    for _ in range(45):
        X_train, y_train = get_minibatch(doc_stream, size=1000)
        if not X_train:
            break
        X_train = vect.transform(X_train)
        clf.partial_fit(X_train, y_train, classes=classes)
        pbar.update()

    X_test, y_test = get_minibatch(doc_stream, size=5000)
    X_test = vect.transform(X_test)
    print('정확도: %.3f' % clf.score(X_test, y_test))

    clf = clf.partial_fit(X_test, y_test)
    print('정확도(partial_fit): %.3f' % clf.score(X_test, y_test))


             



