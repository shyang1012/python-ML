import pickle
import re
import os
from vectorizer import vect
import numpy as np

def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path
    if not os.path.exists(result):
        os.makedirs(result)
    return result


clf = pickle.load(open(get_base_dir('pkl_objects/classifier.pkl'), 'rb'))


label = {0: 'negative', 1: 'positive'}
example = ['I love this movie']

X = vect.transform(example)

print('Prediction: %s\nProbability: %.2f%%' %
      (label[clf.predict(X)[0]],
       np.max(clf.predict_proba(X)) * 100))
