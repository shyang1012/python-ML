import numpy as np
from .models import MovieReview
import pickle
import os
from .vectorizer import vect



def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path
    if not os.path.exists(result):
        os.makedirs(result)
    return result


def update_model(model):
    results = MovieReview.objects.fetchall()
    print(results)
    