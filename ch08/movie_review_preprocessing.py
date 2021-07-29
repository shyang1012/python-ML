import pyprind
import pandas as pd
import os
import numpy as np

def get_base_dir(path):
    result = os.path.join(os.path.dirname(os.path.abspath(__file__)),path)
    if not os.path.exists(result):
        os.makedirs(result)
    return result

if __name__ == '__main__':
    basepath =  path = os.path.join(get_base_dir('data'),'aclImdb')
    print(basepath)

    labels = {'pos':1, 'neg':0}

    pbar = pyprind.ProgBar(50000)
    df = pd.DataFrame()

    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = os.path.join(basepath, s, l)
            for file in sorted(os.listdir(path)):
                with open(os.path.join(path, file),'r',encoding='utf-8') as infile:
                    txt = infile.read()
                df = df.append([[txt, labels[l]]], ignore_index=True)
                pbar.update()
    df.columns = ['review', 'sentiment']

    np.random.seed(0)

    df = df.reindex(np.random.permutation(df.index))
    df.to_csv(get_base_dir('data')+'/movie_data.csv', index=False, encoding = 'utf-8')