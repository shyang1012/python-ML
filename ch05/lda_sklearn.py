import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import scipy

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))+"/"+"common")

from CmnUtils import CmnUtils


def load_wine_data():
    df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']
    print('Class labels', np.unique(df_wine['Class label']))
    print(df_wine.head())
    return df_wine


def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path;
    if not os.path.exists(result):
        os.makedirs(result)
    return result


if __name__ == '__main__':
    df_wine = load_wine_data()

    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

    X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, 
                     stratify=y,
                     random_state=0)

    # 1. d 차원 데이터셋을 표준화 전처리
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    lda = LDA(n_components=2) #solver 매개변수 기본값은 'svd'

    X_train_lda  = lda.fit_transform(X_train_std, y_train)

    lr = LogisticRegression(random_state=1)
    lr = lr.fit(X_train_lda, y_train)

    CmnUtils.plot_decision_regions(X_train_lda, y_train, classifier=lr)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/wine_LogisticRegression_train_lda.png', dpi = 300)
    # plt.show()
    plt.close()

    X_test_lda = lda.transform(X_test_std)
    CmnUtils.plot_decision_regions(X_test_lda, y_test, classifier=lr)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/wine_LogisticRegression_test_lda.png', dpi = 300)
    # plt.show()
    plt.close()
