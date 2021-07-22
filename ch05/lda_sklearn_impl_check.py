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

    
    # 사이킷런의 LDA 구현 방식
    print("사이킷런의 LDA 구현 방식")
    # 클래스 비율 계산
    y_uniq, y_count = np.unique(y_train, return_counts=True)
    priors = y_count / X_train_std.shape[0]
    print(priors)

    #  산포행렬 s_w 계산
    s_w = np.zeros((X_train_std.shape[1], X_train_std.shape[1]))
    # print(s_w)

    for i, label in enumerate(y_uniq):
         # 1/n로 나눈 공분산 행렬을 얻기 위해 bias=True로 지정합니다.
        s_w += priors[i] * np.cov(X_train_std[y_train == label].T, bias=True)
    
    # print("s_w:", s_w)

    # 클래스간의 산포 행렬도 클래스 비율을 곱해 계산
    s_b = np.zeros((X_train_std.shape[1], X_train_std.shape[1]))

    d = len(df_wine.columns[1:])
    mean_vecs = []  

    for label in range(1, 4):
        mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
        print('MV %s: %s\n' % (label, mean_vecs[label - 1]))

    mean_overall = np.mean(X_train_std, axis=0)
    mean_overall = mean_overall.reshape(d, 1)  # 열 벡터로 만들기

    for i, mean_vec in enumerate(mean_vecs):
        n = X_train_std[y_train == i + 1].shape[0]
        mean_vec = mean_vec.reshape(-1, 1)
        s_b += priors[i] * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
    print('클래스 간의 산포 행렬: %sx%s' % (s_b.shape[0], s_b.shape[1]))

    # s_b를 직접 구해 고윳값을 분해하는 대신 scipy.linalg.eigh 함수에 s_b와 s_w를 전달하면
    # s_b * w = lamda*S_w *w식의 고윳값을 바로 계산 가능
    ei_val, ei_vec = scipy.linalg.eigh(s_b, s_w)
    print("ei_val:",ei_val)
    ei_vec = ei_vec[:, np.argsort(ei_val)[::-1]]
    # print("ei_vec:",ei_vec)

    lda_eigen = LDA(solver='eigen')
    lda_eigen.fit(X_train_std, y_train)

    # 클래스 내의 산포 행렬은 covariance_ 속성에 저장되어 있습니다.
    print(np.allclose(s_w, lda_eigen.covariance_))

    Sb = np.cov(X_train_std.T, bias=True) - lda_eigen.covariance_
    print(np.allclose(Sb, s_b))

    # 구해진 고유백터는 scalings_ 속성에 저장되어 있습니다.
    # 클래스가 세 개이므로 두 개의 고유백터(선형 판별 벡터) 비교
    print(np.allclose(lda_eigen.scalings_[:, :2], ei_vec[:, :2]))

    # transform 메서드는 단순히 샘풀과 고유백터의 점곱으로 구할 수 있습ㄴ디ㅏ.
    print(np.allclose(lda_eigen.transform(X_test_std), np.dot(X_test_std, ei_vec[:, :2])))

    # LinearDiscriminantAnalysis 클래스의 solver 매개변수 기본값은 'svd'로 특이 값 분해를
    # 사용합니다.
    # 산포 행렬을 직접 계산하지 않기 때문에 특성이 많은 데이터셋도 잘 작동됩니다.
