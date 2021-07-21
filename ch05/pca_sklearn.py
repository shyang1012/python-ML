import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))+"/"+"common")

from CmnUtils import CmnUtils

"""PCA(Principal Component Analysis, PCA): 비지도(unsupervised) 선형 변환 기법
    처리단계
    1. d 차원 데이터셋을 표준화 전처리
    2. 공분한 행렬(Covariance matrix)을 만듬
    3. 공분산 행렬을 고유 백터(eigenvector)와 고윳값(eigenvalue)으로 분해
    4. 고윳값을 내림차순으로 정렬하고 그에 해당하는 고유 백터의 순위를 매김
    5. 고윳값이 가장 큰 k개의 고유 백터 선택, 여기서 k는 새로운 특성부분 공간의 차원(k <= d)
    6. 최상위 k개의 고유 백터로 투영 행렬(projection matrix) W를 만듬
    7. 투영행렬 W를 사용해서 d 차원 입력 데이터셋 X를 새로운 k차원의 특성부분 공간으로 변환

"""


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

    pca = PCA()

    X_train_pca  = pca.fit_transform(X_train_std)
    print(pca.explained_variance_ratio_)

    plt.bar(range(1, 14), pca.explained_variance_ratio_, alpha=0.5, align='center')
    plt.step(range(1, 14), np.cumsum(pca.explained_variance_ratio_), where='mid')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.savefig(get_base_dir('images')+'/explained_variance_ratio_sklearn.png', dpi = 300)
    # plt.show()
    plt.close()
    

    pca2 = PCA(n_components=2)
    X_train_pca2 = pca2.fit_transform(X_train_std)
    X_test_pca2 = pca2.transform(X_test_std)

    lr = LogisticRegression(random_state=1)
    lr = lr.fit(X_train_pca2, y_train)

    CmnUtils.plot_decision_regions(X_train_pca2, y_train, classifier= lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/wine_LogisticRegression_train_pca.png', dpi = 300)
    # plt.show()
    plt.close()
    
    lr.predict(X_test_pca2)

    CmnUtils.plot_decision_regions(X_test_pca2, y_test, classifier=lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/wine_LogisticRegression_test_pca.png', dpi = 300)
    # plt.show()
    plt.close()

    pca = PCA(n_components=None)
    X_train_pca = pca.fit_transform(X_train_std)
    print(pca.explained_variance_ratio_)

    # n_components에 (0, 1) 사이 실수를 입력하면 설명된 분산의 비율을 나타냅니다. 
    # 이 비율을 달성하기 위해 필요한 주성분 개수를 선택합니다.

    pca = PCA(n_components=0.95)
    pca.fit(X_train_std)
    print('주성분 개수:', pca.n_components_)
    print('설명된 분산 비율:', np.sum(pca.explained_variance_ratio_))

    # n_components='mle'로 지정하면 토마스 민카(Thomas Minka)가 제안한 차원 선택 방식을 사용
    pca = PCA(n_components='mle')
    pca.fit(X_train_std)
    print('주성분 개수:', pca.n_components_)
    print('설명된 분산 비율:', np.sum(pca.explained_variance_ratio_))

    """PCA의 가장 큰 제약 사항 중 하나는 배치로만 실행되기 때문에 대용량 데이터셋을 처리하려면 많은 
    메모리가 필요합니다. IncrementalPCA를 사용하면 데이터셋의 일부를 사용하여 반복적으로 
    훈련할 수 있습니다.

    partial_fit() 메서드는 네트워크나 로컬 파일 시스템으로부터 조금씩 데이터를 받아와 훈련할 수 
    있습니다. fit() 메서드는 numpy.memmap을 사용하여 로컬 파일로부터 데이터를 조금씩 읽어 올 수 
    있습니다. 한 번에 읽어 올 데이터 크기는 IncrementalPCA 클래스의 batch_size로 지정합니다. 
    기본값은 특성 개수의 5배입니다.

    IncrementalPCA의 n_components 매개변수는 정수 값만 입력할 수 있습니다. 
    """
    ipca = IncrementalPCA(n_components=9)
    for batch in range(len(X_train_std)//25+1):
        X_batch = X_train_std[batch*25:(batch+1)*25]
        ipca.partial_fit(X_batch)
    
    print('주성분 개수:', ipca.n_components_)
    print('설명된 분산 비율:', np.sum(ipca.explained_variance_ratio_))