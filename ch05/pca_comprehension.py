import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

    # 2. 공분한 행렬(Covariance matrix)을 만듬
    cov_mat = np.cov(X_train_std.T)

    # 3. 공분산 행렬을 고유 백터(eigenvector)와 고윳값(eigenvalue)으로 분해
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    print(f"""고윳값: {eigen_vals}""") # 설명된 분산비율, 고유백터(주성분) 순서

    # 4. 고윳값을 내림차순으로 정렬하고 그에 해당하는 고유 백터의 순위를 매김
    tot = sum(eigen_vals)
    var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    # 주성분의 설명된 분산 그래프 표시
    plt.bar(range(1, 14), var_exp, alpha = 0.5, align='center', label='Indivdual explained variance')
    plt.step(range(1,14), cum_var_exp, where='mid', label='Cumulative explained variance')
    plt.ylabel("Explained variance ratio")
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/Explained variance ratio and Principal component index.png', dpi = 300)
    # plt.show()
    plt.close()

    #  5. 고윳값이 가장 큰 k개의 고유 백터 선택, 여기서 k는 새로운 특성부분 공간의 차원(k <= d)
    # (고윳값, 고유백터) 튜플의 리스트 작성
    eigen_pairs=[(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    # 높은 값에서 낮은 값으로 (고윳값, 고유 백터) 튜플을 정렬
    eigen_pairs.sort(key = lambda k:k[0], reverse=True)
    
    # 6. 최상위 k개의 고유 백터로 투영 행렬(projection matrix) W를 만듬
    """여기서는 산점도를 그리기 위해 두개의 고유백터만 선택,
    실전에서는 계산 효율성과 모델 성능사이의 절충점을 찾아 
    주성분 갯수 결정
    """
    # 투영행렬 w 선택, 여기선 2개만 선택
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis])) # 행을 열로 변환
    print("투영행렬\n", w)

    # 7. 투영행렬 W를 사용해서 d 차원 입력 데이터셋 X를 새로운 k차원의 특성부분 공간으로 변환
    X_train_pca = X_train_std.dot(w)

    # pca 변환된 wine 데이터 시각화
    # 산점도를 그릴 목적으로 클래스 레이블 정보를 사용하였으나 PCA는 비지도 학습기법
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']

    for l, c, m in zip(np.unique(y_train), colors, markers):
         plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=l, marker=m)
    
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/wine_pca.png', dpi = 300)
    plt.show()
    plt.close()