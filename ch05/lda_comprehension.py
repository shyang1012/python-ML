import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

"""LDA(Linear Discriminant Analysis, PCA): 선형 판별 분석(지도학습 방법)
    가정: LDA는 데이터가 정규분포라고 가정
    클래스가 동일한 공분산 행렬을 가지고 훈련샘플은 서로 통계적으로 독립적이라고 가정
    
    일반적인 성능: 하나 이상의 가정이 (조금) 위반되더라도 여전히 LDA는 차원축소를 상당히 잘 수행
    
    내부 동작 방식
    1. d 차원 데이터셋을 표준화 전처리(d는 차원(feature, demension) 갯수)
    2. 각 클래스에 대해 d차원의 평균백터 계산
    3. 클래스 간의 산포행렬(scatter matrix) Sᵦ와 클래스 내 산포 행렬 S_w를 구성
    4. S_w^-1Sᵦ 행렬의 고유 백터와 고윳값을 계산
    5. 고윳값을 내림차순으로 정렬하여 고유 백터의 순서를 매김
    6. 고윳값이 가장 큰 k개의 고유 백터 선택하여 d X k 차원의 변환행렬 W를 구성
        이 행렬의 열이 고유 백터
    7. 변환행렬 W를 사용하여 샘플을 새로운 특성 부분공간으로 투영
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


    # 2. 각 클래스에 대해 d차원의 평균백터 계산
    np.set_printoptions(precision=4)

    mean_vecs = []

    for label in range(1, 4):
        mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
        print('MV %s: %s\n' % (label, mean_vecs[label - 1]))

    # 3. 클래스 간의 산포행렬(scatter matrix) Sᵦ와 클래스 내 산포 행렬 S_w를 구성
    d = len(df_wine.columns[1:])
    
    # 클래스 내 산포 행렬 S_w를 계산
    # 산포행렬을 계산할 때 훈련 데이터셋의 클래스 레이블이 균등하게 분포되었다고 가정.
    S_W = np.zeros((d,d))
    for label, mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.zeros((d,d)) # 각 클래스에 대한 산포 행렬
        for row in X_train_std[y_train == label]:
            row, mv = row.reshape(d, 1), mv.reshape(d, 1)  # 열 벡터를 만듭니다
            class_scatter += (row - mv).dot((row - mv).T) # 클래스 산포 행렬을 더합니다.
        S_W += class_scatter

    print('클래스 내의 산포 행렬: %sx%s' % (S_W.shape[0], S_W.shape[1]))
    
    print(f'''클래스 레이블 분포: {np.bincount(y_train)[1:]}''') # [41 50 33]
    # 클래스가 균일하게 분포되어 있지 않기 때문에 공분산 행렬을 사용하는 것이 더 낫습니다

    S_W = np.zeros((d,d))
    for label, mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.cov(X_train_std[y_train==label].T) # 각 클래스에 대한 산포 행렬
        S_W += class_scatter
    print('스케일이 조절된 클래스 내의 산포 행렬: %sx%s' % (S_W.shape[0], S_W.shape[1]))

    #클래스간 산포행렬 계산
    mean_overall =np.mean(X_train_std, axis=0)
    mean_overall = mean_overall.reshape(d, 1) # 열 백터로 만들기
    d = len(df_wine.columns[1:]) #클래스레이블을 제외한 feature 갯수
    S_B = np.zeros((d, d))

    for i, mean_vec in enumerate(mean_vecs):
        n = X_train_std[y_train == i + 1, :].shape[0]
        mean_vec = mean_vec.reshape(d, 1)  # 열 벡터로 만들기
        S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

    print('클래스 간의 산포 행렬: %sx%s' % (S_B.shape[0], S_B.shape[1]))

    # 4. S_w^-1Sᵦ 행렬의 고유 백터와 고윳값을 계산

    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    # (고윳값, 고유벡터) 튜플의 리스트를 만듭니다.
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

    #  5. 고윳값을 내림차순으로 정렬하여 고유 백터의 순서를 매김
    # (고윳값, 고유벡터) 튜플을 큰 값에서 작은 값 순서대로 정렬합니다.
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

    # 고윳값의 역순으로 올바르게 정렬되었는지 확인합니다.
    print('내림차순의 고윳값:\n')
    for eigen_val in eigen_pairs:
        print(eigen_val[0])

    tot = sum(eigen_vals.real)
    discr = [(i /tot) for i in sorted(eigen_vals.real, reverse= True)]
    cum_discr = np.cumsum(discr)

    plt.bar(range(1, 14), discr, alpha=0.5, align='center',
        label='Individual "discriminability"')
    plt.step(range(1, 14), cum_discr, where='mid',
            label='Cumulative "discriminability"')
    plt.ylabel('"Discriminability" ratio')
    plt.xlabel('Linear discriminants')
    plt.ylim([-0.1, 1.1])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/wine_lda.png', dpi = 300)    
    # plt.show()
    plt.close()

    # 6. 고윳값이 가장 큰 k개의 고유 백터 선택하여 d X k 차원의 변환행렬 W를 구성
    # 이 행렬의 열이 고유 백터

    w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
              eigen_pairs[1][1][:, np.newaxis].real))

    print('행렬 W:\n', w)

    #  7. 변환행렬 W를 사용하여 샘플을 새로운 특성 부분공간으로 투영
    X_train_lda = X_train_std.dot(w)

    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']

    for l, c, m in zip(np.unique(y_train), colors, markers):
         plt.scatter(X_train_lda[y_train == l, 0],
                X_train_lda[y_train == l, 1] * (-1),
                c=c, label=l, marker=m)
    
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.savefig(get_base_dir('images')+'/transformed_wine_lda.png', dpi = 300)
    plt.show()
    plt.close()