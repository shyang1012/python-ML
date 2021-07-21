from sklearn.model_selection import * 
from sklearn.preprocessing import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

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


if __name__ == '__main__':
    print("4.4 특성 스케일 맞추기")
    """특성 스케일 조정은 전처리 파이프라인에서 잊어버리기 쉽지만 아주 중요한 단계
       Decision Tree와 Random forest는 특성 스케일 조정에 대해 걱정할 필요가 없는 몇안되는 머신러닝 알고리즘
       2장의 경사하강법(Gradient Descent) 알고리즘을 구현하면서 확인하였듯 대부분의 머신러닝 알고리즘과 
       최적화 알고리즘은 특성의 스케일이 같을 때 훨씬 성능이 좋음.
    """

    df_wine = load_wine_data()
    print("4.3 데이터셋을 훈련 데이터셋과 테스트 데이터셋으로 나누기")
    # 인덱스 1에서 인덱스 13까지의  특성(feture) 슬라이싱하여 넘파이 배열로 변환하여 X에 할당.
    # 첫번째 열의 클래스 레이블은 변수 y에 할당.
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    print("X(data)",X)
    print("#".center(20,'#'))
    print("Y(target)",y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y
                                                        , test_size= 0.3 #와인샘플의 30%가 테스트에 할당
                                                        , random_state= 0
                                                        , stratify=y)
    print("#".center(20,'#'))
    # 최소 최대 스케일 변환을 통한 정규화는 정해진 범위 내의 값이 필요할 때 유용
    mms = MinMaxScaler()
    X_train_norm = mms.fit_transform(X_train)
    X_test_norm = mms.transform(X_test)

    print(X_train_norm)
    print(X_test_norm)
    print("#".center(20,'#'))

    # 표준화를 사용하면 특성의 평균을 0에 맞추고 표준편차를 1로 만들어 정규분포와 같은 특성을 갖도록 만듬
    # 가중치를 더 쉽게 학습할 수 있음
    # 표준화는 이상치 정보가 유지되기 때문에 제한된 범위내로 데이터를 조정하는 최소, 최대 스케일 변환에 비해 
    # 알고리즘이 이상치에 덜 민감
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    print("#".center(20,'#'))

    rbs = RobustScaler()
    X_train_rbs = rbs.fit_transform(X_train)
    X_test_rbs = rbs.transform(X_test)
    print(X_test_rbs)

    ex = np.array([0, 1, 2, 3, 4, 5])

    print('표준화:', (ex - ex.mean()) / ex.std())

    # 판다스는 기본적으로 ddof=1를 사용합니다(샘플 표준 편차).
    # 반면 넘파이 std 메서드와 StandardScaler는 ddof=0를 사용합니다.

    # 정규화합니다
    print('정규화:', (ex - ex.min()) / (ex.max() - ex.min()))
    print('StandardScaler:', scale(ex))
    print('MinMaxScaler:', minmax_scale(ex))
    print('RobustScaler:', robust_scale(ex))
    print('MaxAbsScaler:', maxabs_scale(ex))

    # MaxAbsScaler는 각 특성별로 데이터를 최대 절댓값으로 나눕니다. 
    # 따라서 각 특성의 최댓값은 1이 됩니다. 
    # 전체 특성은 [-1, 1] 범위로 변경됩니다.
    mas = MaxAbsScaler()
    X_train_maxabs = mas.fit_transform(X_train)
    X_test_maxabs = mas.transform(X_test)

    # MaxAbsScaler, maxabs_scaler()는 데이터를 중앙에 맞추지 않기 때문에 
    # 희소 행렬을 사용할 수 있습니다.
    X_train_sparse = sparse.csr_matrix(X_train)
    X_train_maxabs = mas.fit_transform(X_train_sparse)

    # RobustScaler는 희소 행렬을 사용해 훈련할 수 없지만 변환은 가능합니다.
    X_train_robust = rbs.transform(X_train_sparse)

    # 마지막으로 Normalizer 클래스와 normalize() 함수는 특성이 아니라 샘플별로 정규화를
    #  수행합니다. 또한 희소 행렬도 처리할 수 있습니다. 
    # 기본적으로 샘플의 L2 노름이 1이 되도록 정규화합니다.
    nrm = Normalizer()
    x_train_l2 = nrm.fit_transform(X_train)

    ex_2f = np.vstack((ex[1:], ex[1:]**2))
    print(ex_2f)

    l2_norm = np.sqrt(np.sum(ex_2f ** 2, axis=1))
    print(l2_norm)
    print(ex_2f / l2_norm.reshape(-1, 1))