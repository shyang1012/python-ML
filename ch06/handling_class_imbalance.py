import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.decomposition import *
from sklearn.svm import *
from sklearn.pipeline import *
from sklearn.utils.fixes import *
from sklearn.metrics import *
from sklearn.utils import resample

import os
""" 불균형한 클래스 다루기
"""

def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path
    if not os.path.exists(result):
        os.makedirs(result)
    return result


def load_breast_cancer_data():
    # 판다스를 사용하여 UCI 서버에서 직접 데이터 셋을 읽어 들임
    df_wdbc = pd.read_csv('https://archive.ics.uci.edu/ml/'
                    'machine-learning-databases'
                    '/breast-cancer-wisconsin/wdbc.data',
                      header=None)
    df_wdbc.columns = ["id"
                    ,"diagnosis"
                    ,"radius_mean"
                    ,"texture_mean"
                    ,"perimeter_mean"
                    ,"area_mean"
                    ,"smoothness_mean"
                    ,"compactness_mean"
                    ,"concavity_mean"
                    ,"concave_points_mean"
                    ,"symmetry_mean"
                    ,"fractal_dimension_mean"
                    ,"radius_se"
                    ,"texture_se"
                    ,"perimeter_se"
                    ,"area_se"
                    ,"smoothness_se"
                    ,"compactness_se"
                    ,"concavity_se"
                    ,"concave_points_se"
                    ,"symmetry_se"
                    ,"fractal_dimension_se"
                    ,"radius_worst"
                    ,"texture_worst"
                    ,"perimeter_worst"
                    ,"area_worst"
                    ,"smoothness_worst"
                    ,"compactness_worst"
                    ,"concavity_worst"
                    ,"concave_points_worst"
                    ,"symmetry_worst"
                    ,"fractal_dimension_worst"
                ]
    print('diagnosis', np.unique(df_wdbc['diagnosis']))
    print(df_wdbc.head())
    return df_wdbc


def main():

    df = load_breast_cancer_data()
    # X에 데이터 세트(30개 특성), y에 레이블 정보를 잘라서 담음
    # y = diagnosis, X는 radius_mean 컬럼부터 담음.
    X, y = df.iloc[:, 2:].values, df.iloc[:, 1].values
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(le.classes_)
    converted_label = le.transform(['M', 'B'])
    print(converted_label)

    # 훈련데이터셋 80%, 테스트데이터셋 20%로 나눔
    X_train, X_test, y_train, y_test = train_test_split(X
                                        , y
                                        , test_size=0.20
                                        ,stratify=y
                                        , random_state=1)
    X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
    y_imb = np.hstack((y[y == 0], y[y == 1][:40]))

    y_pred = np.zeros(y_imb.shape[0])
    print(np.mean(y_pred == y_imb) * 100)

    print('샘플링하기 전 클래스 1의 샘플 개수:', X_imb[y_imb == 1].shape[0])

    X_upsampled, y_upsampled = resample(X_imb[y_imb == 1],
                                    y_imb[y_imb == 1],
                                    replace=True,
                                    n_samples=X_imb[y_imb == 0].shape[0],
                                    random_state=123)

    print('샘플링하기 후 클래스 1의 샘플 개수:', X_upsampled.shape[0])

    X_bal = np.vstack((X[y == 0], X_upsampled))
    y_bal = np.hstack((y[y == 0], y_upsampled))
    y_pred = np.zeros(y_bal.shape[0])
    print(np.mean(y_pred == y_bal) * 100)


if __name__ == '__main__':
    main()
        