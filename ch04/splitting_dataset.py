from sklearn.model_selection import * 
from sklearn.preprocessing import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    df_wine = load_wine_data()
    print("4.3 데이터셋을 훈련 데이터셋과 테스트 데이터셋으로 나누기")
    # 인덱스 1에서 인덱스 13까지의  특성(feture) 슬라이싱하여 넘파이 배열로 변환하여 X에 할당.
    # 첫번째 열의 클래스 레이블은 변수 y에 할당.
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    print("X(data)",X)
    print("Y(target)",y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y
                                                        , test_size= 0.3 #와인샘플의 30%가 테스트에 할당
                                                        , random_state= 0
                                                        , stratify=y)

    print("훈련 데이터")
    print(len(X_train))
    print(len(y_train))
    print("시험데이터")
    print(len(X_test))
    print(len(y_test))

   
