from sklearn.model_selection import * 
from sklearn.preprocessing import *
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import sbs as sb

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
    
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    # print("X(data)",X)
    # print("Y(target)",y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y
                                                        , test_size= 0.3 #와인샘플의 30%가 테스트에 할당
                                                        , random_state= 0
                                                        , stratify=y)
    
    # C = 1.0이 기본
    # 규제 효과를 높이거나 낮추려면 C 값을 증가시기커나 감소
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)

    # feature selection
    sbs= sb.SBS(knn, k_features=1)
    sbs.fit(X_train_std, y_train)

    # 선택한 특성의 성능 출력

    k_feat= [len(k) for k in sbs.subsets_]

    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.ylim([0.7, 1.02])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.tight_layout()
    # plt.savefig('images/04_08.png', dpi=300)
    plt.show()

    k3 = list(sbs.subsets_[10])
    print(df_wine.columns[1:][k3])

    knn.fit(X_train_std, y_train)
    print('훈련 정확도:', knn.score(X_train_std, y_train))
    print('테스트 정확도:', knn.score(X_test_std, y_test))

    knn.fit(X_train_std[:, k3], y_train)
    print('훈련 정확도:', knn.score(X_train_std[:, k3], y_train))
    print('테스트 정확도:', knn.score(X_test_std[:, k3], y_test))