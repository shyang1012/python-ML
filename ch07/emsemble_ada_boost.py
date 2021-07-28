import pandas as pd
import numpy as np
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.tree import *
from sklearn.model_selection import *
import matplotlib.pyplot as plt
import os
"""
Ada Boost(Adaptive Boosting)
부스트에서 앙상블은 약한 학습기(week learner)라고도 하는 매우 간단한 분류기로 구성
약한 학습기는 램덤 추축보다 조금 성능이 좋은 정도(예, depth = 1인 decision tree)

* 부스팅의 핵심 아이디어 *
  - 분류하기 어려운 훈련 샘플에 초점을 맞추는 것
  - 잘못 분류된 훈련 샘플을 그 약한 학습기가 학습하여 앙상블 성능 향상

부스팅 작동 원리
1. 훈련 데이터셋 D에서 중복을 허용하지 않고 랜덤한 부분집합 d_1을 뽑아 약한 학습기 C_1을 훈련
2. 훈련 데이터셋에서 중복을 허용하지 않고 두 번째 랜덤한 훈련 부분집합 d_2를 뽑고 이전에 잘못
  분류된 샘플의 50%를 더해서 약한 학습기 C_2를 학습
3. 훈련 데이터셋 D에서 C_1과 C_2에서 잘못 분류한 훈련샘플 d_3을 찾아 세 번째 약한 학습기인 
  C_3을 훈련
4. 약한 학습기 C_1, C_2, C_3를 다수결 투표로 연결. 
"""

def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path
    if not os.path.exists(result):
        os.makedirs(result)
    return result

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
    # 클래스 1 제외하기
    # feature는 ['Alcohol','OD280/OD315 of diluted wines']만 사용
    df_wine = df_wine[df_wine['Class label']!= 1]
    y = df_wine['Class label'].values
    X = df_wine[['Alcohol','OD280/OD315 of diluted wines']].values

    le = LabelEncoder()
    y = le.fit_transform(y)
    # print(y)
    
    # 80%는 훈련데이터셋, 20%는 테스트 데이터셋으로 나누기

    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    tree = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=1) # week learner 설정.

    # 500개의 week learner decision tree를 학습하여 앙상블 만들기
    ada = AdaBoostClassifier(base_estimator=tree
                            ,n_estimators=500
                            ,learning_rate=0.1
                            ,random_state=1
    )

    # ada와 약한학습 의사결정트리에서 훈련 데이터셋과 테스트 데이터셋의 예측정확도 비교
    X_tree = tree.fit(X_train, y_train)
    y_train_pred = X_tree.predict(X_train)
    y_test_pred = X_tree.predict(X_test)

    tree_train= accuracy_score(y_train, y_train_pred)
    tree_test= accuracy_score(y_test, y_test_pred)
    print("결정 트리의 훈련 정확도/테스트 정확도 %.3f/%.3f" %(tree_train, tree_test))

    X_ada = ada.fit(X_train, y_train)
    y_train_pred = X_ada.predict(X_train)
    y_test_pred = X_ada.predict(X_test)

    ada_train= accuracy_score(y_train, y_train_pred)
    ada_test= accuracy_score(y_test, y_test_pred)
    print("에이다의 훈련 정확도/테스트 정확도 %.3f/%.3f" %(ada_train, ada_test))

    # 결정트리와 에이다의 결정경계 비교
    x_min = X_train[:, 0].min() -1
    x_max = X_train[:, 0].max() +1
    y_min = X_train[:, 1].min() -1
    y_max = X_train[:, 1].max() +1

    xx, yy = np.meshgrid(np.arange(x_min,x_max, 0.1)
                        , np.arange(y_min,y_max, 0.1)
    )

    f, axarr = plt.subplots(nrows=1, ncols=2, 
                        sharex='col', 
                        sharey='row', 
                        figsize=(8, 3))

    for idx, clf, tt in zip([0, 1], [tree, ada],['Decision tree', 'Ada']):
        clf.fit(X_train, y_train)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[idx].contourf(xx, yy, Z, alpha=0.3)
        axarr[idx].scatter(X_train[y_train==0, 0]
                        , X_train[y_train== 0, 1]
                        , c='blue'
                        , marker='^'
        )
        axarr[idx].scatter(X_train[y_train==1, 0]
                        , X_train[y_train== 1, 1]
                        , c='green'
                        , marker='o'
        )

        axarr[idx].set_title(tt)

    axarr[0].set_ylabel('Alcohol', fontsize = 10)
    plt.tight_layout()
    plt.text(0, -0.149, s='OD280/OD315 of diluted wines'
    , ha='center'
    , va='center'
    , fontsize=10
    , transform=axarr[1].transAxes)

    plt.savefig(get_base_dir('images')+'/compare_ada_model_performance.png', dpi = 300)
    plt.show()
    plt.close()




