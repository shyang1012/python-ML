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
배깅: 부트스르랩 샘플링을 통한 분류 앙상블
앙상블에 있는 개별 분류기를 동일한 훈련 데이터셋으로 학습하는 것이 아니라 원본 훈련데이터셋에서
부트스트랩(bootstrap) 샘플(중복을 허용한 랜덤 샘플)을 뽑아서 사용
배깅은 bootstrap aggregating이라고도 함.

배깅의 개별분류기는 일반적으로 가지치기를 하지 않은 full decision tree를 사용

랜덤 포레스트와 배깅은 모두 기본적으로 부트스트랩 샘플링을 사용하기 때문에 분류기마다 사용하지 않은 여분을 샘플 존재
Out Of Bag(OOB): 분류기마다 사용하지 않은 여분을 샘플
- OOB를 사용하면 검증 세트를 만들지 않고 앙상블 모델을 평가할 수 있음.
- 사이킷런에서는 oob_score 매개변수를 True로 설정하면 이를 사용 가능(기본은 False)
- 사이킷련의 랜덤 포레스트는 분류일 경우 OOB 샘플에 대한 각 트리의 예측 확률을 누적하여 가장 큰 확률을 가진
- 클래스를 타깃과 비교하여 정확도 계산
- 회귀일 경우에는 각 트리의 예측 평균에 대한 R2 점수 계산
- 이점수는 oob_score_ 속성에 저장됨.
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

    tree = DecisionTreeClassifier(criterion='entropy', random_state=1) # 가지치기를 하지않은 decision tree

    # 500개의 가지치기를 하지 않은 decision tree를 학습하여 앙상블 만들기
    bag = BaggingClassifier(base_estimator=tree
                            ,n_estimators=500
                            ,max_samples=1.0
                            ,max_features=1.0
                            ,bootstrap=True
                            ,bootstrap_features=False
                            ,n_jobs=1
                            ,random_state=1
    )

    # 배깅과 가지치기 없는 단일 의사결정트리에서 훈련 데이터셋과 테스트 데이터셋의 예측정확도 비교
    X_tree = tree.fit(X_train, y_train)
    y_train_pred = X_tree.predict(X_train)
    y_test_pred = X_tree.predict(X_test)

    tree_train= accuracy_score(y_train, y_train_pred)
    tree_test= accuracy_score(y_test, y_test_pred)
    print("결정 트리의 훈련 정확도/테스트 정확도 %.3f/%.3f" %(tree_train, tree_test))

    X_bag = bag.fit(X_train, y_train)
    y_train_pred = X_bag.predict(X_train)
    y_test_pred = X_bag.predict(X_test)

    bag_train= accuracy_score(y_train, y_train_pred)
    bag_test= accuracy_score(y_test, y_test_pred)
    print("배깅의 훈련 정확도/테스트 정확도 %.3f/%.3f" %(bag_train, bag_test))

    # 결정트리와 배깅의 결정경계 비교
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

    for idx, clf, tt in zip([0, 1], [tree, bag],['Decision tree', 'Bagging']):
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

    plt.savefig(get_base_dir('images')+'/compare_model_performance.png', dpi = 300)
    plt.show()
    plt.close()




