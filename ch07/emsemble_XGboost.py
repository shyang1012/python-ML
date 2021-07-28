import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.tree import *
from sklearn.model_selection import *
import matplotlib.pyplot as plt
import os


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

    xgb = XGBClassifier(tree_method='hist',random_state=1)

    # ada와 약한학습 의사결정트리에서 훈련 데이터셋과 테스트 데이터셋의 예측정확도 비교
    X_tree = tree.fit(X_train, y_train)
    y_train_pred = X_tree.predict(X_train)
    y_test_pred = X_tree.predict(X_test)

    tree_train= accuracy_score(y_train, y_train_pred)
    tree_test= accuracy_score(y_test, y_test_pred)
    print("결정 트리의 훈련 정확도/테스트 정확도 %.3f/%.3f" %(tree_train, tree_test))

    X_xgb = xgb.fit(X_train, y_train)
    y_train_pred = X_xgb.predict(X_train)
    y_test_pred = X_xgb.predict(X_test)

    xgb_train= accuracy_score(y_train, y_train_pred)
    xgb_test= accuracy_score(y_test, y_test_pred)
    print("XGBoost 훈련 정확도/테스트 정확도 %.3f/%.3f" %(xgb_train, xgb_test))

    # 결정트리와 XGBoost의 결정경계 비교
    x_min = X_train[:, 0].min() -1
    x_max = X_train[:, 0].max() +1
    y_min = X_train[:, 1].min() -1
    y_max = X_train[:, 1].max() +1

    xx, yy = np.meshgrid( np.arange(x_min,x_max, 0.1)
                        , np.arange(y_min,y_max, 0.1)
    )

    f, axarr = plt.subplots(nrows=1, ncols=2, 
                        sharex='col', 
                        sharey='row', 
                        figsize=(8, 3))

    for idx, clf, tt in zip([0, 1], [tree, xgb],['Decision tree', 'XGBoost']):
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

    plt.savefig(get_base_dir('images')+'/compare_XGBoost_model_performance.png', dpi = 300)
    plt.show()
    plt.close()



