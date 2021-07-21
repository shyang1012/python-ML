from sklearn.model_selection import * 
from sklearn.preprocessing import *
from sklearn.ensemble import *
from sklearn.feature_selection import *
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

    feat_labels = df_wine.columns[1:]

    forest = RandomForestClassifier(n_estimators=500, random_state=1)

    forest.fit(X_train, y_train)

    importances = forest.feature_importances_

    indices =np.argsort(importances)[::-1]

    
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))
    

    sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
    X_selected = sfm.transform(X_train)
    print('이 임계 조건을 만족하는 샘플의 수:', X_selected.shape[1])
    for f in range(X_selected.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))
    

    rfe = RFE(forest, n_features_to_select=5)
    print(rfe.fit(X_train, y_train))
    print("rfe.ranking_",rfe.ranking_)

    f_mask = rfe.support_

    importances2 = rfe.estimator_.feature_importances_
    indices2 = np.argsort(importances2)[::-1]

    for i in indices2:
        print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[f_mask][i], 
                            importances2[i]))

    plt.title('Feature Importance')
    plt.bar(range(X_train.shape[1]), 
            importances[indices],
            align='center')

    
    plt.xticks(range(X_train.shape[1]), 
            feat_labels[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    # plt.savefig('images/04_09.png', dpi=300)
    plt.show()

    