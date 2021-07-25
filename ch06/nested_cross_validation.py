import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.decomposition import *
from sklearn.svm import *
from sklearn.pipeline import *
from sklearn.utils.fixes import *

import os
""" Grid search와 K-fold Closs Validation을 함께 사용하면 머신 러니 모델의 성능 세부튜닝이 용이
    여러종류의 머신 러닝 알고리즘을 비교하려면 중첩교차검증(Nested cross validation) 방법 권장

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
    try:
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

        pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))
        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

        param_grid = [{'svc__C': param_range, 
               'svc__kernel': ['linear']},
              {'svc__C': param_range, 
               'svc__gamma': param_range, 
               'svc__kernel': ['rbf']}]

        gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  refit=True,
                  return_train_score=True,
                  cv=2,
                  n_jobs=-1)

        scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)


        print('CV 정확도: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

        
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()
    