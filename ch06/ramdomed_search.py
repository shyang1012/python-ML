import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.decomposition import *
from sklearn.svm import *
from sklearn.pipeline import *
from sklearn.utils.fixes import *

import os
"""랜덤 서치는 일반적으로 그리드 서치와 비슷한 성능을 내지만 훨씬 비용과 시간이 적게 듬
    특히 랜덤 서치에서 60개의 파라미터 조합이 있다면 최적의 성능에서 5% 이내에 있는 
    솔루션을 얻을 확률은 95%
    매개변수 탐색 범위가 넓거나 규제 매개변수 C와 같이 연속적인 값을 탐색해야 하는 경우에 
    RandomizedSearchCV가 GridSearchCV보다 더 효율적
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
        distribution = loguniform(0.0001, 1000.0)

        param_dist = [{'svc__C': distribution, 
               'svc__kernel': ['linear']},
              {'svc__C': distribution, 
               'svc__gamma': distribution, 
               'svc__kernel': ['rbf']}]

        rs = RandomizedSearchCV(estimator=pipe_svc,
                        param_distributions=param_dist,
                        n_iter=30,
                        cv=10,
                        random_state=1,
                        return_train_score=True,
                        n_jobs=-1)

        rs = rs.fit(X_train, y_train)
        print(rs.best_score_)
        print(rs.best_params_)
        clf = rs.best_estimator_
        print('테스트 정확도: %.3f' % clf.score(X_test, y_test))
        print('첫 번째 폴드의 점수:',rs.cv_results_['split0_train_score'])
        print('첫 번째 폴드의 테스트점수:',rs.cv_results_['split0_test_score'])
        print('전체 훈련 점수의 평균 값:',rs.cv_results_['mean_train_score'])

        
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()
    