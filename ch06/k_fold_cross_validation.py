import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.decomposition import *
from sklearn.linear_model import *
from sklearn.pipeline import *

import os


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
    
   
    pipe_lr = make_pipeline(StandardScaler()
                            , PCA(n_components=2)
                            , LogisticRegression(random_state = 1)
                            )
    # kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
    # scores = []

    # for k, (train, test) in enumerate(kfold):
    #     pipe_lr.fit(X_train[train], y_train[train])
    #     score = pipe_lr.score(X_train[test],y_train[test])
    #     scores.append(score)
    #     print('폴드: %2d, 클래스 분포: %s, 정확도: %.3f' % (k+1,np.bincount(y_train[train]), score))

    # print('\nCV 정확도: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    #10 fold cross validation => 모델 생성
    scores = cross_validate(estimator=pipe_lr, 
                        X=X_train, 
                        y=y_train, 
                        scoring=['accuracy'], 
                        cv=10, 
                        n_jobs=-1)

    print('CV 정확도 점수: %s' % scores['test_accuracy'])
    print('CV 정확도: %.3f +/- %.3f' % (np.mean(scores['test_accuracy']), 
                                 np.std(scores['test_accuracy'])))

    # cross_val_predict 함수는 cross_val_score와 비슷한 인터페이스를 제공하지만 
    # 훈련 데이터셋의 각 샘플이 테스트 폴드가 되었을 때 만들어진 예측을 반환
    # method를 별도 지정하지 않으면 기본값은 'predict'
    # 기본 method 이외의 값은 ‘predict_proba’, ‘predict_log_proba’, ‘decision_function’ 등
    preds = cross_val_predict(estimator=pipe_lr,
                          X=X_train, 
                          y=y_train,
                          cv=10,
                          method='predict',
                          n_jobs=-1)
    print(preds[:10])

if __name__ == '__main__':
    main()
