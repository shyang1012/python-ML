import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.decomposition import *
from sklearn.linear_model import *
from sklearn.pipeline import *
import matplotlib.pyplot as plt

import os

"""검증 곡선으로 과대적합(overfit)과 과소적합(underfit) 조사
    검증곡선은 학습곡선과 관련이 있지만 샘플 크기의 함수로 훈련 정확도와 테스트 정확도를 그리는 대신
    모델 파라미터 값의 함수로 그래프를 그림
    예를 들어 로지스틱 회기에 있는 규제 매개변수 C
    검증곡선은 기본적으로 K fold Cross Validation을 사용하여 모델성능을 추정

    C(규제값) 값이 바뀜에 따라 정확도의 차이가 미묘하지만 규제강도를 높이면(C값을 줄이면)
    모델이 데이터에 조금 과소적합되는 것을 볼 수 있음.
    규제강도가 낮아지는 큰 C값에서는 모델이 데이터에 조금 과대적합되는 경향을 보임.
    이 예제에서 적절한 C값은 0.01과 0.1 사이
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

        pipe_lr = make_pipeline(StandardScaler(),
                            LogisticRegression(penalty='l2', random_state=1,
                                            max_iter=10000))

        param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

        train_scores, test_scores = validation_curve(
                estimator=pipe_lr, 
                X=X_train, 
                y=y_train, 
                param_name='logisticregression__C', 
                param_range=param_range,
                cv=10)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)


        print("train_mean:",train_mean, "train_std:", train_std)
        print("test_mean:",test_mean, "test_std:", test_std)

        plt.plot(param_range, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='Training accuracy')

        plt.fill_between(param_range, train_mean + train_std,
                        train_mean - train_std, alpha=0.15,
                        color='blue')

        plt.plot(param_range, test_mean, 
                color='green', linestyle='--', 
                marker='s', markersize=5, 
                label='Validation accuracy')

        plt.fill_between(param_range, 
                        test_mean + test_std,
                        test_mean - test_std, 
                        alpha=0.15, color='green')

        plt.grid()
        plt.xscale('log')
        plt.legend(loc='lower right')
        plt.xlabel('Parameter C')
        plt.ylabel('Accuracy')
        plt.ylim([0.8, 1.0])
        plt.tight_layout()
        plt.savefig(get_base_dir('images')+'/validation_curves.png', dpi = 300)
        plt.show()
        plt.close()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()
    