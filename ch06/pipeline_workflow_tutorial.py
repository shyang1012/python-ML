import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.decomposition import *
from sklearn.linear_model import *
from sklearn.pipeline import *
from sklearn import set_config
from sklearn.utils._estimator_html_repr import *

from IPython.core.display import display, HTML 
import os
from html2image import Html2Image




def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path;
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

if __name__ == '__main__':
    
    df = load_breast_cancer_data()
    # X에 데이터 세트(30개 특성), y에 레이블 정보를 잘라서 담음
    # y = diagnosis, X는 radius_mean 컬럼부터 담음.
    X, y = df.iloc[:, 2
    :].values, df.iloc[:, 1].values
    
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

    # 파이프라인으로 변환기와 추정기 연결
    pipe_lr = make_pipeline(StandardScaler()
                            , PCA(n_components=2)
                            , LogisticRegression(random_state = 1)
                            )
    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    print('테스트 정확도: %.3f' % pipe_lr.score(X_test, y_test))
    set_config(display='diagram') #주피터에서는 잘 작동

    html = estimator_html_repr(pipe_lr)
    #html파일을 저장.
    # with open(get_base_dir('data')+"/pipeline_tutorial.html","w") as w :
    #     w.write(html)

    # with open(get_base_dir('data')+"/pipeline_tutorial.html","r") as r :
    #     pipeline_html = r.readlines()
    # pipeline_html = "".join(pipeline_html)

    # html을 이미지로 저장
    hti = Html2Image(size=(200, 200))
    hti.output_path=get_base_dir('images')
    hti.temp_path = get_base_dir('temp')
    hti.screenshot(html_str=html, save_as='pipeline_tutorial.png')