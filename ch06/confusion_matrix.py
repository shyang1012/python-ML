import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.decomposition import *
from sklearn.svm import *
from sklearn.pipeline import *
from sklearn.utils.fixes import *
from sklearn.metrics import *
import matplotlib.pyplot as plt

import os
""" Confusion Matrix(혼돈행렬, 오차행렬)
                                Predicted class
                            Class = Y          Class = N
    Actual  Class= Y    True Positive          False Negative
    Class   Class= N    False Positive         True Negative

    Precision(정밀도)   = TP / (TP + FP)  # how many are really positive?
    Recall(재현율)      = TP / (TP + FN)  # how many are predicted as positive?
    Specificity(특이도) = TN / (TN + FP)  # how many got negative results?
    Sesitivity(민감도)  = TP / (TP + FN)  # how many got positive test results?

    Error Rate(예측 오차;ERR) = (FP + FN) / (TP + FN + FP + TN)  
    Accuracy(예측 정확도;ACC) = (TP + TN) / (TP + FN + FP + TN) = 1 - ERR

    F1-score(precision 과 recall의 조화평균) = 2 * ((Precision * Recall) / (Precision + Recall))
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
    pipe_svc.fit(X_train, y_train)
    y_pred = pipe_svc.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred = y_pred, labels=[1, 0])
    tp, fn, fp, tn = confmat[0,0], confmat[0,1], confmat[1,0], confmat[1,1]
    
    print('Precision(정밀도): %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
    print('Recall(재현율): %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
    print("Specificity(특이도):%.3f" % (tn/(tn+fp)))
    print("Sesitivity(민감도):%.3f" % (tp/(tp+fn)))
    print('F1-score(precision 과 recall의 조화평균): %.3f' % f1_score(y_true=y_test, y_pred=y_pred))
    # print('Accuracy(예측정확도;ACC): %.3f' % ((tp + tn) / (tp + fn + fp + tn)))
    print('Accuracy(예측정확도;ACC): %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred))
    print("Confusion Matrix")
    print(confmat)
    print(confmat[0,0])
  


    scorer = make_scorer(f1_score, pos_label=0)

    c_gamma_range = [0.01, 0.1, 1.0, 10.0]

    param_grid = [{'svc__C': c_gamma_range,
                'svc__kernel': ['linear']},
                {'svc__C': c_gamma_range,
                'svc__gamma': c_gamma_range,
                'svc__kernel': ['rbf']}]

    gs = GridSearchCV(estimator=pipe_svc,
                    param_grid=param_grid,
                    scoring=scorer,
                    cv=10,
                    n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    print('GridSearch f1_score: %.3f' % gs.best_score_)
    print(gs.best_params_)


    plot_confusion_matrix(pipe_svc, X_test, y_test, labels=[1, 0])
    plt.savefig(get_base_dir('images')+'/confution_metrix.png', dpi = 300)
    plt.show()
    plt.close()

    plot_confusion_matrix(pipe_svc, X_test, y_test, normalize='all', labels=[1, 0])
    plt.savefig(get_base_dir('images')+'/confution_metrix_normalize.png', dpi = 300)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    main()
        