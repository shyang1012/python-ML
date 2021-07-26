from numpy.lib.function_base import interp
import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.decomposition import *
from sklearn.linear_model import *
from sklearn.pipeline import *
from sklearn.utils.fixes import *
from sklearn.metrics import *
import matplotlib.pyplot as plt

import os

from distutils.version import LooseVersion as Version
from scipy import __version__ as scipy_version


if scipy_version >= Version('1.4.1'):
    from numpy import interp
else:
    from scipy import interp



""" ROC(Receiver Operating Characteristic) Curve
    분류기의 임계값을 바꾸어 가며 계산된 FPR(FP/(FP+TN))과 TPR(Sesitivity(TP / (TP + FN)))점수를 기반으로 분류모델을 선택하는 유용한 도구
    - ROC 그래프의 대각선은 랜덤 추측으로 해석
    - 대각선 아래에 위치한 분류모델을 랜덤 추측보다 성능이 나쁨(쓸데 없는거, 버려~~)
    - ROC AUC(ROC Area Under the Curve): ROC Curve 아래 면적을 계산하여 분류모델의 성능 종합 가능(예측 정확도랑 비슷하게 사용)
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

    pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(penalty='l2', 
                                           random_state=1,
                                           C=100.0))
    
    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
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

    pre_scorer = make_scorer(score_func=precision_score, 
                         pos_label=1, 
                         greater_is_better=True, 
                         average='micro')
    print('다중 분류의 성능지표')
    print(pre_scorer)

    X_train2 = X_train[:, [4, 14]]
        

    cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))

    fig, ax = plt.subplots(figsize=(7, 5))

    mean_precision = 0.0
    mean_recall = np.linspace(0, 1, 100)

    for i, (train, test) in enumerate(cv):
        pipe_lr.fit(X_train2[train], y_train[train])
        pr_disp = plot_precision_recall_curve(
            pipe_lr, X_train2[test], y_train[test], 
            name=f'Fold {i}', ax=ax)
        mean_precision += interp(mean_recall, pr_disp.recall[::-1], 
                                pr_disp.precision[::-1])

    plt.plot([0, 1], [1, 0],
            linestyle='--', color=(0.6, 0.6, 0.6),
            label='Random guessing')

    mean_precision /= len(cv)
    mean_auc = auc(mean_recall, mean_precision)
    plt.plot(mean_recall, mean_precision, 'k--',
            label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.plot([0, 1, 1], [1, 1, 0],
            linestyle=':', color='black',
            label='Perfect performance')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.savefig(get_base_dir('images')+'/precision_recall_curve.png', dpi = 300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
        