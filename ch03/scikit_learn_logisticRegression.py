from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.linear_model import *
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))+"/"+"common")

from CmnUtils import CmnUtils


def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path;
    if not os.path.exists(result):
        os.makedirs(result)
    return result


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data[:,[2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # feature nomalization
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # 매개변수 C는 규제(Regularization;정규화) 하이퍼파라미터 람다 λ의 역수
    # C의 값이 작아지면 Regularization 강도가 증가.
    # Regularization은 정규화지만, 책에서는 규제로 번역하므로 해당 용어를 사용
    lr = LogisticRegression(C=100.0, random_state=1, multi_class='ovr')
    lr.fit(X_train_std, y_train)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined_std = np.hstack((y_train, y_test))

    print(lr.predict_proba(X_test_std[:3, :]))

    y_pred = lr.predict(X_test_std)

    print("confution matrix \n",confusion_matrix(y_test, y_pred))
    print("정확도: ",accuracy_score(y_test, y_pred))

    cmmnUtils = CmnUtils()

    cmmnUtils.plot_decision_regions_with_test_marker(X=X_combined_std, y= y_combined_std, classifier=lr, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/iris_decision_regisons_LogisticRegression_scikit.png', dpi = 300)
    # plt.show()
    plt.close()

    #L2 Regularization 효과 확인

    weights, params = [], []

    for c in np.arange(-5, 5):
        lr = LogisticRegression(C=10.0 ** c, random_state= 1, multi_class='ovr')
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        params.append(10.0**c)
    
    weights = np.array(weights)
    plt.plot(params, weights[:, 0], label='petal length')
    plt.plot(params, weights[:, 1], linestyle='--',label='petal width')
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.legend(loc='upper left')
    plt.xscale('log')
    plt.savefig(get_base_dir('images')+'/iris_weight_coefficient_LogisticRegression_scikit.png', dpi = 300)
    plt.show()
    plt.close()



    

  

    