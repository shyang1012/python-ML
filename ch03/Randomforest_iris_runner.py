from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from pydotplus import *
from sklearn.ensemble import *
from sklearn import tree
from sklearn.metrics import *
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))+"/"+"common")

# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin'

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

    print('클래스레이블:', np.unique(y))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y
    , test_size=0.3
    , random_state=1
    , stratify=y
    )

    print('y의 레이블 카운트', np.bincount(y))
    print('y_train의 레이블 카운트', np.bincount(y_train))
    print('y_test의 레이블 카운트', np.bincount(y_test))

    # feature nomalization
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # print("X_train_std", X_train_std)
    # print("X_test_std", X_test_std)

    forest = RandomForestClassifier(criterion='gini', n_estimators=25,n_jobs=2, random_state=1)
    forest.fit(X_train_std, y_train)



    y_pred = forest.predict(X_test_std)
    print('잘못 분류된 샘플 개수: %d' % (y_test != y_pred).sum())
    print("정확도 %.3f" % accuracy_score(y_test, y_pred))
    print("정확도 %.3f" % forest.score(X_test_std, y_test))

    target_names = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']
    print("confution matrix \n",multilabel_confusion_matrix(y_test, y_pred,labels=[0, 1, 2]))
    print(classification_report(y_test,y_pred, target_names=target_names))

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined_std = np.hstack((y_train, y_test))

    cmmnUtils = CmnUtils()

    cmmnUtils.plot_decision_regions_with_test_marker(X=X_combined_std, y= y_combined_std, classifier=forest, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/iris_decision_regisons_RandomForest_gini.png', dpi = 300)
    plt.show()
    plt.close()
 
