from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.svm import *
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


def make_mock_data():
    np.random.seed(1)
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:,0]>0, X_xor[:,1]>0)
    y_xor = np.where(y_xor, 1, -1)
    return X_xor, y_xor


 


if __name__ == '__main__':
    X_xor, y_xor = make_mock_data()

    print("X_xor",X_xor)
    print("y_xor",y_xor)

    plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor==1, 1], c='b', marker='x', label='1')
    plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor==-1, 1], c='r', marker='s', label='-1')
    plt.xlim([-3,4])
    plt.ylim([-3,3])
    plt.legend(loc='best')
    plt.tight_layout()
    # plt.show()
    plt.savefig(get_base_dir('images')+'/mockdata_scatter.png', dpi = 300)
    plt.close()

    X = X_xor
    y = y_xor
    
    svm = SVC(kernel='rbf', gamma=0.25, C=1.0)
    svm.fit(X_xor, y_xor)

    y_pred = svm.predict(X_xor)
    print('잘못 분류된 샘플 개수: %d' % (y_xor != y_pred).sum())
    print("정확도 %.3f" % accuracy_score(y_xor, y_pred))
    print("confution matrix \n",confusion_matrix(y_xor, y_pred))

    cmmnUtils = CmnUtils()

    cmmnUtils.plot_decision_regions(X=X_xor, y= y_xor, classifier=svm, test_idx=range(105, 150))
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/mockdata_decision_regisons_svm_rbf.png', dpi = 300)
    plt.show()
    plt.close()
    

