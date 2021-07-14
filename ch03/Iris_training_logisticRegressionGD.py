from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from matplotlib import markers
from matplotlib import colors
from matplotlib.colors import ListedColormap
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import LogisticRegressionGD as logit


def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path;
    if not os.path.exists(result):
        os.makedirs(result)
    return result


def training_model(X, y):
    ig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ada_gd = logit.LogisticRegressionGD(n_iter=15, eta=0.01)
    
    ada_gd.fit(X, y)
    plt.plot(range(1, len(ada_gd.cost_)+1), ada_gd.cost_,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.savefig(get_base_dir('images')+'/iris_eval_logisticRegressionGD.png', dpi = 300)
    # plt.show()
    plt.close()
    return ada_gd
    
    
def plot_decision_regions(X, y, classifier, resolution=0.02):
    
    # 마커와 컬러맵을 설정합니다
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정 경계를 그립니다
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 꽃받침 길이 최소/최대
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # 꽃잎 길이 최소/최대
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 샘플의 산점도를 그립니다
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
    
    plt.title('LogisticRegression - Gradient Descent')
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/iris_decision_regisons_logisticRegressionGD.png', dpi = 300)
    plt.show()
    plt.close()

    # plt.plot(range(1, len(classifier.cost_) + 1), classifier.cost_, marker='o')
    # plt.xlabel('Epochs')
    # plt.ylabel('Sum-squared-error')
    # plt.savefig(get_base_dir('images')+'/iris_sum-squared-error_logisticRegressionGD.png', dpi = 300)

    # plt.tight_layout()
    # plt.show()
    # plt.close()
    

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

    X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
    y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

    lrgd = logit.LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd.fit(X_train_01_subset,
         y_train_01_subset)
    
    plot_decision_regions(X=X_train_01_subset, 
                      y=y_train_01_subset,
                      classifier=lrgd)
