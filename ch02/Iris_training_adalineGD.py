
from matplotlib import markers
from matplotlib import colors
from matplotlib.colors import ListedColormap
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import AdalineGD as ada


def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path;
    if not os.path.exists(result):
        os.makedirs(result)
    return result


def get_iris_data():
    # s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    s = get_base_dir('data')+'/iris.data'
    print("URL:", s)

    df = pd.read_csv(s, header=None, encoding='utf-8')
    print(df.tail())
    
    return df


def drow_adaline_Learning_rate(X, y):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ada1 = ada.AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    ada2 = ada.AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.0001')

    plt.savefig(get_base_dir('images')+'/iris_adaline_Learning_rate.png', dpi = 300)
    plt.close()
    # plt.show()


def training_model(X, y):
    ig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ada_gd = ada.AdalineGD(n_iter=15, eta=0.01)
    
    ada_gd.fit(X, y)
    plt.plot(range(1, len(ada_gd.cost_)+1), ada_gd.cost_,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.savefig(get_base_dir('images')+'/iris_eval_adalineGD.png', dpi = 300)
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
    
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/iris_decision_regisons_adalineGD.png', dpi = 300)
    # plt.show()
    plt.close()

    plt.plot(range(1, len(classifier.cost_) + 1), classifier.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.savefig(get_base_dir('images')+'/iris_sum-squared-error_adalineGD.png', dpi = 300)

    plt.tight_layout()
    plt.show()
    plt.close()
    

if __name__ == '__main__':
    
    df = get_iris_data();
    y = df.iloc[0:100, 4].values
    y = np.where(y== 'Iris-setosa', -1, 1)

    X = df.iloc[0:100, [0, 2]].values

    # 특성을 표준화합니다.
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    drow_adaline_Learning_rate(X, y)

    ada_gd = ada.AdalineGD(n_iter=15, eta=0.01)
    ada_gd.fit(X_std, y)
    
    plot_decision_regions(X_std, y, classifier=ada_gd)
