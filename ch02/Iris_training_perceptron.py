
from matplotlib import markers
from matplotlib import colors
from matplotlib.colors import ListedColormap
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Perceptron as pc


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


def drow_iris_plot(X, y):

    # 산점도(plot) 그리기, setosa - 음성, versicolor -양성
    plt.scatter(X[:50,0], X[:50, 1], color='red',marker='o', label='setosa')
    plt.scatter(X[50:100,0], X[50:100, 1], color='blue',marker='x', label='versicolor')

    plt.xlabel('sepal length [cm]')
    plt.ylabel('pepal length [cm]')
    plt.legend(loc ='upper left')

    plt.savefig(get_base_dir('images')+'/iris_scatterplot_perceptron.png', dpi = 300)
    plt.close()
    # plt.show()


def training_model(X, y):
    ppn = pc.Perceptron(eta=0.1, n_iter=10)
    
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.savefig(get_base_dir('images')+'/iris_eval_perceptron.png', dpi = 300)
    # plt.show()
    plt.close()
    return ppn
    
    
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
                    edgecolor=None)
    
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.savefig(get_base_dir('images')+'/iris_decision_regisons_perceptron.png', dpi = 300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    
    df = get_iris_data();
    y = df.iloc[0:100, 4].values
    y = np.where(y== 'Iris-setosa', -1, 1)

    X = df.iloc[0:100, [0, 2]].values

    drow_iris_plot(X, y)

    ppn = training_model(X, y)
    
    plot_decision_regions(X, y, ppn)

    
