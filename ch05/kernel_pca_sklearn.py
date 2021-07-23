from sklearn.decomposition import KernelPCA
from sklearn.datasets import *
import matplotlib.pyplot as plt
import os


def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path;
    if not os.path.exists(result):
        os.makedirs(result)
    return result



def main():
    X, y = make_moons(n_samples=100, random_state=123)
    scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
    X_scikit_kpca = scikit_kpca.fit_transform(X)

    plt.scatter(X_scikit_kpca[y==0, 0], X_scikit_kpca[y==0, 1], color = 'red', marker='^', alpha=0.5)
    plt.scatter(X_scikit_kpca[y==1, 0], X_scikit_kpca[y==1, 1], color = 'blue', marker='o', alpha=0.5)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/make_moons_kpca_sklearn.png', dpi = 300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
    