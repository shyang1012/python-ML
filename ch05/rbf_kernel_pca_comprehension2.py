import os
from ch05.rbf_kernel_pca import RbfKernalPCA as KPCA
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.decomposition import PCA
import numpy as np



def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path;
    if not os.path.exists(result):
        os.makedirs(result)
    return result


def main():
    X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)

    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/make_circles.png', dpi = 300)
    # plt.show()
    plt.close()

    scikit_pca =PCA(n_components=2)
    X_spca = scikit_pca.fit_transform(X)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
    ax[0].scatter(X_spca[y == 0, 0], X_spca[y==0, 1], color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_spca[y == 1, 0], X_spca[y==1, 1], color='blue', marker='o', alpha=0.5)
    
    ax[1].scatter(X_spca[y == 0, 0], np.zeros((500, 1)) + 0.02,color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_spca[y == 1, 0], np.zeros((500, 1)) - 0.02,color='blue', marker='o', alpha=0.5)

    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')

    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/make_circles_sk_pca.png', dpi = 300)
    # plt.show()
    plt.close()

    X_kpca = KPCA(gamma=15, n_components=2)
    X_kpca = X_kpca.fit_transform(X)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
    ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],color='blue', marker='o', alpha=0.5)

    ax[1].scatter(X_kpca[y==0, 0], np.zeros((500, 1))+0.02, color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_kpca[y==1, 0], np.zeros((500, 1))-0.02,color='blue', marker='o', alpha=0.5)

    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')

    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/make_circles_kpca.png', dpi = 300)
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
    
