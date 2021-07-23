import os
from ch05.rbf_kernel_pca import RbfKernalPCA as KPCA
import matplotlib.pyplot as plt
from sklearn.datasets import *
import numpy as np



def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path;
    if not os.path.exists(result):
        os.makedirs(result)
    return result

def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)

def main():
    X, y = make_moons(n_samples=100, random_state=123)

    X_kpca = KPCA(gamma=15, n_components=1)
    alphas, lambdas = X_kpca.fit_Transform_with_lambdas(X)

    x_new = X[25]
    print(x_new)

    x_proj = alphas[25] # 원본 투영
    print(x_proj)

    # 새로운 데이터포인트를 투영합니다.
    x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
    print(x_reproj)

    plt.scatter(alphas[y == 0, 0], np.zeros((50)),color='red', marker='^', alpha=0.5)
    plt.scatter(alphas[y == 1, 0], np.zeros((50)),color='blue', marker='o', alpha=0.5)
    plt.scatter(x_proj, 0, color='black',label='Original projection of point X[25]', marker='^', s=100)
    plt.scatter(x_reproj, 0, color='green',label='Remapped point X[25]', marker='x', s=500)
    plt.yticks([], [])
    plt.legend(scatterpoints=1)

    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/make_moons_kpca_project.png', dpi = 300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
    
