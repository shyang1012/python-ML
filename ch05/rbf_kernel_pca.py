from scipy.linalg.decomp import eigvals
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import numpy as np

from distutils.version import LooseVersion as Version
from scipy import __version__ as scipy_version

# scipy 2.0.0에서 삭제될 예정이므로 대신 numpy.exp를 사용합니다.
if scipy_version >= Version('1.4.1'):
    from numpy import exp
else:
    from scipy import exp


class RbfKernalPCA(object):
    
    def __init__(self, gamma = 15, n_components = 2):
        """RBF 커널 PCA 구현

        Args:
            gamma (int, optional): RBF 커널 튜닝 매개변수. Defaults to 15.
            n_components (int, optional): 반환할 주성분 개수. Defaults to 2.
        """
        self.gamma = gamma
        self.n_components = n_components
    

    def fit_transform(self,X):
        """[summary]

        Args:
            X ([ndarray]): {넘파이 ndarray}, shape = [n_samples, n_features]

        Returns:
           alphas ([ndarray]): shape = [n_samples, k_features] 투영된 데이터셋
        """


        # MxN 차원의 데이터셋에서 샘플 간의 유클리디안 거리의 제곱을 계산합니다.
        sq_dists = pdist(X,'sqeuclidean')

        # 샘플 간의 거리를 정방 대칭 행렬로 변환합니다.
        mat_sq_dists = squareform(sq_dists)


        # 커널 행렬을 계산합니다.
        K = exp(-self.gamma * mat_sq_dists)

        # 커널 행렬을 중앙에 맞춥니다.
        N = K.shape[0]
        one_n = np.ones((N, N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)


        # 중앙에 맞춰진 커널 행렬의 고윳값과 고유벡터를 구합니다.
        # scipy.linalg.eigh 함수는 오름차순으로 반환합니다.
        eigvals, eigvecs = eigh(K)
        eigvals, eigvecs = eigvals[::-1], eigvecs[:,::-1]


        # 최상위 k 개의 고유벡터를 선택합니다(결과값은 투영된 샘플입니다).
        alphas = np.column_stack([eigvecs[:, i] for i in range(self.n_components)])

        return alphas 

    
    def fit_Transform_with_lambdas(self,X):
        """[summary]

        Args:
            X ([ndarray]): {넘파이 ndarray}, shape = [n_samples, n_features]

        Returns:
            alphas: {넘파이 ndarray}, shape = [n_samples, k_features]
            투영된 데이터셋
            
            lambdas: list
            고윳값
        """


        # MxN 차원의 데이터셋에서 샘플 간의 유클리디안 거리의 제곱을 계산합니다.
        sq_dists = pdist(X,'sqeuclidean')

        # 샘플 간의 거리를 정방 대칭 행렬로 변환합니다.
        mat_sq_dists = squareform(sq_dists)


        # 커널 행렬을 계산합니다.
        K = exp(-self.gamma * mat_sq_dists)

        # 커널 행렬을 중앙에 맞춥니다.
        N = K.shape[0]
        one_n = np.ones((N, N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)


        # 중앙에 맞춰진 커널 행렬의 고윳값과 고유벡터를 구합니다.
        # scipy.linalg.eigh 함수는 오름차순으로 반환합니다.
        eigvals, eigvecs = eigh(K)
        eigvals, eigvecs = eigvals[::-1], eigvecs[:,::-1]


        # 최상위 k 개의 고유벡터를 선택합니다(결과값은 투영된 샘플입니다).
        alphas = np.column_stack([eigvecs[:, i] for i in range(self.n_components)])

        # 고유 벡터에 상응하는 고윳값을 선택합니다.
        lambdas = [eigvals[i] for i in range(self.n_components)]

        return alphas, lambdas
