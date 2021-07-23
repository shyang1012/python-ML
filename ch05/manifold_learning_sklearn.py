import matplotlib.pyplot as plt
import os
from sklearn.manifold import *
from sklearn.datasets import *

def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path;
    if not os.path.exists(result):
        os.makedirs(result)
    return result

def plot_manifold(X, y, savefig_name=None):
    
    plt.scatter(X[y == 0, 0], X[y == 0, 1], 
                color='red', marker='^', alpha=0.5)
    plt.scatter(X[y == 1, 0], X[y == 1, 1],
                color='blue', marker='o', alpha=0.5)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    if savefig_name is not None:
        plt.savefig(savefig_name, dpi=300)
    else:
        plt.show()
    plt.close()

def main():
    X, y = make_moons(n_samples=100, random_state=123)

    # 지역 선형 임베딩 (Locally Linear Embedding, LLE): 이웃한 샘플 
    # 간의 거리를 유지하는 저차원 투영 검색
    lle = LocallyLinearEmbedding(n_components=2, random_state=1)
    X_lle = lle.fit_transform(X)
    plot_manifold(X_lle, y,get_base_dir('images')+'/lle_moon.png')

    # t-SNE(t-distributed Stochastic Neighbor Embedding)는 데이터 포인트 간의 유사도를 
    # 결합 확률(joint probability)로 변환하고, 저차원과 고차원의 확률 사이에서 
    # 쿨백-라이블러(Kullback-Leibler) 발산을 최소화
    # t-SNE는 특히 고차원 데이터셋을 시각화하는데 뛰어난 성능
    tsne = TSNE(n_components=2, random_state=1)
    X_tsne = tsne.fit_transform(X)

    plot_manifold(X_tsne, y, get_base_dir('images')+'/tsne_moon.png')


if __name__ == '__main__':
    main()
    
