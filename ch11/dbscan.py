import os
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path
    if not os.path.exists(result):
        os.makedirs(result)
    return result



if __name__ == '__main__':
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    plt.scatter(X[:, 0], X[:, 1])
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/make_moons.png', dpi = 300)
    plt.close()

    # k-means, hierarchical-clustering

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    km = KMeans(n_clusters=2, random_state=0)
    y_km = km.fit_predict(X)
    ax1.scatter(X[y_km == 0, 0], X[y_km == 0, 1],
                edgecolor='black',
                c='lightblue', marker='o', s=40, label='cluster 1')
    ax1.scatter(X[y_km == 1, 0], X[y_km == 1, 1],
                edgecolor='black',
                c='red', marker='s', s=40, label='cluster 2')
    ax1.set_title('K-means clustering')

    ac = AgglomerativeClustering(n_clusters=2,
                                affinity='euclidean',
                                linkage='complete')
    y_ac = ac.fit_predict(X)
    ax2.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], c='lightblue',
                edgecolor='black',
                marker='o', s=40, label='Cluster 1')
    ax2.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], c='red',
                edgecolor='black',
                marker='s', s=40, label='Cluster 2')
    ax2.set_title('Agglomerative clustering')

    plt.legend()
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/k-means_hierarchical.png', dpi = 300)
    plt.close()

    db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
    y_db = db.fit_predict(X)
    plt.scatter(X[y_db == 0, 0], X[y_db == 0, 1],
                c='lightblue', marker='o', s=40,
                edgecolor='black', 
                label='Cluster 1')
    plt.scatter(X[y_db == 1, 0], X[y_db == 1, 1],
                c='red', marker='s', s=40,
                edgecolor='black', 
                label='Cluster 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/dbscan.png', dpi = 300)
    plt.show()
    plt.close()