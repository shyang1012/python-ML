from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path
    if not os.path.exists(result):
        os.makedirs(result)
    return result

if __name__ == '__main__':
    X, y = make_blobs(n_samples=150, 
                  n_features=2, 
                  centers=3, 
                  cluster_std=0.5, 
                  shuffle=True, 
                  random_state=0)

    plt.scatter(X[:, 0], X[:, 1], c='white', marker='o', edgecolor='black', s=50)
    plt.grid()
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/keans-data.png', dpi = 300)
    # plt.show()
    plt.close()

    km = KMeans(n_clusters=3, 
            init='random', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)

    y_km = km.fit_predict(X)

    plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='Cluster 1')
    plt.scatter(X[y_km == 1, 0],
                X[y_km == 1, 1],
                s=50, c='orange',
                marker='o', edgecolor='black',
                label='Cluster 2')
    plt.scatter(X[y_km == 2, 0],
                X[y_km == 2, 1],
                s=50, c='lightblue',
                marker='v', edgecolor='black',
                label='Cluster 3')
    plt.scatter(km.cluster_centers_[:, 0],
                km.cluster_centers_[:, 1],
                s=250, marker='*',
                c='red', edgecolor='black',
                label='Centroids')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/keans-result.png', dpi = 300)
    # plt.show()
    plt.close()

    print('왜곡: %.2f' % km.inertia_)

    distortions = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, 
                    init='k-means++', 
                    n_init=10, 
                    max_iter=300, 
                    random_state=0)
        km.fit(X)
        distortions.append(km.inertia_)
    
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/keans-distortion.png', dpi = 300)
    plt.show()
    plt.close()