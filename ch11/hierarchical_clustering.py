import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path
    if not os.path.exists(result):
        os.makedirs(result)
    return result


if __name__ == '__main__':
    np.random.seed(123)
    variable = ['X', 'Y', 'Z']
    labels = ['ID_0', 'ID_1', 'ID_2','ID_3', 'ID_4']
    
    X = np.random.random_sample([5, 3]) * 10

    df = pd.DataFrame(X, columns=variable, index=labels)
    print(df)

    row_dist = pd.DataFrame(squareform(
        pdist(df, metric='euclidean')),
        columns = labels, index = labels
        )
    
    print(row_dist)

    # 1. 잘못된 방식: squareform 거리 행렬
    # row_clusters = linkage(row_dist, method='complete', metric='euclidean')
    # pd.DataFrame(row_clusters,
    #          columns=['row label 1', 'row label 2',
    #                   'distance', 'no. of items in clust.'],
    #          index=['cluster %d' % (i + 1)
    #                 for i in range(row_clusters.shape[0])])

    # 2. 올바른 방식: 축약된 거리 행렬

    # row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
    # df= pd.DataFrame(row_clusters,
    #             columns=['row label 1', 'row label 2',
    #                     'distance', 'no. of items in clust.'],
    #             index=['cluster %d' % (i + 1) 
    #                     for i in range(row_clusters.shape[0])])
    # print(df)

    # 3. 올바른 방식: 입력 샘플 행렬

    row_clusters = linkage(df.values, method='complete', metric='euclidean')
    pd.DataFrame(row_clusters,
                columns=['row label 1', 'row label 2',
                        'distance', 'no. of items in clust.'],
                index=['cluster %d' % (i + 1)
                        for i in range(row_clusters.shape[0])])
    
    # 검은색 덴드로그램 만들기 (1/2 부분만)
    row_dendr = dendrogram(row_clusters, 
                       labels=labels,
                       # make dendrogram black (part 2/2)
                       # color_threshold=np.inf
                       )
    plt.tight_layout()
    plt.ylabel('Euclidean distance')
    plt.savefig(get_base_dir('images')+'/hierarchical-diagram.png', dpi = 300)
    plt.close()

    fig = plt.figure(figsize=(8, 8), facecolor='white')
    axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])

    # 노트: matplotlib < v1.5.1일 때는 use orientation='right'를 사용하세요
    row_dendr = dendrogram(row_clusters, orientation='left')
    print(row_dendr)

    # df_rowclust = df.iloc[row_dendr['leaves']]
    data= sorted(row_dendr['leaves'][::-1])
    print('data:',data)

    df_rowclust = df.iloc[row_dendr['leaves'][::-1]]


    print(df_rowclust)
   
    axd.set_xticks([])
    axd.set_yticks([])

    # 덴드로그램의 축을 제거합니다.
    for i in axd.spines.values():
        i.set_visible(False)

    # 히트맵을 출력합니다.
    axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])  # x-위치, y-위치, 너비, 높이
    cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
    fig.colorbar(cax)
    axm.set_xticklabels([''] + list(df_rowclust.columns))
    axm.set_yticklabels([''] + list(df_rowclust.index))

    plt.savefig(get_base_dir('images')+'/hierarchical-heatmap.png', dpi = 300)
    # plt.show()
    plt.close()

    # 사이킷런에서 병합 군집 적용
    ac = AgglomerativeClustering(n_clusters=3, 
                             affinity='euclidean', 
                             linkage='complete')
    labels = ac.fit_predict(X)
    print('클러스터 레이블: %s' % labels)

    ac = AgglomerativeClustering(n_clusters=2, 
                             affinity='euclidean', 
                             linkage='complete')
    labels = ac.fit_predict(X)
    print('클러스터 레이블: %s' % labels)

   


