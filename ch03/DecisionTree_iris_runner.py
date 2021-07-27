from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from pydotplus import *
from sklearn.tree import *
from sklearn import tree
from sklearn.metrics import *
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))+"/"+"common")

# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin'

from CmnUtils import CmnUtils



def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path;
    if not os.path.exists(result):
        os.makedirs(result)
    return result

if __name__ == '__main__':
    print(sys.path)

    iris = datasets.load_iris()
    X = iris.data[:,[2, 3]]
    y = iris.target

    print('클래스레이블:', np.unique(y))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y
    , test_size=0.3
    , random_state=1
    , stratify=y
    )

    print('y의 레이블 카운트', np.bincount(y))
    print('y_train의 레이블 카운트', np.bincount(y_train))
    print('y_test의 레이블 카운트', np.bincount(y_test))

    # feature nomalization
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # print("X_train_std", X_train_std)
    # print("X_test_std", X_test_std)

    tree_model = DecisionTreeClassifier(criterion='gini', ccp_alpha=0.05, random_state=0)
    # tree_model = DecisionTreeClassifier(random_state=0)
    path = tree_model.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
   
    #  적정 ccp_alpha 값을 찾기 위한 소스 시작
    # fig, ax = plt.subplots()
    # ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
    # ax.set_xlabel("effective alpha")
    # ax.set_ylabel("total impurity of leaves")
    # ax.set_title("Total Impurity vs effective alpha for training set")
    # plt.savefig(get_base_dir('images')+'/total impurity of leaves.png', dpi = 300)
    # # plt.show()
    # plt.close()

    # clfs = []
    # for ccp_alpha in ccp_alphas:
    #     clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    #     clf.fit(X_train, y_train)
    #     clfs.append(clf)
    # print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
    #     clfs[-1].tree_.node_count, ccp_alphas[-1]))

    # clfs = clfs[:-1]
    # ccp_alphas = ccp_alphas[:-1]

    # node_counts = [clf.tree_.node_count for clf in clfs]
    # depth = [clf.tree_.max_depth for clf in clfs]
    # fig, ax = plt.subplots(2, 1)
    # ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
    # ax[0].set_xlabel("alpha")
    # ax[0].set_ylabel("number of nodes")
    # ax[0].set_title("Number of nodes vs alpha")
    # ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
    # ax[1].set_xlabel("alpha")
    # ax[1].set_ylabel("depth of tree")
    # ax[1].set_title("Depth vs alpha")
    # fig.tight_layout()
    # plt.savefig(get_base_dir('images')+'/Number of nodes vs alpha.png', dpi = 300)
    # # plt.show()
    # plt.close()

    # train_scores = [clf.score(X_train, y_train) for clf in clfs]
    # test_scores = [clf.score(X_test, y_test) for clf in clfs]

    # fig, ax = plt.subplots()
    # ax.set_xlabel("alpha")
    # ax.set_ylabel("accuracy")
    # ax.set_title("Accuracy vs alpha for training and testing sets")
    # ax.plot(ccp_alphas, train_scores, marker='o', label="train",
    #         drawstyle="steps-post")
    # ax.plot(ccp_alphas, test_scores, marker='o', label="test",
    #         drawstyle="steps-post")
    # ax.legend()
    # plt.savefig(get_base_dir('images')+'/Accuracy vs alpha.png', dpi = 300)
    # plt.show()
    # plt.close()
    #  적정 ccp_alpha 값을 찾기 위한 소스 끝

    tree_model.fit(X_train_std, y_train)
    dot_data = export_graphviz(tree_model
        , filled=True
        , rounded=True
        , class_names=['Setosa', 'Versicolor', 'Virginica']
        , feature_names=['petal length', 'petal width']
        , out_file= None
        )
    
    graph = graph_from_dot_data(dot_data)
    graph.write_png(get_base_dir('images')+'/iris_decisionTree_model.png')

    y_pred = tree_model.predict(X_test_std)
    print('잘못 분류된 샘플 개수: %d' % (y_test != y_pred).sum())
    # print("정확도 %.3f" % accuracy_score(y_test, y_pred))
    print("정확도 %.3f" % tree_model.score(X_test_std, y_test))

    target_names = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']
    print("confution matrix \n",multilabel_confusion_matrix(y_test, y_pred,labels=[2, 1, 0]))
    print(classification_report(y_test,y_pred, target_names=target_names))

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined_std = np.hstack((y_train, y_test))

    cmmnUtils = CmnUtils()

    cmmnUtils.plot_decision_regions_with_test_marker(X=X_combined_std, y= y_combined_std, classifier=tree_model, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/iris_decision_regisons_decisionTree_gini.png', dpi = 300)
    # plt.show()
    plt.close()
    
   

    # tree.plot_tree(tree_model)
    # plt.savefig(get_base_dir('images')+'/iris_decisionTree_model.png', dpi = 300)
    # plt.show()
    # plt.close()
