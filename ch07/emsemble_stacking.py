from sklearn import datasets
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.tree import *
from sklearn.linear_model import *
from sklearn.neighbors import *
import numpy as np
from sklearn.pipeline import *
from sklearn.ensemble import *
from sklearn.metrics import *
import matplotlib.pyplot as plt
from itertools import product
import os

def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path
    if not os.path.exists(result):
        os.makedirs(result)
    return result


def main():
    iris = datasets.load_iris()
    # 꽃받침 너비와 꽃잎길이 두개의 feature만 사용
    # multi-label을 활용하도 되지만, ROC AUC 계산을 위해 Iris-versicolor와 Iris-virginica 클래스에 해당하는
    # 샘플만 사용
    iris = datasets.load_iris()
    X, y = iris.data[50:, [1, 2]], iris.target[50:]
    le = LabelEncoder()
    y = le.fit_transform(y)

    # training dataset : test dataset = 50%: 50%
    X_train, X_test, y_train, y_test =\
       train_test_split(X, y, 
                        test_size=0.5, 
                        random_state=1,
                        stratify=y)

    clf1 = LogisticRegression(penalty='l2', 
                            C=0.001,
                            random_state=1)

    # 책의 예제는 max_length 설정을 통한 pre-pruning 방식을 소개하지만
    # under fit 될 우려가 있기 때문에 full tree 완성 후 over-fit을 방지하기 위한 
    # post puruning 실행. ID3, C4.5, C5.0 ,CART 등 대부분의 의사결정트리 알고리즘은
    # post puruning 채택
    # sklearn에서는 ccp_alpha 값 설정을 통해서 사후 가지치기 실행 가능.
    # 적정한 값 설정은 https://scikit-learn.org/stable/modules/tree.html을 참고하여
    # 찾은 후 셋팅.
    clf2 = DecisionTreeClassifier(ccp_alpha=0.05,
                                criterion='entropy',
                                random_state=0) #ccp_alpha값을 설정하여 post puruning 실행

    clf3 = KNeighborsClassifier(n_neighbors=2,
                                p=2,
                                metric='minkowski')

    pipe1 = Pipeline([['sc', StandardScaler()],
                    ['clf', clf1]])
    pipe3 = Pipeline([['sc', StandardScaler()],
                    ['clf', clf3]])

    stack= StackingClassifier(estimators=[('lr',pipe1)
                                        ,('dt',clf2)
                                        ,('knn',pipe3)]
                                        ,final_estimator=LogisticRegression())
  


    clf_labels = ['Logistic regression', 'Decision tree', 'KNN','Stacking']
    all_clf= [pipe1, clf2, pipe3,stack]

    print('10-겹 교차 검증:\n')
    for clf, label in zip(all_clf, clf_labels):
        scores = cross_val_score(estimator=clf,
                                X=X_train,
                                y=y_train,
                                cv=10,
                                scoring='roc_auc')
        print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
            % (scores.mean(), scores.std(), label))

    
    params = {'dt__max_depth': [1, 2],
          'lr__clf__C': [0.001, 0.1, 100.0]}

    grid = GridSearchCV(estimator=stack,
                        param_grid=params,
                        cv=10,
                        scoring='roc_auc')
    grid.fit(X_train, y_train)

    for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        print("%0.3f +/- %0.2f %r"
            % (grid.cv_results_['mean_test_score'][r], 
                grid.cv_results_['std_test_score'][r] / 2.0, 
                grid.cv_results_['params'][r]))
    print('최적의 매개변수: %s' % grid.best_params_)
    print('정확도: %.2f' % grid.best_score_)

    # 앙상블 분류기의 평가와 튜닝
    colors = ['black', 'orange', 'blue', 'green']
    linestyles = [':','--','-.','-']

    for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
        # 양성 클래스의 레이블이 1이라고 가정합니다
        y_pred = clf.fit(X_train,
                        y_train).predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_test,
                                        y_score=y_pred)
        roc_auc = auc(x=fpr, y=tpr)
        plt.plot(fpr, tpr,
                color=clr,
                linestyle=ls,
                label='%s (auc = %0.2f)' % (label, roc_auc))
    
    plt.legend(loc='lower right')

    plt.plot([0, 1], [0, 1],
         linestyle='--',
         color='gray',
         linewidth=2)

    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid(alpha=0.5)
    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True positive rate (TPR)')
    plt.savefig(get_base_dir('images')+'/evaluate_classifier_ROC_curve_Stacking.png', dpi = 300)
    # plt.show()
    plt.close()

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)

    x_min = X_train_std[:, 0].min() - 1
    x_max = X_train_std[:, 0].max() + 1
    y_min = X_train_std[:, 1].min() - 1
    y_max = X_train_std[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))

    f, axarr = plt.subplots(nrows=2, ncols=2, 
                            sharex='col', 
                            sharey='row', 
                            figsize=(7, 5))

    for idx, clf, tt in zip(product([0, 1], [0, 1]),
                            all_clf, clf_labels):
        clf.fit(X_train_std, y_train)
        
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
        
        axarr[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0], 
                                    X_train_std[y_train==0, 1], 
                                    c='blue', 
                                    marker='^',
                                    s=50)
        
        axarr[idx[0], idx[1]].scatter(X_train_std[y_train==1, 0], 
                                    X_train_std[y_train==1, 1], 
                                    c='green', 
                                    marker='o',
                                    s=50)
        
        axarr[idx[0], idx[1]].set_title(tt)

    plt.text(-3.5, -5., 
            s='Sepal width [standardized]', 
            ha='center', va='center', fontsize=12)
    plt.text(-12.5, 4.5, 
            s='Petal length [standardized]', 
            ha='center', va='center', 
            fontsize=12, rotation=90)

    plt.savefig(get_base_dir('images')+'/decision_regisons_Stacking.png', dpi = 300)
    plt.show()
    plt.close()


    




if __name__ == '__main__':
    main()
