from sklearn.model_selection import * 
from sklearn.preprocessing import *
from sklearn.linear_model import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

def load_wine_data():
    df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']
    print('Class labels', np.unique(df_wine['Class label']))
    print(df_wine.head())
    return df_wine


if __name__ == '__main__':
    df_wine = load_wine_data()
    
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    # print("X(data)",X)
    # print("Y(target)",y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y
                                                        , test_size= 0.3 #와인샘플의 30%가 테스트에 할당
                                                        , random_state= 0
                                                        , stratify=y)
    
    lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=1)
    # C = 1.0이 기본
    # 규제 효과를 높이거나 낮추려면 C 값을 증가시기커나 감소
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    lr.fit(X_train_std, y_train)

    print('훈련 정확도:', lr.score(X_train_std, y_train))
    print('테스트 정확도:', lr.score(X_test_std, y_test))


    # 규제에 대한 강도를 달리하여 특성의 가중치 변화를 그래프로 표현
    fig = plt.figure()
    ax= plt.subplot(111)

    colors = ['blue', 'green', 'red', 'cyan', 
          'magenta', 'yellow', 'black', 
          'pink', 'lightgreen', 'lightblue', 
          'gray', 'indigo', 'orange']
    
    weights, params = [], []
    
    for c in np.arange(-4., 6.):
        lr = LogisticRegression(penalty='l1', C=10.**c, solver='liblinear', 
                            multi_class='ovr', random_state=0)
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        params.append(10**c)
    
    weights = np.array(weights)

    for column, color in zip(range(weights.shape[1]), colors):
        plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color)
    plt.rcParams["figure.figsize"] = (14,4)
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.color'] = 'r'
    plt.rcParams['axes.grid'] = True 

    plt.axhline(0, color='black', linestyle='--', linewidth=3)
    plt.xlim([10**(-5), 10**5])
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.xscale('log')
    plt.legend(loc='upper center')
    ax.legend(loc='upper center', 
            bbox_to_anchor=(1.38, 1.03),
            ncol=1, fancybox=True)
    # plt.savefig('images/04_07.png', dpi=300, 
    #             bbox_inches='tight', pad_inches=0.2)
    plt.show()