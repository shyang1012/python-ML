import os
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap
import numpy as np

from LinearRegressionGD import LinearRegressionGD
from sklearn.preprocessing import StandardScaler


def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path
    if not os.path.exists(result):
        os.makedirs(result)
    return result


def get_data():
    df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-3rd-edition/'
                 'master/ch10/housing.data.txt',
                 header=None,
                 sep='\s+')

    df.columns = ['CRIM' # 도시의 인당 범죄율
                , 'ZN' # 2만5000평방 피트가 넘는 주택 비율
                , 'INDUS' # 도시에서 소매 업종이 아닌 지역 비율
                , 'CHAS' # 찰스 강 인접 여부 (강주변 = 1, 그외 =0)
                , 'NOX' # 일산화질소 농도 (10ppm당)
                , 'RM' # 주택 평균 방 갯수
                , 'AGE' # 1940년 이전에 지어진 자가 주택 비율
                , 'DIS' # 다섯 개의 보스턴 고용 센터까지 가중치가 적용된 거리
                , 'RAD' # 방사형으로 뻗은 고속도로까지 접근성 지수
                , 'TAX' # 10만 달러당 재산세율
                , 'PTRATIO' # 도시의 학생-교사 비율
                , 'B' # 1000(Bk -0.063)^2. 여기서 Bk는 도시의 아프리카계 미국인 비율
                , 'LSTAT' # 저소득 계층의 비율
                , 'MEDV' # 자가 주택의 중간 가격(1000달러 단위), 여기서는 타겟값으로 사용
            ]
    print(df.head())
    return df

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return 


if __name__ == '__main__':
    df = get_data()
    cols =['LSTAT','INDUS','NOX','RM','MEDV']

    #주택 데이터셋의 산점도 행렬
    scatterplotmatrix(df[cols].values, figsize=(10, 8), 
                  names=cols, alpha=0.5)
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/scatterplotmatrix_house.png', dpi = 300)
    # plt.show()
    plt.close()

    # 상관관계 행렬의 히트맵 그리기
    cm = np.corrcoef(df[cols].values.T)
    hm = heatmap(cm, row_names=cols, column_names=cols)
    plt.savefig(get_base_dir('images')+'/heatmap_house.png', dpi = 300)
    # plt.show()
    plt.close()

    X = df[['RM']].values #방 갯수
    y = df['MEDV'].values # 주택가격
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_std = sc_x.fit_transform(X)
    y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
    lr = LinearRegressionGD()
    lr.fit(X_std, y_std)

    plt.plot(range(1, lr.n_iter+1), lr.cost_)
    plt.ylabel('SSE')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/LinearRegressionGD_house.png', dpi = 300)
    # plt.show()
    plt.close()

    lin_regplot(X_std, y_std, lr)
    plt.xlabel('Average number of rooms [RM] (standardized)')
    plt.ylabel('Price in $1000s [MEDV] (standardized)')

    plt.savefig(get_base_dir('images')+'/lin_regplot_house.png', dpi = 300)
    # plt.show()
    plt.close()

    print('기울기: %.3f' % lr.w_[1])
    print('절편: %.3f' % lr.w_[0])

    num_rooms_std = sc_x.transform(np.array([[5.0]]))
    price_std = lr.predict(num_rooms_std)
    print("$1,000 단위 가격: %.3f" % sc_y.inverse_transform(price_std))

