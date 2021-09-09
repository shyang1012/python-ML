import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


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
    X = df[['LSTAT']].values
    y = df['MEDV'].values
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)

    forest = RandomForestRegressor(n_estimators=1000, 
                               criterion='mse', 
                               random_state=1, 
                               n_jobs=-1)
    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)

    print('훈련 MSE: %.3f, 테스트 MSE: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    print('훈련 R^2: %.3f, 테스트 R^2: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))
    
    plt.scatter(y_train_pred,  
            y_train_pred - y_train, 
            c='steelblue',
            edgecolor='white',
            marker='o', 
            s=35,
            alpha=0.9,
            label='Training data')
    plt.scatter(y_test_pred,  
                y_test_pred - y_test, 
                c='limegreen',
                edgecolor='white',
                marker='s', 
                s=35,
                alpha=0.9,
                label='Test data')

    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
    plt.xlim([-10, 50])
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/RamdomForestRegression.png', dpi = 300)
    plt.show()
    plt.close()
