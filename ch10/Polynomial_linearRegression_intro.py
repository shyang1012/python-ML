from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import os

def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path
    if not os.path.exists(result):
        os.makedirs(result)
    return result

if __name__ == '__main__':
    X = np.array([258.0, 270.0, 294.0, 
              320.0, 342.0, 368.0, 
              396.0, 446.0, 480.0, 586.0])\
             [:, np.newaxis]

    y = np.array([236.4, 234.4, 252.8, 
              298.6, 314.2, 342.2, 
              360.8, 368.0, 391.2,
              390.8])
    lr = LinearRegression()
    pr = LinearRegression()

    quadratic = PolynomialFeatures(degree=2)
    X_quad = quadratic.fit_transform(X)

    #선형 특성 학습
    lr.fit(X, y)

    X_fit = np.arange(250, 600, 10)[:, np.newaxis]
    y_lin_fit = lr.predict(X_fit)

    # 이차항 특성 학습
    pr.fit(X_quad, y)
    y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))
    
    # 결과 그래프
    plt.scatter(X, y, label='Training points')
    plt.plot(X_fit, y_lin_fit, label='Linear fit', linestyle='--')
    plt.plot(X_fit, y_quad_fit, label='Quadratic fit')
    plt.xlabel('Explanatory variable')
    plt.ylabel('Predicted or known target values')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/Polynomial_linearRegression_intro.png', dpi = 300)
    # plt.show()
    plt.close()

    y_lin_pred = lr.predict(X)
    y_quad_pred = pr.predict(X_quad)

    print('훈련 MSE 비교 - 선형 모델: %.3f, 다항 모델: %.3f' % (
        mean_squared_error(y, y_lin_pred),
        mean_squared_error(y, y_quad_pred)))
    print('훈련 R^2 비교 - 선형 모델: %.3f, 다항 모델: %.3f' % (
            r2_score(y, y_lin_pred),
            r2_score(y, y_quad_pred)))