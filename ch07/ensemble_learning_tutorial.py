from scipy.special import comb
import math
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt
import os

"""
앙상블학습(ensemble learning)
여러 분류기를 하나의 메타 분류기로 연결하여 개별분류기보다 더 좋은 일반화 성능을 달성하는 것.
가장 인기가 있는 앙상블 방법: majerity volting(과반수 투표)
        - 분류기의 과반수가 예측한 클래스 레이블을 선택하는 단순한 방법(50% 이상 투표받은 클래스 레이블 선택)
         --> 다수결투표(pluraity volting)
"""

def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path
    if not os.path.exists(result):
        os.makedirs(result)
    return result

def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.))
    probs = [comb(n_classifier, k) * error**k * (1-error)**(n_classifier - k)
             for k in range(k_start, n_classifier + 1)]
    return sum(probs)



if __name__ == '__main__':
    print(ensemble_error(n_classifier=11, error=0.25))
    print(binom.cdf(5, 11, 0.75))
    
    error_range = np.arange(0.0, 1.01, 0.01)
    ens_errors = [ensemble_error(n_classifier=11, error=error) for error in error_range]
   


    plt.plot(error_range, ens_errors, label='Ensemble error', linewidth=2)
    plt.plot(error_range, error_range, label='Base error',linestyle='--', linewidth=2)
    plt.xlabel("Base error")
    plt.ylabel("Base/Ensemble error")
    plt.legend(loc='upper left')
    plt.grid(alpha=0.5)


    plt.savefig(get_base_dir('images')+'/Ensemble_learning_error_rate.png', dpi = 300)
    # plt.show()
    plt.close()


    #majerity vote 핵심 알고리즘 간단히 구현 시작
    
    # argmax와 bincount 함수를 사용하여 가중치가 적용된 다수결 투표 구현 1
    print("가중치가 적용된 다수결투표 구현결과:"
        ,np.argmax(np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6])))


    # 클래스 확률 기반으로 가중치가 적용된 다수결 투표 구현
    # shape: 3X2, axis= 0 : 수직방향, axis=1 : 수평방향
    ex = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.4, 0.6]
    ])
    p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])
    print("각 클래스 확률:",p)
    print("레이블 예측결과:",np.argmax(p))
    
    #majerity vote 핵심 알고리즘 간단히 구현 끝
