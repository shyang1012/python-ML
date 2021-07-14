import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np
import os
from scipy.special import expit


def sigmoid(z):
    """시스모이드 함수
    오즈비(odds ratio;교환비;승산비)는 특정 이벤트가 발생할 확률, 두 오즈(odds)의 비율
    어떤 이벤트가 발생할  P/(1-P)로 표현됨
    오즈(Odds) = 성공확률 / 실패확률
            -------Cancer--------
            |    |   O   |   X   |
            |--------------------|
            |  O |   a   |   b   |
    Exposure| -------------------|    
            |  x |   c   |   d   |
            ----------------------
    Odds 계산식
    OR = (a/b)/(c/d) = ad / bc  
    예시: 
      위험인자에 노출된 사람중에서 암환자인 오즈값 = a/b = Odds1
      위험인자에 노출되지 않은 사람 중에서 암환자인 오즈값 c/d = Odds2
      OR = Odds1 / Odds2  = (a/b) / (c/d)
    오즈비는 샘플링에서 생길 수 있는 Bias를 최소화하여, 통계적 의미를 강화
    
    시그모이드는 오즈비에 자연 log를 취한 값이다.

    Args:
        z : 가중치와 입력(training samples's feature)의 선형조합으로 이루어진 
        최종입력

    Returns:
        로짓(logit) 계산 결과.
    """
    return 1.0 / (1.0 + np.exp(-z))


def get_base_dir(path):
    result = os.path.dirname(os.path.abspath(__file__))+'/'+path;
    if not os.path.exists(result):
        os.makedirs(result)
    return result

def cost_1(z):
    return - np.log(sigmoid(z))

def cost_0(z):
    return - np.log(1 - sigmoid(z))


if __name__ == '__main__': 
    # font_path = "C:/Windows/Fonts/NGULIM.TTF"
    # font = font_manager.FontProperties(fname=font_path).get_name()
    # print(font)
    rc('font', family="New Gulim")

    z = np.arange(-7, 7, 0.1)
    # phi_z = sigmoid(z)
    phi_z = expit(z)
    plt.plot(z, phi_z)
    plt.axvline(0.0, color='k')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('$\phi (z)$')
    # y 축의 눈금과 격자선
    plt.yticks([0.0, 0.5, 1.0])
    ax = plt.gca()
    ax.yaxis.grid(True)

    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/sigmoid.png', dpi = 300)
    plt.close()

    z = np.arange(-10, 10, 0.1)
    # phi_z = sigmoid(z)
    phi_z = expit(z)
    c1 = [cost_1(x) for x in z]
    plt.plot(phi_z, c1, label='J(w) y=1 일때')
    c0 = [cost_0(x) for x in z]
    plt.plot(phi_z, c0, linestyle='--',label='J(w) y=0 일때')
    plt.ylim(0.0, 5.1)
    plt.xlim([0, 1])
    plt.xlabel('$\phi (z)$')
    plt.ylabel('J(w)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(get_base_dir('images')+'/sigmoid_logit_cost.png', dpi = 300)
    plt.show()
