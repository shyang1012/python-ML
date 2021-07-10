import numpy as np

class Perceptron(object):
    """퍼셉트론 분류기

    Args:
        eta (float, optional): 학습률(0.0과 1.0 사이). Defaults to 0.01.
        n_iter (int, optional): 훈련 데이터셋 반복 횟수. Defaults to 50.
        random_state (int, optional): 가중치 무작위 초기화를 위한 난수 생성기 시드. Defaults to 1.
    
    Attribute:
        w_(1d-array)   :  학습된 가중치
        errors_ (list) : 에포크마다 누적된 분류 오류 
    """
    
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
       self.eta= eta
       self.n_iter=n_iter
       self.random_state = random_state



    def fit(self,x, y):
        """훈련데이터 학습
        Args:
            x: array-like, shape = [n_samples, n_features]
                n_samples개의 샘플과 n_features개의 특성으로 이루어진 훈련 데이터
            y: array-like, shape = [n_samples]
                타깃값(class label)
        return: self
        """
        rgen = np.random.RandomState(self.random_state)
        
        # 가중치를 백터 R^m+1로 초기화 여기서 m은 데이터 셋에 있는 차원(특성) 개수
        # 백터의 첫 번째 원소인 절편을 위해 1을 더함. 즉 절편 == self.w_[0]
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size= 1 + x.shape[1]) # 표준편차가 0.01인 정규분포(gaussian distribution)에서 뽑은 랜덤한 작은 수
        # 


        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        print(self.errors_)
        return self
    
    def net_input(self, x):
        """입력 계산
            단위 계단 함수(unit step function) 또는 헤비사이드 계단 함수(Heaviside step function)은 
            0보다 작은 실수에 대해서 0, 0보다 큰 실수에 대해서 1, 0에 대해서 1/2의 값을 갖는 함수
        Args:
            x: n_samples개의 샘플과 n_features개의 특성으로 이루어진 훈련 데이터

        Returns:
            [type]: [description]
        """
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        """단위 계단 함수를 사용하여 클래스 레이블 반환
        Args:
            x: n_samples개의 샘플과 n_features개의 특성으로 이루어진 훈련 데이터

        Returns:
            [int]: 클래스 레이블
        """
       
        result = np.where(self.net_input(x) >= 0.0, 1, -1)
        # print("예측 데이터",x, "예측 결과:", result)
        return result
    
            