import numpy as np


class AdalineSGD(object):
    """ADAptive LInear NEuron 분류기
        확률적 경사하강법을 이용한 분류기

        확률적 경사하강법 장점: 배치 경사 하강법에 비해 빠름, 
        하나의 샘플에 대한 그래디언트만 계산, 즉 데이터가 매우 클 때 유용함, 
        배치 경사 하강법에 비해 비용함수가 불규칙할 경우 알고리즘이 
        지역 최솟값(local minimum)을 건너뛸 가능성이 높음

        단점: 배치 경사 하강법에 비해 불안정, 일정하게 비용함수가 감소하는 것이 아니라 
        요동치기 때문에 평균적으로 감소 배치 경사 하강법에 비해 
        전역 최솟값(global minimum)에 다다르지 못함, 
        하이퍼 파라미터 증가(ex stochastic index, learning schedule)

        단점 극복방법: 학습률 조정
        처음에는 학습률을 크게 하여 지역 최솟값에 빠지지 않도록 한 후 점차 작게 줄여서 전역 최솟값에 도달
        => 학습률을 점진적으로 줄이는 학습 스케줄(learning schedule)을 설정


    """
    
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        """
            ADAptive LInear NEuron 분류기 초기화
        Args:
            eta (float, optional): 학습률 (0.0과 1.0 사이). Defaults to 0.01.
            n_iter (int, optional): 훈련 데이터셋 반복 횟수. Defaults to 10.
            shuffle (bool, optional): True로 설정하면 같은 반복이 되지 않도록 에포크마다 훈련 데이터를 섞습니다. Defaults to True.
            random_state ([type], optional): 가중치 무작위 초기화를 위한 난수 생성기 시드. 튜토리얼용 Defaults to None.
        """

        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        

    def fit(self, X, y):
        """훈련 데이터 학습

        Args:
            X ({array-like}, shape = [n_samples, n_features]): n_samples 개의 샘플과 n_features 개의 특성으로 이루어진 훈련 데이터
            y (array-like, shape = [n_samples]): 타깃 벡터

        Returns:
            self: object
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X, y):
        """가중치를 다시 초기화하지 않고 훈련 데이터를 학습합니다
            스트리밍 데이터를 사용하는 혼라인 학습방식으로 모델을 훈련하려면 개개의 
            샘플마다 partial_fit 메소드 호출
        Args:
             X ({array-like}, shape = [n_samples, n_features]): n_samples 개의 샘플과 n_features 개의 특성으로 이루어진 훈련 데이터
             y (array-like, shape = [n_samples]): 타깃 벡터

        Returns:
            self: object
        """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)

        return self
    

    def _shuffle(self, X, y):
        """훈련 데이터를 섞습니다

        Args:
            X ({array-like}, shape = [n_samples, n_features]): n_samples 개의 샘플과 n_features 개의 특성으로 이루어진 훈련 데이터
            y (array-like, shape = [n_samples]): 타깃 벡터

        Returns:
            X[r], y[r]
        """
        r = self.rgen.permutation(len(y))
        return X[r], y[r]


    def _initialize_weights(self, m):
        """랜덤한 작은 수로 가중치를 초기화합니다

        Args:
            m ({array-like}, shape = [n_samples, n_features]): 훈련데이터
        """
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size= 1+m)
        self.w_initialized= True

    def _update_weights(self, xi, target):
        """아달린 학습 규칙을 적용하여 가중치를 업데이트합니다

        Args:
            xi ([type]): [description]
            target ([type]): [description]

        Returns:
            [type]: [description]
        """
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    

    def net_input(self, X):
        """입력 계산

        Args:
            X ([type]): [description]

        Returns:
            [type]: [description]
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """선형 활성화 계산"""
        return X

    def predict(self, X):
        """단위 계단 함수를 사용하여 클래스 레이블을 반환합니다"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)