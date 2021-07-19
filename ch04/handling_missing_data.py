import pandas as pd
from io import StringIO
import sys
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,IterativeImputer, KNNImputer
from sklearn.preprocessing import FunctionTransformer
import numpy as np

if __name__ == '__main__':
    csv_data = """A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    10.0,11.0,12.0"""

    if(sys.version_info<(3, 0)):
        csv_data =unicode(csv_data)

    df = pd.read_csv(StringIO(csv_data))

    print("4.1 누락된 데이터 다루기")
    print(df)
    print("4.1.1 테이블 형태 데이터에서 누락된 값 식별")
    print("특성별 결측값의 합계를 출력합니다.")
    print(df.isnull().sum())
    print("# `values` 속성으로 넘파이 배열을 얻을 수 있습니다\n",df.values)

    print("4.1.2 누락된 값이 있는 훈련 샘플이나 특성 제외")
    print("누락된 값이 있는 행을 삭제합니다\n",df.dropna(axis=0))
    print("누락된 값이 있는 열을 삭제합니다\n",df.dropna(axis=1))
    print("모든 열이 NaN인 행을 삭제합니다\n",df.dropna(how='all'))
    print("NaN 아닌 값이 네 개보다 작은 행을 삭제합니다\n",df.dropna(thresh=4))
    print("특정 열에 NaN이 있는 행만 삭제합니다(여기서는 'C'열)\n",df.dropna(subset=['C']))
    
    print("4.1.3 누락된 값 대체")
    print("원래배열\n",df.values)

    # 행을 평균으로 누락된 값 대체하기

    imr = SimpleImputer(missing_values=np.nan, strategy='mean')
    imr = imr.fit(df.values)
    imputed_data = imr.transform(df.values)
    print("행을 평균으로 누락된 값 대체하기\n", imputed_data)

    ftr_imr = FunctionTransformer(lambda X: imr.fit_transform(X.T).T)
    imputed_data = ftr_imr.fit_transform(df.values)
    print("FunctionTransformer\n", imputed_data)

    # SimpleImputer 클래스의 add_indicator 매개변수를 True로 지정하면 indicator_ 속성이 추가되고 
    # transform() 메서드가 누락된 값의 위치를 포함된 배열을 반환

    imr =SimpleImputer(add_indicator=True)
    imputed_data = imr.fit_transform(df.values)
    print("add_indicator\n", imputed_data)
    print(imr.indicator_)
    print(imr.indicator_.features_)
    # `MissingIndicator` 객체의 `fit_transform()` 메서드를 호출하면 `features_` 
    # 속성에 담긴 특성에서 누락된 값의 위치를 나타내는 배열을 반환
    print(imr.indicator_.fit_transform(df.values))
    print(imr.inverse_transform(imputed_data))

    iimr = IterativeImputer()
    print("IterativeImputer\n",iimr.fit_transform(df.values))

    kimr =KNNImputer()
    print("KNNImputer\n",kimr.fit_transform(df.values))

    print("평균으로 결측값 채우기\n", df.fillna(df.mean()))
    print("평균으로 결측값 채우기(행) bfill method='backfill'\n", df.fillna(method='bfill')) # 누락된 값을 다음 행의 값으로 채움
    print("평균으로 결측값 채우기(행) ffill method='pad'\n", df.fillna(method='pad')) # 이전 행의 값을 채움
    print("평균으로 결측값 채우기(열) bfill method='backfill'\n", df.fillna(method='ffill', axis=1)) # 이전 열의 값으로 누락값 채움
    print("평균으로 결측값 채우기(열) ffill method='pad'\n", df.fillna(method='pad', axis=1))