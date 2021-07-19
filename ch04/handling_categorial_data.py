import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder


if __name__ == '__main__':
    print("4.2 범주형 데이터 다루기")
    df = pd.DataFrame([
       ['green', 'M', 10.1, 'class2']
        ,['red', 'L', 13.5, 'class1']
        ,['blue', 'XL', 15.3, 'class2']
    ])
    df.columns = ['color','size','price','classlabel']
    print('예제데이터')
    print(df)

    # XL = L+1 = M+2
    size_mapping = {
        "XL": 3,
        "L": 2,
        "M": 1
    }

    df['size'] = df['size'].map(size_mapping)
    print("순서가 있는 특성 맵핑(XL = L+1 = M+2)")
    print(df)

    inv_size_mapping = {v: k for k, v in size_mapping.items()}
    print("원본데이터 확인")
    print(df['size'].map(inv_size_mapping))

    
    print('class label 맵핑')
    # 클래스 레이블을 문자열에서 정수로 바꾸기 위해
    # 매핑 딕셔너리를 만듭니다
    class_mapping = {label:idx for idx, label in enumerate(np.unique(df['classlabel']))}
    print(class_mapping)

    df['classlabel'] = df['classlabel'].map(class_mapping)
    print(df)

    inv_class_mapping = {v: k for k, v in class_mapping.items()}
    print(inv_class_mapping)
    print("원본데이터 확인")
    df['classlabel'] = df['classlabel'].map(inv_class_mapping)
    print(df)

    print("LabelEncoder를 활용한 클래스 변환")
    class_le = LabelEncoder()
    y = class_le.fit_transform(df['classlabel'])
    print(y)

    print("LabelEncoder를 활용한 클래스 역변환")
    print(class_le.inverse_transform(y))

    # 4.2.4 순서가 없는 특성에 원-핫 인코딩 적용
   

    print("4.2.4 순서가 없는 특성에 원-핫 인코딩 적용")
    X = df[['color','size', 'price']].values
    color_le = LabelEncoder()
    X[:,0] = class_le.fit_transform(X[:,0])
    print(X)
    print("OrdinalEncoder와 ColumnTransformer 함께 사용하기")
    ord_enc = OrdinalEncoder(dtype=np.int)
    col_trans = ColumnTransformer([('ord_enc', ord_enc, ['color'])])
    X_trans = col_trans.fit_transform(df)
    print(X_trans)
    print("역변환")
    print(col_trans.named_transformers_['ord_enc'].inverse_transform(X_trans))

    X = df[['color', 'size', 'price']].values
    color_ohe = OneHotEncoder()
    one_hot_array= color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()
    print(one_hot_array)

    print("OneHotEncoder의 dtype 매개변수를 np.int로 지정하여 정수로 원-핫 인코딩합니다.")
    """원-핫 인코딩은 단어 집합의 크기를 벡터의 차원으로 하고, 
        표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 다른 인덱스에는 0을 
        부여하는 단어의 벡터 표현 방식

        원-핫 인코딩을 두 가지 과정으로 정리해보겠습니다.
        (1) 각 단어에 고유한 인덱스를 부여합니다. (정수 인코딩)
        (2) 표현하고 싶은 단어의 인덱스의 위치에 1을 부여하고, 다른 단어의 인덱스의 위치에는 0을 부여합니다.
        """  

    X = df[['color', 'size', 'price']].values
    c_transf = ColumnTransformer([ ('onehot', OneHotEncoder(), [0]), # color에는 OneHotEncoder 적용
                                ('nothing', 'passthrough', [1, 2])]) # 나머지엔 아무것도 하지마라.
    print(c_transf.fit_transform(X))

    # 원-핫 인코딩 via 판다스
    print("원-핫 인코딩 via 판다스")
    dummy =  pd.get_dummies(df[['price','color','size']])
    print (dummy)
    print("원-핫 인코딩 via 판다스에서 다중 공선성 문제 처리")
    dummy =  pd.get_dummies(df[['price', 'color', 'size']], drop_first=True)
    print (dummy)

   # OneHotEncoder에서 다중 공선성 문제 처리
    print("OneHotEncoder에서 다중 공선성 문제 처리")
    color_ohe = OneHotEncoder(categories='auto', drop='first')
    c_transf = ColumnTransformer([ ('onehot', color_ohe, [0]),
                                ('nothing', 'passthrough', [1, 2])])
    print(c_transf.fit_transform(X))

    print("추가내용: 순서가 있는 특성 인코딩")
    df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2']])

    df.columns = ['color', 'size', 'price', 'classlabel']
    print(df)

    df['x > M'] = df['size'].apply(lambda x: 1 if x in {'L', 'XL'} else 0)
    df['x > L'] = df['size'].apply(lambda x: 1 if x == 'XL' else 0)

    del df['size']
    print(df)