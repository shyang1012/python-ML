import pandas as pd
from sklearn.feature_extraction.text import *
from sklearn.decomposition import LatentDirichletAllocation
import os
import re
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.pipeline import *
from sklearn.linear_model import *
from sklearn.model_selection import *

"""
토픽 모델링(topic modeling)
 - 레이블이 없는 텍스트 문서에 토픽을 할당하는 광범위한 분야
 - 예시: 대량의 뉴스 기사 데이터셋을 분류(스포츠, 금융, 세계뉴스, 정치, 지역뉴스 등)

잠재 디리클래 할당(Latent Dirichlet Allocation, LDA)
- 여러 문서에 걸쳐 자주 등장하는 단어의 그룹을 찾는 확률적 생성 모델
- LDA의 입력은 BoW 모델, LDA는 BoW 행렬을 두개의 행렬로 분해
    * 문서-토픽 행렬
    * 단어-토픽 행렬
- 이 두행렬을 곱해서 가능한 작은 오차로 BoW 입력 행렬을 재구성할 수 있도록 LDA가 BoW 행렬 분해
- 단점: 미리 토픽의 갯수를 수동으로 지정해야 함
"""

def get_base_dir(path):
    result = os.path.join(os.path.dirname(os.path.abspath(__file__)),path)
    if not os.path.exists(result):
        os.makedirs(result)
    return result

def preprocesser(text):
    text = re.sub('<[^>]*>','', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) +' '.join(emoticons).replace('-', ''))
    return text

def load_movie_review():
    df = pd.read_csv(get_base_dir('data')+'/movie_data.csv', encoding='utf-8')
    df['review'] = df['review'].apply(preprocesser)
    print(df.head(3))
    return df

if __name__ == '__main__':
    df = load_movie_review()

    count = CountVectorizer(stop_words='english',
                        max_df=.1,
                        max_features=5000)
    X = count.fit_transform(df['review'].values)

    lda = LatentDirichletAllocation(n_components=10, random_state=123, learning_method='batch')

    X_topics = lda.fit_transform(X)

    print(lda.components_.shape)

    n_top_words = 5
    feature_names = count.get_feature_names()

    for topic_idx, topic in enumerate(lda.components_):
        print("Topic %d:" % (topic_idx + 1))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()\
                            [:-n_top_words - 1:-1]]))
    
    horror = X_topics[:, 5].argsort()[::-1]
    for iter_idx, movie_idx in enumerate(horror[:3]):
        print('\n공포 영화 #%d:' % (iter_idx + 1))
        print(df['review'][movie_idx][:300], '...')