import pandas as pd
import os

def get_base_dir(path):
    result = os.path.join(os.path.dirname(os.path.abspath(__file__)),path)
    if not os.path.exists(result):
        os.makedirs(result)
    return result


if __name__ == '__main__':
    df = pd.read_csv(get_base_dir('data')+'/movie_data.csv', encoding='utf-8')
    print(df.head(3))
    print(df.shape)
    