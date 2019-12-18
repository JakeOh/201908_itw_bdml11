import numpy as np
import pandas as pd


def fill_group_mean(df):
    group_mean = df['data'].mean()
    print(group_mean)
    return df.fillna(group_mean)


if __name__ == '__main__':
    # Series 객체 생성
    s = pd.Series(np.random.randint(1, 10, 5))
    print(s)
    s[3] = np.nan  # 원소 한개를 NA로 변경
    print(s)
    # NA를 평균값으로 대체하기 위해서, 평균을 먼저 계산.
    m = s.mean()  # numpy, pandas의 집계 함수들은 NA를 제거하고 계산함.
    print('평균 =', m)
    s = s.fillna(m)  # Series s의 모든 NA들을 파라미터 m으로 채워줌.
    print(s)

    df = pd.DataFrame({
        'province': ['서울', '경기', '충청', '전라', '강원', '경상', '부산'],
        'division': ['west'] * 4 + ['east'] * 3,
        'data': np.random.randint(1, 10, 7)
    })
    print(df)

    # 데이터 2개를 NA로 대체
    df.loc[[0, 6], 'data'] = np.nan
    print(df)

    # 데이터 프레임의 NA를 각 그룹별 평균으로 대체
    grouped = df.groupby('division')  # DataFrameGroupBy 객체
    # GroupBy.apply(fn)는 함수 fn의 첫번째 파라미터에 DataFrameGroupBy 객체를
    # 전달함!
    cleaned = grouped.apply(fill_group_mean)
    print(cleaned)
