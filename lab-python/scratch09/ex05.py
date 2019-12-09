"""
gapminder.tsv 파일을 pandas 패키지의 read_csv() 함수를 사용해서
DataFrame으로 변환.
DataFrame의 행과 열의 개수 확인
DataFrame의 앞쪽 데이터 5개를 출력
DataFrame의 뒷쪽 데이터 5개를 출력
DtatFrame의 컬럼 이름들을 출력
DataFrame의 각 컬럼의 데이터 타입들을 출력
DataFrame에서 'country', 'lifeExp', 'gdpPercap' 컬럼들만 출력
DataFrame에서  행 인덱스가 0, 99, 999인 행들을 출력
DataFrame에서 행 레이블이 840~851인 행들의
나라이름, 기대 수명, 1인당 GDP를 출력
DataFrame에서 연도(year)별 기대 수명의 평균을 출력
DataFrame에서 연도(year)별, 대륙(continent)별 기대 수명의 평균
"""
import pandas as pd
import matplotlib.pyplot as plt

# sep 파라미터: 데이터 구분자(separator)
gapminder = pd.read_csv('gapminder.tsv', sep='\t', encoding='UTF-8')

# DataFrame.shape: (row 개수, column 개수)
print(gapminder.shape)
nrows, ncols = gapminder.shape
print(f'nrows={nrows}, ncols={ncols}')

# DataFrame.head(n): 첫 n개의 row를 출력. n의 기본값은 5.
print(gapminder.head())

# DataFrame.tail(n): 마지막 n개의 row를 출력. n의 기본값은 5.
print(gapminder.tail())

# 행 인덱스를 이용한 출력
# DataFrame.iloc[row index, column index]
# 만약 column index를 생략하면, 선택한 row index의 모든 컬럼이 선택됨.
print(gapminder.iloc[0:5])
print(gapminder.iloc[nrows - 5:nrows])  # 인덱스 a:b는 a <= x < b를 의미함.

# DataFrame.columns: pandas.Index 클래스 객체. 컬럼 이름들의 리스트를 가지고 있음.
print(gapminder.columns)

# DataFrame.dtypes: 각 컬럼의 이름과 데이터 타입을 저장하고 있는 프로퍼티
# pandas.read_csv() 함수는 파일의 문자열들을 타입에 맞게끔 변환하는 기능을 가지고 있음.
# pandas 데이터 타입: object(문자열), int64(64비트 정수), float64(64비트 실수)
print(gapminder.dtypes)

# DataFrame[column names]: 데이터 프레임에서 컬럼을 선택.
col_names = ['country', 'lifeExp', 'gdpPercap']
print(gapminder[col_names])

# 데이터 프레임에서 특정 행(row)들을 인덱스로 선택
# row_indices = [0, 9, 999]
# print(gapminder.iloc[row_indices])
print(gapminder.iloc[[0, 9, 999]])

# DataFrame.iloc[row index, column index]
# DataFrame.loc[row label, column label]
# loc에서 범위 연산자(:)가 사용되면, 이름(label)로 취급하기 때문에 양쪽 숫자 모두 포함
# iloc에서  범위 연산자(:)가 사용되면, 인덱스로 취급하기 때문에 뒤쪽 숫자는 미포함
print(gapminder.loc[840:851, ['country', 'lifeExp', 'gdpPercap']])
print(gapminder.iloc[840:852, [0, 3, 5]])

# 연도별 기대수명(lifeExp)의 평균
gapminder_by_year = gapminder.groupby('year')
print(gapminder_by_year)  # DataFrameGroupBy 객체
print(gapminder_by_year['lifeExp'])  # SeriesGroupBy 객체
print(gapminder_by_year['lifeExp'].mean())

gapminder_by_year_continent = gapminder.groupby(['year', 'continent'])
print(gapminder_by_year_continent['lifeExp'].mean())

# 연도별 기대수명 그래프
year_lifeExp = gapminder.groupby('year')['lifeExp'].mean()
print(year_lifeExp)
plt.plot(year_lifeExp)
plt.titile('lifeExp by year')
plt.show()

# 연도별 전세계 인구수(pop)를 그래프
year_pop = gapminder.groupby('year')['pop'].sum()
print(year_pop)
plt.plot(year_pop)
plt.title('pop by year')
plt.show()
