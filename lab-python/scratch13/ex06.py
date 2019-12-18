import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston

boston = load_boston()
X = boston['data']  # boston.data
y = boston['target']  # boston.target
features = boston['feature_names']  # boston.feature_names

# numpy.ndarray 타입을 pandas.DataFrame 타입으로 변환
boston_df = pd.DataFrame(X, columns=features)
# 데이터 프레임에 컬럼 추가
boston_df['Price'] = y

print(boston_df.head())
print(boston_df.shape)
print(boston_df.describe())

# 전체 데이터 프레임에서 관심 컬럼(변수, 특성)들만 선택
columns = ['LSTAT', 'INDUS', 'NOX', 'RM', 'Price']
subset_df = boston_df[columns]
print(subset_df.head())

sns.pairplot(subset_df)
plt.show()

# sns.pairplot(boston_df)
# plt.show()

# 상관 행렬(correlation matrix): 상관 계수들로 만든 행렬
# DataFrame.corr(): 상관 계수들을 계산
corr_matrix = subset_df.corr().round(2)

# heatmap: 상관 계수(correlation coefficient)들을 색상으로 표시한 그래프
sns.heatmap(corr_matrix, annot=True)
plt.show()


