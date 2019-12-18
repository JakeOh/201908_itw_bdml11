"""
y = b + a1 * x1 : 단순 선형 회귀
y = b + a1 * x1 + a2 * x2 + ...: 다중 선형 회귀
"""
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(1216)
X1 = np.random.rand(100, 1)
print('X1 =', X1[:5])

X2 = np.random.rand(100, 1)
print('X2 =', X2[:5])

y = 3 + 4 * X1 + 5 * X2 + np.random.randn(100, 1)  # target
print('y =', y[:5])

X = np.c_[X1, X2]  # data
print('X =', X[:5])

lin_reg = LinearRegression()  # LR 객체 생성
lin_reg.fit(X, y)  # model fitting, 학습
print('절편(intercept) =', lin_reg.intercept_)
print('계수(coefficients) =', lin_reg.coef_)








