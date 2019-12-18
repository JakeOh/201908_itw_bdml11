"""
y = b + a * x: linear regression
y = b + a1 * x + a2 * x^2 -> 선형 회귀로 b, a1, a2를 결정할 수 있다.

y = b + a1 * x1 + a2 * x2: 선형 회귀
y = b + a1 * x1 + a2 * x2 + a3 * x1^2 + a4 * x1 * x2 + a5 * x2^2
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(1216)
# Training Set - data
X = 6 * np.random.rand(100, 1) - 3  # -3 <= x < 3
print('X =', X[:5])

# target
y = 0.5 + 2 * X + X**2 + np.random.randn(100, 1)

A = np.array([[1], [2], [3]])  # 3x1 행렬(2차원 리스트)
print('A =', A)
poly_feature = PolynomialFeatures(degree=2, include_bias=False)
A_poly = poly_feature.fit_transform(A)  # x^2 컬럼이 추가됨
print('A_poly =', A_poly)

B = np.array([[1, 2], [3, 4]])  # 2x2 행렬(2차원 리스트
print('B =', B)
B_poly = poly_feature.fit_transform(B)  # x1^2, x1*x2, x2^2 컬럼이 추가됨
print('B_poly =', B_poly)

X_poly = poly_feature.fit_transform(X)
print('X_poly =', X_poly[:5])

lin_reg = LinearRegression()  # LR 객체 생성
lin_reg.fit(X_poly, y)  # Train data fitting, 학습
print('intercept:', lin_reg.intercept_)
print('coefficients:', lin_reg.coef_)
# y = b + a1 * x + a2 * x^2

X_test = np.linspace(-3, 3, 100).reshape(100, 1)
print(X_test[:5])
print(X_test[-5:])
X_test_poly = poly_feature.fit_transform(X_test)
y_pred = lin_reg.predict(X_test_poly)

plt.scatter(X, y)
plt.plot(X_test, y_pred, 'r')
plt.show()




