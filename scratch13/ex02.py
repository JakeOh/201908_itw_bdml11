import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(1216)
X = 2 * np.random.rand(100, 1)
print('X shape:', X.shape)  # 0.0 ~ 2.0 숫자들로 이루어진 100x1 행렬(2차원 ndarray)

y = 4 + 3 * X + np.random.randn(100, 1)
print('y shape', y.shape)

plt.scatter(X, y)
plt.show()

X_b = np.c_[np.ones((100, 1)), X]
print('X_b shape:', X_b.shape)
print(X_b[:5])

# linalg 모듈: Linear Algebra(선형 대수)
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print('theta =', theta_best)

# 행렬식을 이용해서 찾은 theta 값과 LinearRegression 클래스에서 계산되 theta 비교
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(f'y절편: {lin_reg.intercept_}, 기울기: {lin_reg.coef_}')

X_test = [[0],
          [1],
          [2]]
#  행렬식: y = X_b @ theta
X_test_b = np.c_[np.ones((3, 1)), X_test]
print(X_test_b)
y_pred = X_test_b.dot(theta_best)
print(y_pred)

# scikit-learn 패키지를 사용한 예측
predictions = lin_reg.predict(X_test)
print(predictions)

plt.scatter(X, y)
plt.plot(X_test, y_pred, 'ro-')
plt.show()

