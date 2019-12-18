import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import load_diabetes

# X, y = load_diabetes(return_X_y=True)
# print(X[:5])
# print(X.shape, y.shape)

datasets = load_diabetes()
X = datasets.data
y = datasets.target
print('X:', X.shape, 'y:', y.shape)
features = datasets.feature_names
print('feature names:', features)
print('X[0] =', X[0])
# 모든 특성(컬럼)들이 평균=0, 표준 편차=1 로 전처리가 되어 있는 데이터 세트.
print('y[0] =', y[0])

# 선형 회귀(linear regression)
# y = b + a * x
# y = b0 + b1 * x1 + b2 * x2 + ...

# 1개의 figure에 10개의 subplot를 그려서, 변수들과 당뇨병(y)의 대략적이 관계를 파악.
# y ~ age, y ~ sex, y ~ bmi, ...
fig, ax = plt.subplots(3, 4)
for row in range(3):
    for col in range(4):
        axis = ax[row, col]
        idx = 4 * row + col
        if idx > 9:
            break
        x = X[:, idx]
        axis.scatter(x, y)
        axis.set_title(features[idx])
plt.show()

# y = b + a * bmi: y와 bmi 간의 선형 관계식
bmi = X[:, np.newaxis, 2]  # data에서 'bmi' 컬럼만 선택
# scikit-learn의 LinearRegression은
# 2차원 배열 형태의 훈련 데이터 세트만 사용하기 때문에
print('bmi.shape:', bmi.shape)
print('bmi[5] =', bmi[:5])

# bmi를 학습(training)/검증(test) 세트로 분리
bmi_train = bmi[:-40]
bmi_test = bmi[-40:]
y_train = y[:-40]
y_test = y[-40:]

# 선형 회귀 모델(linear regression model) 객체 생성
regr = linear_model.LinearRegression()

# traing set를 학습(fit)
# y = b + a * bmi 선형 관계식에서 y절편 b와 기울기 a를 결정
regr.fit(bmi_train, y_train)
print('coefficients:', regr.coef_)

# 검증 세트로 테스트
y_pred = regr.predict(bmi_test)

plt.scatter(bmi_test, y_test)  # 실제 값
plt.plot(bmi_test, y_pred, 'ro-')  # 예측 값
plt.title('Diabetes vs BMI')
plt.xlabel('bmi')
plt.show()

# y ~ s5 선형 관계식을 찾고, 그래프를 그림.
s5 = X[:, np.newaxis, 8]  # 2차원 배열 형태로 's5' 컬럼을 선택

s5_train = s5[:-40]  # train set
s5_test = s5[-40:]  # test set
y_train = y[:-40]
y_test = y[-40:]

regr = linear_model.LinearRegression()
regr.fit(s5_train, y_train)
print('Coefficient:', regr.coef_)  # 선형 회귀식에서 기울기

y_pred = regr.predict(s5_test)  # 예측값

plt.scatter(s5_test, y_test)  # 실제 값
plt.plot(s5_test, y_pred, 'ro-')  # 예측 값
plt.title('Diabetes vs S5')
plt.xlabel('s5')
plt.show()

array = np.array([[1, 2],
                   [3, 4]])
print(array)  # 2x2 행렬(2차원 배열)
for row in range(2):
    for col in range(2):
        print(array[row, col], end=' ')
print()

array_flatten = array.flatten()
print(array_flatten)
for i in range(4):
    print(array_flatten[i], end=' ')
print()

fig, ax = plt.subplots(3, 4)
# ax: 3x4 형태의 2차원 배열(ndarray)
ax_flat = ax.flatten()
for i in range(len(features)):
    subplot = ax_flat[i]
    subplot.scatter(X[:, i], y)
    subplot.set_title(features[i])
plt.show()


