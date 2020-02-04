# -*- coding: utf-8 -*-
"""house_price.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Vp4NHaz49ks6wH60_SbSEdhRHJpDHOSA
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

# 업로드한 housing.csv 파일을 읽어서 DataFrame 생성
import pandas as pd

df = pd.read_csv('housing.csv', header=None, delim_whitespace=True)
# header=None: csv 파일에 컬럼 이름들이 포함되어 있지 않기 때문에.
# delim_whitespace=True: csv 파일의 데이터들이 comma가 아니라 공백으로 구분되고 있기 때문에.

df.head()

df.describe()

print(df.shape)

# DataFrame을 데이터(집 값에 영향을 미치는 변수들)과 집 값을 분리
dataset = df.to_numpy() 
X = dataset[:, :-1]
Y = dataset[:, -1]
print(f'X: {X.shape}, Y: {Y.shape}')

# X, Y를 학습 데이터, 테스트 데이터 세트로 분리
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
print(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}')
print(f'X_test: {X_test.shape}, Y_test: {Y_test.shape}')

# 신경망 생성
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

# fully-connected layer를 추가 - 은닉층 2개, 출력층
model.add(Dense(30, activation='relu', input_dim=X_train.shape[1]))  # 은닉층
model.add(Dense(6, activation='relu'))  # 은닉층
model.add(Dense(1))  # 출력층

# 모델 컴파일
model.compile(loss='mean_squared_error',  # 회귀(regression) - 수치 예측
              optimizer='adam')

# 모델 학습
history = model.fit(X_train, Y_train, batch_size=10, epochs=200,
          validation_split=0.2)

# 모델 평가 - 학습시킨 모델에 테스트 데이터로 테스트
eval = model.evaluate(X_test, Y_test)
print(eval)

# 주택 가격의 예측값
Y_pred = model.predict(X_test)  # 2차원 배열
Y_pred = Y_pred.flatten()  # 1차원 배열로 변환
# print(Y_pred)

# 실제값과 예측값 10개 비교
for i in range(10):
    true_val = Y_test[i]
    pred_val = Y_pred[i]
    squared_error = (true_val - pred_val) ** 2
    print(f't={true_val}, p={pred_val}, se={squared_error}')

# MSE - epoch plot
losses = history.history['loss']
val_losses = history.history['val_loss']

import matplotlib.pyplot as plt

plt.plot(losses, label='Train MSE')
plt.plot(val_losses, label='Test MSE')
plt.legend()
plt.show()

# 모델 개선
# 1) X_train을 z-score로 변환(z = (x - mean)/std) -> 변환된 데이터로 학습
# X_test 데이터는 학습 데이터의 평균과 표준편차를 사용해서 변환하고, 평가/예측에 사용
import numpy as np

mean = X_train.mean(axis=0)  # np.mean(X_train, axis=0)
std = X_train.std(axis=0)

train_data = (X_train - mean) / std
test_data = (X_test - mean) / std

model = Sequential()
# fully-connected layer를 추가 - 은닉층 2개, 출력층
model.add(Dense(30, activation='relu', input_dim=X_train.shape[1]))  # 은닉층
model.add(Dense(6, activation='relu'))  # 은닉층
model.add(Dense(1))  # 출력층

# 모델 컴파일
model.compile(loss='mean_squared_error',  # 회귀(regression) - 수치 예측
              optimizer='adam')

# 모델 학습
history = model.fit(train_data, Y_train, batch_size=10, epochs=200,
          validation_split=0.2)

y_pred = model.predict(test_data)
y_pred = y_pred.flatten()
for i in range(10):
    true_val = Y_test[i]
    pred_val = round(y_pred[i], 2)
    se = round((true_val - pred_val) ** 2, 4)
    print(f't={true_val}, p={pred_val}, se={se}')

def build_model():
    model = Sequential()
    # fully-connected layer를 추가 - 은닉층 2개, 출력층
    model.add(Dense(30, activation='relu', 
                    input_dim=X_train.shape[1]))  # 은닉층
    model.add(Dense(6, activation='relu'))  # 은닉층
    model.add(Dense(1))  # 출력층
    # 모델 컴파일
    model.compile(loss='mean_squared_error',  # 회귀(regression) - 수치 예측
                optimizer='adam')
    return model

import numpy as np

k = 4  # k-fold cross-validation
num_val_samples = len(train_data) // k
num_epochs = 200
all_scores = []
for i in range(k):
    print(f'processing {i}-fold ...')
    # k-fold CV에서 사용할 검증(validation) 데이터
    val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = Y_train[i * num_val_samples : (i + 1) * num_val_samples]
    # k-fold CV에서 사용할 학습(train) 데이터:
    # 원래 학습 데이터에서 검증 데이터를 제외한 나머지
    part_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    part_train_targets = np.concatenate(
        [Y_train[:i * num_val_samples],
         Y_train[(i + 1) * num_val_samples:]],
        axis=0)
    # 모델 생성 & 컴파일
    model = build_model()
    # 모델 학습
    fitted = model.fit(part_train_data, part_train_targets,
              epochs=num_epochs, verbose=0)
    loss = fitted.history['loss']
    all_scores.append(loss)

all_scores = np.array(all_scores)
print(all_scores)

average_scores = all_scores.mean(axis=0)
plt.plot(average_scores)
plt.show()

eval = model.evaluate(test_data, Y_test)
print(eval)
