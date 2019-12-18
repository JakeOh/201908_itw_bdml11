"""
R을 활용한 머신 러닝 - 암 데이터 파일(csv)
scikit-learn 패키지 활용, kNN 결과
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

if __name__ == '__main__':
    # 데이터 준비
    dataset = pd.read_csv('wisc_bc_data.csv')
    print(dataset.info())  # 569 x 32
    print(dataset.describe())  # scaling이 필요
    print(dataset.head())

    # n차원 상의 point와 레이블로 구분
    X = dataset.iloc[:, 2:].to_numpy()
    y = dataset.iloc[:, 1].to_numpy()  # M, B
    print(X[0])
    print(y[0])

    # train/test set로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Scaling
    scaler = StandardScaler()
    scaler.fit(X_train)  # X_train 세트의 평균/표준편차 계산
    X_train = scaler.transform(X_train)  # 학습 세트 변환
    X_test = scaler.transform(X_test)  # 테스트 세트 변환
    print(np.mean(X_train[:, 0]), np.std(X_train[:, 0]))
    print(np.mean(X_test[:, 0]), np.std(X_test[:, 0]))

    # 학습/예측
    knn = KNeighborsClassifier(n_neighbors=5)  # 분류기 생성
    knn.fit(X_train, y_train)  # 분류기 학습
    y_pred = knn.predict(X_test)  # 검증 데이터의 예측 결과 추출

    # 모델 평가
    print(confusion_matrix(y_test, y_pred))  # 혼동 행렬
    print(classification_report(y_test, y_pred))

    # k값 변경에 따른 모델 성능
    errors = []
    for i in range(1, 41):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        errors.append(np.mean(pred_i != y_test))

    plt.plot(range(1, 41), errors, marker='o')
    plt.show()



