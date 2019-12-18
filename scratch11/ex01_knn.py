"""
scikit-learn 패키지를 이용한 k-NN(k-Nearest Neighbor: 최근접 이웃)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

if __name__ == '__main__':
    # 1. 데이터 준비
    # csv 파일에 컬럼 이름들(헤더 정보)가 없기 때문에 컬럼 이름들을 정의
    col_names = ['sepal-length', 'sepal-width',
                 'petal-length', 'petal-width',
                 'Class']
    # csv 파일에서 DataFrame을 생성
    dataset = pd.read_csv('iris.csv',
                          header=None,
                          names=col_names)
    # DataFrame 확인
    print(dataset.shape)  # (row 개수, column 개수)
    print(dataset.info())  # 데이터 타입, row 개수, column 개수, 컬럼 데이터 타입
    print(dataset.describe())  # 요약 통계 정보
    print(dataset.iloc[:5])  # dataset.head()
    print(dataset.iloc[-5:])  # dataset.tail()

    # 2. 데이터 전처리:
    # 데이터 세트를 데이터(포인트)와 레이블로 구분
    # X = 전체 행, 마지막 열 제외한 모든 열 데이터 -> n차원 공간의 포인트
    X = dataset.iloc[:, :-1].to_numpy()  # DataFrame을 np.ndarray로 변환
    # print(X)
    # y = 전체 행, 마지막 열 데이터 -> 레이블
    y = dataset.iloc[:, 4].to_numpy()
    # print(y)
    # 전체 데이터 세트를 학습 세트(training set)와 검증 세트(test set)로 나눔.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(len(X_train), len(X_test), len(y_train), len(y_test))
    print(X_train[:3])
    print(y_train[:3])
    
    # 3. 거리 계산을 위해서 각 특성(변수)들을 스케일링(표준화)
    # Z-score 표준화: 평균을 0, 표준편차 1로 변환
    scaler = StandardScaler()  # Scaler 객체 생성
    scaler.fit(X_train)  # 스케일링(표준화)를 위한 평균과 표준 편차 계산
    X_train = scaler.transform(X_train)  # 스케일링(표준화) 수행
    X_test = scaler.transform(X_test)
    # 스케일링(z-score 표준화 수행 결과 확인)
    for col in range(4):
        print(f'train 평균 = {X_train[:, col].mean()}, 표준편차 = {X_train[:, col].std()}')
        print(f'test 평균 = {X_test[:, col].mean()}, 표준편차 = {X_test[:, col].std()}')

    # 4. 학습/예측(Training/Prdiction)
    # k-NN 분류기(classifier)를 생성
    classifier = KNeighborsClassifier(n_neighbors=5)
    # 분류기 학습
    classifier.fit(X_train, y_train)
    # 예측
    y_pred = classifier.predict(X_test)
    print(y_pred)

    # 5. 모델 평가
    conf_matrix = confusion_matrix(y_test, y_pred)  # 혼동 행렬
    print(conf_matrix)
    report = classification_report(y_test, y_pred)  # 리포트
    print(report)
    # 정확도(accuracy) = (TP + TN) / (TP + FP + FN + TN)
    # 정밀도(precision) = TP / (TP + FP)
    # 재현율(recall) = TP / (TP + FN)

    # 6. 모델 개선(향상) - k값을 변화시킬 때 에러가 줄어드는 지.
    errors = []
    for i in range(1, 31):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        errors.append(np.mean(pred_i != y_test))
    print(errors)

    plt.plot(range(1, 31), errors, marker='o')
    plt.title('Mean Error with K-Value')
    plt.xlabel('k-value')
    plt.ylabel('mean error')
    plt.show()


