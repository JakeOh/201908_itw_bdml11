from collections import Counter

import numpy as np
import pandas as pd
import os

from scratch12.ex04 import summarize_by_class, calculate_class_probability
from sklearn.metrics import confusion_matrix, classification_report

def train_test_split(df, test_size):
    """df=데이터 프레임, test_size=테스트 세트의 비율
    학습 세트(X_train)와 검증 세트(X_test)를 리턴
    train/test set: 리스트 또는 np.ndarray
    [[x1, x2, ..., lable1], [x1, x2, ..., label2], [], ...], [[], [], [], ...]
    """
    # DataFrame을 numpy.ndarray 타입으로 변환
    array = df.to_numpy()
    # array의 순서를 무작위로 섞음.
    np.random.seed(1213)
    np.random.shuffle(array)
    # 학습 세트/테스트 세트를 나누기 위한 인덱스
    cut = int(len(array) * (1 - test_size))
    # 학습 세트/테스트 세트로 나눔
    train_set = array[:cut]
    test_set = array[cut:]
    # 결과 리턴
    return train_set, test_set


def predict(summaries, X_test):
    """테스트 세트의 예측값들의 배열(리스트)을 리턴
    [0, 1, 1, 2, 0, 0, 2, ...]"""
    predicts = []  # 빈 리스트 생성
    # X_test의 원소 개수만큼 반복하면서
    for test in X_test:
        # 각 원소(예측값을 찾으려는 데이터)의 클래스에 속할 확률을 계산
        probabilities = calculate_class_probability(summaries, test)
        # 각 클래스에 속할 확률들 중에서 최댓값을 찾음
        best_label, _ = sorted(probabilities.items(), key=lambda x: -x[1])[0]
        # sorted(Iterable, key=정렬기준함수):
        #   정렬 기준 함수의 리턴값을 기준으로 Iterable 타입을 정렬함
        # sorted(dict): dict의 키들을 오름차순 정렬한 리스트
        # sorted(dict.values()): dict의 값들을 오름차순 정렬한 리스트
        # sorted(dict.items()): (key, value) 튜플을 key값을 기준으로 정렬
        # best_label, best_prob = None, -1
        # for k, v in probabilities.items():
        #     if v > best_prob:  # 더 큰 확률값을 찾을 경우
        #         best_prob = v  # 찾은 확률값으로 best_prob을 업데이트
        #         best_label = k  # 찾은 확률값의 키로 예측 레이블을 업데이트
        # # 확률 최댓값의 키값을 예측값 리스트에 추가
        predicts.append(best_label)

    return predicts


if __name__ == '__main__':
    iris_file = os.path.join('..', 'scratch11', 'iris.csv')
    cancer_file = os.path.join('..', 'scratch11', 'wisc_bc_data.csv')

    iris_dataset = pd.read_csv(iris_file,
                               header=None,
                               names=['sl', 'sw', 'pl', 'pw', 'Class'])
    print(iris_dataset.head())
    print(iris_dataset.shape)
    print(iris_dataset.iloc[:, -1])  # DataFrame의 가장 마지막 컬럼
    iris_dataset.info()
    # iris 품종: Iris-setosa, Iris-versicolor, Iris-virginica
    species = set(iris_dataset.iloc[:, -1])
    print(species)

    # Iris-setosa=0, Iris-versicolor=1, Iris-virginica=2로 변경
    iris_dataset.loc[iris_dataset['Class'] == 'Iris-setosa', 'Class'] = 0
    iris_dataset.loc[iris_dataset['Class'] == 'Iris-versicolor', 'Class'] = 1
    iris_dataset.loc[iris_dataset['Class'] == 'Iris-virginica', 'Class'] = 2
    print(iris_dataset.head())

    species = set(iris_dataset.iloc[:, -1])  # 'Class' 컬럼
    print(species)
    species_counts = Counter(iris_dataset.iloc[:, -1])
    print(species_counts)

    iris_train, iris_test = train_test_split(iris_dataset, test_size=0.2)
    print('Train set:', iris_train.shape)
    print('Test set:', iris_test.shape)
    print(iris_train[-5:])  # [:5] - 처음 5개 원소, [-5:] - 마지막 5개 원소

    # 학습 데이터 세트만으로 summaries(평균, 표준 편차, 개수)를 찾음.
    model = summarize_by_class(iris_train)  # GaussianNB에서 사용할 mu, sigma
    print(model)
    print(model[0.0])

    # 검증(테스트) 데이터 세트로 모델에서 예측하는 값들을 찾음.
    iris_pred = predict(model, iris_test)
    print(iris_pred)
    print(iris_test[:, -1] == iris_pred)
    print(confusion_matrix(iris_test[:, -1], iris_pred))
    print(classification_report(iris_test[:, -1], iris_pred))

    # 데이터 준비
    cancer_dataset = pd.read_csv(cancer_file)

    # 데이터 확인
    print(cancer_dataset.head())
    cancer_dataset.info()

    # 데이터 전처리 - 'id' 삭제, 'diagnosis' 변수(컬럼) 값들을 숫자로 변환
    ret = cancer_dataset.drop(columns=['id'])
    # 원본 데이터 프레임은 그대로 남아 있고, 컬럼이 삭제된 새로운 데이터 프레임을 리턴
    print(ret.head())

    del cancer_dataset['id']  # 원본 데이터 프레임에서 컬럼을 삭제
    print(cancer_dataset.head())

    diagnosis = set(cancer_dataset['diagnosis'])
    print(diagnosis)
    # B(Benign)=0, M(Malignant)=1
    cancer_dataset.loc[cancer_dataset['diagnosis'] == 'B', 'diagnosis'] = 0
    cancer_dataset.loc[cancer_dataset['diagnosis'] == 'M', 'diagnosis'] = 1

    # cancer_dataset.loc[cancer_dataset['diagnosis'] == 'B', 'Class'] = 0
    # cancer_dataset.loc[cancer_dataset['diagnosis'] == 'M', 'Class'] = 1
    # del cancer_dataset['diagnosis']
    # print(cancer_dataset.tail())

    # 'diagnosis' 컬럼을 데이터 프레임의 마지막 컬럼으로
    column_names = cancer_dataset.columns.tolist()
    # [name for name in cancer_dataset.columns]
    print(column_names)
    column_names.remove('diagnosis')
    column_names.append('diagnosis')
    print(column_names)
    # df = cancer_dataset.reindex(columns=column_names)
    # print(df.head())
    df = cancer_dataset[column_names]
    print(df.head())

    df = cancer_dataset.loc[:, ::-1]
    print(df.head())

    cancer_train, cancer_test = train_test_split(df, test_size=0.2)
    model = summarize_by_class(cancer_train)
    predicts = predict(model, cancer_test)
    print(confusion_matrix(cancer_test[:, -1], predicts))
    print(classification_report(cancer_test[:, -1], predicts))



