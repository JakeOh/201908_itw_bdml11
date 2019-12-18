"""
연속형 변수에서의 Naive Bayes 예측 원리
"""
import random
from collections import defaultdict
from math import exp, pi, sqrt

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


def separate_by_class(dataset):
    """데이터 세트를 클래스 별로 분류한 사전(dict)를 리턴
    {class_0: [[], [], ...],
     class_1: [[], [], ...], ...}
     dataset: 리스트라고 가정.
    """
    separated = dict()  # 빈 dict를 생성
    for i in range(len(dataset)):  # dataset의 길이(원소의 개수)만큼 반복
        vector = dataset[i]  # dataset의 i번째 row(원소)
        class_value = vector[-1]  # 벡터의 가장 마지막 원소가 레이블(클래스)
        if class_value not in separated:
            # 클래스 값이 dict의 키로 존재하지 않으면
            separated[class_value] = []  # 비어 있는 리스트를 생성
        separated[class_value].append(vector)
    return separated


def separate_by_class2(dataset):
    separated = defaultdict(list)  # defaultdict 객체 생성
    for i in range(len(dataset)):  # 리스트의 원소 개수만큼 반복
        vector = dataset[i]  # 리스트의 i번째 원소
        class_value = vector[-1]  # 리스트의 마지막 원소는 클래스(레이블)
        # 클래스를 key를 갖는 리스트에 vector를 추가
        separated[class_value].append(vector)
    return separated


def summarize_dataset(dataset):
    """데이터 세트의 각 컬럼(변수, 특성)의 평균과 표준 편차, 개수들을 계산, 리턴
    [(mean, std, count), (), ...]
    """
    # for col in zip(*dataset):
    #     print(col)
    # *: unpacking 연산자
    #   *[1, 2] -> 1, 2
    #   *[[1,2], [3,4]] -> [1,2], [3,4]
    # zip(*[[1,2], [3,4]]) -> zip([1,2], [3,4]) -> [1, 3], [2, 4]
    summaries = [(np.mean(col), np.std(col), len(col))
                 for col in zip(*dataset)]
    # 마지막 컬럼은 데이터가 아니라 클래스(레이블)이므로 평균, 표준편차가 필요 없음
    del summaries[-1]  # 리스트의 마지막 원소를 삭제
    return summaries


def summarize_by_class(dataset):
    """데이터 세트의 컬럼(변수, 특성)들에 대해서, 각 클래스 별로
    평균, 표준 편차, 개수 요약
    {class_0: [(x1_mean, x1_std, x1_len), (x2_mean, x2_std, x2_len), ...],
     class_1: [(x1_mean, x1_std, x1_len), (x2_mean, x2_std, x2_len), ...],
     ...}
    """
    # 데이터 세트를 클래스 별로 분류
    separated = separate_by_class2(dataset)
    summaries = dict()
    for class_value, vectors in separated.items():
        summaries[class_value] = summarize_dataset(vectors)
    return summaries


def calculate_probability(x, mu, sigma):
    """Gaussian Normal Distribution"""
    exponent = exp(-(x - mu)**2 / (2 * sigma**2))
    return (1 / (sqrt(2 * pi) * sigma)) * exponent


def calculate_class_probability(summaries, vector):
    """주어진 vector의 각 클래스별 예측값을 계산
    P(class|x1,x2) ~ P(class) * P(x1|class) * P(x2|class)
    """
    total_rows = sum([vectors[0][2] for _, vectors in summaries.items()])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        # p = P(class)
        probabilities[class_value] = class_summaries[0][2] / total_rows
        for i in range(len(class_summaries)):
            mu, sigma, count = class_summaries[i]
            # prob = P(x1|class)
            prob = calculate_probability(vector[i], mu, sigma)
            # p = P(class) * P(x1|class)
            probabilities[class_value] *= prob
    return probabilities


if __name__ == '__main__':
    # 테스트 하기 위한 더미 데이터를 생성
    # [[x1, x2, 0], ... [x1, x2, 1], ...]
    random.seed(1212)
    dataset = [[random.random(), random.random(), x // 5]
               for x in range(10)]
    # print(dataset)

    df = pd.DataFrame(dataset, columns=['x1', 'x2', 'Class'])
    print(df)

    separated = separate_by_class2(dataset)
    print(separated)

    summary = summarize_dataset(dataset)
    print(summary)

    summaries = summarize_by_class(dataset)
    print(summaries)

    print(dataset[0])
    probabilities = calculate_class_probability(summaries, dataset[0])
    print(probabilities)
    probabilities = calculate_class_probability(summaries, dataset[5])
    print(probabilities)

    # iris 데이터에 적용
    X, y = load_iris(return_X_y=True)
    print(X[:5])
    print(y[:5])
    iris_dataset = np.c_[X, y]
    print(iris_dataset[:5])
    print(iris_dataset[-5:])

    # iris_dataset[0], iris_dataset[50], iris_dataset[100]
    summaries = summarize_by_class(iris_dataset)
    probabilities = calculate_class_probability(summaries, iris_dataset[0])
    print(probabilities)
    probabilities = calculate_class_probability(summaries, iris_dataset[50])
    print(probabilities)
    probabilities = calculate_class_probability(summaries, iris_dataset[100])
    print(probabilities)





