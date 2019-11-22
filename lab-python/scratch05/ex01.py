"""
scratch05.ex01.py

통계

중심 경향성: 평균, 중앙값, 분위수(4분위, 100분위=퍼센트), 최빈값
산포도: 분산(variance), 표준편차(standard deviation), 범위(range)
상관 관계: 공분산(covariance), 상관 계수(correlation)
"""
import math
from collections import Counter
from scratch04.ex01 import dot


def mean(x):
    """
    리스트 x의 모든 아이템들의 평균을 계산해서 리턴
    x = [x1, x2, ..., xn]
    mean = (x1 + x2 + ... + xn) / n

    :param x: 원소 n개인 (1차원) 리스트
    :return: 평균
    """
    return sum(x) / len(x)


def median(x):
    """
    리스트 x를 정렬했을 때 중앙에 있는 값을 찾아서 리턴
    n이 홀수이면, 중앙값을 찾아서 리턴
    n이 짝수이면, 중앙에 있는 두 개 값의 평균을 리턴
    
    :param x: 원소 n개인 (1차원) 리스트
    :return: 중앙값
    """
    n = len(x)  # 리스트의 아이템 개수
    sorted_x = sorted(x)  # 데이터 크기 순으로 정렬(오름차순)
    mid_point = n // 2  # 리스트의 가운데 위치(인덱스)
    if n % 2:  # n이 홀수인 경우
        median_value = sorted_x[mid_point]
    else:  # n이 짝수인 경우
        median_value = (sorted_x[mid_point - 1] +
                        sorted_x[mid_point]) / 2
    return median_value


def quantile(x, p):
    """
    리스트 x의 p분위에 속하는 값을 찾아서 리턴

    :param x: 원소 n개인 (1차원) 리스트
    :param p: 0 ~ 1.0 사이의 값
    :return: 해당 분위수의 값
    """
    n = len(x)  # 리스트의 아이템 개수
    p_index = int(n * p)  # 해당 퍼센트의 인덱스 - 소수점 버리기
    sorted_x = sorted(x)  # 오름차순 정렬된 리스트
    return sorted_x[p_index]


def mode(x):
    """
    리스트에서 가장 자주 나타나는 값(최빈값)을 리턴.
    최빈값이 여러개인 경우, 최빈값들의 리스트를 리턴.
    from collections import Counter

    :param x: 원소가 n개인 (1차원) 리스트
    :return: 최빈값들의 리스트
    """
    counts = Counter(x)  # Counter 객체(인스턴스) 생성
    # print(counts)
    # print(counts.keys(), counts.values())
    # # Counter.keys(): 데이터(아이템), Counter.values(): 빈도수
    # print(counts.items())  # (값, 빈도수) 튜플들의 리스트
    max_count = max(counts.values())  # 빈도수의 최대값
    return [val for val, cnt in counts.items()
            if cnt == max_count]
    # freq = []  # 최빈값들을 저장할 리스트
    # for val, cnt in counts.items():  # Counter 객체에 대해서 반복
    #     if cnt == max_count:  # 빈도수가 최대 빈도수와 같으면
    #         freq.append(val)  # 리스트에 저장
    # return freq


def data_range(x):
    """
    최댓값과 최솟값의 차이

    :param x: 원소 n개인 (1차원) 리스트
    :return: 리스트의 최댓값 - 리스트의 최솟값
    """
    # return max(x) - min(x)
    sorted_x = sorted(x)
    return sorted_x[len(x) - 1] - sorted_x[0]


def de_mean(x):
    """
    편차(데이터 - 평균)들의 리스트

    :param x: 원소 n개인 (1차원) 리스트
    :return: 편차(deviation)들의 리스트
    """
    mu = mean(x)  # 평균
    return [x_i - mu for x_i in x]


def variance(x):
    """
    ((x1 - mean)**2 + (x2 - mean)**2 + ... + (xn - mean)**2)/(n-1)

    :param x: 원소 n개인 (1차원) 리스트
    :return: 분산
    """
    n = len(x)  # 원소 개수
    # x_bar = mean(x)  # 평균
    # return sum([(x_i - x_bar) ** 2 for x_i in x]) / (n - 1)
    deviations = de_mean(x)  # 편차들의 리스트
    return sum([d ** 2 for d in deviations]) / (n - 1)


def variance(x):
    print('dot 사용')
    n = len(x)
    deviations = de_mean(x)
    return dot(deviations, deviations) / (n - 1)


def standard_deviation(x):
    """
    sqrt(variance)

    :param x: 원소 n개인 (1차원) 리스트
    :return: 표준편차
    """
    return math.sqrt(variance(x))


def covariance(x, y):
    """
    공분산(covariance)
    Cov = sum((xi - x_bar)(yi - y_bar)) / (n - 1)

    :param x: 원소 n개인 (1차원) 리스트
    :param y: 원소 n개인 (1차원) 리스트
    :return: 공분산
    """
    x_bar = mean(x)  # x의 평균
    y_bar = mean(y)  # y의 평균
    x_deviations = [x_i - x_bar for x_i in x]  # 편차들의 리스트
    y_deviations = [y_i - y_bar for y_i in y]
    sum_of_deviations = dot(x_deviations, y_deviations)
    # sum_of_deviations = 0
    # for xd, yd in zip(x, y):
    #     sum_of_deviations += (xd - x_bar) * (yd - y_bar)
    return sum_of_deviations / (len(x) - 1)


def correlation(x, y):
    """
    상관 계수(correlation)
    Corr = Cov(x, y) / (SD(x) * SD(y))

    :param x: 원소 n개인 (1차원) 리스트
    :param y: 원소 n개인 (1차원) 리스트
    :return: 상관 계수
    """
    sd_x = standard_deviation(x)
    sd_y = standard_deviation(y)
    if sd_x != 0 and sd_y != 0:
        corr = covariance(x, y) / (sd_x * sd_y)
    else:
        corr = 0

    return corr


if __name__ == '__main__':
    data = [2, 2, 3, 3, 4, 4, 4, 6, 6, 6, 100]
    mean_data = mean(data)
    print(mean_data)

    median_data = median(data)
    print(median_data)

    quantile_1 = quantile(data, 0.25)  # 1사분위 값
    print(quantile_1)
    quantile_3 = quantile(data, 0.75)  # 3사분위 값
    print(quantile_3)

    most_frequent = mode(data)
    print(most_frequent)

    print('variance =', variance(data))
    print('standard deviation =', standard_deviation(data))

    x = [2, 2, 3, 3, 4, 4, 4, 6, 6, 6, 100]
    y = [10, 2, 5, 3, 4, 1, 4, 0, 6, 1, 7]
    cov = covariance(x, y)
    print('covariance =', cov)
    # 27.872727272727275
    print('correlation =', correlation(x, y))

    x = [-3, -2, -1, 0, 1, 2, 3]
    y = [3, 2, 1, 0, 1, 2, 3]
    # y = |x|
    print(correlation(x, y))
