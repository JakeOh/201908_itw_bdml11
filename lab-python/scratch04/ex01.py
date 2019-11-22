"""
선형 대수(Linear Algebra)
"""
import math


def add(v, w):
    """
    주어진 두 개의 n차원 벡터에서 성분별로 더하기를 해서,
    새로운 n차원 벡터를 리턴s

    :param v: n차원 벡터(성분이 n개인 벡터)
    :param w: n차원 벡터(성분이 n개인 벡터)
    :return: 각 성분별로 더하기 결과를 갖는 벡터
    """
    # result = []
    # for i in range(len(v)):
    #     result.append(v[i] + w[i])
    # return result
    if len(v) != len(w):
        raise ValueError('v와 w는 같은 length를 가져야 함.')
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def subtract(v, w):
    """
    주어진 두 개의 n차원 벡터에서 성분별로 뺄셈을 수행

    :param v: n차원 벡터
    :param w: n차원 벡터
    :return: n차원 벡터
    """
    if len(v) != len(w):
        raise ValueError('v와 w는 같은 length를 가져야 함.')
    return [v_i - w_i for v_i, w_i in zip(v, w)]


def vector_sum(vectors):
    """
    모든 벡터들에서 각 성분별 더하기를 수행
    vector_sum([[1, 2], [3, 4], [5, 6]]) = [9, 12]
    
    :param vectors: n차원 벡터들의 리스트(2차원 리스트)
    :return: n차원 벡터
    """
    num_of_elements = len(vectors[0])
    for vector in vectors[1:]:
        if num_of_elements != len(vector):
            raise ValueError('모든 벡터는 길이가 같아야 함.')

    # result = [0 for _ in range(num_of_elements)]  # [0, 0, 0 , ...]
    # for i in range(num_of_elements):
    #     for vector in vectors:
    #         result[i] += vector[i]
    # return result
    result = vectors[0]
    for vector in vectors[1:]:
        result = add(result, vector)
    return result


def scalar_multiply(c, vector):
    """
    c * [x1, x2, x3, ...] = [c*x1, c*x2, c*x3, ...]

    :param c: 숫자(스칼라)
    :param vector: n차원 벡터(n개의 아이템을 갖는 1차원 리스트)
    :return: n차원 벡터
    """
    # result = []
    # for x_i in vector:
    #     result.append(c * x_i)
    # return result
    return [c * x_i for x_i in vector]


def dot(v, w):
    """
    [v1, v2, v3, ...] @ [w1, w2, w3, ...] = v1*w1 + v2*w2 + v3*w3 + ...

    :param v: n차원 벡터
    :param w: n차원 벡터
    :return: 숫자(스칼라)
    """
    if len(v) != len(w):
        raise ValueError('두 벡터의 길이(length)는 같아야 함')
    sum = 0
    for v_i, w_i in zip(v, w):
        sum += v_i * w_i
    return sum


def vector_mean(vectors):
    """
    주어진 벡터들의 리스트에서 각 항목별 평균으로 이루어진 벡터

    :param vectors: n차원 벡터들의 리스트
    (길이가 n인 1차원 리스트를 아이템으로 갖는 2차원 리스트)
    :return: n차원 벡터(길이가 n인 1차원 리스트)
    """
    length = len(vectors)
    return scalar_multiply(1/length, vector_sum(vectors))


def sum_of_squares(vector):
    """
    vector = [x1, x2, ..., xn]일 때,
    x1 ** 2 + x2 ** 2 + ... + xn ** 2을 리턴

    :param vector: n차원 벡터(길이가 n인 1차원 리스트)
    :return: 숫자
    """
    # sum = 0
    # for x_i in vector:
    #     sum += x_i ** 2  # x_i * x_i
    # return sum
    return dot(vector, vector)


def magnitude(vector):
    """
    벡터의 크기를 리턴 - sqrt(sum_of_squares)

    :param vector: n차원 벡터(길이가 n인 1차원 리스트)
    :return: 숫자
    """
    return math.sqrt(sum_of_squares(vector))


def squared_distance(v, w):
    """
    v = [v1, v2, ..., vn), w = [w1, w2, ..., wn]일 때,
    (v1-w1)**2 + (v2-w2)**2 + ... + (vn-wn)**2 리턴

    :param v: n차원 벡터(길이가 n인 1차원 리스트)
    :param w: n차원 벡터(길이가 n인 1차원 리스트)
    :return: 숫자
    """
    # sum = 0
    # for v_i, w_i in zip(v, w):
    #     sum += (v_i - w_i) ** 2
    # return sum
    return sum_of_squares(subtract(v, w))


def distance(v, w):
    """
    두 벡터 v와 w 사이의 거리를 리턴 - sqrt(squared_distance)

    :param v: n차원 벡터(길이가 n인 1차원 리스트)
    :param w: n차원 벡터(길이가 n인 1차원 리스트)
    :return: 숫자
    """
    return math.sqrt(squared_distance(v, w))


if __name__ == '__main__':
    v = [1, 2]
    w = [3, 4]
    result = add(v, w)
    print('add =', result)
    result = subtract(v, w)
    print('subtract =', result)

    v = [1, 2, 3]
    w = [4, 5, 6]
    result = add(v, w)
    print('add =', result)
    result = subtract(v, w)
    print('subtract =', result)

    vectors = [[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]]
    result = vector_sum(vectors)
    print(result)

    v = [1, 3, 5]
    sm = scalar_multiply(2, v)
    print('scalar multiply =', sm)

    v = [2, 3]
    unit_x = [1, 0]  # x축 단위 벡터
    unit_y = [0, 1]  # y축 단위 벡터
    dot1 = dot(v, unit_x)
    print('dot1 =', dot1)
    dot2 = dot(v, unit_y)
    print('dot2 =', dot2)

    vectors = [
        [1, 2, 3],
        [3, 4, 5],
        [5, 6, 7],
        [7, 8, 9]
    ]
    vm = vector_mean(vectors)
    print('vector mean =', vm)

    v = [3, 4]
    ss = sum_of_squares(v)
    print('sum of squares =', ss)
    norm = magnitude(v)
    print('magnitude =', norm)

    v = [1, 2]
    w = [3, 4]
    sqd = squared_distance(v, w)
    print('squared distance =', sqd)
    dist = distance(v, w)
    print('distance = ', dist)
