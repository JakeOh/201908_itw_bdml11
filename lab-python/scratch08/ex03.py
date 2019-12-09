"""
편미분(Partial Differentiation)을 이용한 경사 하강법
"""
import random

from scratch04.ex01 import scalar_multiply, add, distance


def partial_difference_quotient(f, v, i, h=0.001):
    """
    (f([x1, ..., xi + h, ..., xn]) - f([x1, ..., xi, ..., xn])) / h

    :param f: f(vector) = float인 함수
    :param v: 기울기(gradient)를 계산할 점의 위치 - 벡터(리스트)
    :param i: 기울기(gradient)를 계산할 성분의 인덱스 - 정수
    :param h: i번째 성분의 변화량
    :return: 편미분 결과 -> i번째 성분 방향의 gradient
    """
    # w = [v_j + (h if i == j else 0)
    #      for j, v_j in enumerate(v)]
    w = []
    for j, v_j in enumerate(v):
        if j == i:
            v_j += h
        w.append(v_j)
    return (f(w) - f(v)) / h


def estimate_gradient(f, v, h=0.001):
    """
    [df/dx1, df/dx2, ..., df/dxi, ..., df/dxn]

    :param f: f(vector) = float 함수
    :param v: 기울기를 구하려는 점의 좌표. [x1, ...., xn]
    :param h: 증분(increment)
    :return: 모든 성분의 gradient들로 이루어진 벡터(리스트)
    """
    return [partial_difference_quotient(f, v, i, h)
            for i, _ in enumerate(v)]


def gradient_step(v, gradient, step=-0.1):
    """
    [xi + step * df/dxi]

    :param v: 이동 전 점의 위치
    :param gradient: 점 v에서의 기울기
    :param step: 이동시키는 가중치(학습률)
    :return: 기울기의 방향으로 이동한 점의 위치
    """
    # step * df/dxi
    increment = scalar_multiply(step, gradient)
    # xi + step * df/dxi
    return add(v, increment)


if __name__ == '__main__':
    # f([x1, x2]) = x1 ** 2 + x2 ** 2 의 최솟값: (0, 0)
    # g([x1, x2]) = (x1 - 1) ** 2 + (x2 + 1) ** 2 의 최솟값: (1, -1)
    def f(v):
        """ v= [x1, x2] 가정 """
        return v[0] ** 2 + v[1] ** 2

    def g(v):
        return (v[0] - 1) ** 2 + (v[1] + 1) ** 2

    random.seed(1129)
    # 기울기를 계산할 최초의 (x1, y1) 좌표를 임의로 선택
    init_v = [random.randint(-10, 10), random.randint(-10, 10)]
    print('init_v =', init_v)

    tolerance = 0.000001  # 반복문을 종료할 임계값

    count = 0
    while True:  # 무한 반복문
        count += 1
        # 선택한 좌표(x1, y1)에서 기울기 계산
        gradient = estimate_gradient(g, init_v)
        # 다음 좌표로 점을 이동시킴
        next_v = gradient_step(init_v, gradient, step=-0.1)
        print(f'{count}: next_v = {next_v}')
        # 이동 전 위치와 이동 후 위치 사이의 거리를 계산
        if distance(init_v, next_v) < tolerance:
            # 이동 거리가 임계값보다 작다면 반복문을 종료
            break
        else:
            # 이동시킨 위치에서 gradient를 다시 계산하기 위해서
            init_v = next_v





