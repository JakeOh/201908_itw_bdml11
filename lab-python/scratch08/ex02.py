"""
gradient descent 연습
"""
import random

import matplotlib.pyplot as plt

from scratch08.ex01 import difference_quotient, move


def g(x):
    """ y = (1/3)x**3 - x """
    return x**3 / 3 - x


if __name__ == '__main__':
    # ex01에서 작성한 함수들을 이용
    # 함수 g(X)의 그래프를 그림
    xs = [x / 10 for x in range(-30, 31)]
    ys = [g(x) for x in xs]
    plt.plot(xs, ys)
    plt.axhline(y=0, color='0.3')
    plt.axvline(x=0, color='0.3')
    plt.axvline(x=-1, color='0.75')
    plt.axvline(x=1, color='0.75')
    plt.ylim(bottom=-2, top=2)
    plt.title('y = x**3 / 3 - x')
    plt.show()

    # 극값(local 최소/최대)를 경사 하강법으로 찾음
    tolerance = 0.0000001
    # 국소(지역) 최솟값을 찾기 위한 x 시작 좌표
    init_x = 1.9
    count = 0
    while True:  # 반복
        count += 1
        # x좌표 init_x에서의 접선의 기울기를 찾음
        gradient = difference_quotient(g, init_x, h=0.0001)
        # x좌표를 이동: 최솟값 문제이기 때문에 기울기 반대 방향
        next_x = move(init_x, gradient, step=-0.7)
        print(f'{count}: x = {next_x}')
        # 이동 전후의 x좌표 사이의 거리가 tolerance(임계값)보다
        # 작은 지를 확인. 만약 임계값보다 작다면 반복 종료
        if abs(next_x - init_x) < tolerance:
            break
        else:
            # 이동 후의 x의 위치를 새로운 기울기를 찾기 위한 시작값으로 설정
            init_x = next_x

    # 최댓값을 찾기 위한 x 시작 위치
    init_x = -1.9
    count = 0
    while True:
        count += 1
        gradient = difference_quotient(g, init_x, h=0.001)
        # 최댓값을 찾는 문제이므로 기울기의 부호와 같은 방향으로 이동
        next_x = move(init_x, gradient, step=0.1)
        print(f'{count}: x = {next_x}')
        if abs(next_x - init_x) < tolerance:
            break
        else:
            init_x = next_x
