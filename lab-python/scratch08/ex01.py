"""
Gradient Descent(경사 하강법):
데이터 과학을 하다보면, 최적화 문제들을 만나게 됨.
최적화 문제 - 특정 상황에서 가장 적합한 모델을 찾는 문제
(예) 모델의 오류(error)를 "최소화", liklihood(우도)를 "최대화"
타겟 함수를 최소(혹은 최대)로 만들어 주는 파라미터를 찾는 문제

곡선의 접선을 찾고, 접선의 기울기 방향으로 점을 이동시켜 나가면서
최소(최대)값을 찾음
"""
import random

import matplotlib.pyplot as plt


def f(x):
    """ y = x**2 """
    return x ** 2


def f_derivative(x):
    """ y = x**2의 도함수(미분): y = 2x """
    return 2 * x


def tangent(x, a, x1, y1):
    """ 기울기가 a이고, 점 (x1, y1)을 지나는 직선의 방정식
        y - y1 = a(x - x1)
    """
    return a * (x - x1) + y1


def difference_quotient(f, x, h):
    """ 함수 f의 도함수의 근사값 """
    return (f(x + h) - f(x)) / h


def move(x, direction, step=-0.1):
    """x좌표를 새로운 x로 이동.
    direction: 접선의 기울기.
    step > 0인 경우는 접선과 같은 방향으로 이동 -> 최댓값 찾기
    step < 0인 경우는 접선과 반대 방향으로 이동 -> 최솟값 찾기"""
    return x + step * direction


if __name__ == '__main__':
    # 그래프를 그릴 x 좌표들: (-3.0, -2.9, ..., 2.9, 3.0)
    xs = [x / 10 for x in range(-30, 31)]
    # 그래프를 그릴 y 좌표들
    ys = [f(x) for x in xs]

    # x**2 그래프에서 x=-1에서의 접선을 그리기 위해서
    # 직선의 방정식 y = ax + b에서 기울기 a와 y절편 b를 찾아야 함.
    # 기울기 a는 x**2의 미분값으로 찾음.
    # y절편은 (x1, f(x1))을 지나는 직선임을 이용해서 찾음.
    a = f_derivative(-1)  # x=-1에서 접선의 기울기
    x1, y1 = -1, f(-1)  # x=-1에서 곡선과 접선이 만나는 점의 좌표
    tangents = [tangent(x, a, x1, y1) for x in xs]

    # 도함수의 근사값을 사용해서 x=-1에서 기울기를 찾음
    # h값이 0에 가까울 수록 더 정확한 접선의 기울기가 됨
    h = 0.1
    a2 = difference_quotient(f, x=-1, h=h)
    tangent_estimates = [tangent(x, a2, x1, y1)
                         for x in xs]

    plt.plot(xs, ys)
    plt.plot(xs, tangents, label='actual')
    plt.plot(xs, tangent_estimates, label=f'estimate: h={h}')
    plt.ylim(bottom=-2)
    plt.axhline(y=0, color='black')  # y=0인 수평 보조선
    plt.axvline(x=0, color='black')  # x=0인 수직 보조선
    plt.legend()
    plt.show()

    # 실제 기울기(f_derivative)와 기울기 근사값 비교
    xs = [x for x in range(-10, 11)]
    actuals = [f_derivative(x) for x in xs]
    estimates_1 = [difference_quotient(f, x, h=1)
                   for x in xs]
    estimates_2 = [difference_quotient(f, x, h=0.1)
                   for x in xs]

    plt.scatter(xs, actuals, label='actual', marker="x")
    plt.scatter(xs, estimates_1, label='h=1', marker='+')
    plt.scatter(xs, estimates_2, label='h=0.1', marker='o')

    plt.legend()
    plt.show()

    # 경사 하강법(gradient descent):
    xs = [x / 10 for x in range(-30, 31)]  # [-3, -2.9, ..., 2.9, 3]
    ys = [f(x) for x in xs]  # y=x**2 그래프의 y값들
    init_x = 2  # 최솟값을 찾기위해 시작할 x 좌표
    for _ in range(5):
        # x = init_x에서의 접선의 기울기
        gradient = difference_quotient(f, init_x, h=0.01)
        # 접선을 그래프로 그리기 위해서
        tangent_estimates = [tangent(x, gradient, init_x, f(init_x))
                             for x in xs]
        plt.plot(xs, tangent_estimates, label=f'x={init_x}')
        # x 좌표를 새로운 좌표로 이동하고 반복
        init_x = move(init_x, gradient, step=-0.9)

    plt.plot(xs, ys, color='black')
    plt.legend()
    plt.ylim(bottom=-1)
    plt.show()

    # 임의의 점에서 시작해서 y=x**2의 최솟값을 찾음
    random.seed(1128)
    init_x = random.randint(-10, 10)  # 시작 값
    print(f'init_x = {init_x}')
    tolerance = 0.0000001
    # 두 x좌표 사이의 거리가 tolerance 이하이면 반복문 종료
    count = 0
    while True:  # 반복
        count += 1
        # x좌표에서의 접선의 기울기를 계산
        gradient = difference_quotient(f, init_x, h=0.001)
        # 찾은 기울기를 사용해서 x좌표를 이동
        next_x = move(init_x, gradient, step=-0.1)
        print(f'{count}: x = {next_x}')
        if abs(next_x - init_x) < tolerance:
            # 이동 전과 후의 x값의 차이가 tolerance 미만이면 반복문 종료
            break
        else:
            # 이동한 점이 다음 반복에서는 시작점이 되어야 함!
            init_x = next_x
