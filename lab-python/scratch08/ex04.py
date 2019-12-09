"""
선형 회귀(Linear Regression)
모델 y = ax + b에서, 기울기(slope) a와 y절편 b를 찾는 문제.

(a, b)를 특정 값으로 가정했을 때의 예상값과 실제값 사이의 오차들의
제곱의 합을 최소로 하는 파라미터 a와 b를 찾는 문제임.
실제값: (x, y)
예상값: y_hat = theta1 * x + theta2
오차: e = y_hat - y = theta1 * x + theta2 - y
오차 제곱: f = e**2 = (theta1 * x + theta2 - y)**2
기울기 theta1에 대한 편미분: df/dt1 ~ e * x
y절편 theta2에 대한 편미분: df/dt2 ~ 0

1) 확률적 경사 하강법(Stochastic Gradient Descent)
    전체 데이터 세트(훈련 세트)에서 샘플 데이터 1개씩 gradient를
    계산. 파라미터(기울기, 절편) 변경.
    위 과정을 임의의 회수(epoch)만큼 반복.
2) 배치 경사 하강법(Batch GD):
    전체 데이터 세트를 사용해서 전체 gradient들의 평균을 gradient로
    사용해서 파라미터 theta를 변경.
    위 과정을 임의의 회수(epoch)만큼 반복
3) 미니 배치 경사 하강법(Mini-batch GD):
    전체 데이터 세트를 크기를 작세 샘플링해서 처리하는 방식.
    각각의 epoch마다 데이터 세트의 순서를 섞어서 파라미터(theta)의
    최적값을 찾음.
"""
import random

from scratch04.ex01 import vector_mean
from scratch08.ex03 import gradient_step


def linear_gradient(x, y, theta):
    """
    특정 데이터 (x, y)에서 기울기와 y절편에 대한
    편미분 벡터 리턴

    :param x: 실제 데이터
    :param y: 실제 데이터
    :param theta: [theta1, theta2] 벡터(리스트). [기울기, y절편]
    """
    slope, intersect = theta
    y_hat = slope * x + intersect  # 예상값
    error = y_hat - y  # 오차
    # error**2을 최소화하는 slope(기울기), intersect(절편) 찾기
    # 점(x,y)에서 [기울기에 대한 편미분, 절편에 대한 편미분]
    gradient = [error * x, error]
    return gradient


def minibatches(dataset, batch_size, shuffle=True):
    # 데이터 세트를 무작위로 섞음
    if shuffle:
        random.shuffle(dataset)

    # 배치 시작 인덱스: 0, batch_size, 2*batch_size, ...
    batch_starts = [s for s in range(0, len(dataset), batch_size)]
    # [[ds[0], ..., ds[size-1]], [ds[size], ... ds[2size-1]], ...]
    mini = [dataset[s:s+batch_size] for s in batch_starts]
    return mini


dataset = [(x, 20 * x + 5) for x in range(-50, 51)]

if __name__ == '__main__':
    print('=== 확률적 경사 하강법 ===')
    # 임의의 파라미터 초깃값
    theta = [1, 1]  # [기울기, 절편], y = x + 1
    # 파라미터의 값을 변경할 때 사용할 가중치(학습률, learning rate)
    step = 0.001  # next_x = init_x + step * gradient

    for epoch in range(200):  # 임의의 회수(epoch)만큼 반복
        random.shuffle(dataset)  # 데이터 세트를 랜덤하게 섞음
        # 각각의 epoch마다
        # 데이터 세트에서 샘플(x, y)를 추출
        for x, y in dataset:
            # 각 점에서 gradient를 계산
            gradient = linear_gradient(x, y, theta)
            # 파라미터 theta(직선의 기울기와 절편)를 변경
            theta = gradient_step(theta, gradient, -step)
        if (epoch + 1) % 10 == 0:
            print(f'{epoch}: {theta}')

    print('\n=== 배치 경사 하강법 ===')
    step = 0.001
    theta = [1, 1]  # 임의의 값으로 [기울기, 절편] 설정
    for epoch in range(5000):
        # 모든 샘플에서의 gradient를 계산
        gradients = [linear_gradient(x, y, theta)
                     for x, y in dataset]
        # gradients의 평균을 계산
        gradient = vector_mean(gradients)
        # 파라미터 theta(기울기, 절편)을 변경
        theta = gradient_step(theta, gradient, -step)
        if (epoch + 1) % 100 == 0:
            print(f'{epoch}: {theta}')

    print('\n=== 미니 배치 경사 하강법 ===')
    theta = [1, 1]  # 임의의 파라미터 시작값
    step = 0.001  # 학습률
    for epoch in range(1000):
        mini_batches = minibatches(dataset, 20, True)
        for batch in mini_batches:
            gradients = [linear_gradient(x, y, theta)
                         for x, y in batch]
            gradient = vector_mean(gradients)
            theta = gradient_step(theta, gradient, -step)
        if (epoch + 1) % 100 == 0:
            print(f'{epoch}: {theta}')
