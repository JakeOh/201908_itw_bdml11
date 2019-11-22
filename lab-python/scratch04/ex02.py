"""
numpy 패키지를 사용한 벡터 연산
"""
import math

import numpy as np

print('numpy version:', np.__version__)

# 파이썬 list 데이터 타입의 연산
v = [1, 2]  # type: class list
print(type(v))
print('v =', v)

w = [2, 3]
print('w =', w)

print(v + w)
# list는 + 연산을 사용할 수 있음
# + 연산자는 extend 함수와 비슷한 기능
# + 연산자는 v나 w를 변경하지 않고, 새로운 리스트를 리턴
# v.extend(w) 함수는 v를 변경함
# print(v - w)  # list는 - 연산을 사용할 수 없음!
v.extend(w)
print(v)

# numpy 패키지의 ndarray 타입을 사용
# n-dimensional array(n차원 배열)
v = np.array([1, 2])
print('type:', type(v))
print(v)
print('dimension:', v.ndim)
print('shape:', v.shape)
# 1차원 ndarray인 경우 shape은 (원소 개수,)

v = np.array([
    [1, 2],
    [2, 3],
    [3, 4]
])
print('type:', type(v))
print('dimension:', v.ndim)
print('shape:', v.shape)
# 2차원 ndarray인 경우 shape은 (row 개수, column 개수)
print(v)

l = [
    [1, 2],
    [3, 4],
    [5, 6]
]
print(l)

# ndarray 타입을 사용한 벡터 연산
v = np.array([1, 2, 3])
w = np.array([3, 4, 5])
vector_add = v + w
print('vector add =', vector_add)
vector_sub = v - w
print('vector subtract =', vector_sub)

vectors = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

np_sum = np.sum(vectors)  # 2차원 배열의 모든 원소들의 합
print('np_sum =', np_sum)

# axis=0: 2차원 배열의 각 컬럼들의 합으로 이루어진 배열
np_sum_by_col = np.sum(vectors, axis=0)
print('np_sum_by_col =', np_sum_by_col)

# axis=1: 2차원 배열의 각 행(row)들의 합으로 이루어진 배열
np_sum_by_row = np.sum(vectors, axis=1)
print('np_sum_by_row =', np_sum_by_row)

np_mean = np.mean(vectors)
print('np_mean =', np_mean)

np_mean_by_col = np.mean(vectors, axis=0)
print('np_mean_by_col =', np_mean_by_col)

np_mean_by_row = np.mean(vectors, axis=1)
print('np_mean_by_row =', np_mean_by_row)

v = np.array([1, 2, 3])
scalar_mul = v * 3  # 3 * v
print('scalar multiplication =', scalar_mul)
scalar_div = 3 / v  # v / 3
print('scalar division =', scalar_div)

v = np.array([1, 2])
w = np.array([3, 4])
print('dot =', v.dot(w))

# numpy를 사용한 벡터의 크기
def norm(v):
    return math.sqrt(v.dot(v))

v = np.array([1, 1])
print('norm =', norm(v))

# numpy를 사용한 두 벡터 간의 거리
def dist(v, w):
    return norm(v - w)
    # return math.sqrt((v - w).dot(v - w))

v = np.array([1, 2])
w = np.array([3, 4])
print('distance =', dist(v, w))
