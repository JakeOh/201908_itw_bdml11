import numpy as np

# numpy.c_ (column bind)와 numpy_r (row bind)의 비교
a = np.array([1, 2, 3])
print(a, type(a), a.shape)
b = np.array([4, 5, 6])
print(b, type(b), b.shape)

c = np.c_[a, b]
print(c, type(c), c.shape)

d = np.r_[a, b]
print(d, type(d), d.shape)

e = np.array([[1, 2, 3],
              [4, 5, 6]])
f = np.array([[10, 20],
              [30, 40]])
print(np.c_[e, f])  # e와 f의 row 개수가 같아야 column 방향으로 붙일 수 있음.
# print(np.r_[e, f])  # e와 f의 column 개수가 다르면 row 방향으로 붙일 수 없음!

g = np.array([[100, 200, 300]])
# print(np.c_[e, g])  # row의 개수가 다르기 때문에 오른쪽으로 붙일 수 없음!
print(np.r_[e, g])  # column의 개수가 같아야 밑으로 붙일 수 있음.

# (2, 3) shape의 모든 원소가 1인 array를 생성해서 출력: A
print(np.ones((2, 3), dtype=np.int))

# (2, 3) shape의 모든 원소가 0인 array를 생성해서 출력: B
print(np.zeros((2, 3), dtype=np.int))

# (3, 2) shape의 원소는 1 ~ 6인 array를 생성해서 출력: C
print(np.arange(1, 7).reshape((3, 2)))

# (3, 2) shape의 난수들로 이루어진 array를 생성해서 출력: D
print(np.random.random((3, 2)))

"""다음과 같은 결과가 나올 수 있도록 
numpy를 사용하지 않고 add(x, y), subtract(), multiply(), divide(), dot() 함수를 구현
|1 2| + |5 6|= |6  8 | 
|3 4|   |7 8|  |10 12|

|1 2| - |5 6|= |-4 -4| 
|3 4|   |7 8|  |-4 -4|

|1 2| * |5 6|= |5  12| 
|3 4|   |7 8|  |21 32|

|1 2| / |5 6|= |0.2   0.333| 
|3 4|   |7 8|  |0.428 0.5  |

|1 2| @ |5 6|= |19 22| 
|3 4|   |7 8|  |43 50|
위의 결과와 같은 결과를 주는 numpy 코드를 작성
"""

"""
항등 행렬(Indentity matrix): 대각선의 원소는 1이고, 나머지 원소는 0인 정사각행렬
    A @ I = I @ A = A를 만족
역행렬(Inverse matrix): A @ A- = A- @ A = I를 만족하는 행렬
전치 행렬(Transpose matrix): 행렬의 row와 column을 서로 바꾼 행렬
"""


def my_dot(A, B):
    """두 행렬 A와 B의 dot 연산 결과를 리턴
        dot_ik = sum(j)[a_ij * b_jk]
    """
    print('A shape:', A.shape)
    print('B shape:', B.shape)
    if A.shape[1] != B.shape[0]:
        raise ValueError('A의 column과 B의 row 개수는 같아야 함!')
    n_row = A.shape[0]  # dot 결과 행렬의 row 개수
    n_col = B.shape[1]  # dot 결과 행렬의 column 개수
    temp = A.shape[1]  # 각 원소들끼리 곱한 결과를 더하는 회수
    numbers = []  # 결과값들을 저장할 리스트
    # 결과를 저장할 (n_row, n_col) 모양의 행렬
    result = np.zeros((n_row, n_col))
    for i in range(n_row):  # A 행렬의 row 개수만큼 반복
        for k in range(n_col):  # B 행렬의 column 개수만큼 반복
            n = 0
            for j in range(temp):
                # dot 결과 행렬의 [i,k]번째 원소의 값을 계산
                n += A[i, j] * B[j, k]
            numbers.append(n)  # [i,j]번째 원소를 리스트에 추가
            result[i, k] = n
    # 결과를 (n_row, n_col) 모양의 행렬로 변환해서 리턴
    return np.array(numbers).reshape(n_row, n_col)


A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])
print(my_dot(A, B))
print(A @ B)  # A.dot(B)

print(my_dot(B, A))
print(B @ A)  # B.dot(A)


def dot_1d(X, Y):
    if len(X) != len(Y):
        raise ValueError('len(X) != len(Y)')
    result = 0
    for i in range(len(X)):
        result += X[i] * Y[i]
    return result


def dot_2d(X, Y):
    n_row = X.shape[0]
    n_col = Y.shape[1]
    if X.shape[1] != Y.shape[0]:
        raise ValueError('X.shape[1] != Y.shape[0]')
    result = np.zeros((n_row, n_col))
    for i, row in enumerate(X):
        for j, col in enumerate(zip(*Y)):
            print(f'i={i}, row={row}, j={j}, col={col}')
            result[i, j] = dot_1d(row, col)
    return result


print(dot_2d(A, B))
print(dot_2d(B, A))

