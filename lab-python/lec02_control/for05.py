"""
list comprehension
"""
import numpy as np

numbers = [1, 2, 3, 4, 5]
print(numbers)

numbers2 = []
for n in range(1, 6):
    numbers2.append(n)
print(numbers2)

numbers3 = [n for n in range(1, 6)]
print(numbers3)

# 2, 4, 6, 8, 10
even = [2 * n for n in range(1, 6)]
print(even)

# 1, 4, 9, 16, 25
squares = [n ** 2 for n in range(1, 6)]
print(squares)

# 난수 10개 배열
randoms = [np.random.randint(0, 101) for _ in range(10)]
print(randoms)

even2 = []
for n in range(1, 11):
    if n % 2 == 0:
        even2.append(n)
print(even2)

even3 = [n for n in range(1, 11) if n % 2 == 0]
print(even3)

# 주사위 2개를 던졌을 때 경우의 수
# (1, 1), (1, 2), (1, 3), ..., (1, 6)
# (2, 1), (2, 2), (2, 3), ..., (2, 6)
# ...
# (6, 1), (6, 2), (6, 3), ..., (6, 6)
dice1 = []  # 빈 리스트 생성
for x in range(1, 7):
    for y in range(1, 7):
        dice1.append((x, y))  # (x,y) tuple을 리스트에 추가
print(dice1)

dice2 = [(x, y) for x in range(1, 7) for y in range(1, 7)]
print(dice2)

# (1, 1)
# (2, 1), (2, 2)
# (3, 1), (3, 2), (3, 3)
# ...
# (x, y) if x >= y
dice3 = []
for x in range(1, 7):
    for y in range(1, 7):
        if x >= y:
            dice3.append((x, y))
print(dice3)

dice4 = [(x, y)
         for x in range(1, 7)
         for y in range(1, 7)
         if x >= y]
print(dice4)

# 시험점수(0 ~ 100) 10개를 가지고 있는 리스트
scores = [np.random.randint(0, 101) for _ in range(10)]
print(scores)
# 평균보다 높은 점수들의 리스트
mean = np.mean(scores)
above_mean = [s for s in scores if s > mean]
# above_mean = []
# for s in scores:
#     if s > mean:
#         above_mean.append(s)
print(above_mean)



