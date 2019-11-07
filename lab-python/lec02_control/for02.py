import numpy as np
from math import sqrt

# 빈 리스트(scores)를 선언
scores = []

# 난수(0 <= x <= 100) 10개를 리스트에 저장
for _ in range(10):
    # x = np.random.randint(0, 101)
    scores.append(np.random.randint(0, 101))
print(scores)

# 리스트에 저장된 시험 점수 10개의 총점을 계산, 출력
total = 0
for score in scores:
    total += score
print(f'총점 = {total}')
print(f'총점2 = {sum(scores)}')

# 리스트에 저장된 시험 점수 10개의 평균을 계산, 출력
avg = total / len(scores)
print(f'평균 = {avg}')
print('평균2 =', np.mean(scores))

# 표준 편차 계산: from math import sqrt
sum_of_squares = 0
for score in scores:
    sum_of_squares += (score - avg) ** 2
standard_deviation = sqrt(sum_of_squares / len(scores))
print(f'표준편차 = {standard_deviation}')
print('표준편차2 =', np.std(scores))

# 리스트에 저장된 시험 점수 10개 중에서 최댓값, 최솟값을 찾아서 출력
print('최댓값 =', max(scores))
print('최솟값 =', min(scores))

max_score = scores[0]
min_score = scores[0]
for score in scores:
    if score > max_score:
        # 리스트에서 현재 최댓값보다 더 큰 수를 찾은 경우
        max_score = score
    if score < min_score:
        # 리스트에서 현재 최솟값보다 더 작은 수를 찾은 경우
        min_score = score
print(max_score, min_score)

sorted_scores = sorted(scores)
print(sorted_scores)
print(scores)
