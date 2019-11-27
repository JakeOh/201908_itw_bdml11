"""
조건부 확률(conditional probability)
P(A): 사건 A가 일어날 확률
P(B): 사건 B가 일어날 확률
P(A,B): 사건 A와 사건 B가 동시에 일어날 확률(A와 B의 교집합 확률)
P(A|B): 사건 B가 일어났을 때, 사건 A가 일어날 (조건부) 확률
    P(A|B) = P(A,B) / P(B)
    만약 A와 B가 독립 사건이면, P(A,B) = P(A)P(B)
    P(A|B) = P(A)P(B) / P(B) = P(A)

조건부 확률 예시: 자녀가 2명 있는 가정
A: 첫째가 딸인 경우. P(A) = 2/4 = 1/2
B: 두 자녀가 모두 딸인 경우. P(B) = 1/4
C: 두 자녀 중 최소 한 명이 딸인 경우. P(C) = 3/4
P(A,B) = 1/4 = P(B,A)
P(B,C) = 1/4 = P(C,B)
P(B|A) = P(첫째가 딸일 때, 두 자녀 모두 딸) = 1/2
       = P(B,A)/P(A) = (1/4)) / (2/4) = 1/2
P(B|C) = P(적어도 한 명이 딸일 때, 두 자녀 모두 딸) = 1/3
       = P(B,C)/P(C) = (1/4) / (3/4) = 1/3
"""
import random

kid = ('boy', 'girl')
trials = 10_000
older_girl = 0  # 첫째가 딸인 경우
both_girl = 0  # 두 자녀 모두 딸인 경우
either_girl = 0  # 두 자녀 중 적어도 한 명이 딸인 경우
for _ in range(trials):
    older = random.choice(kid)
    younger = random.choice(kid)
    if older == 'girl':
        older_girl += 1
    if older == 'girl' and younger == 'girl':
        both_girl += 1
    if older == 'girl' or younger == 'girl':
        either_girl += 1

# 첫째가 딸일 때, 두 자녀 모두 딸일 확률
p1 = both_girl / older_girl
print('P(첫째가 딸일 때, 두 자녀 모두 딸) =', p1)

# 적어도 한 명이 딸일 때, 두 자녀 모두 딸일 확률
p2 = both_girl / either_girl
print('P(적어도 한명이 딸일 때, 두 자녀 모두 딸) =', p2)
