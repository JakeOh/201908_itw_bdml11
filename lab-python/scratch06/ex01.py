"""
scratch06\ex01.py
확률

사건 공간(universe of events)
사건(event)
확률(probability)
"""
import random
from collections import Counter

coin = ['H', 'T']
dice = [1, 2, 3, 4, 5, 6]
trials = 10_000

# 동전 1개를 10,000번 던지는 실험
# 앞면(H)이 나올 확률과 뒷면(T)이 나올 확률이 1/2임을 증명
heads, tails = 0, 0  # 앞면과 뒷면이 나오는 회수를 저장할 변수
for _ in range(trials):  # 10,000번 반복
    random_coin = random.choice(coin)  # 동전 던진 결과(앞 또는 뒤)
    if random_coin == 'H':
        heads += 1
    else:
        tails += 1
p_h = heads / trials  # 앞면이 나올 확률 = 앞면 회수 / 전체 회수
p_t = tails / trials  # 뒷면이 나올 확률
print('P(H) =', p_h)
print('P(T) =', p_t)

# 동전 2개를 던지는 실험(10,000번)
# 1) 앞면의 개수가 1개일 확률 1/2 (HT, TH)
# 2) 첫번째 동전이 앞면일 확률 1/2 (HH, HT)
# 3) 적어도 한 개의 동전이 앞면일 확률 3/4 (HH, HT, TH)
# = 1 - 두 개 동전 모두 뒷면일 확률

# 동전 3개를 던지는 실험(10,000번)
# 앞면의 개수가 1개일 확률 3/8 (HTT, THT, TTH)


def experiment(type, n, t):
    """

    :param type: 실험 타입(동전 던지기 or 주사위 던지기, ...)
    :param n: 동전/주사위의 개수
    :param t: 실험 회수
    :return: 리스트
    """
    cases = []  # 동전 던지기 실험 결과를 저장
    for _ in range(t):  # 실험 회수만큼 반복
        case = []  # 각 실험의 결과를 저장
        for _ in range(n):  # 동전 개수만큼 반복
            rand = random.choice(type)  # 'H' or 'T'
            case.append(rand)  # 1회 실험 결과에 저장
        # 1회 실험이 끝날 때마다 각 결과를 tuple로 변환 후 저장
        # Counter 클래스는 tuple의 개수는 셀 수 있지만,
        # list의 개수는 셀 수 없음!
        cases.append(tuple(case))
    return cases


coin_exp = experiment(coin, 2, 10_000)
print(coin_exp[0:10])  # 첫 10개의 실험 결과 확인

# 동전 던지기 실험 경우의 수
coin_event_counts = Counter(coin_exp)
print(coin_event_counts)


def how_many_heads(x):
    counter = Counter(x)
    print(counter)
    return counter['H']

num_of_cases = 0
for ev, cnt in coin_event_counts.items():
    if how_many_heads(ev) == 1:
        num_of_cases += cnt
p_h1 = num_of_cases / trials
print('P(앞면이 1개일 확률) =', p_h1)

num_of_cases = 0
for ev, cnt in coin_event_counts.items():
    # if ev == ('H', 'H') or ev == ('H', 'T'):
    if ev[0] == 'H':
        num_of_cases += cnt
p_first_h = num_of_cases / trials
print('P(첫번째 동전이 앞면) =', p_first_h)

num_of_cases = 0
for ev, cnt in coin_event_counts.items():
    if how_many_heads(ev) == 1 or how_many_heads(ev) == 2:
        num_of_cases += cnt
p = num_of_cases / trials
print('P(적어도 1개가 앞면) =', p)

# 여사건 이용
num_of_cases = 0
for ev, cnt in coin_event_counts.items():
    if how_many_heads(ev) == 0:
        num_of_cases += cnt
p = num_of_cases / trials
print('P(적어도 1개가 앞면) =', 1 - p)

# H = 1, T = 0 약속 -> coin = [1, 0]
coin2 = [1, 0]
cases = []
for _ in range(trials):
    num_of_heads = 0
    for _ in range(2):
        num_of_heads += random.choice(coin2)
    cases.append(num_of_heads)
print(cases[0:10])
coin_event_counts = Counter(cases)
print('P(H=0) =', coin_event_counts[0] / trials)
print('P(H=1) =', coin_event_counts[1] / trials)
print('P(H=2) =', coin_event_counts[2] / trials)

# 주사위 2개를 던졌을 때, 두 눈의 합이 8일 확률
#   (2, 6), (3, 5), (4, 4), (5, 3), (6, 2) => 5/36
# 주사위 2개를 던졌을 때, 적어도 하나가 짝수가 나올 확률
#   (1, 1), (1, 3), (1, 5), (3, 1), (3, 3), (3, 5),
#   (5, 1), (5, 3), (5, 5)를 제외한 모든 경우 => 27/36
dice_exp = experiment(dice, 2, 10_000)
print(dice_exp[0:5])

event_counts = Counter(dice_exp)
print(len(event_counts))
print(event_counts)
num_of_cases = 0
for ev, cnt in event_counts.items():
    # if ev[0] + ev[1] == 8:
    if sum(ev) == 8:
        num_of_cases += cnt
p = num_of_cases / trials
print('P(두 눈의 합=8) =', p, 5/36)

num_of_cases = 0
for ev, cnt in event_counts.items():
    if ev[0] % 2 == 0 or ev[1] % 2 == 0:
        num_of_cases += cnt
p = num_of_cases / trials
print('P(적어도 하나가 짝수) =', p, 27/36)
