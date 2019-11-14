"""
재귀 함수(recursive function)
"""


# factorial
# 0! = 1
# n! = 1 x 2 x 3 x ... x (n-1) x n = (n-1)! x n
def factorial1(n: int) -> int:
    result = 1
    for x in range(1, n + 1):
        result *= x  # result = result * n
    return result


def factorial2(n: int) -> int:
    if n == 0:
        return 1
    elif n > 0:
        return factorial2(n-1) * n


for x in range(6):
    print(f'{x}! = {factorial2(x)}')


# 재귀 함수를 사용한 하노이 탑
def hanoi_tower(n, start, target, aux):
    """
    재귀 함수를 사용해서 하노이 탑 문제 해결 방법 출력

    :param n: 옮길 원반 갯수(양의 정수)
    :param start: 원반들이 있는 시작 기둥 번호
    :param target: 원반들을 모두 옮겨 놓을 타겟 기둥 번호
    :param aux: 보조 기둥으로 사용할 기둥 번호
    :return: None
    """
    if n == 1:
        print(f'{start} -> {target}')
        return  # 함수 종료

    # (n-1)개의 원반을 target을 보조 기둥으로 사용해서
    # aux 기둥으로 모두 옮김
    hanoi_tower(n - 1, start, aux, target)
    # 시작 기둥에 남아 있는 한개의 원반을 목표 기둥으로 옮김
    print(f'{start} -> {target}')
    # aux 기둥에 남아 있는 (n-1)개의 원반을
    # start 기둥을 보조 기둥으로 사용해서 target으로 옮김
    hanoi_tower(n - 1, aux, target, start)


for n in range(1, 5):
    print('하노이 탑 n=', n)
    hanoi_tower(n, start=1, target=3, aux=2)
    print('================================')

