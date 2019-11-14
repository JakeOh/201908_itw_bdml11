"""
선택 정렬 알고리즘
"""
import numpy as np


def find_max(numbers: list) -> tuple:
    max_id, max_value = 0, numbers[0]
    for i, v in enumerate(numbers):
        if v > max_value:
            max_id, max_value = i, v
    return max_id, max_value


def find_min(numbers):
    """
    주어진 리스트에서 최솟값과 최솟값의 인덱스를 찾아서 리턴

    :param numbers: 숫자들의 리스트
    :return: (최솟값의 인덱스, 최솟값)의 쌍(tuple)
    """
    min_id, min_value = 0, numbers[0]
    for i, v in enumerate(numbers):
        if v < min_value:
            min_id, min_value = i, v
    return min_id, min_value


def sel_sort(numbers: list, reverse: bool = False) -> list:
    """
    주어진 리스트의 원소들이 정렬된 새로운 리스트를 생성해서 리턴
    주어진 리스트(파라미터에 전달된 리스트)의 순서는 바뀌지 않음

    :param numbers:
    :param reverse: False이면 오름차순, True이면 내림차순 정렬.
    기본값은 False(즉, 오름차순 정렬)
    :return:
    """
    numbers_copy = numbers.copy()
    result = []  # 빈 리스트 생성
    while numbers_copy:  # numbers의 원소가 있는 동안에
        # print('numbers=', numbers_copy)
        # print('result=', result)
        if reverse:  # True: 내림차순 정렬
            _, found = find_max(numbers_copy)  # 최댓값 찾음
        else:  # False: 오름차순 정렬
            _, found = find_min(numbers_copy)  # 최솟값 찾음
        result.append(found)  # 결과 리스트에 추가
        numbers_copy.remove(found)  # 카피본에서 최대/최소 값 삭제
    return result


numbers = [np.random.randint(0, 100) for _ in range(10)]
print(numbers)
sorted_numbers = sel_sort(numbers)  # 오름차순 정렬
print('ascending =', sorted_numbers)
sorted_numbers = sel_sort(numbers, reverse=True)  # 내림차순 정렬
print('descending =', sorted_numbers)


def sel_sort2(numbers: list, reverse: bool = False) -> None:
    """
    주어진 리스트를 정렬하는 함수
    새로운 리스트를 생성하지 않고, 원본 리스트의 순서를 바꿈

    :param numbers:
    :param reverse: False이면 오름차순, True이면 내림차순 정렬.
    기본값은 False(오름차순 정렬)
    :return:
    """
    length = len(numbers)
    for i in range(0, length - 1):
        # i: 최댓값/최솟값을 옮길 위치
        for j in range(i + 1, length):
            # j: 최댓값/최솟값을 찾기 위해서 비교할 원소들의 인덱스
            if reverse:  # True(내림차순)
                if numbers[i] < numbers[j]:
                    # 리스트 맨 앞에 있는 값보다 더 큰 값을 찾았다면
                    # 서로 위치를 바꿔줌(큰 값을 앞으로 이동)
                    numbers[i], numbers[j] = numbers[j], numbers[i]
            else:  # False(오름차순)
                if numbers[i] > numbers[j]:
                    # 리스트의 맨 앞에 있는 값보다 더 작은 값을 찾았다면
                    # 서로 위치를 바꿔줌(작은 값을 앞으로 이동)
                    numbers[i], numbers[j] = numbers[j], numbers[i]
            print(numbers)


print('==============================')
numbers = [np.random.randint(0, 100) for _ in range(5)]
print(numbers)
sel_sort2(numbers)
print(numbers)
print('==============================')
numbers = [np.random.randint(0, 100) for _ in range(5)]
print(numbers)
sel_sort2(numbers, reverse=True)
print(numbers)
