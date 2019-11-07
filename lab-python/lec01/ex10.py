"""
set(집합): 저장하는 순서가 중요하지 않고,
같은 값이 중복 저장되지 않는 데이터 타입
"""

s1 = {1, 2, 3, 3, 2, 1}
print(s1)
s2 = {4, 3, 2}
print(s2)

# 집합 연산: 합집합, 교집합, 차집합
print(s1 | s2)  # 합집합(|)
print(s1 & s2)  # 교집합(&)
print(s1 - s2)  # 차집합(-)

# 집합에 원소 추가/삭제
s1.add(100)  # add(value)
print(s1)
s1.remove(3)  # remove(value)
print(s1)
