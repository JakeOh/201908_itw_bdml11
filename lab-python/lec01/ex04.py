"""
연산자(operator)
- 할당(assignment): =
- 산술 연산: +, -, *, **, /, //, %
- 복합 할당 연산: +=, -=, *=, /=, ...
- 비교 연산: >, >=, <, <=, ==, !=
- 논리 연산: and, or, not
- identity 연산: is, is not(id() 함수의 리턴값이 같은 지 다른 지)
"""

x = 1  # 연산자 오른쪽의 값을 연산자 왼쪽의 변수에 저장(할당)
# 1 = x
print(2 ** 3)  # 2 x 2 x 2
print(10 / 3)
print(10 // 3)  # 정수 나눗셈 몫
print(10 % 3)  # 정수 나눗셈 나머지

x = 1
print('x =', x)

x += 10  # x = x + 10
print('x =', x)

print(1 == 2)
print(1 != 2)

x = 50
print(0 < x < 100)  # x > 0 and x < 100
print(x < 0 or x > 10)
# Alt+Enter (마법키)
