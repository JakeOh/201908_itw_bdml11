"""
함수(function): 기능을 수행해서 값을 반환(return)하는 코드 블록
인수(argument): 함수를 호출할 때 전달하는 값
매개변수(parameter): argument를 저장하기 위해서,
함수를 정의할 때 선언하는 변수
"""

result = print('Hello, Python!')  # 함수 호출(call, invoke)
print(result)  # argument 1개
print()  # argument 0개
print('hello', 'python', 123)  # argument 3개
print('hello', end=',')
print('python')

# Python 내장(built-in) 함수
# Ctrl+Q(맥: Ctrl+J): 함수/클래스 문서(documentation) 보기
result = sum([1, 2, 3, 4, 5])
# result: 함수 sum의 리턴 값(반환 값)
print(result)

result = abs(-5)
print(result)

result = pow(2, 4)  # 2 ** 4
print(result)

result = pow(2, 4, 3)  # 2 ** 4 % 3
print(result)
