"""
error, exception: 프로그램 실행 중에 발생할 수 있는 오류
프로그램 실행 중에 오류가 발생했을 때 해결 방법:
1) 오류가 발생한 코드 위치를 찾아서 오류가 발생하지 않도록 수정
2) 오류가 발생하더라도, 오류를 무시하고 프로그램이 실행될 수 있도록
프로그램을 작성 -> try 구문
"""

# prnit(1)  # NameError

# int(): 문자열 -> 정수
# float(): 문자열 -> 실수
# n = int('123.')  # ValueError

numbers = [1, 2, 3]
# print(numbers[3])  # IndexError

# print('123' + 456)  # TypeError

print(123 / 0)  # ZeroDivisionError
