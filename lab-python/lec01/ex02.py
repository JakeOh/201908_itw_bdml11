"""
여러가지 print() 방법
"""

print('Hello, python!')

age = 16
name = '오쌤'
print('나이:', age, ', 이름:', name)
print(f'나이: {age}, 이름: {name}')  # formatted string
print('나이: {}, 이름: {}'.format(age, name))
print('나이: %d, 이름: %s' % (age, name))
# %d: 정수, %f: 실수, %s: 문자열
# digit, floating-point number, string

"""
사용자 입력(키보드 입력) 처리
"""
print('>>> 이름 입력:')
name = input()
print(f'name: {name}')

age = input('>>> 나이 입력:')
print(f'age: {age}')
# print(age + 1)  # 실행 중 오류(TypeError) 발생
# Ctrl+/ : 주석 토글
