"""
사용자 정의 오류를 발생시키는 방법
"""

try:
    age = int(input('나이를 입력>>'))
    if age < 0:
        raise ValueError('나이는 0또는 양의 정수여야 합니다!')
    print('입력한 나이:', age)
except ValueError as e:
    print(e.args)
