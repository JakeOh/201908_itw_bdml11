"""
try-except 구문 활용
"""

while True:
    try:
        n = int(input('정수 입력>>'))
        print('n =', n)
        break
    except ValueError:
        print('입력값은 정수여야 합니다!')

print('프로그램 정상 종료...')
