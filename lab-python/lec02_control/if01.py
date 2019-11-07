"""
python if 구문(statement)
if 조건식:
    조건식이 참일 때 실행할 문장

if 조건식:
    참일 때 실행할 문장
else:
    거짓일 때  실행할 문장

if 조건식1:
    조건식1이 참일 때 실행할 문장
elif 조건식2:
    조건식2이 참일 때 실행할 문장
...
else:
    모든 조건식들이 거짓일 때 실행할 문장
"""

# 숫자를 입력받아서 양수인 경우에만 출력
num = int(input('>>> 정수 입력:'))
if num > 0:
    print(f'num = {num}')

if num > 0:
    print('양수')
else:
    print('0 또는 음수')

# if-elif-else
score = 85
if score >= 90:
    print('A')
elif score >= 80:
    print('B')
elif score >= 70:
    print('C')
else:
    print('F')

# if, elif, else 블록 안에서 또다른 if 구문을 사용할 수도 있음.
if num % 2 == 0:  # 짝수이면
    if num % 4 == 0:
        print('4의 배수')
    else:
        print('4의 배수가 아닌 짝수')
else:  # 홀수이면
    print('홀수')

print('프로그램 종료')

