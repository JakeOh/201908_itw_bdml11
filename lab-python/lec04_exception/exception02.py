"""
try 구문:
try:
    실행할 문장들
except 에러종류 [as 별명]:
    에러가 발생했을 때 실행할 문장들
[except 다른에러: 코드]
[else:
    에러 없이 try 블록 안의 모든 문장이 정상적으로 실행됐을 때
    실행할 코드들]
[finally:
    에러 발생 여부와 상관없이 반드시 실행할 문장들]
"""

try:
    numbers = [1, 2, 3]
    for i in range(1, 4):
        print(i, ':', numbers[i])
except ValueError:
    print('값 에러 처리')
except IndexError:
    print('인덱스 에러 처리')
else:
    print('try의 모든 내용을 정상적으로 실행')
finally:
    print('finally 블록')





