"""
for line in file:
    실행문
file에서 readline()을 호출한 결과를 line 변수에 저장
line이 빈 문자열('')이 아닐 때 실행문 실행
line이 빈 문자열이면 for 루프 종료
"""

with open('test.txt', mode='r', encoding='utf-8') as f:
    for line in f:
        print(line.strip())
