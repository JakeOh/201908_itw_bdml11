"""
file open 모드(mode)
r: read, 읽기 모드
   읽기 모드는 파일이 없으면 FileNotFoundError가 발생
w: write, 쓰기 모드
   쓰기 모드는 파일이 없으면, 새로운 파일을 생성함
   파일이 있으면, 기존 파일을 열어 줌. 단, 기존 파일의 내용이 삭제됨.
   덮어쓰기(overwrite)
a: append, 추가 모드
   추가 모드는 파일이 없으면, 새로운 파일을 생성함.
   파일이 있으면, 기존 파일의 가장 마지막에 file pointer가 위치함.
   새로운 내용은 파일 끝에 추가(append)
"""

try:
    with open('NoFile.txt', mode='r') as f:
        pass
except FileNotFoundError:
    pass

with open('NewFile.txt', mode='w', encoding='utf-8') as f:
    f.write('test 테스트 ...')

with open('NewFile.txt', mode='w', encoding='utf-8') as f:
    pass  # 아무일도 하지 않더라도 파일의 내용이 삭제됨.

with open('Append.txt', mode='a', encoding='utf-8') as f:
    f.write('test\n')

with open('Append.txt', mode='a', encoding='utf-8') as f:
    f.write('추가...\n')
