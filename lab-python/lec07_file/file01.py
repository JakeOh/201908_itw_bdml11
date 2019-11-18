"""
os 모듈의 변수와 함수들
"""
import os

# CWD: Current Working Directory(현재 작업 디렉토리/폴더)
print(os.getcwd())

# 절대 경로(absolute path):
#   시스템의 루트(root)부터 전체 디렉토리 경로를 표시하는 방법
#   C:\dev\lab-python\lec07_file  (Windows)
#   /Users/user/Documents  (MacOS 또는 Linux)
# 상대 경로(relative path):
#   현재 작업 디렉토리(cwd)를 기준으로 경로를 표시하는 방법
#   .(현재 디렉토리), ..(상위 디렉토리)
#   ..\lec06_class\inheritance01.py
#   .\file01.py
#   file01.py

print(os.name)  # OS 종류 확인
if os.name == 'nt':  # Windows OS인 경우
    file_path = '.\\temp\\temp.txt'
else:  # Windows 이외의 OS인 경우
    file_path = './temp/temp.txt'
print(file_path)

# 파일 구분자(file separator)를 해당 OS에 맞게끔 경로를 만들어 줌
file_path = os.path.join('.', 'temp', 'temp.txt')
print(file_path)

print(os.path.isdir('.'))
print(os.path.isdir('file01.py'))  # .\\file01.py
print(os.path.isfile('.'))
print(os.path.isfile('file01.py'))

with os.scandir('.') as my_dir:
    for entry in my_dir:
        print(entry.name, '\t', entry.is_file())

# 파일(디렉토리) 이름 변경:
# os.rename(원본이름, 바꿀이름)
# 원본 파일(디렉토리)가 없는 경우에 에러 발생
try:
    os.rename('temp', 'test')
except FileNotFoundError:
    print('temp 폴더가 없음')

# 파일 삭제: os.remove(삭제할 파일 이름)
# 디렉토리 삭제: os.rmdir(삭제할 폴더 이름)
try:
    os.rmdir('test')
    print('test 폴더가 삭제됨')
except FileNotFoundError:
    print('삭제 권한 없음')

# 디렉토리 만들기:
# os.mkdir(디렉토리 이름)
# os.makedirs(디렉토리 이름)
try:
    os.mkdir('test2')
except FileExistsError:
    print('test2 폴더가 이미 있음')

# os.mkdir('test1\\temp')
# test1 폴더가 없기 때문 그 하위 폴더를 생성할 수 없음
# os.makedirs(폴더 이름) 함수는
# 중간 단계 폴더가 없어도 해당 경로에 필요한 모든 폴더들을 생성할 수 있음

try:
    # os.makedirs('test1\\temp')
    os.makedirs(os.path.join('test1', 'temp'))
    print('test1\\temp 폴더 생성 성공')
except FileExistsError:
    print('test1\\temp 폴더가 이미 있음 ')
