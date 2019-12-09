"""
csv 모듈을 사용한 mpg.csv 파일 읽기
"""
import csv
import os

file_path = os.path.join('..', 'scratch08', 'mpg.csv')
# Windows OS: ..\scratch08\mpg.csv
# Linux, macOS: ../scratch08/mpg.csv
with open(file_path, mode='r', encoding='UTF-8') as f:
    reader = csv.reader(f)
    reader.__next__()  # 한 줄 읽고 건너뜀.
    # 첫번째 줄은 컬럼 이름들이기 때문에.
    df = [line for line in reader]

print(df[0:5])  # 리스트 df에서 인덱스 0 ~ 4까지 행을 출력
#  리스트 df에서 0번째 행의 0, 1, 2번째 컬럼 아이템만 출력
print(df[0][0], df[0][1], df[0][2])

# 리스트에서 각 행마다 반복하면서,
# 각 행의 인덱스 2번 아이템을 숫자로 변환해서 새로운 리스트에 저장
displ = [float(row[2]) for row in df]
print(displ)

with open(file_path, mode='r', encoding='UTF-8') as f:
    # 사전(dict) 타입으로 데이터들을 읽어주는 reader 객체
    # 보통 csv 파일에 컬럼 이름이 포함된 경우 사용
    reader = csv.DictReader(f)
    # DictReader 객체의 read 기능을 사용하면,
    # 각 행은 '컬럼이름: 값'의 쌍으로 이루어진 dict(사전)가 됨.
    df = [row for row in reader]

print(df[0:5])
print(df[0])
# 각 행의 특정 원소를 찾아갈 때, 인덱스가 아닌 컬럼 이름으로 찾을 수 있다.
print(df[0]['manufacturer'])
print(df[0]['model'])
print(df[0]['displ'])
displ = [float(row['displ']) for row in df]
print(displ)





