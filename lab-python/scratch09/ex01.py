import csv

# 문자열(string)을 아이템으로 갖는 리스트
row1 = ['test1', 'success', 'Mon']
row2 = ['test2', 'failure, kind of', 'Tue']
row3 = ['test3', 'success, kind of', 'Wed']
result = [row1, row2, row3]
print(result)

# 파일을 쓰기 모드로 열기
# csv 파일을 쓸(write) 때는 불필요한 라인이 써지지 않도록 하기 위해서
# 파일을 오픈할 때 newline='' 파라미터를 추가!
with open('test_result.csv', mode='w',
          encoding='UTF-8', newline='') as f:
    # csv writer 객체 생성
    writer = csv.writer(f, delimiter=',')
    for row in result:
        # writer 객체의 writerow() 메소드를 사용해서 한 줄씩 쓰기
        writer.writerow(row)

print('\ncsv 모듈을 사용하지 않을 때')
# csv 모듈을 사용하지 않고 CSV 파일을 읽었을 때 문제점
with open('test_result.csv', mode='r', encoding='UTF-8') as f:
    for line in f:
        print(line.strip().split(','))
        # 'failure, kind of'라는 하나의 문자열이
        # '"failure'와 ' kind of"'라는 두개의 문자열로 쪼개짐.
        # 원래 데이터에 없어야 할 "가 문자열에 포함됨.

print('\ncsv 모듈을 사용할 때')
with open('test_result.csv', mode='r', encoding='UTF-8') as f:
    # csv reader 객체를 생성
    reader = csv.reader(f)
    for row in reader:
        # reader 객체의 read 기능을 이용해서 한줄씩 읽음
        print(row)


