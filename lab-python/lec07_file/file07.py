"""
file.readline() 사용해서 csv 파일 읽기
"""
import os


def my_csv_reader(fn: str, header=True, encoding='utf-8') -> list:
    """
    csv 파일의 데이터를 2차원 행렬 형태로 리턴
    
    :param fn: 읽을 csv 파일 이름(예: data\\exam.csv)
    :param header: csv 파일의 헤더 존재 여부
    :param encoding: 파일 인코딩
    :return: csv 파일에서 헤더는 제외한 데이터들로 이루어진 2차원 리스트
    """
    data = []  # 빈 리스트를 만든다.
    with open(fn, mode='r', encoding=encoding) as f:
        if header:  # csv 파일에 컬럼 이름들이 헤더에 있으면
            f.readline()  # 한 줄을 읽고 지나간다.
        for line in f:  # 파일 끝까지 한 줄씩 읽어서
            # 읽은 줄의 앞/뒤 공백문자를 제거하고(strip),
            # 문자열을 쉼표(,)로 분리(split)해서 문자열 리스트를 만들고,
            # 리스트 data에 추가한다.
            data.append(line.strip().split(','))

    return data


def print_data(data: list) -> None:
    """
    2차원 리스트의 내용을 출력
    1 10 20 30 40
    2 11 21 31 41
    ...

    :param data: 2차원 행렬 형태의 리스트
    :return: None
    """
    for row in data:  # 2차원 리스트의 각 행들에 대해서 반복
        for x in row:  # 각 행(1차원 리스트)의 아이템들에 대해서 반복
            print(x, end=' ')  # 아이템을 같은 줄에 출력
        print()  # 한 줄 출력 후 줄바꿈


def get_sum_mean(data: list, col: int) -> tuple:
    """
    주어진 2차원 리스트(data)에서 해당 컬럼(col)의 데이터들의
    총합(sum)과 평균(mean)을 계산해서 리턴

    :param data: 2차원 행렬 형태의 리스트
    :param col: 컬럼 인덱스(0, 1, 2, ...)
    :return: 컬럼 데이터의 합과 평균
    """
    column_sum = 0
    for row in data:  # 2차원 리스트의 각 행들에 대해서 반복
        column_sum += int(row[col])  # 그 행의 col 위치에 있는 아이템을 int로 변환 후 더함.

    column_mean = column_sum / len(data)  # 평균 계산

    return column_sum, column_mean


if __name__ == '__main__':
    # 작성한 함수들을 테스트
    input_file = os.path.join('data', 'exam.csv')
    exam = my_csv_reader(input_file)
    print(exam)

    print_data(exam)

    sum_kor, mean_kor = get_sum_mean(exam, 1)
    print(sum_kor, mean_kor)
