"""
함수 정의(definition)/선언(declaration)
def 함수이름(파라미터 선언, ...):
    함수의 기능 작성
    [return 값]
"""


# 함수 정의(선언)
def say_hello():
    """
    '안녕하세요'를 출력하는 함수

    :return: None
    """
    print('안녕하세요')


# 함수는 호출해야만 실행됨
say_hello()  # 함수 호출(call, invoke)


def print_msg(msg):
    """
    인수(argument) msg를 화면에 출력하는 함수

    :param msg: 출력할 메시지(str 타입)
    :return: None
    """
    print(msg)


print_msg('hello')


def add(x, y):
    """
    숫자 2개를 전달받아서 그 숫자들의 합을 리턴하는 함수

    :param x: int
    :param y: int
    :return: x + y를 리턴(int)
    """
    return x + y


result = add(10, 20)
print(f'add 결과 = {result}')


def sum_and_product(x, y):
    """
    두 수 x와 y의 합(summation)과 곱(product)을 리턴하는 함수

    :param x: int
    :param y: int
    :return: x+y, x*y 순서로 리턴
    """
    return x + y, x * y


sum, product = sum_and_product(11, 22)
print(f'sum = {sum}, product = {product}')

result = sum_and_product(11, 22)
print(result, result[0], result[1])


def make_person(name, age):
    """
    이름(name), 나이(age)를 전달받아서 dict 타입을 리턴하는 함수

    :param name: 이름(str)
    :param age: 나이(int)
    :return: {'name': name, 'age': age}
    """
    return {'name': name, 'age': age}


person = make_person('오쌤', 16)
print(person)
