"""
가변 길이 인수(variable-length argument):
함수를 호출할 때 전달하는 argument의 갯수가 다양하게 변하는 것.
파라미터 이름 앞에 *를 붙임.
"""
print('a')
print('a', 'b', 'c', 'd', sep=':')


def fn_vararg(*varargs):
    print(varargs)
    print(*varargs)
    # 가변길이 인수들은 tuple처럼 취급하면 됨
    for arg in varargs:
        print(arg)


fn_vararg(1, 2, 3)


def summation(*args):
    """
    임의의 갯수의 숫자들을 전달받아서 그 숫자들의 총합을 리턴하는 함수

    :param args: 합계를 계산할 숫자들(갯수 제한 없음)
    :return: 숫자들의 합
    """
    total = 0
    for arg in args:
        total += arg
    return total


print(summation())
print(summation(1))
print(summation(1, 3, 5, 7, 9))


def fn_vararg2(a, *b):
    print(f'a = {a}')
    print(f'b = {b}')


# fn_vararg2()  # a에 값을 전달하지 않으면 에러 발생
fn_vararg2(1)  # b는 가변길이 파라미터이므로 인수를 전달하지 않아도 됨.


def fn_varargs3(*a, b):
    print(f'a = {a}')
    print(f'b = {b}')


# fn_varargs3()
# fn_varargs3(1)
# fn_varargs3(1, 2)
fn_varargs3(1, b=2)
# 가변길이 파라미터 뒤에 선언된 파라미터에 값을 전달할 때는
# keyword 방식으로만 값(argument)를 전달할 수 있음


def calculator(*values, operator):
    """
    operator가 '+'인 경우에는 values들의 합계를 리턴하고,
    operator가 '*'인 경우에는 values들의 곱을 리턴하는 함수
    operator가 '+', '*'가 아니면 None을 리턴

    :param values:
    :param operator:
    :return:
    """
    result = None
    if operator == '+':
        result = 0
        for x in values:
            result += x  # result = result + x
        # return result
    elif operator == '*':
        result = 1
        for x in values:
            result *= x  # result = result * x
        # return result

    return result


result = calculator(1, 2, 3, 4, 5, operator='+')
print(result)
result = calculator(1, 2, 3, 4, 5, operator='*')
print(result)
result = calculator(1, 2, 3, 4, 5, operator='-')
print(result)


