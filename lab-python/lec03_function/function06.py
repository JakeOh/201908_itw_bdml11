"""
variable-length keyword argument:
함수 정의할 때 파라미터 이름 앞에 **를 사용
함수 내부에서는 dict처럼 취급함.
"""


def test(**kargs):
    print(kargs)
    for key in kargs:
        print(key, kargs[key])


test()
test(name='오쌤', age=16)
