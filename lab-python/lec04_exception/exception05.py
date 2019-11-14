def user_input():
    """
    사용자에게 1, 2, 3 중 하나의 숫자를 입력하도록 안내.
    사용자가 입력한 숫자를 리턴.
    사용자가 숫자로 변환할 수 없는 문자를 입력하면, 안내문 출력 후 다시 입력받음.
    사용자가 1, 2, 3 이외의 숫자를 입력하면, 안내문 출력 후 다시 입력받음.

    :return: 1, 2, 3 중의 하나
    """
    while True:
        try:
            try:
                n = int(input('1, 2, 3 중에 숫자 하나를 입력>>'))
            except ValueError:
                raise ValueError('문자는 입력하면 안되요~')
            if n in (1, 2, 3):
                return n  # break
            else:
                raise ValueError('1, 2, 3만 가능')
        except ValueError as e:
            print(e.args)
    # return n


user = user_input()
print('입력 값 =', user)

