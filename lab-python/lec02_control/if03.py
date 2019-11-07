"""
가위(1)/바위(2)/보(3)
"""
import numpy as np


print('가위 바위 보 게임 ')
print('[1] 가위')
print('[2] 바위')
print('[3] 보')
print('---------------')
print('선택>>')

user = int(input())

computer = np.random.randint(1, 4)  # 1 <= com < 4 난수
print(computer)

if user == 1:  # user: 가위
    if computer == 1:  # computer: 가위
        print('Tie')
    elif computer == 2:  # computer: 바위
        print('Lose')
    else:  # computer: 보
        print('Win')
elif user == 2:  # user: 바위
    if computer == 1:  # computer: 가위
        print('Win')
    elif computer == 2:  # computer: 바위
        print('Tie')
    else:  # computer: 보
        print('Lose')
else:  # user: 보
    if computer == 1:  # computer: 가위
        print('Lose')
    elif computer == 2:  # computer: 바위
        print('Win')
    else:  # computer: 보
        print('Tie')


if user == computer:
    print('Tie')
elif user == 1:
    if computer == 2:  # 가위 vs 바위
        print('Lose')
    else:  # 가위 vs 보(computer == 3)
        print('Win')
elif user == 2:
    if computer == 1:  # 바위 vs 가위
        print('Win')
    else:  # 바위 vs 보(computer == 3)
        print('Lose')
else:  # user == 3
    if computer == 1:  # 보 vs 가위
        print('Lose')
    else:  # 보 vs 바위(computer == 2)
        print('Win')

result = user - computer
if result == 0:  # 비김
    print('Tie')
elif result == 1 or result == -2:  # user
    print('User Win!')
else:
    print('User Lose...')
