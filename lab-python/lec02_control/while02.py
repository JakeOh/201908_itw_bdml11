"""
while 문 연습
"""
while True:
    print('[1] 입력')
    print('[2] 수정')
    print('[3] 삭제')
    print('[0] 종료')
    print('----------')
    print('메뉴 선택>>')
    menu = input()
    if menu == '0':
        break

print('프로그램 종료')
