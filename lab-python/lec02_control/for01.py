"""
Python 반복문 - for 구문
for 변수 in Iterable:
    반복할 문장들

Iterable(반복 가능한 타입): list, tuple, set, dict, str, ...
"""

# range(to): 0부터 (to - 1)까지 범위의 숫자들
# range(from, to): from부터  (to - 1)까지 범위의 숫자들
# range(from, to, step): from부터 (to - 1)까지 step만큼씩 증가
for i in range(5):  # (0, 1, 2, 3, 4)
    print(i, end=' ')
print()

for i in range(1, 5):  # (1, 2, 3, 4)
    print(i, end=' ')
print()

for i in range(1, 5, 2):  # (1, 3)
    print(i, end=' ')
print()

for s in 'Hello, Python!':
    print(s, end=' ')
print()

languages = ['PL/SQL', 'R', 'Python', 'Java']
for lang in languages:
    print(lang, end=' ')
print()

for i in range(len(languages)):
    print(i, languages[i])

alphabets = {1: 'a', 2: 'b', 3: 'c'}
print(alphabets.keys())  # dict의 키(key)들
for key in alphabets.keys():
    print(key, alphabets[key])

# in dict는 딕셔너리의 key들을 반복!!!
for key in alphabets:
    print(key)

for item in alphabets.items():
    print(item)

# key, value = (1, 'a')
for key, value in alphabets.items():
    print(key, value)

