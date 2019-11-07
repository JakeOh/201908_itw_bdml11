"""
dict: key-value의 쌍으로 이루어진 데이터들을 저장하는
사전(dictionary)식 데이터 타입
"""

person = {'name': '오쌤', 'age': 16, 'height': 170.5}
print(person)
print(type(person))

# dict의 데이터 참조 - key를 사용
print(person['name'])
print(person['age'])

print(person.keys())  # dict의 key를 알아낼 때
print(person.values())  # dict의 value들만 알아낼 때
print(person.items())  # (key, value)를 알아낼 때

students = {1: '강다혜', 2: '김수인', 3: '김영광', 10: '안도연'}
print(students[1])

# dict에 값을 추가
students[4] = '김재성'
print(students)

# dict의 값을 변경
students[4] = 'gildong'
print(students)

# dict의 값을 삭제 - pop(key) 메소드 사용
students.pop(4)
print(students)

book = {
    'title': '파이썬 프로그래밍 교과서',
    'authors': ['제니퍼', '폴', '제이슨'],
    'company': '길벗',
    'isbn': 97911
}
print(book['authors'])
print(book['authors'][0])
