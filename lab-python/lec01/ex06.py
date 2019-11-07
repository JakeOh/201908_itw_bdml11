"""
문자열(str) 타입
"""

s = '''\
    def my_function(x: int) -> int:
        return x + 1
'''
print(s)

s = '\thello\n\tpython'
print(s)

# 문자열의 인덱스, 자르기(slicing)
s = 'hello'
print(s[0])
print(s[1])
print(s[4])
# print(s[5])  # IndexError 발생
print(s[0:2])
# x:y - from x(포함, include) to y(미포함, exclude)
print(s[1:5])
print(s[1:])
# 범위 연산자에서 끝 인덱스가 없는 경우는 문자열의 끝까지
print(s[:3])
# 범위 연산자에서 시작 인덱스가 없는 경우는 문자열의 시작(0)부터

print(s[-3:-1])
