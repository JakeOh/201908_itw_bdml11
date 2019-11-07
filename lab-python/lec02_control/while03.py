"""
반복문 연습
"""

# Shift+F6 (Shift+Fn+F6): refactor/rename
# 1 + 2 + 3 + ... + 100 = ?
n = 100
print(n * (n + 1) / 2)

numbers = [x for x in range(1, 101)]
print(sum(numbers))

total = 0
for x in range(1, 101):
    total += x
print(f'sum = {total}')

total, n = 0, 1
while n <= 100:
    total += n
    n += 1
print(f'total = {total}')

# 1 + 2 + 3 + ... + x <= 100
total = 0
x = 1
while True:
    total += x
    print(f'x = {x}, sum = {total}')
    if total > 100:
        break
    x += 1

total, x = 0, 1
while total <= 100:
    total += x
    print(f'x = {x}, sum = {total}')
    x += 1
