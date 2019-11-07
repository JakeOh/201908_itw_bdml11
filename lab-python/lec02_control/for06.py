"""
dictionary comprehension
"""

numbers = [1, 2, 3, 4, 5]
names = ['a', 'b', 'c', 'd', 'e']

students = {}  # empty dict
for i in range(len(numbers)):
    students[numbers[i]] = names[i]
print(students)

# dictionary comprehension
students2 = {numbers[i]: names[i]
             for i in range(len(numbers))}
print(students2)

# num_name = zip(numbers, names)
# print(num_name)
for x in zip(numbers, names):
    print(x)

students3 = {}
for key, value in zip(numbers, names):
    students3[key] = value
print(students3)

students4 = {k: v for k, v in zip(numbers, names)}
print(students4)

students5 = {k: v
             for k, v in zip(numbers, names)
             if k % 2 == 1}
print(students5)
