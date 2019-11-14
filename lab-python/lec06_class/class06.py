# from math import pi
import math

class Circle:
    # __init__(): 초기화(객체 생성) 함수
    def __init__(self, radius):
        self.radius = radius
        if self.radius < 0:
            raise ValueError('반지름은 0 또는 양수')

    # area(): 원의 넓이를 리턴
    def area(self):
        return math.pi * self.radius ** 2

    # perimeter(): 원의 둘레 길이를 리턴
    def perimeter(self):
        return 2 * math.pi * self.radius

    # __str__(): Circle(r=123) 형식
    def __str__(self):
        return f'Circle(r={self.radius})'

    # representation
    def __repr__(self):
        return f'원({self.radius})'

    # __eq__(): 반지름 같으면 equal(True)
    def __eq__(self, other):  # equal
        print('__eq__ 호출')
        return self.radius == other.radius

    def __ne__(self, other):  # not equal
        # != 연산자를 사용하면 자동으로 호출되는 메소드
        print('__ne__ 호출')
        return self.radius != other.radius

    def __gt__(self, other):
        # greater than: > 연산자를 사용하면 자동으로 호출되는 메소드
        print('__gt__ 호출')
        return self.radius > other.radius

    def __ge__(self, other):
        # greater than or equal to
        # >= 연산자를 사용하면 자동으로 호출되는 메소드
        return self.__gt__(other) or self.__eq__(other)
        # return self.radius >= other.radius

    # __lt__: less than (<)
    # __le__: less than or equal to (<=)


if __name__ == '__main__':
    c1 = Circle(1)
    print(c1)
    print('c1 area:', c1.area())
    print('c1 perimeter:', c1.perimeter())
    print('c1 id:', id(c1))  # 생성된 객체의 주소

    c2 = Circle(1)
    print('c2 id:', id(c2))

    print(c1 == c2)  # c1.__eq__(c2)
    print(c1 != c2)
    # __eq__ 메소드만 작성한 경우,
    # != 연산자는 __eq__ 메소드를 호출한 후 그 결과값의 반대(not)를 사용
    # __ne__ 메소드가 있는 경우,
    # != 연산자는 __ne__ 메소드의 리턴 값을 사용

    # print(c1 > c2)
    # print(c1 >= c2)
    # print(c1 < c2)
    # print(c1 <= c2)

    circles = [
        Circle(10),
        Circle(7),
        Circle(100),
        Circle(50),
        Circle(0),
        Circle(50)
    ]
    print(circles)
    print(sorted(circles))  # 오름차순 정렬
    print(sorted(circles, reverse=True))  # 내림차순 정렬
