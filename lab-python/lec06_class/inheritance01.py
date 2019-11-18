"""
상속(inheritance):
부모 클래스로부터 데이터(field)와 기능(method)를 물려받아서
자식 클래스에서 사용할 수 있도록 하는 개념
- parent(부모), super(상위), base(기본) class
- child(자식), sub(하위), derived(유도) class
"""


class Shape:
    def __init__(self, x=0, y=0):
        print('Shape.__init__ 호출')
        self.x = x
        self.y = y

    def __repr__(self):
        return f'Shape(x={self.x}, y={self.y})'

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def area(self):
        """
        Shape 객체는 넓이를 계산할 수 없고,
        Shape의 sub 타입들인 Rectangle, Circle 객체가
        각자의 방식으로 넓이를 계산해야 됨.

        :return: 도형의 넓이
        """
        raise NotImplementedError('area 메소드는 반드시 override')

    def draw(self):
        """
        넓이를 계산하는 area 메소드를 사용해서 도형 내부를
        그려주는 메소드

        :return: None
        """
        print(f'Drawing area {self.area()} ...')


# 상속:
# class Child(Parent):
#   body
class Rectangle(Shape):
    # Child 클래스에서 __init__ 메소드를 작성하지 않은 경우에는
    # 파이썬 인터프리터가 Parent 클래스의  __init__ 메소드를
    # 호출해서 부모 객체를 자동으로 생성함.
    # 개발자 child 클래스에서 __init__ 메소드를 정의한 경우에는
    # 파이썬 인터프리터가 parent 클래스의 __init_ 메소드를
    # 자동으로 호출하지 않음!
    # child 클래스에서 parent 클래스의 __init__ 메소드를 명시적으로
    # 호출해야 함!
    def __init__(self, w=0, h=0, x=0, y=0):
        print('Rectangle.__init__ 호출')
        super().__init__(x, y)  # 부모 클래스의 __init_ 호출
        self.w = w
        self.h = h

    # override: 부모 클래스로부터 상속받은 메소드를
    # 자식 클래스에서 재정의하는 것.
    def __repr__(self):
        return f'사각형(가로={self.w}, 세로={self.h}, x={self.x}, y={self.y})'

    def area(self):
        return self.w * self.h


class Circle(Shape):
    def __init__(self, r=0, x=0, y=0):
        print('Circle.__init__ 호출')
        # super 클래스의 __init__ 메소드를 반드시(!) 호출해야 함.
        # super().__init__(x, y)
        Shape.__init__(self, x, y)  # self를 생략 불가!
        # sub 클래스만 갖는 field를 초기화
        self.r = r

    def __repr__(self):
        return f'동그라미(반지름={self.r}, x={self.x}, y={self.y})'

    def area(self):
        return 3.14159 * self.r ** 2


if __name__ == '__main__':
    shape1 = Shape()
    print(shape1)
    shape1.move(1, 2)
    print(shape1)

    rect1 = Rectangle(w=3, h=4, x=0, y=0)
    print('rect1 타입:', type(rect1))
    print('rect1:', rect1)  # override한 __repr__ 메소드가 호출됨.
    rect1.move(-1, -2)  # 부모에게서 상속받은 move 메소드가 호출됨.
    print(rect1)

    circle1 = Circle(r=10, x=0, y=0)
    print('circle1 type:', type(circle1))
    print(circle1)

    rect1.draw()
    circle1.draw()
