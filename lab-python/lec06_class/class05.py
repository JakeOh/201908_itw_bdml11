class Rectangle:
    """ 직사각형 클래스 """
    def __init__(self, width=0, height=0):
        self.width = width
        self.height = height

    def info(self):
        print(f'Rectangle(w={self.width}, h={self.height})')

    def area(self):
        return self.width * self.height

    # == 연산자를 사용했을 때 자동으로 호출되는 메소드
    def __eq__(self, other):
        return self.width == other.width and \
                self.height == other.height

    # 객체의 내용을 print할 때 자동으로 호출되는 메소드
    def __str__(self):
        return f'<직사각형 가로={self.width}, 세로={self.height}>'


if __name__ == '__main__':
    rect1 = Rectangle()
    # argument를 아무것도 전달하지 않으면
    # 모든 parameter는 기본값(default argument)를 사용하게 됨
    print(type(rect1))
    print(id(rect1))
    rect1.info()

    rect2 = Rectangle(1)  # height만 default argument가 사용됨
    rect2.info()

    rect3 = Rectangle(height=2)  # keyword argument를 사용한 호출
    rect3.info()

    rect4 = Rectangle(2, 3)  # positional argument를 사용한 호출
    rect4.info()
    print('rect4 넓이:', rect4.area())
    print(id(rect4))

    rect5 = Rectangle(width=2, height=3)
    rect5.info()
    print('rect5 넓이:', rect5.area())
    print(id(rect5))

    print(rect4 == rect5)
    # obj1 == obj2 비교하는 방법(== 연산자의 동작 방식):
    # obj1.__eq__(obj2)
    # == 연산자는 클래스의 __eq__ 메소드를 호출
    # 개발자가 클래스를 정의할 때 __eq__ 메소드를 정의하지 않아도
    # 모든 클래스는 __eq__ 메소드를 가지고 있음
    # 기본 __eq__ 메소드는 개체들의 주소값(id)를 비교함
    # 개발자가 __eq__ 메소드를 다른 방식으로 작성하면
    # == 연산자는 개발자의 의도대로 True/False를 리턴하게 됨

    print(rect5)
