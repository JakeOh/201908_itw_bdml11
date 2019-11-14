"""
클래스 작성, 테스트
"""
from math import sqrt

class Point:
    """
    2차원(x-y) 평면 상의 점 1개를 저장할 수 클래스
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def print_point(self):
        """
        Point 객체가 가지고 있는 점의 좌표를 (x, y) 형식으로 출력

        :return: None
        """
        print(f'({self.x}, {self.y})')

    def distance(self, other):
        """
        두 점 사이의 거리를 계산해서 리턴하는 함수

        :param other: Point 객체
        :return: self와 other 사이의 거리
        """
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


if __name__ == '__main__':
    pt1 = Point(1, 1)
    pt1.print_point()

    pt2 = Point(2, 3)
    pt2.print_point()

    print(pt1.distance(pt2))
