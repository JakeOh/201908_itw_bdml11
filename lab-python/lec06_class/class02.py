"""
클래스 = 데이터(필드, field) + 기능(메소드, method) => 데이터 타입
"""

class Score:
    # 생성자를 호출했을 때 필드들을 초기화하는 함수
    def __init__(self, korean, english, math):
        self.korean = korean  # field
        self.english = english
        self.math = math

    # method
    def calc_total(self):
        return self.korean + self.english + self.math

    def calc_average(self):
        return self.calc_total() / 3


# Score 클래스의 객체를 생성
score1 = Score(99, 88, 77)  # 생성자 호출
score1.math = 79
print(score1.calc_total())
print(score1.calc_average())

score2 = Score(90, 85, 70)
print(score2.calc_total())
print(score2.calc_average())
