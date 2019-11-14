"""
클래스 작성, 객체 생성, 메소드 사용 연습
"""
class Employee:
    """
    사원 정보와 급여 인상 메소드를 가지고 있는 클래스
    field: empno, ename, salary, deptno
    method: raise_salary(self, pct)
    """
    def __init__(self, empno, ename, salary, deptno):
        self.empno = empno
        self.ename = ename
        self.salary = salary
        self.deptno = deptno

    def raise_salary(self, pct):
        """
        인상된 급여를 리턴

        :param pct: 급여 인상율(0.1 = 10%, 0.5 = 50%, ...)
        :return: 인상된 급여
        """
        # self.salary = (1 + pct) * self.salary
        self.salary *= (1 + pct)
        return self.salary

    def __repr__(self):
        return f'(사번: {self.empno}, 이름: {self.ename}, 급여: {self.salary}, 부서번호: {self.deptno})'


gil_dong = Employee(1010, '홍길동', 1000, 10)  # 객체 생성
print(gil_dong.__repr__())  # 직원 정보 출력
gil_dong.raise_salary(0.1)  # 급여 인상
print(gil_dong.__repr__())  # 직원 정보 출력

scott = Employee(1011, 'Scott', 10000, 20)
print(scott.__repr__())
scott.raise_salary(-0.1)
print(scott.__repr__())

ohssam = Employee(1012, '오쌤', 500, 30)

employees = [ohssam, gil_dong, scott]
print(employees)

print(sorted(employees, key=lambda x: x.empno))
print(sorted(employees, key=lambda x: x.salary))
print(sorted(employees, key=lambda x: x.ename))

