"""
emp.csv 파일을 읽어서, DataFrame을 생성
- 급여(sal)가 2000 이상인 직원들의 모든 정보를 출력
- 부서 번호(deptno)가 10인 직원들의 모든 정보를 출력
- 급여가 전체 직원의 급여의 평균보다 많은 직원의 사번, 이름, 급여를 출력
- 30번 부서에서 일하는, 직책이 SALESMAN인 직원들의 사번, 이름, 급여, 부서번호를 출력
- 20, 30번 부서에 근무하는 직원들 중 급여가 2000을 초과하는 직원들의 사번, 이름,
급여, 부서번호를 출력
- 수당이 없는 직원들 중에서, 매니저가 있고, 직책이 'MANAGER' 또는 'CLERK'인
직원들의 모든 정보를 검색
- 사원 이름에 'E'가 포함된 직원들의 이름만 출력 (str.contains() 이용)
- DataFrame을 csv 형식으로 파일에 write하는 함수를 찾아서 데이터 프레임을 파일로 저장
"""
import pandas as pd

if __name__ == '__main__':
    file_path = 'emp.csv'
    # pandas.read_csv() 함수는 csv 파일의 첫번째 줄을 헤더(컬럼 이름)으로 취급.
    # csv 파일의 첫번째 줄이 헤더(컬럼 이름)가 아니고 실제 레코드인 경우에는,
    # read_csf() 함수를 호출할 때 header=None 파라미터를 추가해야 함.
    emp = pd.read_csv(file_path, header=None)
    print('shape:', emp.shape)
    print(emp.head())

    # DataFrame 컬럼 이름을 설정
    emp.columns = ['empno', 'ename', 'job', 'mgr',
                   'hiredate', 'sal', 'comm', 'deptno']
    print(emp.iloc[0:2])

    print('\n급여(sal)가 2000 이상인 직원들의 모든 정보')
    print(emp[emp['sal'] >= 2000])  # DataFrame[조건식]

    print('\n부서 번호(deptno)가 10인 직원들의 모든 정보')
    print(emp[emp['deptno'] == 10])

    print('\n급여가 전체 직원의 급여의 평균보다 많은 직원의 사번, 이름, 급여')
    # select empno, ename, sal from emp where sal > (select avg(sal) from emp)
    subset = emp[emp['sal'] > emp['sal'].mean()]
    print(subset[['empno', 'ename', 'sal']])

    print('\n30번 부서에서 일하는, 직책이 SALESMAN인 직원들의 사번, 이름, 급여, 부서번호')
    # select empno, ename, job, sal, deptno
    # from emp
    # where deptno == 30 and job == 'SALESMAN'
    c1 = emp['deptno'] == 30  # 30번 부서에서 일하는
    c2 = emp['job'] == 'SALESMAN'  # 직책이 SALESMAN인
    subset = emp[c1 & c2]
    print(subset[['empno', 'ename', 'job', 'sal', 'deptno']])

    print('\n20, 30번 부서에 근무하는 직원들 중 급여가 2000을 초과하는 직원들')
    # select * from emp
    # where (deptno = 20 or deptno = 30) and sal > 2000;
    # select * from emp
    # where deptno in (20, 30) and sal > 2000;

    # c1 = emp['deptno'] == 20
    # c2 = emp['deptno'] == 30
    # c3 = emp['sal'] > 2000
    # subset = emp[(c1 | c2) & c3]
    c1 = emp['deptno'].isin([20, 30])
    c2 = emp['sal'] > 2000
    subset = emp[c1 & c2]
    print(subset[['empno', 'ename', 'sal', 'deptno']])

    print('\n수당이 없는 직원들 중에서, 매니저가 있고, 직책이 MANAGER 또는 CLERK인 직원들')
    # select * from emp
    # where comm is null and mgr is not null and job in ('MANAGER', 'CLERK');

    c1 = emp['comm'].isna()  # 수당이 없는 직원. emp['comm'] == 'NaN' 이라고 하면 안됨.
    c2 = ~emp['mgr'].isna()  # mgr 컬럼의 값이 NaN(null)이 아닌 경우
    # c3 = (emp['job'] == 'MANAGER') | (emp['job'] == 'CLERK')
    c3 = emp['job'].isin(['MANAGER', 'CLERK'])  # job이 manager 또는 clerk
    subset = emp[c1 & c2 & c3]
    print(subset)

    print('\n사원 이름에 E가 포함된 직원들의 이름')
    # select ename from emp where ename like '%E%'
    subset = emp[emp['ename'].str.contains('E')]
    print(subset['ename'])  # Series
    print(subset[['ename']])  # DataFrame

    # DataFrame.to_csv(file_path): 데이터 프레임을 csv 파일로 저장
    # to_csv() 함수는 행 이름(row index)를 파일에 씀.
    # row index를 파일에 쓰지 않으려면, 함수를 호출할 때 index=False 파라미터 추가.
    emp.to_csv('emp2.csv', index=False)
