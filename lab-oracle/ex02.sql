-- 도구 -> SQL워크시트(Alt+F10): 새 워크시트(접속) 생성

-- 테이블에서 데이터 검색
-- 1) 컬럼 별로 검색 - 프로젝션(projection)
-- 2) 행(row)별로 검색 - 실렉션(selection)

-- select 컬럼이름1, 컬럼이름2, ... from 테이블이름;
-- 사원 테이블(emp)에서 사번(empno)과 사원이름(ename)을 검색
select empno, ename from emp;
-- 사원 테이블에서 사번, 이름, 입사날짜, 급여를 검색 
select empno, ename, hiredate, sal from emp;

select ename, empno from emp;

-- 테이블에서 모든 컬럼을 검색할 때: select * from 테이블이름;
select * from emp;

-- select 컬럼이름 as 별명, ... from 테이블 이름;
select empno as 사번, ename as 이름
from emp;

-- 사원 테이블에서 사번, 이름, 급여(sal), 연봉(sal * 12)을 검색
select empno, ename, sal, sal * 12 as ANNUAL_SAL
from emp;
-- 수당(comm)을 포함한 1년 연봉을 계산해서 출력
select empno, ename, sal, 
       sal * 12 + comm as ANNUAL_SAL
from emp;


-- 사원 테이블에서 직책(job)을 검색
select job from emp;

-- 중복된 행의 값을 제거하고 검색
select distinct job from emp;

select job, deptno from emp;
select distinct job, deptno from emp;
-- select job, distinct deptno from emp;  -- 오류가 발생하는 문장

