-- 조건을 만족하는 레코드 검색
-- select 컬럼, ... from 테이블이름 where 조건;

-- 사원 테이블에서 부서 번호가 10인 사원들의 모든 컬럼을 검색
select * from emp where deptno = 10;

-- 사원 테이블에서 급여가 2500 이상(>=)인 직원들을 검색 
select * from emp where sal >= 2500;

-- 사원 테이블에서 급여가 1000 이상, 2000 미만인 직원들의 모든 컬럼을 검색
select * from emp
where sal >= 1000 and sal < 2000;

-- 사원 테이블에서 급여가 3000 이상 5000 이하인 직원들의 모든 컬럼을 검색
select * from emp
where sal >= 3000 and sal <= 5000;

select * from emp
where sal between 3000 and 5000;

-- 사원 테이블에서 급여가 1000 이하이거나 또는 5000 이상인 직원들의 모든 컬럼 검색
select *
from emp
where sal <= 1000 or sal >= 5000;

-- 사원 테이블에서 부서번호가 10 또는 20인 직원들의 모든 컬럼 검색 
select *
from emp
where deptno = 10 or deptno = 20;

select *
from emp
where deptno in (10, 20);

-- 직원 테이블에서 부서번호가 10번이 아닌 직원들의 모든 정보 검색
-- 다름: !=, <>, ^=
select * 
from emp
where deptno != 10;

select *
from emp
where not deptno = 10;

-- 직원 테이블에서 직책이 'CLERK'인 직원들의 모든 정보를 검색
select *
from emp
where job = 'CLERK';
-- Oracle에서 예약어(select, from, where, ....), 테이블 이름, 컬럼 이름은 대소문자를 구별하지 않음.
-- 테이블에 저장된 값은 대소문자를 구별함. 'CLERK' != 'Clerk'

select *
from emp
where ename > 'G';
-- 문자열도 크기 비교가 가능함.

-- 1) 직원 테이블에서 부서번호가 20번이거나 직책이 'SALESMAN'인 
-- 사원들의 모든 정보 검색
select *
from emp
where deptno = 20 or job = 'SALESMAN';

-- 2) 직원 테이블에서 직책이 'CLERK'이거나 또는 'ANALYST' 또는 'MANAGER'인
-- 직원들의 모든 정보를 검색
select * 
from emp
where job = 'CLERK'
      or job = 'ANALYST'
      or job = 'MANAGER';

select * 
from emp
where job in ('CLERK', 'ANALYST', 'MANAGER');

-- 3) 직원 테이블에서 직책이 'CLERK', 'ANALYST', 'MANAGER'가 아닌
-- 직원들의 모든 정보를 검색
select * 
from emp
where job != 'CLERK'
      and job != 'ANALYST'
      and job != 'MANAGER';

select * 
from emp
where job not in ('CLERK', 'ANALYST', 'MANAGER');

-- 오라클에서 문자열은 항상 작은 따옴표('')를 사용
-- 컬럼의 별명(alias)에서만 큰따옴표("")를 사용함
select empno 사번  from emp;
select empno "사번"  from emp;
select empno as 사번  from emp;
select empno as "사번"  from emp;


-- 직원 테이블에서 이름이 'A'로 시작하는 직원들의 모든 정보 검색
select *
from emp
where ename like 'A%';

select *
from emp
where ename like 'A_';

select *
from emp
where job like '_LERK';

select *
from emp
where job like '_ERK';

select *
from emp
where job like '%ERK';

select *
from emp
where job like '%E%';

-- 직원 테이블에서 수당(comm)이 null인 직원들의 모든 정보를 검색
select * from emp where comm is null;

-- 직원 테이블에서 수당(comm)이 null이 아닌 직원들의 모든 정보를 검색
select * from emp where comm is not null;
select * from emp where not (comm is null);
-- select * from emp where comm not is null;  -- 오류

-- 직원 테이블에서 수당을 받는 직원들 중에서 급여가 1500 초과하는 직원의
-- 사번, 이름, 급여를 검색
select empno, ename, sal
from emp
where (comm is not null) and (sal > 1500);

-- 직원 테이블에서 수당을 받지 않는 직원들 중에서 급여가 1500 초과하는 직원의
-- 사번, 이름, 급여를 검색
select empno, ename, sal
from emp
where (comm is null) and (sal > 1500);

-- 집합연산자: 합집합(union), 교집합(intersect), 차집합(minus)
select * from emp where deptno = 10 or deptno = 20;

select * from emp where deptno = 10
union
select * from emp where deptno = 20;

-- 집합연산자를 사용할 때 두개의 select 문장은 같은 컬럼들을 같은 순서로 검색해야 함!
select empno, ename, deptno from emp where deptno = 10
union
select empno, ename, deptno from emp where deptno = 20;

-- 이름이 'E'포함하는 모든 직원 정보
select * from emp where ename like '%E%';
-- job이 'E'포함하는 모든 직원 정보
select * from emp where job like '%E%';

select * from emp where ename like '%E%'
intersect
select * from emp where job like '%E%';

select * from emp where ename like '%E%'
minus
select * from emp where job like '%E%';

select * from emp where job like '%E%'
minus
select * from emp where ename like '%E%';


-- 1) 30번 부서에서 일하는, 직책이 SALESMAN인 직원들의
-- 사번, 이름, 급여, 부서번호를 검색
select empno, ename, sal, deptno
from emp
where deptno = 30 and job = 'SALESMAN';

-- 2) 20, 30번 부서에 근무하는 직원들 중 급여가 2000을 초과하는
-- 직원들의 사번, 이름, 급여, 부서번호를 검색
select empno, ename, sal, deptno
from emp
where deptno in (20, 30) and sal > 2000;

select empno, ename, sal, deptno
from emp
where (deptno = 20 or deptno = 30) and sal > 2000;

-- 3) 수당이 없는 직원들 중에서, 매니저가 있고, 직책이 'MANAGER' 또는 'CLERK'인
-- 직원들의 모든 정보를 검색
select *
from emp
where comm is null
    and mgr is not null
    and job in ('MANAGER', 'CLERK');




