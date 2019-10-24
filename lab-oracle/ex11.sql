-- join

-- Oracle 방식의 Cross Join(Cartesian Product)
select *
from emp, dept;

-- ANSI(표준) 방식의 Cross Join
select *
from emp cross join dept;

-- Oracle 방식의 Join
select *
from emp, dept
where emp.deptno = dept.deptno
order by emp.empno;

-- ANSI 방식의 Join
select *
from emp e join dept d
    on e.deptno = d.deptno
order by e.empno;

-- emp 테이블과 dept 테이블에서
-- 사번, 이름, 부서번호, 부서이름을 출력
-- 1) Oracle 방식
select e.empno, e.ename, e.deptno, d.dname
from emp e, dept d
where e.deptno = d.deptno
order by e.empno;
-- 2) ANSI 방식
select e.empno, e.ename, e.deptno, d.dname
from emp e join dept d
    on e.deptno = d.deptno
order by e.empno;

-- emp 테이블과 dept 테이블에서
-- 사번, 이름, 급여, 부서번호, 부서이름을 검색
-- 단, 급여가 2000 이상인 직원만 선택
-- 1) Oracle 방식
select e.empno, e.ename, e.sal, d.deptno, d.dname
from emp e, dept d
where e.deptno = d.deptno 
      and
      e.sal >= 2000;
-- 2) ANSI 방식
select e.empno, e.ename, e.sal, d.deptno, d.dname
from emp e join dept d
    on e.deptno = d.deptno
where e.sal >= 2000;


-- emp 테이블과 salgrade 테이블에서
-- 사번, 이름, 급여, 급여 등급을 출력
-- 1) Oracel
select e.empno, e.ename, e.sal, s.grade
from emp e, salgrade s
where e.sal between s.losal and s.hisal;
-- 2) ANSI
select e.empno, e.ename, e.sal, s.grade
from emp e join salgrade s
           on e.sal >= s.losal and e.sal <= s.hisal;
           -- on e.sal between s.losal and s.hisal;


-- emp, dept, salgrade 테이블에서
-- 사번, 이름, 부서번호, 부서이름, 급여, 급여 등급을 출력
-- 1) Oracle
select e.empno, e.ename, e.deptno, d.dname, e.sal, s.grade
from emp e, dept d, salgrade s
where e.deptno = d.deptno
      and
      e.sal between s.losal and s.hisal;
-- 2) ANSI
select e.empno, e.ename, e.deptno, d.dname, e.sal, s.grade
from emp e join dept d
           on e.deptno = d.deptno
     join salgrade s
           on e.sal between s.losal and s.hisal;

           
-- emp 테이블에서 사번, 이름, 매니저 사번, 매니저 이름 출력
-- 1) Oracle
select e1.empno, e1.ename, e1.mgr, e2.ename
from emp e1, emp e2
where e1.mgr = e2.empno;
-- 2) ANSI
select e1.empno, e1.ename, e1.mgr, e2.ename
from emp e1 join emp e2
     on e1.mgr = e2.empno;


-- emp, dept테이블에서 사번, 이름, 부서번호, 부서이름을 출겨
-- dept 테이블의 모든 부서 정보를 출력
-- 1) Oracle
select e.empno, e.ename, d.deptno, d.dname
from emp e, dept d
where e.deptno(+) = d.deptno;
-- 2) ANSI
select e.empno, e.ename, d.deptno, d.dname
from emp e right join dept d
    on e.deptno = d.deptno;
    
select e.empno, e.ename, d.deptno, d.dname
from dept d, emp e
where d.deptno = e.deptno(+);

select e.empno, e.ename, d.deptno, d.dname
from dept d left join emp e
    on d.deptno = e.deptno;


-- emp 테이블에서 사번, 이름, 매니저 사번, 매니저 이름 출력
-- 매니저가 존재하지 않는 King의 정보도 함께 출력
-- 1) Oracle
select e1.empno, e1.ename, e1.mgr, e2.ename
from emp e1, emp e2
where e1.mgr = e2.empno(+);
-- 2) ANSI
select e1.empno, e1.ename, e1.mgr, e2.ename
from emp e1 left join emp e2
    on e1.mgr = e2.empno;

-- 부하 직원이 없는 사원들의 정보를 함께 출력(21 records)
select e1.empno, e1.ename, e2.empno, e2.ename
from emp e1, emp e2
where e1.mgr(+) = e2.empno;

select e1.empno, e1.ename, e2.empno, e2.ename
from emp e1 right join emp e2
    on e1.mgr = e2.empno;


-- full (outer) join
-- full join은 ANSI 방식의 문법만 있고, Oracle 방식의 문법은 없음
select e1.empno, e1.ename, e2.empno, e2.ename
from emp e1 full join emp e2
    on e1.mgr = e2.empno;
-- Oracle 방식으로 full join을 하려면 UNION(합집합) 연산자를 사용해야 함
select e1.empno, e1.ename, e2.empno, e2.ename
from emp e1, emp e2 where e1.mgr = e2.empno(+)
union
select e1.empno, e1.ename, e2.empno, e2.ename
from emp e1, emp e2 where e1.mgr(+) = e2.empno;


-- Ex 1.
select e.deptno, dname, round(avg(sal), 1), max(sal), min(sal), count(*)
from emp e, dept d
where e.deptno = d.deptno
group by e.deptno, dname;

select e.deptno, dname, round(avg(sal), 1), max(sal), min(sal), count(*)
from emp e join dept d
    on e.deptno = d.deptno
group by e.deptno, dname;


-- Ex 2
select d.deptno, d.dname, e.empno, e.ename, e.job, e.sal
from dept d, emp e
where d.deptno = e.deptno(+);

select d.deptno, d.dname, e.empno, e.ename, e.job, e.sal
from dept d left join emp e
    on d.deptno = e.deptno;

select d.deptno, d.dname, e.empno, e.ename, e.job, e.sal
from emp e, dept d
where e.deptno(+) = d.deptno;

select d.deptno, d.dname, e.empno, e.ename, e.job, e.sal
from emp e right join dept d
    on d.deptno = e.deptno;

-- ex3
select d.deptno, d.dname, e1.empno, e1.ename, e1.mgr, 
       e2.ename as mgr_name,
       e1.sal,
       s.grade
from dept d, emp e1, emp e2, salgrade s
where d.deptno = e1.deptno(+)
      and
      e1.mgr = e2.empno(+)
      and
      e1.sal between s.losal(+) and s.hisal(+);

select d.deptno, d.dname, e1.empno, e1.ename, e1.mgr,
       e2.ename as mgr_name,
       e1.sal, s.grade
from dept d left join emp e1
            on d.deptno = e1.deptno
            left join emp e2
            on e1.mgr = e2.empno
            left join salgrade s
            on e1.sal between s.losal and s.hisal;





