-- ex 1
create table emp2
as (select * from emp);

create table dept2
as (select * from dept);

create table salgrade2
as (select * from salgrade);

-- ex 2
insert into dept2(deptno, dname, loc)
values(50, 'ORACLE', 'Seoul');

insert into dept2(deptno, dname, loc)
values(60, 'SQL', 'Jeju');

insert into dept2(deptno, dname, loc)
values(70, 'DDL', 'Pusan');

insert into dept2(deptno, dname, loc)
values(80, 'DML', 'Incheon');

-- ex 3
insert into emp2(empno, ename, job, mgr, hiredate, sal, comm, deptno)
values(7201, 'test1', 'CLERK', 7788, 
      to_date('2019-09-04', 'YYYY-MM-DD'), 4500, null, 50);

insert into emp2(empno, ename, job, mgr, hiredate, sal, comm, deptno)
values(7202, 'test2', 'CLERK', 7201, 
      to_date('2019-09-04', 'YYYY-MM-DD'), 1500, null, 60);

insert into emp2(empno, ename, job, mgr, hiredate, sal, comm, deptno)
values(7203, 'test3', 'CLERK', 7201, 
      to_date('2019-09-04', 'YYYY-MM-DD'), 2000, null, 70);

insert into emp2(empno, ename, job, mgr, hiredate, sal, comm, deptno)
values(7204, 'test1', 'CLERK', 7201, 
      to_date('2019-09-04', 'YYYY-MM-DD'), 3000, null, 80);

insert into emp2(empno, ename, job, mgr, hiredate, sal, comm, deptno)
values(7205, 'test5', 'CLERK', 7201, 
      to_date('2019-09-04', 'YYYY-MM-DD'), 500, 1000, 50);

insert into emp2(empno, ename, job, mgr, hiredate, sal, comm, deptno)
values(7206, 'test6', 'CLERK', 7201, 
      to_date('2019-09-04', 'YYYY-MM-DD'), 1500, 100, 60);

select * from emp2;

-- ex 4
select avg(sal) from emp2 where deptno = 20;

update emp2
set deptno = 80
where sal > (
    select avg(sal) from emp2 where deptno = 20
);

select * from emp2;

-- ex 5
select max(hiredate) from emp2 where deptno = 10;

update emp2
set sal = sal * 1.1, deptno = 70
where hiredate > (
    select max(hiredate) from emp2 where deptno = 10
);

select * from emp2;

-- ex 6
select e.empno, e.sal, s.grade
from emp2 e join salgrade2 s
on e.sal between s.losal and s.hisal
where s.grade = 5;

delete from emp2
where empno in (
    select e.empno
    from emp2 e join salgrade2 s
    on e.sal between s.losal and s.hisal
    where s.grade = 5
);

delete from emp2
where sal between 
            (select losal from salgrade where grade = 5) 
            and 
            (select hisal from salgrade where grade = 5);


-- ex 7.
commit;