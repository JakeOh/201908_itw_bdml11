set serveroutput on;

-- ex 5. (1)
select e.ename, d.loc
from emp e join dept d on e.deptno = d.deptno
where e.ename = 'SCOTT';
/

-- ex5. (2)
accept p_ename prompt 'Enter emp name:';
declare
    v_ename emp.ename%type;
    v_loc dept.loc%type;
begin
    select e.ename, d.loc into v_ename, v_loc
    from emp e join dept d on e.deptno = d.deptno
    where e.ename = upper('&p_ename');

    DBMS_OUTPUT.put_line(v_ename || ' is working at ' || v_loc);
end;
/


-- ex 6. (1)
select empno, ename, hiredate,
       trunc(months_between(sysdate, hiredate) / 12) YEAR
from emp;

-- ex 6. (2)
select deptno, min(hiredate) min_hiredate
from emp
group by deptno
order by deptno;

-- ex 6. (3)
select deptno, empno, ename,
       trunc(months_between(sysdate, hiredate) / 12) year
from emp
where deptno = 10
    and
      hiredate = (select min(hiredate) from emp where deptno = 10);

-- ex 6. (4)
declare
    v_deptno emp.deptno%type;
    v_empno emp.empno%type;
    v_ename emp.ename%type;
    v_yos number;  -- year of service

    cursor cur_deptno_min_hiredate is
        select deptno, min(hiredate) min_hiredate
        from emp
        group by deptno
        order by deptno;
begin
    for r in cur_deptno_min_hiredate loop
        select deptno, empno, ename,
               trunc(months_between(sysdate, hiredate) / 12) year
        into v_deptno, v_empno, v_ename, v_yos
        from emp
        where deptno = r.deptno
          and hiredate = r.min_hiredate;

        DBMS_OUTPUT.put_line(v_deptno
            || ', ' || v_empno
            || ', ' || v_ename
            || ', ' || v_yos);
    end loop;
end;
/


-- ex 7. (1)
select e.ename, m.ename
from emp e left join emp m on e.mgr = m.empno;

select e.ename, m.ename
from emp e, emp m
where e.mgr = m.empno(+);

-- ex 7. (2)
select e.ename, m.ename
from emp e left join emp m on e.mgr = m.empno
where e.ename = upper('scott');

select e.ename, m.ename
from emp e, emp m
where e.mgr = m.empno(+)
  and e.ename = upper('scott');

-- ex 7. (3)
create or replace function get_mgr_name(p_ename varchar2) return varchar2
is
    v_mgr_name emp.ename%type;
begin
    select m.ename into v_mgr_name
    from emp e
             left join emp m on e.mgr = m.empno
    where e.ename = upper(p_ename);

    return v_mgr_name;
end;
/

declare
    v_mgr_name emp.ename%type;

    cursor cur_get_ename is
        select ename from emp;
begin
    for r in cur_get_ename loop
        v_mgr_name := get_mgr_name(r.ename);
        DBMS_OUTPUT.PUT_LINE(r.ename || ', ' || v_mgr_name);
    end loop;
end;
/

-- ex 8. (1)
select empno, ename
from emp
where empno in (select mgr from emp);

-- ex 8. (2)
select empno, ename
from emp
where empno not in (select mgr from emp where mgr is not null);

-- ex 8. (3)
create or replace function is_manager(p_empno number) return varchar2
is
    v_mgr_name emp.ename%type;
begin
    select ename into v_mgr_name
    from emp
    where empno in (select mgr from emp)
      and empno = p_empno;

    return 'm';
exception
    when others then
        return 'e';
end;
/
select is_manager(7788) from dual;
select is_manager(7369) from dual;


-- ex 8. (4)
declare
    v_is_mgr varchar2(1);
    cursor cur is
        select empno, ename from emp;
begin
    for r in cur loop
        v_is_mgr := is_manager(r.empno);
        if v_is_mgr = 'm' then
            DBMS_OUTPUT.PUT_LINE(r.ename || ', ' || '관리자');
        else
            DBMS_OUTPUT.PUT_LINE(r.ename || ', ' || '사원');
        end if;
    end loop;
end;
/


-- ex 9. (1)
select e.empno, e.ename, e.hiredate, m.ename, m.hiredate
from emp e
    join emp m on e.mgr = m.empno
where e.hiredate < m.hiredate;

select e.empno, e.ename, e.hiredate, m.ename, m.hiredate
from emp e, emp m
where e.mgr = m.empno
  and e.hiredate < m.hiredate;

-- ex 9. (2)
select e.empno, e.ename, e.hiredate, m.ename, m.hiredate
from emp e
    join emp m on e.mgr = m.empno
where e.hiredate < m.hiredate
  and e.empno = 7369;

select e.empno, e.ename, e.hiredate, m.ename, m.hiredate
from emp e
    join emp m on e.mgr = m.empno
where e.hiredate < m.hiredate
  and e.empno = 7839;

-- ex 9. (3)
create or replace function is_longer(p_empno number) return number
is
    v_empno emp.empno%type;
begin
    select e.empno into v_empno
    from emp e
             join emp m on e.mgr = m.empno
    where e.hiredate < m.hiredate
      and e.empno = p_empno;

    return 1;
exception
    when others then
        return 0;
end;
/

select is_longer(7369) from dual;
select is_longer(7839) from dual;

-- ex 9. (4)
declare
    cursor cursor1 is
        select empno from emp;

    type rec_result is record (
        ename       emp.ename%type,
        hiredate    emp.hiredate%type,
        is_long     number
    );
    v_result rec_result;
begin
    for r1 in cursor1 loop
        select ename, hiredate, is_longer(r1.empno) is_long
        into v_result
        from emp
        where empno = r1.empno;
        DBMS_OUTPUT.PUT_LINE(v_result.ename
                                 || ', ' || v_result.hiredate
                                 || ', ' || v_result.is_long);
    end loop;
end;
/


-- ex 10.
select job, sum(sal) from emp group by job order by job;
select ename, sal, job from emp where job = 'CLERK';

declare
    cursor cursor1 is
        select job, sum(sal) tot_sal from emp group by job order by job;
    cursor cursor2(p_job emp.job%type) is
        select ename, sal, job from emp where job = p_job;
    type rec_result is record (
        ename   emp.ename%type,
        sal     emp.sal%type,
        job     emp.job%type
    );
    v_result rec_result;
begin
    for r1 in cursor1 loop
        DBMS_OUTPUT.PUT_LINE('직업: ' || r1.job);

        for r2 in cursor2(r1.job) loop
            DBMS_OUTPUT.PUT_LINE(r2.ename || ', ' || r2.sal || ', ' || r2.job);
        end loop;

        DBMS_OUTPUT.PUT_LINE('---------------');
        DBMS_OUTPUT.PUT_LINE('토탈: ' || r1.tot_sal);
        DBMS_OUTPUT.PUT_LINE(' ');
    end loop;
end;
/

