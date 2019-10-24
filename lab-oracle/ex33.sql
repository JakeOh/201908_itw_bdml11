/*
커서(cursor): PL/SQL 내부에서 SQL 문장을 처리하는 정보를 저장한 메모리 공간
(1) 명시적 커서(explicit cursor): 개발자가 직접 이름을 선언하고 사용하는 커서
(2) 묵시적 커서(implicit cursor): 별다른 선언 없이 사용하는 커서

명시적 커서: 선언 -> open -> fetch -> close

*/

set serveroutput on;

select * from dept;

declare
    v_dept dept%rowtype;
begin 
    select * into v_dept from dept where deptno = 10;
    -- 오라클 DB 내부에서 묵시적 커서가 자동으로 생성됨.
    dbms_output.put_line(v_dept.deptno);
end;
/


-- select into 구문은 1개 이상의 행(row)가 select되는 경우는 사용할 수 없다.
-- 명시적 커서는 select의 결과 행의 갯수에 상관없이 사용할 수 있다.
declare
    -- 명시적 커서 선언:
    -- cursor 커서이름 is (SQL 문장);
    cursor my_cursor is (
        select * from dept where deptno = 10
    );
    
    v_row dept%rowtype;  -- 커서의 데이터를 저장(fetch)할 때 사용할 변수
begin
    -- 커서 open(열기, SQL 문장 실행)
    open my_cursor;
    
    -- 커서 fetch(데이터 읽기)
    fetch my_cursor into v_row;
    dbms_output.put_line(v_row.deptno || ', ' ||
                        v_row.dname || ', ' ||
                        v_row.loc);
    
    -- 커서 close(닫기)
    close my_cursor;
end;
/


-- 결과 행이 1개인 명시적 커서
declare
    -- (1) 커서 선언
    cursor sel_emp is (
        select * from emp where empno = 7788
    );
    v_row emp%rowtype;  -- 커서 결과를 저장(fetch)할 변수
begin
    -- (2) 커서 open
    open sel_emp;
    
    -- (3) 커서 실행 결과 읽기(fetch)
    fetch sel_emp into v_row;
    dbms_output.put_line(v_row.empno || ', ' || v_row.ename);
    
    -- (4) 커서 close
    close sel_emp;
end;
/

-- 결과 행이 여러개가 될 수 있는 명시적 커서
declare
    -- 명시적 커서 선언
    cursor sel_dept is (
        select * from dept
    );
    -- 커서 실행 결과를 fetch할 때 사용할 변수
    v_row dept%rowtype;
begin
    -- (2) open
    open sel_dept;
    
    -- (3) fetch -- 결과 행이 여러개일 경우 반복문 안에서 fetch
    loop
        fetch sel_dept into v_row;
        -- 커서에서 더이상 읽을 레코드가 없을 때 loop를 종료
        exit when sel_dept%notfound;
        
        dbms_output.put_line(v_row.deptno || ', ' 
                        || v_row.dname || ', ' 
                        || v_row.loc);
    end loop;
    
    -- (4) close
    close sel_dept;
end;
/


-- emp 테이블에서 부서번호가 30번인 직원들의 사번, 이름, 급여를 출력 
declare
    cursor sel_emp is (
        select * from emp where deptno = 30
    );

    v_row emp%rowtype;
begin
    open sel_emp;
    
    loop
        fetch sel_emp into v_row;
        exit when sel_emp%notfound;
        dbms_output.put_line(v_row.empno || ', ' 
                        || v_row.ename || ', '
                        || v_row.sal);
    end loop;
    
    close sel_emp;
end;
/


declare
    cursor my_cursor is (
        select empno, ename, sal from emp where deptno = 10
    );
    type my_record is record (
        empno   emp.empno%type,
        ename   emp.ename%type,
        sal     emp.sal%type
    );
    v_row my_record;
begin
    open my_cursor;
    
--    loop 
--        fetch my_cursor into v_row;
--        exit when my_cursor%notfound;
--    end loop;

    fetch my_cursor into v_row;
    while my_cursor%found loop
        dbms_output.put_line(v_row.empno || ', ' 
                        || v_row.ename || ', '
                        || v_row.sal);
        fetch my_cursor into v_row;
    end loop;
    
    close my_cursor;
end;
/


-- 명시적 커서를 for loop에서 사용
-- open, fetch, close가 자동으로 수행됨
-- for 변수 in 커서 loop ... end loop;
declare
    cursor my_cursor is (
        select * from dept
    );
begin
    for row in my_cursor loop
        dbms_output.put_line(row.deptno || ', ' || row.dname || ', ' || row.loc);
    end loop;
end;
/

-- 명시적 커서, for loop를 사용해서 
-- 20번 부서의 직원들의 사번, 이름, 급여를 출력 
declare
    cursor my_cursor is (
        select empno, ename name, sal from emp where deptno = 20
    );
begin
    for row in my_cursor loop
        dbms_output.put_line(row.empno || ', ' || row.name || ', ' || row.sal);
    end loop;
end;
/

-- 전체 직원의 급여 평균보다 적은 급여를 받는 직원들의 이름을 출력하는 PL/SQL
select avg(sal) from emp;
select ename from emp where sal < ?;
select ename from emp where sal < (select avg(sal) from emp);

declare
    cursor my_cursor is (
        select ename from emp
        where sal < (
            select avg(sal) from emp
        )
    );
begin
    for result in my_cursor loop
        dbms_output.put_line(result.ename);
    end loop;
end;
/

--  파라미터(parameter, 매개변수)를 갖는 커서 선언
-- cursor 커서이름(변수 타입, 변수 타입, ...) is SQL문장;
declare
    v_avg number;
    cursor my_cursor(p_avg number) is
        select ename from emp where sal < p_avg;
begin
    -- 전체 직원의 급여 평균을 알아냄
    select avg(sal) into v_avg from emp;
    dbms_output.put_line(v_avg);
    
    -- 파라미터를 갖는 커서를 open/fetch/close
    for row in my_cursor(v_avg) loop
        dbms_output.put_line(row.ename);
    end loop;
end;
/


-- 10번 부서에서, 10번 부서 직원들의 급여 평균보다 적은 급여를 받는 직원들의 
-- 이름을 출력하는 PL/SQL
select deptno, ename from emp
where sal < (
    select avg(sal) from emp where deptno = 10
) and deptno = 10;

declare
    cursor my_cursor is
        select deptno, ename from emp
        where sal < (select avg(sal) from emp where deptno = 10)
        and
        deptno = 10;
begin
    for row in my_cursor loop
        dbms_output.put_line(row.deptno || ', ' || row.ename);
    end loop;
end;
/

declare
    cursor my_cursor(p_deptno emp.deptno%type) is
        select deptno, ename from emp
        where sal < (select avg(sal) from emp where deptno = p_deptno)
        and deptno = p_deptno;
begin
    for row in my_cursor(20) loop
        dbms_output.put_line(row.deptno || ', ' || row.ename);
    end loop;
end;
/

-- 각 부서에서, 그 부서 직원들의 급여 평균보다 적은 급여를 받는 직원들의 
-- 부서번호와 직원 이름을 출력하는 PL/SQL
declare
    cursor get_deptno is 
        select deptno from dept;
    cursor my_cursor(p_deptno emp.deptno%type) is 
        select deptno, ename from emp
        where sal < (select avg(sal) from emp where deptno = p_deptno)
        and deptno = p_deptno;
begin
    for rslt in get_deptno loop
--        dbms_output.put_line(rslt.deptno);
        for row in my_cursor(rslt.deptno) loop
            dbms_output.put_line(row.deptno || ', ' || row.ename);
        end loop;
    end loop;
end;
/

select deptno from dept;
select deptno from emp;
select distinct deptno from emp;

select deptno, avg(sal) from emp group by deptno;

declare
    cursor cur1 is
        select deptno, avg(sal) average from emp
        group by deptno
        order by deptno;
    cursor cur2(p_deptno emp.deptno%type, p_avg number) is
        select deptno, ename, sal from emp
        where deptno = p_deptno and sal < p_avg;
begin
    for r1 in cur1 loop
        dbms_output.put_line(r1.deptno || ', ' || r1.average);
        for r2 in cur2(r1.deptno, r1.average) loop
            dbms_output.put_line(r2.deptno || ', ' || r2.ename || ', ' || r2.sal);
        end loop;
    end loop;
end;
/


select deptno dno, avg(sal) average from emp 
group by deptno;

select a.dno
from (select deptno dno, avg(sal) average from emp group by deptno) a;

select e.deptno, e.ename, e.sal, a.average
from emp e
    join (select deptno, avg(sal) average 
            from emp group by deptno) a
    on e.deptno = a.deptno
where e.sal < a.average
order by e.deptno;
    
declare
    cursor my_cursor is
        select e.deptno, e.ename, e.sal, a.average
        from emp e
            join (select deptno, avg(sal) average 
                    from emp group by deptno) a
            on e.deptno = a.deptno
        where e.sal < a.average
        order by e.deptno;
begin
    for r in my_cursor loop
        dbms_output.put_line(r.deptno || ', ' || r.ename);
    end loop;
end;
/
    
    
    
    