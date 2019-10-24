-- PL/SQL의 데이터 타입
-- 1) Scalar(스칼라) 타입: number, varchar2, date, ...
-- 2) Reference(참조) 타입: 특정 테이블의 행(row) 또는 열(column)을 참조하는 타입
-- 3) Compound(복합) 타입: collection, record

-- 컬럼 참조 타입 변수 선언: 변수이름 테이블이름.컬럼이름%type := 값;
-- 행 참조 타입 변수 선언: 변수이름 테이블이름%rowtype := 값;

set serveroutput on;

declare
    -- dept 테이블의 각 컬럼의 값들을 저장할 수 있는 변수 선언
    v_deptno dept.deptno%type := 10;
    v_dname dept.dname%type;
    v_loc dept.loc%type;
begin
    select dname, loc into v_dname, v_loc
    from dept
    where deptno = v_deptno;
    dbms_output.put_line(v_deptno || ' : ' || v_dname || ' : ' || v_loc);
end;
/


declare
    -- emp 테이블의 행(row)의 모든 정보를 저장할 수 있는 변수 선언
    v_emp7788 emp%rowtype;
begin
    select * into v_emp7788 from emp where empno = 7788;
    dbms_output.put_line(v_emp7788.ename);
    dbms_output.put_line(v_emp7788.sal);
    dbms_output.put_line(v_emp7788.deptno);
    dbms_output.put_line(v_emp7788.hiredate);
end;
/





