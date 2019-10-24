/*
패키지(package):
기능 면에서 연관성이 높은 프로시저, 함수, 타입(레코드, 컬렉션), 커서 등을 
하나로 모아둔 서브 프로그램

패키지를 생성할 때는 패키지의 명세(specification)와 본문(body)를 선언해야 함

패키지 명세:
create [or replace] package 패키지이름
is
    타입 선언;
    프로시저 프로토타입 선언;
    함수 프로토타입 선언;
    ...
end;

패키지 본문(본체):
create [or replace] package body 패키지이름
is
     패키지 명세(스펙)에서 선언된 프로시저/함수 등을 구현
end;
*/

-- 패키지 명세(스펙) 작성
create or replace package my_pkg
is
    -- 타입 선언
    type rec_emp_sal is record (
        empno   emp.empno%type,
        ename   emp.ename%type,
        sal     emp.sal%type
    );
    
    -- 변수 선언
    num1 number := 0;
    
    -- 프로시저 프로토타입(prototype) 선언
    procedure my_proc(p_empno number);
    
    -- 함수 프로토타입 선언
    function my_add(p_x number, p_y number) return number;
    
end;
/


-- 패키지의 본문(본체, body)를 작성
create or replace package body my_pkg
is
    -- 패키지 명세에 선언된 프로시저 구현
    procedure my_proc(p_empno number)
    is
        v_result my_pkg.rec_emp_sal;
    begin
        select empno, ename, sal 
        into v_result
        from emp
        where empno = p_empno;
        
        dbms_output.put_line(v_result.empno || ', '
                            || v_result.ename || ', '
                            || v_result.sal);
    end my_proc;
    
    -- 패키지 명세에 선언된 함수를 구현
    function my_add(p_x number, p_y number) return number
    is
    begin
        return p_x + p_y;
    end my_add;
end;
/


select my_pkg.my_add(1, 2) from dual;

declare
    v_result my_pkg.rec_emp_sal;
begin
    dbms_output.put_line(my_pkg.num1);
    my_pkg.my_proc(7788);
end;
/

























