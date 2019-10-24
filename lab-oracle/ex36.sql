/*
Procedure(프로시저) vs Function(함수)
1. Procedure(프로시저)
  (1) 특정한 로직을 수행하기만 하고, 명시적으로 반환(return)하지 않는 서브 프로그램
  (2) 실행: 다른 PL/SQL (프로그램) 내부에서 실행됨. SQL 문에서는 실행될 수 없음.
  (3) 파라미터: in, out, in out 모드들을 사용할 수 있음.
  (4) 반환: return 문을 사용한 반환을 할 수 없음.
  out/in out 파라미터를 통해서 1개 이상의 값을 반환할 수는 있음.
  
2. Function(함수)
  (1) 특정한 기능을 수행한 뒤 반드시 결과값을 반환(return)하는 서브 프로그램
  (2) 실행: 다른 PL/SQL (프로그램) 내부 또는 SQL 문장 안에서 실행될 수 있음.
  (3) 파라미터: in 모드의 파라미터만 사용. in은 생략.
  문법적으로는 함수에서도 out, in out 파라미터를 사용할 수 있지만,
  out/in out 파라미터를 갖는 함수는 SQL 문장에서 사용할 수 없음!
  오라클에서는 out/in out 파라미터를 함수에서 사용하지 말라고 권장함.
  (4) 반환: 반드시 return 문을 사용해서 1개의 값을 반환해야 함!

함수를 선언하는 방법:
create [or replace] function 함수이름 [(
    파라미터1 타입 [:= 기본값1],
    파라미터2 타입 [:= 기본값2],
    ...
)]
return 리턴타입
is
begin
[exception -- 예외 처리부]
end;
*/

create or replace function my_add(
    p_x number,
    p_y number
) return number
is
--    v_result number;  -- 리턴값을 저장하기 위한 변수
begin
--    v_result := p_x + p_y;
--    return v_result; -- 함수를 호출한 곳으로 값을 반환(return)
    return p_x + p_y;
end;
/

-- 오라클 내장 함수의 결과값을 확인할 때 dual 더미 테이블 사용.
select upper('itwill') from dual;

-- 우리가 작성하는 함수도 SQL문으로 결과값(return value)을 확인할 수 있음.
select my_add(1, 2) from dual;

-- 함수는 PL/SQL에서도 사용 가능
set serveroutput on;
declare
    v_result number;
begin
    v_result := my_add(100, 200);
    dbms_output.put_line(v_result);
end;
/


-- 숫자 두개를 파라미터로 전달받아서 뺄셈의 결과를 리턴하는 함수를 작성/테스트
create or replace function func_sub(
    p_x number,
    p_y number
) return number
is
begin
    return p_x - p_y;
end;
/

select func_sub(1, 2), func_sub(2, 1) from dual;

declare
    v_result number;
begin
    v_result := func_sub(123, 100);
    dbms_output.put_line(v_result);
end;
/


-- 문자열 두개를 파라미터로 전달받아서 
-- 두 문자열을 하나로 합친 문자열을 리턴하는 함수를 작성/테스트
-- 'Hello', 'World' -> 'HelloWorld'
create or replace function my_concat(
    str1 varchar2,
    str2 varchar2
) return varchar2
is
--    v_result varchar2(32767);
begin
--    v_result := str1 || str2;
--    return v_result;
    return str1 || str2;
end;
/

select my_concat('Hello', 'World') from dual;

declare
    v_result varchar2(100);
begin
    select my_concat(ename, job) into v_result
    from emp
    where empno = 7788;
    dbms_output.put_line(v_result);
end;
/


-- 사번과 세율을 파라미터로 전달받아서
-- 해당 사원의 총 급여( (sal + comm) * (1 - tax) )를 계산해서 리턴하는 함수
create or replace function func_total_sal(
    p_empno number,
    p_tax number := 0.05
) return number
is
    v_total number;  -- 총 급여를 계산한 결과를 저장할 변수
begin
    select (sal + nvl(comm, 0)) * (1 - p_tax)
    into v_total
    from emp
    where empno = p_empno;
    
    return v_total;
end;
/


select empno, ename, sal, comm, func_total_sal(7788)
from emp
where empno = 7788;

declare
    cursor cur_empno is
        select empno from emp;
    type rec_emp_sal is record (
        empno   emp.empno%type,
        ename   emp.ename%type,
        sal     emp.sal%type,
        comm    emp.comm%type,
        total   number
    );
    v_result rec_emp_sal;
begin
    for r in cur_empno loop
        select empno, ename, sal, comm, func_total_sal(r.empno)
        into v_result
        from emp
        where empno = r.empno;
        
        dbms_output.put_line(v_result.empno || ' : ' 
                        || v_result.ename || ' : '
                        || v_result.sal || ' : '
                        || v_result.comm || ' : '
                        || v_result.total);
    end loop;
end;
/


-- factorial 계산 결과를 리턴하는 함수
-- 0! = 1
-- 1! = 1 x 1 = 1
-- 2! = 1 x 2 = 2
-- 3! = 1 x 2 x 3 = 6

create or replace function my_factorial(n number) return number
is
    v_result number := 1;  -- 계산 결과를 저장할 변수
begin
    for i in 1..n loop
        v_result := v_result * i;
    end loop;

    return v_result;
end;
/

begin
    for n in 0..5 loop
        dbms_output.put_line(my_factorial(n));
    end loop;
end;
/

create or replace function factorial2(n number) return number
is
    v_result number := -1; -- 계산 결과를 저장할 변수
begin
    if n = 0 then
        v_result := 1;
    elsif n > 0 then
        v_result := factorial2(n - 1) * n;
        -- 재귀 호출(recursion): 함수 내부에서 자기 자신을 다시 호출하는 것
    end if;
    
    return v_result;
end;
/

begin
    for n in 0..5 loop
        dbms_output.put_line(factorial2(n));
    end loop;
end;
/

/*
Parameter(매개변수, 인자) vs Argument(인수, 전달인자)
- parameter: 함수, 프로시저를 선언할 때 전달받는 값을 저장하기 위해서 
    선언하는 변수
    create procedure proc_name(param1 number, param2 varchar2, ...)
    create function func_name(param1 number, param2 varchar2, ...) return number
- argument: 함수, 프로시저를 호출할 때 전달하는 값
    proc_name(123, 'abc');
    func_name(100, '가나다');
*/








