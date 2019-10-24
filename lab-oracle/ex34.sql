/*
error
(1) 컴파일 에러(compilation error): SQL 또는PL/SQL 문법에 맞지 않아서 
실행 자체가 되지 않는 오류
-> 문법에 틀린 부분을 찾아서 문법에 맞게끔 수정해야만 함
-> 문법에 맞게끔 고치지 않으면 실행이 되지 않기 때문에 exception 처리가 되지 않음

(2) 실행중 에러(runtime error):
문법에 맞게 작성이 되서 컴파일 되고 실행은 되지만, 
실행 중에 어떤 이유로 발생하는 에러
-> 예외(exception) 처리의 대상이 됨.


예외/오류(exception/error) 처리
declare
    -- 선언부: 변수 선언, 타입(record, collection) 선언, 커서 선언
begin
    -- 실행부: 절차적으로 진행할 코드들
exception
    -- 예외처리부: 실행부(begin ~)의 코드들을 실행하다가 오류가 발생했을 때 
    -- 그 예외를 처리하는 부분
    when 예외이름1 then
        예외1가 발생했을 때 처리할 코드;
    when 예외이름2 then
        예외2가 발생했을 때 처리할 코드;
    ...
    when others then
        위에서 선언한 예외들이 아니거나 또는 예외 이름을 모를 때
        처리할 코드들;
end;
*/

set serveroutput on;

declare
    v_result number;
begin
    dbms_output.put_line('실행 전...');
    -- v_result := 'abc';
    -- v_result := 123 / 0;
    -- select ename into v_result from emp where empno = 7788;
    select * into v_result from emp;
    dbms_output.put_line(v_result);
end;
/

-- 예외 처리 구문
declare
    v_result number;
begin
    dbms_output.put_line('예외 발생 전');
    v_result := 'abc';
    dbms_output.put_line(v_result);
exception
    when value_error then
        dbms_output.put_line('숫자가 아닙니다!');
end;
/

declare
    v_result number;
begin
    dbms_output.put_line('예외 발생 전...');
    v_result := 123 / 0;
    dbms_output.put_line(v_result);
--exception
--    when zero_divide then
--        dbms_output.put_line('0으로 나눌 수 없습니다.');
--    when others then
--        dbms_output.put_line('에러 코드: ' || sqlcode);
--        dbms_output.put_line('에러 메시지: ' || sqlerrm);
end;
/


declare
    v_result number;
    v_row emp%rowtype;
begin
    -- select * into v_result from emp where empno = 7788;
    -- PL/SQL compilation error
    -- select ename into v_result from emp where empno = 7788;
    select * into v_row from emp;
    dbms_output.put_line(v_row.ename);
exception
    when value_error then
        dbms_output.put_line('Value Error');
    when too_many_rows then
        dbms_output.put_line('결과 행이 너무 많아요.');
    when others then
        dbms_output.put_line(sqlcode || ', ' || sqlerrm);
end;
/






