/*  case ~ when 구문
(1)
case 비교대상
    when 값1 then 
        비교대상 =  값1 만족할 때 실행할 문장;
    when 값2 then 
        비교대상 =  값2 만족할 때 실행할 문장;
    ...
    else 비교대상이 위의 모든 값과 일치하지 않을 때 실행할 문장;
end case;

(2)
case
    when 조건식1 then
        조건식1이 참일 때 실행할 문장;
    when 조건식2 then
        조건식2이 참일 때 실행할 문장;
    ...
    else 위의 모든 조건이 거짓일 때 실행할 문장;
end case;
*/

set serveroutput on;

select trunc(98, -1), trunc(98/10, 0) from dual;

declare
    v_score number := 50;
begin
    case trunc(v_score, -1)
        when 100 then
            dbms_output.put_line('A');
        when 90 then
            dbms_output.put_line('A');
        when 80 then
            dbms_output.put_line('B');
        when 70 then
            dbms_output.put_line('C');
        else
            dbms_output.put_line('D');
    end case;
end;
/


declare
    v_score number := 98;
begin
    case 
        when v_score >= 90 then
            dbms_output.put_line('A');
        when v_score >= 80 then
            dbms_output.put_line('B');
        when v_score >= 70 then
            dbms_output.put_line('C');
        else
            dbms_output.put_line('D');
    end case;
end;
/












