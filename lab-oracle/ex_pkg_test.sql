-- ex_pkg.spl에서 작성한 my_stat 패키지의 함수들을 테스트
set serveroutput on;

declare
    v_score my_stat.NumberArray;
begin
    for i in 1..10 loop
        v_score(i) := trunc(dbms_random.value(0, 11));
    end loop;
    
    for i in 1..10 loop
        dbms_output.put(v_score(i) || ' ');
    end loop;
    dbms_output.new_line;
    
    dbms_output.put_line('sum: ' || my_stat.sum(v_score));
    dbms_output.put_line('avg: ' || my_stat.avg(v_score));
    dbms_output.put_line('var: ' || my_stat.var(v_score));
    dbms_output.put_line('std: ' || my_stat.std(v_score));
    dbms_output.put_line('max: ' || my_stat.max(v_score));
    dbms_output.put_line('min: ' || my_stat.min(v_score));
end;
/


