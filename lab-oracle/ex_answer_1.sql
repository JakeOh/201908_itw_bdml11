set serveroutput on;

-- ex 1.
declare
    x number;
    y number;
begin
    loop
        x := trunc(dbms_random.value(1, 7));
        y := trunc(dbms_random.value(1, 7));
        dbms_output.put_line('(' || x || ', ' || y || ')');
        
        exit when x = y;
    end loop;
end;
/


-- ex 2.
declare
    type NumberArray is table of number index by pls_integer;
    cnt constant number := 20;
    fibo NumberArray;
    ratio NumberArray;
begin
    fibo(1) := 1;
    fibo(2) := 2;
    for i in 3..cnt loop
        fibo(i) := fibo(i - 1) + fibo(i - 2);
    end loop;
    
    for i in 1..cnt loop
        dbms_output.put(fibo(i) || ' ');
    end loop;
    dbms_output.new_line;
    
    for i in 1..(cnt - 1) loop
        ratio(i) := fibo(i + 1) / fibo(i);
        dbms_output.put_line(ratio(i));
    end loop;
end;
/


-- ex 3.
declare
    type CharArray is varray(3) of varchar2(1);
    alphas CharArray := CharArray('a', 'b', 'c');
    cnt number;
begin
    dbms_output.put_line('(1) 중복 순열');
    cnt := 1;
    for i in 1..alphas.count loop
        for j in 1..alphas.count loop
            dbms_output.put_line(cnt || ': ' || alphas(i) || alphas(j));
            cnt := cnt + 1;
        end loop;
    end loop;
    
    dbms_output.new_line;
    dbms_output.put_line('(2) 순열');
    cnt := 1;
    for i in 1..alphas.count loop
        for j in 1..alphas.count loop
            continue when j = i;
            dbms_output.put_line(cnt || ': ' || alphas(i) || alphas(j));
            cnt := cnt + 1;
        end loop;
    end loop;
    
    dbms_output.new_line;
    dbms_output.put_line('(3) 조합');
    cnt := 1;
    for i in 1..alphas.count loop
        for j in (i + 1)..alphas.count loop
            dbms_output.put_line(cnt || ': ' || alphas(i) || alphas(j));
            cnt := cnt + 1;
        end loop;
    end loop;
    
    dbms_output.new_line;
    dbms_output.put_line('(4) 중복 조합');
    cnt := 1;
    for i in 1..alphas.count loop
        for j in i..alphas.count loop
            dbms_output.put_line(cnt || ': ' || alphas(i) || alphas(j));
            cnt := cnt + 1;
        end loop;
    end loop;
end;
/


-- ex 4.
declare
    type CharArray is varray(5) of varchar2(1);
    alphas CharArray := CharArray('a', 'b', 'c', 'd', 'e');
    cnt number;
begin
    dbms_output.put_line('(1) 중복 순열');
    cnt := 1;
    for i in 1..alphas.count loop
        for j in 1..alphas.count loop
            for k in 1..alphas.count loop
                dbms_output.put_line(cnt || ': ' || alphas(i) || alphas(j) || alphas(k));
                cnt := cnt + 1;
            end loop;
        end loop;
    end loop;
    
    dbms_output.new_line;
    dbms_output.put_line('(2) 순열');
    cnt := 1;
    for i in 1..alphas.count loop
        for j in 1..alphas.count loop
            continue when j = i;
            for k in 1..alphas.count loop
                continue when k = j or k = i;
                dbms_output.put_line(cnt || ': ' || alphas(i) || alphas(j) || alphas(k));
                cnt := cnt + 1;
            end loop;
        end loop;
    end loop;
    
    dbms_output.new_line;
    dbms_output.put_line('(3) 조합');
    cnt := 1;
    for i in 1..alphas.count loop
        for j in (i + 1)..alphas.count loop
            for k in (j + 1)..alphas.count loop
                dbms_output.put_line(cnt || ': ' || alphas(i) || alphas(j) || alphas(k));
                cnt := cnt + 1;
            end loop;
        end loop;
    end loop;
    
    dbms_output.new_line;
    dbms_output.put_line('(4) 중복 조합');
    cnt := 1;
    for i in 1..alphas.count loop
        for j in i..alphas.count loop
            for k in j..alphas.count loop
                dbms_output.put_line(cnt || ': ' || alphas(i) || alphas(j) || alphas(k));
            cnt := cnt + 1;
            end loop;
        end loop;
    end loop;
end;
/

