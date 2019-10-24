set serveroutput on;

-- 1. 
-- 구구단(2단 ~ 9단)을 출력
declare
    v_dan number := 2;
    v_x number := 1;
begin
    loop
        v_x := 1;
        loop
            dbms_output.put_line(v_dan || ' x ' || v_x || ' = ' || (v_dan * v_x));
            v_x := v_x + 1;
            exit when v_x = 10;
        end loop;
        
        dbms_output.put_line('---------------');
        v_dan := v_dan + 1;
        exit when v_dan = 10;
    end loop;
end;
/


declare
    dan number;
    n   number;
begin
    dan := 2;  -- 구구단 2단부터 시작
    while dan < 10 loop
        dbms_output.put_line(dan || '단');
        
        -- 해당 단(2단, 3단, ..., 9단) 출력
        n := 1;
        while n < 10 loop
            dbms_output.put_line(dan || ' x ' || n || ' = ' || (dan * n));
            n := n + 1;
        end loop;
        
        dbms_output.put_line('---------------');
        dan := dan + 1;   
    end loop;
end;
/

begin
    for dan in 2..9 loop
        dbms_output.put_line(dan || '단');
        
        for n in 1..9 loop
            dbms_output.put_line(dan || ' x ' || n || ' = ' || (dan * n));
        end loop;
        
        dbms_output.put_line('---------------');
    end loop;
end;
/


-- 2. 
-- 구구단(2단 ~ 9단)을 출력.
-- 2단은 2x2까지, 3단은 3x3까지, 4단은 4x4까지, ...
begin
    for dan in 2..9 loop
        dbms_output.put_line(dan || '단');
        
        for n in 1..9 loop
            dbms_output.put_line(dan || ' x ' || n || ' = ' || (dan * n));
            exit when dan = n;
        end loop;
        
        dbms_output.put_line('--------------');
    end loop;
end;
/


begin
    for dan in 2..9 loop
        dbms_output.put_line(dan || '단');
        
        for n in 1..dan loop
            dbms_output.put_line(dan || ' x ' || n || ' = ' || (dan * n));
        end loop;
        
        dbms_output.put_line('--------------');
    end loop;
end;
/



-- dbms_random.value(x, y) 함수는 x보다 크거나 같고, y보다 작은 난수(임의의 숫자)를 리턴합니다.
-- x <= r < y
select dbms_random.value(0, 1) from dual;  -- 0 <= r < 0.99999...
select dbms_random.value(0, 2) from dual;  -- 0 <= r < 1.99999...
select trunc(dbms_random.value(0, 2), 0) from dual;  -- r = 0 or 1

-- 3. 무작위로 던진 동전에서 앞면과 뒷면이 나올 확률이 1/2임을 증명하세요.
-- (1) dbms_random.value(x, y) 함수를 사용해서 0 또는 1을 무작위로 출력하는 PL/SQL을 작성하세요.
-- (2) (1)에서 작성한 코드를 수정해서, 
-- 난수가 0이면 '동전 앞면(head)', 1이면 '동전 뒷면(tail)'이라고 출력하는 PL/SQL을 작성하세요.
-- (3) 0 또는 1을 랜덤하게 만드는 과정((1) 코드)을 1000번 반복해서, 0과 1의 발생 횟수를 저장하고,
-- 0이 나온 횟수 / 전체 횟수, 1이 나온 횟수 / 전체 횟수 를 계산
declare
    v_trials constant number := 10000;  -- 실험(반복) 횟수
    v_coin number;  -- 동전의 앞/뒷면(0 또는 1)을 저장할 변수
    -- 동전이 앞면일 때(v_coin = 0) 카운트를 1 증가시킬 변수
    v_head number := 0;  
    -- 동전이 뒷면일 때(v_coin = 1) 카운트를 1 증가시킬 변수
    v_tail number := 0;  
begin
    for i in 1..v_trials loop
        v_coin := trunc(dbms_random.value(0, 2), 0);
        -- dbms_output.put_line(v_coin);
        if v_coin = 0 then
            -- dbms_output.put_line('head');
            v_head := v_head + 1;
        else
            -- dbms_output.put_line('tail');
            v_tail := v_tail + 1;
        end if;
    end loop;
    
    dbms_output.put_line('앞면: ' || v_head || ', 확률=' || (v_head / v_trials));
    dbms_output.put_line('뒷면: ' || v_tail || ', 확률=' || (v_tail / v_trials));
end;
/


-- 4. 몬테카를로 알고리즘을 이용한 pi의 근사값(3.141592...) 계산
-- (1) dbms_random.value() 함수를 사용해서, 2차원 좌표평면(x-y) 위의 점의 좌표 (x, y)를 무작위로 만듦.
-- 단, 0 <= x < 1, 0 <= y < 1
-- (2) sqrt(x ** 2 + y ** 2) < 1을 만족하면 v_hit 변수의 값을 1 증가
-- (3) (1), (2)의 과정을 1000번 반복
-- (4) (v_hit / 1000) * 4를 계산한 값을 출력

declare
    x number;  -- x좌표의 값을 저장할 변수
    y number;  -- y좌표의 값을 저장할 변수
    -- 원점에서의 거리가 1 미만인 경우 1씩 증가시킬 변수
    hit number := 0;
    trials constant number := 10000;
    pi number;
begin
    for i in 1..trials loop
        x := dbms_random.value(0, 1);
        y := dbms_random.value(0, 1);
        -- dbms_output.put_line(x || ', ' || y);
        if sqrt(x ** 2 + y ** 2) < 1 then
            hit := hit + 1;
        end if;
        -- dbms_output.put_line(hit);
    end loop;
    
    dbms_output.put_line('hit = ' || hit);
    pi := (hit / trials) * 4;
    dbms_output.put_line('pi = ' || pi);
end;
/









