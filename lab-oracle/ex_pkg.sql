-- 패키지 명세(specification) 작성
create or replace package my_stat
is
    -- 숫자 타입을 저장하고, 인덱스는 양의 정수인 index-by table을 선언: NumberArray
    type NumberArray is table of number index by pls_integer;
    
    /**
     * sum: 파라미터에 전달받은 배열에 있는 모든 원소들의 합을 리턴
     * @param arr   NumberArray
     */
    function sum(arr NumberArray) return number;
    
    /**
     * avg: 파라미터에 전달받은 배열에 있는 모든 원소들의 평균을 계산해서 리턴
     * @param arr   NumberArray
     * @param digit number  리턴값(평균)의 소숫점 이하 자릿수, 기본값은 3
     */
    function avg(arr NumberArray, digit number := 3) return number;
    
    /**
     * var: 파라미터에 전달받은 배열에 있는 모든 원소들의 분산을 계산해서 리턴
     * @param arr   NumberArray
     * @param digit number  리턴값(분산)의 소숫점 이하 자릿수, 기본값은 3
     */
    function var(arr NumberArray, digit number := 3) return number;
   
    /**
     * std: 파라미터에 전달받은 배열에 있는 모든 원소들의 표준편차를 계산해서 리턴
     * @param arr   NumberArray
     * @param digit number  리턴값(표준편차)의 소숫점 이하 자릿수, 기본값은 3
     */
    function std(arr NumberArray, digit number := 3) return number;
     
    /**
     * max: 파라미터에 전달받은 배열에 있는 모든 원소들 중에서 최댓값을 리턴
     * @param arr   NumberArray
     */
    function max(arr NumberArray) return number;
     
     /**
     * min: 파라미터에 전달받은 배열에 있는 모든 원소들 중에서 최솟값을 리턴
     * @param arr   NumberArray
     */ 
    function min(arr NumberArray) return number;
end;
/

-- 패키지 본문(body) 작성
create or replace package body my_stat
is

    function sum(arr NumberArray) return number
    is
        v_result number;
    begin
        v_result := 0;
        for i in 1..arr.count loop
            v_result := v_result + arr(i);
        end loop;
        
        return v_result;
    end sum;

    function avg(arr NumberArray, digit number := 3) return number
    is
        v_result number;
    begin
        -- 평균 = 합계 / 갯수;
        v_result := my_stat.sum(arr) / arr.count;
        return round(v_result, digit);
    end avg;
    
    function var(arr NumberArray, digit number := 3) return number
    is
        v_avg number;  -- 배열 원소들의 평균
        v_ss number;  -- 제곱 합(sum of squares)
        v_var number;  -- 분산(variance)
    begin
        v_avg := my_stat.avg(arr, digit);
        
        v_ss := 0;
        for i in 1..arr.count loop
            v_ss := v_ss + (v_avg - arr(i)) ** 2;
        end loop;
        
        v_var := v_ss / (arr.count - 1);
        
        return round(v_var, digit);
    end var;
    
    function std(arr NumberArray, digit number := 3) return number
    is
        v_std number;
    begin
        v_std := sqrt(my_stat.var(arr, digit));
        return round(v_std, digit);
    end std;
    
    function max(arr NumberArray) return number
    is
        v_result number;  --  최댓값을 저장할 변수
    begin
        v_result := arr(1);
        for i in 2..arr.count loop
            if arr(i) > v_result then
                v_result := arr(i);
            end if;
        end loop;
        
        return v_result;
    end max;
    
    function min(arr NumberArray) return number
    is
        v_result number;
    begin
        v_result := arr(1);
        for i in 2..arr.count loop
            if arr(i) < v_result then
                v_result := arr(i);
            end if;
        end loop;
        
        return v_result;
    end min;
end;
/
































