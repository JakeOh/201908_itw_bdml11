-- 1)
select empno, 
        rpad(trunc(empno, -2) / 100, 4, '*') as mask_no,
        ename,
        rpad(substr(ename, 1, 1), 5, '*') as mask_name
from emp
where length(ename) = 5;

select empno, 
        rpad(substr(empno, 1, 2), 5, '*') as mask_no,
        ename,
        rpad(substr(ename, 1, 1), 5, '*') as mask_name
from emp
where length(ename) = 5;

select substr(sysdate, 1, 3) from dual;

-- 2)
select empno, ename, sal, 
        trunc(sal / 21.5, 2) as day_sal,
        round(sal / (21.5 * 8), 1) as hour_sal
from emp;

-- 3)
select sysdate, next_day(sysdate, '±Ý') from dual;

select empno, ename, 
        to_char(hiredate, 'YYYY-MM-DD') as hiredate,
        to_char(next_day(add_months(hiredate, 3), '¿ù'), 'YYYY-MM-DD') as regular,
        to_char(next_day(add_months(hiredate, 3)-1, '¿ù'), 'YYYY-MM-DD') as regular2
from emp;
