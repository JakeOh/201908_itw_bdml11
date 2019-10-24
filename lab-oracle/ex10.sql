-- hr 계정
-- 부서별 급여 최소값
select min(salary) from employees
group by department_id;

select employee_id, first_name, salary, department_id
from employees
where salary in (
    select min(salary) from employees
    group by department_id
);

select department_id, min(salary)
from employees
group by department_id;

select employee_id, first_name, salary, department_id
from employees
where (department_id, salary) in (
    select department_id, min(salary)
from employees
group by department_id
);






