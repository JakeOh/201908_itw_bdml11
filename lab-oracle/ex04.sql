-- hr 계정 사용
desc employees;

select employee_id, first_name, last_name
from employees
where first_name like 'J%';

select employee_id, first_name, last_name
from employees
where first_name like '%E%' or first_name like '%e%';

select employee_id, first_name, phone_number
from employees
where phone_number like '011%';

select employee_id, first_name, manager_id, department_id
from employees
where manager_id = 120;

select * from employees where employee_id = 120;

select employee_id, first_name, salary
from employees
where salary between 3000 and 5000;

select employee_id, first_name, salary
from employees
where salary >= 3000 and salary <= 5000;

select employee_id, first_name, salary, commission_pct, 
       (salary * 12) * (1 + commission_pct) as ann_sal
from employees
where commission_pct is not null;