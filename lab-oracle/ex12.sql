-- ex 1
select e.employee_id, e.first_name, 
       e.job_id, j.job_title,
       jh.job_id, j2.job_title,
       jh.start_date, jh.end_date
from employees e, jobs j, job_history jh, jobs j2
where e.job_id = j.job_id
      and
      e.employee_id = jh.employee_id
      and
      jh.job_id = j2.job_id
order by e.employee_id, jh.start_date;

select e.employee_id, e.first_name, e.job_id, j.job_title,
       jh.job_id, j2.job_title,
       jh.start_date, jh.end_date
from employees e join jobs j
        on e.job_id = j.job_id
    join job_history jh
        on e.employee_id = jh.employee_id
    join jobs j2
        on jh.job_id = j2.job_id;
        

-- ex 2.
select d.department_id, d.department_name, d.manager_id,
       e.first_name, e.last_name
from departments d, employees e
where d.manager_id = e.employee_id(+)
order by d.department_id;

select d.department_id, d.department_name, d.manager_id,
       e.first_name, e.last_name
from departments d left join employees e
        on d.manager_id = e.employee_id
order by d.department_id;


-- ex 3. 사번, 이름, 지역 이름
select e.employee_id, e.last_name,
       e.department_id,
       d.location_id,
       l.country_id,
       c.region_id,
       r.region_name
from employees e, departments d, locations l, countries c, regions r
where e.department_id = d.department_id(+)
      and
      d.location_id = l.location_id(+)
      and
      l.country_id = c.country_id(+)
      and 
      c.region_id = r.region_id(+)
order by e.employee_id;

select e.employee_id, e.last_name,
       e.department_id,
       d.location_id,
       l.country_id,
       c.region_id,
       r.region_name
from employees e left join departments d
        on e.department_id = d.department_id
    left join  locations l
        on d.location_id = l.location_id
    left join countries c
        on l.country_id = c.country_id
    left join regions r
        on c.region_id = r.region_id
order by e.employee_id;
      














