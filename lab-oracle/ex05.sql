-- hr 계정 사용

-- 1. employees 테이블에서 사번이 179번인 사원의 레코드를 출력
select * from employees where employee_id = 179;

-- 2. 1에서 찾은 정보를 사용해서 179번 사원이 직책 이름을 다른 테이블에서 검색해서 출력
select job_title from jobs where job_id = 'SA_REP';

-- 3. 1에서 찾은 정보를 사용해서 179번 사원이 일하는 부서의 정보를 다른 테이블에서 검색해서 출력
select * from departments where department_id = 80;

-- 4. 1에서 찾은 정보를 사용해서 179번 사원의 매니저 정보를 검색해서 출력
select * from employees where employee_id = 149;

-- 5. departments 테이블에서 manager_id가 존재하는 레코드들을 출력
select * from departments where manager_id is not null;

-- 6. departments 테이블에서 부서번호가 20번인 부서의 레코드를 출력
select * from departments where department_id = 20;

-- 7. 6에서 찾은 정보를 사용해서 20번 부서의 관리자의 정보를 다른 테이블에서 검색해서 출력
select * from employees where employee_id = 201;

-- 8. 6에서 찾은 정보를 사용해서 20번 부서 위치의 주소 정보를 다른 테이블에서 검색해서 출력
select * from locations where location_id = 1800;
