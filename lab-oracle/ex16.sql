-- 레코드 수정/삭제
select * from ex_emp3;

-- ex_emp3 테이블에서 사번이 7369인 사원의 급여를 2000으로 변경
update ex_emp3
set sal = 2000
where empno = 7369;

-- update 문장에서 where 절을 생략하면 테이블의 모든 행을 수정하게 됨!
update ex_emp3 set sal = 1234;

rollback;
-- 바로 이전의 commit 상태(최종 commit 상태)로 데이터들을 되돌리는 명령어


-- ex_emp3 테이블에서 부서번호가 10번인 직원들의 레코드(행)를 삭제
delete from ex_emp3 where deptno = 10;

-- 비교
update ex_emp3 set deptno = null where empno = 7369;
select * from ex_emp3;

commit;
-- delete 구문에서 where 절을 사용하지 않으면 테이블의 모든 행이 삭제됨!

-- ex_emp3 테이블에서 20번 부서에서 일하는 직원들의 수당(comm)을 100으로 수정
update ex_emp3
set comm = 100
where deptno = 20;

select * from ex_emp3;

-- ex_emp3 테이블에서 급여가 평균보다 작은 직원들의 급여를 2배로 인상
select avg(sal) from ex_emp3;
update ex_emp3
set sal = sal * 2
where sal < (
    select avg(sal) from ex_emp3
);










