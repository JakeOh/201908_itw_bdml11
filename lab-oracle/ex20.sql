-- 테이블 수정
-- 테이블의 컬럼 이름 변경
alter table ex02 rename column ex_id to eid;

-- 테이블의 컬럼 데이터 타입 변경
alter table ex02 modify ex_text varchar2(100);

-- 테이블에 새로운 컬럼 추가
alter table ex02 add ex_date date;

-- 테이블에서 컬럼 삭제
alter table ex02 drop column ex_date;

-- 테이블 컬럼에 제약 조건 추가
alter table ex02
add constraint pk_ex02 primary key (eid);
-- 컬럼에 null 또는 중복된 값이 있는 경우는 PK 지정할 수 없음.

select * from ex02;

update ex02 set eid = 2 where eid is null;

-- null 가능 컬럼에 not null 제약 조건 추가
alter table ex02
add constraint nn_ex02 check (ex_text is not null);

-- 컬럼에 지정된 제약 조건 삭제 
alter table ex02
drop constraint nn_ex02;

desc ex02;

-- 테이블 vs 사용자 계정
-- 사용자 계정 create/alter/drop은 system 계정에서 수행
create table ex_test (...);
create user scott2 identified by tiger;

alter table ex_test
rename column a to b;
alter user scott 
identified by tiger account unlock;

drop table ex_test;
drop user scott2;


-- 1.
create table hw_emp
as (select * from emp where 1 != 1);

select * from hw_emp;

-- 2.
alter table hw_emp add etc varchar2(20);
desc hw_emp;

-- 3.
alter table hw_emp modify etc varchar2(100);

-- 4.
alter table hw_emp rename column etc to remark;

-- 5.
insert into hw_emp(empno, ename, job, mgr, hiredate, sal, comm, deptno)
select * from emp;

select * from hw_emp;

-- 6.
alter table hw_emp add constraint pk_hw_emp primary key (empno);

-- 7.
drop table hw_emp;


-- sequence: 어떤 규칙에 맞는 연속된 숫자들을 자동으로 생성해 주는 객체
-- 시퀀스(sequence)의 목적
select max(empno) from emp;

create sequence seq1;
select seq1.nextval from dual;

insert into ex02(eid, ex_text)
values(seq1.nextval, 'aabbcc');

select * from ex02;





