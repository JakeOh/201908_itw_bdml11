-- 제약 조건 2
/*
1.
create table 테이블이름 (
    컬럼이름1 데이터타입 constraint 제약조건이름 제약조건내용, 
    컬럼이름2 데이터타입 constraint 제약조건이름 제약조건내용,
    ...
);


2.
create table 테이블이름 (
    컬럼이름1 데이터타입,
    컬럼이름2 데이터타입,
    ...,
    constraint 제약조건이름 제약조건내용,
    constraint 제약조건이름 제약조건내용,
    ...
);

(주의) not null 제약 조건은 1번 방법만 가능
*/

create table ex08 (
    col1 number,
    col2 number,
    constraint uq_ex8 unique (col2),
    constraint pk_ex08 primary key (col1)
);

create table ex09 (
    col1 varchar2(10),
    constraint nn_ex09 not null (col1)
);

create table ex_emp2 (
    empno number constraint pk_ex_emp2 primary key,
    ename varchar2(20),
    deptno number(2),
    constraint fk_ex_dept2 foreign key (deptno) references ex_dept (deptno)
);


-- 하나의 테이블에서 두개 이상의 컬럼을 묶어서 PK 지정
create table ex09 (
    col1 number primary key,
    col2 varchar2(20) primary key
);  -- 테이블은 오직 한개의 PK만 가질 수 있기 때문에 오류 발생

create table ex09 (
    col1 number,
    col2 varchar2(20),
    constraint pk_ex09 primary key (col1, col2)
);
insert into ex09 values(1, 'a');
insert into ex09 values(1, 'b');
insert into ex09 values(2, 'a');
insert into ex09 values(100, null);
select * from ex09;

insert into ex_dept values(30, '인사부');
-- insert ~ select 구문:
-- 다른 테이블에서 select한 내용을 테이블에 insert하는 방법
insert into ex_emp2 (empno, ename, deptno)
select empno, ename, deptno from emp;

select * from ex_emp2;
select e.empno, e.ename, e.deptno, d.dname
from ex_emp2 e join ex_dept d
        on e.deptno = d.deptno;
        
-- 테이블 복사: 테이블을 생성할 때 다른 테이블의 구조와 데이트를 그대로 가져오는 방법.
-- create ~ as select 구문
create table ex_emp3
as select * from emp;

select * from ex_emp3;

create table ex_emp4
as select * from emp where deptno = 10;

select * from ex_emp4;

select * from emp where 1 != 1;
create table ex_emp5
as select * from emp where 1 != 1;
select * from ex_emp5;














