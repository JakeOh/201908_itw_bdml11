-- 제약 조건(constraint)
-- 접속한 사용자 계정의 테이블들에 적용된 제약조건 확인
select constraint_name, constraint_type, table_name
from user_constraints;

select table_name from user_tables;

-- 제약 조건
-- not null: 반드시 값이 있어야 함
-- unique: 중복된 값이 저장될 수 없음
-- primary key: 고유키. 테이블에서 유일한 한개의 행(레코드)를 검색할 수 있는 컬럼
-- foreign key: 외래키. 외부키. 관계를 맺고 있는 다른 테이블의 PK
-- check: 조건(condition)을 체크하는 제약 조건.
-- default: 컬럼의 기본값(insert하지 않아도 자동으로 저장되는 값)을 설정

-- 테이블 이름: ex03
-- 컬럼: col1 - number, unique
--       col2 - varchar2(20), not null
create table ex03 (
    col1 number unique,
    col2 varchar2(20) not null
);

desc ex03;

insert into ex03 (col1, col2)
values (1, 'aaa');  --  insert 성공

insert into ex03 (col1, col2) 
values (1, 'bbb'); -- unique 제약 조건 위배: insert 실패

insert into ex03 (col2)
values ('bbb');
select * from ex03;

insert into ex03 (col1)
values (10);
commit;


-- 테이블 생성 시 제약 조건에 이름을 주는 방법
create table ex04 (
    col1 number constraint ex04_unq unique,
    col2 varchar2(20) constraint ex04_nn not null
);


-- 테이블 이름:  ex05
-- 컬럼: col1 - number, primary key
--       col2 - varchar2(10)
create table ex05 (
    col1 number constraint pk_ex05 primary key,
    col2 varchar2(10)
);
desc ex05;

insert into ex05
values(1, 'abc');
select * from ex05;

insert into ex05 (col2)
values ('def');
commit;


-- 테이블 이름: ex06
-- 컬럼: col - varchar2(20), 제약 조건은 삽입되는 문자열의 길이가 8 이상
create table ex06 (
    col varchar2(20) constraint ck_ex06 check(length(col) >= 8)
);
desc ex06;
insert into ex06 values('abcdefgh');
insert into ex06 values('abcd');

-- 테이블 이름: ex07
-- 컬럼: ex_id - number, 기본값 0
--       ex_date - date, 기본값 현재시간
create table ex07 (
    ex_id   number default 0,
    ex_date date default sysdate
);
desc ex07;
insert into ex07
values (1, to_date('2019-09-13', 'yyyy-mm-dd'));
select * from ex07;

insert into ex07 (ex_id)
values (100);
insert into ex07 (ex_date)
values (sysdate + 1);

commit;


create table ex_dept (
    deptno  number(2) constraint pk_ex_dept primary key,
    dname   varchar2(20)
);

create table ex_emp (
    empno   number(4) constraint pk_ex_emp primary key,
    ename   varchar2(20),
    deptno  number(2) constraint fk_ex_dept references ex_dept(deptno)
);

-- ex_dept 테이블에 데이터가 없는 경우 (부서 번호가 1개도 없는 경우)
insert into ex_emp
values (1001, '오쌤', 10);
-- ex_emp에는 ex_dept 테이블에 생성되어 있지 않은 부서 번호는 insert할 수 없다!
-- FK가 설정된 ex_emp 테이블의 deptno 컬럼에는 null은 가능
insert into ex_emp (empno, ename)
values (1001, '오쌤');

insert into ex_dept values(10, '개발팀');
insert into ex_dept values(20, '분석팀');

insert into ex_emp values(2001, '홍길동', 10);
insert into ex_emp values(3001, 'scott', 20);

select * from ex_emp;
select * from ex_dept;
commit;

