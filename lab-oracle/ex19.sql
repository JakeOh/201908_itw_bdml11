-- Session, Transaction, Lock
-- SQL 종류
-- DML(Data Manipluation Language): 데이터 조작 언어
--      insert, update, delete
-- DDL(Data Definition Language): 데이터 정의 언어
--      create, alter, truncate, drop
-- DQL(Data Query Language): 데이터 쿼리(질의) 언어. select
-- TCL(Transaction Control Language): 트랜잭션 관리 언어. commit, rollback

select * from dept2;

update dept2 
set loc = '서울Seoul'
where deptno = 50;
select * from dept2;

commit;

insert into dept2
values (90, 'itwill', 'gangnam');
select * from dept2;

update dept2 set loc = '뉴욕' where deptno = 10;

commit;
