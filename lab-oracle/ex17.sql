-- ex 1. customers 테이블
create table customers1 (
    customer_id     varchar2(20) 
            constraint pk1 primary key 
            constraint ch1 check ( lengthb(customer_id) >= 8 ),
    customer_pw     varchar2(20)
            constraint nn1 not null 
            constraint ck2 check ( lengthb(customer_pw) >= 8 ),
    customer_email  varchar2(100),
    customer_gender number(1, 0)
            constraint ch3 check ( customer_gender in (0, 1, 2)),
    customer_age    number(3, 0) default 0
            constraint ck4 check (customer_age between 0 and 200)
);

create table customers2 (
     customer_id     varchar2(20),
     customer_pw     varchar2(20) constraint nn_pw not null,
     customer_email  varchar2(100),
     customer_gender number(1, 0),
     customer_age    number(3, 0) default 0,
     constraint pk_customers primary key (customer_id),
     constraint ck_id check( lengthb(customer_id) >= 8 ),
     constraint ck_pw check( lengthb(customer_pw) >= 8 ),
     constraint ck_gender check( customer_gender in (0, 1, 2) ),
     constraint ck_age check( customer_age between 0 and 200 )
);

-- ex 2. orders
create table orders (
    order_id number(12)
            constraint pk_orders primary key,
    order_date date default sysdate,
    order_method varchar2(8)
            constraint ck_method check( order_method in ('direct', 'online') ),
    customer_id number(6),
    order_status number(2) default 0,
    sales_rep_id number(6)
);


-- ex 3. order_items
create table order_items (
    order_id number(12),
    product_id number(10),
    quantity number(4) default 0,
    price number(10),
    constraint pk_order_items primary key(order_id, product_id)
);

insert into customers1(customer_id, customer_pw)
values('admin0000', 'admin1234');

insert into customers1(customer_id, customer_pw, customer_gender)
values('########', 'pw123456', 2.5);

select * from customers1;

-- customers1 테이블에서 성별의 값이 없는 고객의 성별을 2로 변경
update customers1
set customer_gender = 2
where customer_gender is null;


insert into orders(order_id, customer_id)
values(100001, 123456);

select * from orders;

-- orders 테이블에서 sales_rep_id가 null인 곳을 100으로 수정
update orders
set sales_rep_id = 100
where sales_rep_id is null;

-- order_items 테이블에 레코드 insert
insert into order_items(order_id, product_id, quantity, price)
values(100000, 1, 2, 1000);

insert into order_items(order_id, product_id, quantity, price)
values(100000, 2, 3, 1500);

insert into order_items(order_id, product_id, quantity, price)
values(100001, 123, 10, 10000);

select * from order_items;

-- orders, order_items 테이블에서 
-- 주문 번호, 주문 날짜, 상품 아이디, 상품 수량, 상품 가격을 조회
select o.order_id, o.order_date, i.product_id, i.quantity, i.price
from orders o join order_items i
on o.order_id = i.order_id;

commit;

-- DML(Data Manipulation Language): insert, update, delete
-- commit을 명시적으로 수행한 경우에 데이터베이스에 영구히 저장
-- rollback을 사용해서 최종 commit 상태로 되돌릴 수도 있음.
select * from order_items;
delete from order_items;
rollback;
select * from order_items;


-- DDL(Data Definition Language): create, alter, truncate, drop
-- DDL 문장은 수행 후 자동으로 commit 됨.
truncate table order_items;
select * from order_items;
rollback;
select * from order_items;

drop table ex_emp2;
rollback;
select * from ex_emp2;




