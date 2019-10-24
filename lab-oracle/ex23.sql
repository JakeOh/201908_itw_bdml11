set SERVEROUTPUT ON;

accept p_name prompt '이름 입력하세요...';
begin
    dbms_output.put_line('&p_name');
end;
/

-- 이름을 입력 받아서
-- 사번, 부서번호를 검색해서
-- 사번, 이름, 부서번호를 출력
accept p_name prompt '이름 입력하세요...';
declare
    v_empno number(4); -- 사번을 저장할 변수
    v_deptno number(2); -- 부서 번호를 저장할 변수
    v_ename varchar2(100) := upper('&p_name'); -- 입력받은 이름(문자열)을 저장한 변수 
begin
    select empno, deptno into v_empno, v_deptno
    from emp
    where upper(ename) = v_ename;
    
    dbms_output.put_line('사번: ' || v_empno);
    dbms_output.put_line('이름: ' || v_ename);
    dbms_output.put_line('부서 번호: ' || v_deptno);
end;
/


-- 이름을 입력 받아서
-- emp, dept 테이블을 사용해서(JOIN)
-- 'SCOTT은 DALLAS의 RESEARCH부서에서 근무합니다.'와 같은 형식으로 출력
accept p_name prompt '이름 입력하세요...';
declare
    v_ename varchar2(100) := '&p_name';  -- 입력받은 이름을 저장하는 변수
    v_loc varchar2(13);  -- 부서 위치(loc)를 저장할 변수
    v_dname varchar2(14);  -- 부서 이름을 저장할 변수
begin
    select d.dname, d.loc into v_dname, v_loc
    from emp e , dept d
    where e.deptno = d.deptno
            and upper(e.ename) = upper(trim(v_ename));
    
    dbms_output.put_line(v_ename || '은(는) ' ||
                        v_loc || '의 ' ||
                        v_dname || '부서에서 근무합니다.');
end;
/

-- 입력된 이름이 대/소문자에 상관없이 결과를 출력할 수 있도록

accept p_name prompt '이름 입력하세요...';
declare
    v_ename varchar2(20) := upper(trim('&p_name'));
    v_dname varchar2(14);
    v_loc varchar2(13);
    v_deptno number(2);
begin
    -- 사원 이름으로 그 사원이 근무하는 부서 번호를 검색
    select deptno into v_deptno from emp where upper(ename) = v_ename;
    -- 부서 번호를 사용해서 부서의 이름, 위치를 검색
    select dname, loc into v_dname, v_loc from dept where deptno = v_deptno;
    -- 검색한 정보를 화면 출력
    dbms_output.put_line(v_ename || ' : ' || v_loc || ' : ' || v_dname );
end;
/








