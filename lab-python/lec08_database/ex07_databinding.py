"""
ex07_databinding.py

data bindig을 사용한 SQL 문장 실행
"""

import cx_Oracle
import lec08_database.oracle_config as cfg

with cx_Oracle.connect(cfg.user, cfg.pwd, cfg.dsn) as connection:
    with connection.cursor() as cursor:
        deptno = int(input('부서 번호 입력>> '))
        dname = input('부서 이름 입력>> ')
        loc = input('부서 위치 입력>> ')

        # sql1 = 'insert into dept2 values(:0, :1, :2)'
        # cursor.execute(sql1, [deptno, dname, loc])

        sql2 = 'insert into dept2 values(:dept_no, :dept_name, :loc)'
        cursor.execute(sql2,
                       dept_no=deptno,
                       dept_name=dname,
                       loc=loc)

        connection.commit()
