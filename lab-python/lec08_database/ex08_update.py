"""
ex08_update.py

dept2 테이블에서
부서 번호를 입력받아서, 해당 부서의 위치를 update 실행
"""

import cx_Oracle
import lec08_database.oracle_config as cfg

with cx_Oracle.connect(cfg.user, cfg.pwd, cfg.dsn) as connection:
    with connection.cursor() as cursor:
        deptno = int(input('위치를 수정할 부서 번호>> '))
        loc = input('수정할 위치>> ')

        sql_update = 'update dept2 set loc = :loc where deptno = :deptno'
        cursor.execute(sql_update,
                       loc=loc,
                       deptno=deptno)
        connection.commit()
