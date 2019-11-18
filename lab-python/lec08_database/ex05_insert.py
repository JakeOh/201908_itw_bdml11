"""
ex05_insert.py
"""

import cx_Oracle
import lec08_database.oracle_config as cfg

# 데이터베이스 서버와 연결 설정
with cx_Oracle.connect(cfg.user, cfg.pwd, cfg.dsn) as connection:
    # SQL 문장 실행, 결과 분석할 수 있는 cursor 객체 생성
    with connection.cursor() as cursor:
        sql_insert = "insert into dept2 values(91, '강의장10', 'Seoul')"
        cursor.execute(sql_insert)
        # DML(Data Manipulation Language): insert, update, delete
        # 결과를 영구적으로 반영하기 위해서는 commit을 해야 함.
        connection.commit()

        sql_select = 'select * from dept2'
        cursor.execute(sql_select)
        for row in cursor:
            print(row)
