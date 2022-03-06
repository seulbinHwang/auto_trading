import sqlite3


def check_table_exist(db_name, table_name):
    """
    - Objectives
        - db_name에 table_name이 있는지 확인합니다.
        - 예시
            - db_name: RSIStrategy / ComprehensiveDualMomentumStrategy
            - talbe_name: universe
    """
    with sqlite3.connect('{}.db'.format(db_name)) as con:
        cur = con.cursor()
        # sqlite_master 라는 테이블은, 데이터베이스가 처음 생성되면 자동으로 생성되는 메타데이터 테이블이다. (db의 테이블 정보가 자동으로 저장된다.)
        sql = "SELECT name FROM sqlite_master WHERE type='table' and name=:table_name"
        cur.execute(sql, {"table_name": table_name})

        if len(cur.fetchall()) > 0:
            return True
        else:
            return False

def check_column_exist(db_name, table_name, column_name, value):
    """
    - Objectives
        - db_name에 table_name이 있는지 확인합니다.
        - 예시
            - db_name: RSIStrategy / ComprehensiveDualMomentumStrategy
            - talbe_name: universe
    """
    # sqlite_master 라는 테이블은, 데이터베이스가 처음 생성되면 자동으로 생성되는 메타데이터 테이블이다. (db의 테이블 정보가 자동으로 저장된다.)
    sql = "select `{}` from '{}' where `{}`='{}'".format(column_name, table_name, column_name, value)
    cur = execute_sql(db_name, sql)
    output = cur.fetchone()
    if output is None:
        return False
    else:
        return True

def insert_df_to_db(db_name, table_name, df, option="replace"):
    """
    - Objectives
        - db_name에 table_name에 해당 dataframe(df)를 입력합니다.
        - 예시
            - db_name: RSIStrategy
            - talbe_name: universe
    """
    with sqlite3.connect('{}.db'.format(db_name)) as con: # con: database 연결 객체
       df.to_sql(table_name, con, if_exists=option)


def execute_sql(db_name, sql, param={}):
    """
        sql = "select * from universe"
    """
    with sqlite3.connect('{}.db'.format(db_name)) as con:
       cur = con.cursor()
       cur.execute(sql, param)
       return cur


if __name__ == "__main__":
    pass