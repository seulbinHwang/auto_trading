from db_helper import *
import pandas as pd
from notifier import *
from const import *
from datetime import datetime
import sys

def put_investment_in_db(invest_amount, date):
    send_message('-------역대 투자 금액 확인해요------- \n\n\n', LINE_MESSAGE_TOKEN)
    if not check_table_exist('investment_db', 'monthly'):
        monthly_df = pd.DataFrame({'year_month': [date], '투자금액': [invest_amount]})
        insert_df_to_db('investment_db', 'monthly', monthly_df)
    else:
        if check_column_exist('investment_db', 'monthly', 'year_month', date):
            sql = "select 투자금액 from '{}' where year_month='{}'".format('monthly', date)
            cur = execute_sql('investment_db', sql)
            prev_invest_amount = cur.fetchone()[0]
            present_invest_amount = prev_invest_amount + invest_amount
            sql = "update 'monthly' set 투자금액=:투자금액 where year_month=:year_month"
            execute_sql('investment_db', sql, {"투자금액": present_invest_amount, "year_month": date})
        else:
            # 해당 달의 데이터가 없으면 추가합니다.
            sql = "insert or ignore into 'monthly' (year_month, 투자금액) values(?, ?)"
            execute_sql('investment_db', sql, (date, invest_amount))
    sql = "SELECT * FROM 'monthly' ORDER BY year_month"
    cur = execute_sql('investment_db', sql)
    send_message('[역대 투자 금액] \n\n\n' + str(cur.fetchall()), LINE_MESSAGE_TOKEN)

    sum_total_invest_amount()

def sum_total_invest_amount():
    sql = "select 투자금액 from '{}'".format('monthly')
    cur = execute_sql('investment_db', sql)
    data = cur.fetchall()
    total_invest_amount = 0
    for row in data:
        total_invest_amount += int(row[0])
    send_message('[역대 투자 총 금액] \n\n\n' + str(total_invest_amount), LINE_MESSAGE_TOKEN)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        additional_investment_amount = int(sys.argv[1])
    else:
        additional_investment_amount = 0
    date = input('년도-달 (예:2022-03), 현재 날짜 기준으로 하고싶으면 now 입력:')
    if date == 'now':
        date = datetime.now().strftime("%Y-%m")
    put_investment_in_db(additional_investment_amount, date)