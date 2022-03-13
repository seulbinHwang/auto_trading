from api.FinanceDataReader import *
from universe import comprehensive_dual_momentum_universe
import datetime


universe_dict = comprehensive_dual_momentum_universe.get_universe()
strategy_name = 'comprehensive_dual_momentum_back_test'
finance_data_reader = FinanceDataReader()
now = datetime.datetime.now().strftime("%Y%m%d")
finance_data_reader.check_and_get_universe(strategy_name, universe_dict, now, only_abroad=False)
sql = "select * from 'universe'"
cur = execute_sql(strategy_name, sql)
db_universe_dict = cur.fetchall()  # fetchall: select 문의 결과 객체를 이용하여, 조회 결과 확인 가능
universe= {}
for idx, item in enumerate(db_universe_dict):
    _, code, code_name, country, category, percent, created_at, abs_momentum, rel_momentum, have, have_percent, momentum_month = item
    universe[code] = {
        'code_name': code_name,
        'country': country,
        'category': category,
        'percent': percent,
        'created_at': created_at,
        'abs_momentum': abs_momentum,
        'rel_momentum': rel_momentum,
        'have': have,
        'have_percent': have_percent,
        'momentum_month': momentum_month
    }
for idx, code in enumerate(universe.keys()):
    universe = finance_data_reader.check_and_get_price_data(strategy_name, universe, code, duration_year=30, close_only=True)
