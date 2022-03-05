from api.FinanceDataReader import *
from universe import comprehensive_dual_momentum_universe
from universe import chore_universe
from util.db_helper import *
from util.time_helper import *
from util.notifier import *
from util.crawling import *
import math
import traceback
from threading import Thread
import copy
import numpy as np

class ComprehensiveDualMomentumSrategy():
    def __init__(self, additional_investment_amount, use_kiwoom=False, add_chore_universe=False):
        self.additional_investment_amount = additional_investment_amount
        self.add_chore_universe = add_chore_universe
        self.use_kiwoom = use_kiwoom
        self.strategy_name = "ComprehensiveDualMomentumStrategy"
        nowDatetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        send_message('[전략 코드 시작] \n\n\n' + self.strategy_name + ' 현재 시각' + nowDatetime, LINE_MESSAGE_TOKEN)
        if self.use_kiwoom:
            self.kiwoom = Kiwoom()
        self.finance_data_reader = FinanceDataReader()


        # 유니버스 정보를 담을 딕셔너리
        self.universe = {}

        # 계좌 예수금
        self.deposit = 0

        # 초기화 함수 성공 여부 확인 변수
        self.is_init_success = False

        self.init_strategy()

    def init_strategy(self):
        """
        - Objectives
            - 전략 초기화 기능을 수행하는 함수
                - 유니버스가 없으면 생성하고, 있으면 가져오기
                - 잔고에는 어떤 종목들이 있는가?
        """
        try:
            # 유니버스 조회, 없으면 생성
            self.check_and_get_universe()

            # # 가격 정보를 조회, 필요하면 생성
            self.check_and_get_price_data()
            #
            # Kiwoom > 주문정보 확인 (다음날 장 거래 전까지 유효) -> self.kiwoom.order 을 사용해서 접근할 수 있습니다.
            if self.use_kiwoom:
                self.kiwoom.get_order()
                # Kiwoom > 잔고 확인 -> self.kiwoom.balance 을 사용해서 접근 가능
                self.kiwoom.get_balance()
                # Kiwoom > 예수금 확인
                self.deposit = self.kiwoom.get_deposit()

                # 유니버스 실시간 체결정보 등록
                self.set_universe_real_time()

            self.is_init_success = True

        except Exception as e:
            print(traceback.format_exc())
            # LINE 메시지를 보내는 부분
            send_message(traceback.format_exc(), LINE_MESSAGE_TOKEN)

    def check_and_get_universe(self):
        """유니버스가 존재하는지 확인하고 없으면 생성하는 함수"""
        # ComprehensiveDualMomentumStrategy db 내에 'universe' 테이블이 없으면
        if not check_table_exist(self.strategy_name, 'universe'):
            main_universe_dict = comprehensive_dual_momentum_universe.get_universe()
            universe_dict = copy.deepcopy(main_universe_dict)
            if self.add_chore_universe:
                chore_universe_dict = chore_universe.get_universe()
                for key in universe_dict.keys():
                    universe_dict[key] += chore_universe_dict[key]
            print(universe_dict)
            # 오늘 날짜를 20210101 형태로 지정
            now = datetime.now().strftime("%Y%m%d")
            if self.use_kiwoom:
                self.kiwoom.check_and_get_universe(self.strategy_name, universe_dict, now)
                self.finance_data_reader.check_and_get_universe(self.strategy_name, universe_dict, now, only_abroad=True)
            # FinanceDataReader 로 부터 code(Symbol) 찾기
            else:
                self.finance_data_reader.check_and_get_universe(self.strategy_name, universe_dict, now, only_abroad=False)
            # universe 테이블에서 모든 것을 select 하자.
        sql = "select * from 'universe'"
        cur = execute_sql(self.strategy_name, sql)
        db_universe_dict = cur.fetchall() # fetchall: select 문의 결과 객체를 이용하여, 조회 결과 확인 가능
        for idx, item in enumerate(db_universe_dict):
            _, code, code_name, country, category, percent, created_at, abs_momentum, rel_momentum, have, have_percent, momentum_month = item
            self.universe[code] = {
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
        # send_message('[UNIVERSE LIST] \n\n\n' + str(self.universe), LINE_MESSAGE_TOKEN)

    def check_and_get_price_data(self):
        """일봉 데이터가 존재하는지 확인하고 없다면 생성하는 함수"""
        for idx, code in enumerate(self.universe.keys()):
            print("({}/{}) {}".format(idx + 1, len(self.universe), code))
            if self.use_kiwoom:
                self.universe = self.kiwoom.check_and_get_price_data(self.strategy_name, self.universe, code)
                if self.universe[code]['country'] == 'abroad':
                    self.universe = self.finance_data_reader.check_and_get_price_data(self.strategy_name, self.universe, code)
            else:
                self.universe = self.finance_data_reader.check_and_get_price_data(self.strategy_name, self.universe, code)

    def run(self):
        if self.use_kiwoom:
            self.kiwoom_thread = Thread(target=self.run_kiwoom, args=())
            self.kiwoom_thread.start()
        else:
            self.run_finance_data_reader()
            # self.finance_data_reader_thread = Thread(target=self.run_finance_data_reader, args=())
            # self.finance_data_reader_thread.start()

    def run_finance_data_reader(self):
        """
        :return:

        Objectives
            - universe 내 모든 종목의 현재 가격을 crawling 으로 알아냅니다. ('key' = current_price)
            - 절대 모멘텀
                - 지난 n 개월 전 (n개월 - 15 ~ n개월 + 15) 의 평균 가격을 도출 합니다.
                - (현재 가격 / 과거 가격) 을 저장합니다. ('key' = abs_momentum)
            - 상대 모멘텀
                - 'key' = category 가 같은 종목 중
                    - 'key' = abs_momentum 이 1 이상인 것들 중, 가장 큰 값의 종목을 찾아서
                        - 'key' = rel_momentum 값을 1, 그렇지 않은 종목은 0 으로 저장합니다.
                            - line 메신저로 사야할 종목과 팔아야 할 종목을 보냅니다.
        """
        momentum_month = 3
        momentum_buy_max_num = [1, 2, 2]
        if self.is_init_success:
            category_percent = [0, 0, 0]
            momentum_day = int(momentum_month * (365 / 12))
            try:
                # universe 내 모든 종목의 현재 가격을 crawling 으로 알아냅니다. ('key' = current_price)
                pos_abs_momentum_dicts = [{}, {}, {}]
                for idx, code in enumerate(self.universe.keys()):
                    sql = "update universe set momentum_month=:momentum_month where code=:code"
                    execute_sql(self.strategy_name, sql, {"momentum_month": momentum_month, "code": code})

                    price_df = self.universe[code]['price_df']['Close'].copy()
                    if self.universe[code]['country'] == 'korea':
                        self.universe[code]['current_price'] = float(get_realtime_price_korea(code))
                    else:
                        self.universe[code]['current_price'] = float(price_df[0])
                    # 절대 모멘텀
                    past_average_price = price_df[momentum_day -15: momentum_day + 14].mean()
                    self.universe[code]['abs_momentum'] = np.round(self.universe[code]['current_price'] / past_average_price, 2)
                    sql = "update universe set abs_momentum=:abs_momentum where code=:code"
                    execute_sql(self.strategy_name, sql, {"abs_momentum": self.universe[code]['abs_momentum'], "code": code})

                    # 상대 모멘텀
                    if self.universe[code]['category'] == '주식':
                        category_idx = 0
                    elif self.universe[code]['category'] == '채권':
                        category_idx = 1
                    elif self.universe[code]['category'] == '실물자산':
                        category_idx = 2
                    category_percent[category_idx] += self.universe[code]['percent']
                    if self.universe[code]['abs_momentum'] >= 1:
                        pos_abs_momentum_dicts[category_idx][code] = self.universe[code]['abs_momentum']
                # 상대 모멘텀
                for category_idx in range(len(pos_abs_momentum_dicts)):
                    have_num = 0
                    pos_abs_momentum_dicts[category_idx] = dict(sorted(pos_abs_momentum_dicts[category_idx].items(), key=lambda x: x[1]))
                    pos_abs_moment_code_list = list(pos_abs_momentum_dicts[category_idx].keys())
                    buy_num = min(len(pos_abs_moment_code_list), momentum_buy_max_num[category_idx])
                    if buy_num > 0:
                        have_percent = category_percent[category_idx] / buy_num
                    else:
                        have_percent = 0
                    for temp_idx, code in enumerate(pos_abs_moment_code_list):
                        self.universe[code]['rel_momentum'] = temp_idx
                        if (len(pos_abs_moment_code_list) - temp_idx) <= momentum_buy_max_num[category_idx]: # 3 , 2, 1 < 2
                            self.universe[code]['have'] = 1
                            self.universe[code]['have_percent'] = have_percent
                            have_num += 1
                        else:
                            self.universe[code]['have'] = 0
                            self.universe[code]['have_percent'] = 0

                        
                for idx, code in enumerate(self.universe.keys()):
                    sql = "select have from '{}' where code='{}'".format('universe', code)
                    cur = execute_sql(self.strategy_name, sql)
                    prev_have = cur.fetchone()[0] # int
                    if self.universe[code]['have'] > prev_have:
                        send_message('[매매 고고!!!] \n\n\n' + str(self.universe[code]['code_name']), LINE_MESSAGE_TOKEN)
                    elif self.universe[code]['have'] < prev_have:
                        send_message('[매수 고고!!!] \n\n\n' + str(self.universe[code]['code_name']), LINE_MESSAGE_TOKEN)
                    sql = "update universe set have=:have where code=:code"
                    execute_sql(self.strategy_name, sql,
                                {"have": self.universe[code]['have'], "code": code})
                    sql = "update universe set have_percent=:have_percent where code=:code"
                    execute_sql(self.strategy_name, sql,
                                {"have_percent": self.universe[code]['have_percent'], "code": code})
                    sql = "update universe set rel_momentum=:rel_momentum where code=:code"
                    execute_sql(self.strategy_name, sql,
                                {"rel_momentum": self.universe[code]['rel_momentum'], "code": code})
                for category in ['주식', '채권', '실물자산']:
                    for idx, code in enumerate(self.universe.keys()):
                        if self.universe[code]['category'] == category:
                            universe_data_for_log = dict((i, self.universe[code][i]) for i in self.universe[code] if i != 'price_df')
                            send_message('[UNIVERSE 상태:{}]\n\n\n'.format(category) + str(universe_data_for_log), LINE_MESSAGE_TOKEN)
                            additional_investment_amount_indiv = np.round(self.additional_investment_amount * universe_data_for_log['have_percent'])
                            if additional_investment_amount_indiv > 0:
                                send_message('[새로운 투자 금액!!!!!!! 상태:{} --> 구매금액/새 투자 금액]\n\n\n'.format(universe_data_for_log['code_name']) + str(additional_investment_amount_indiv) + '/' + str(self.additional_investment_amount), LINE_MESSAGE_TOKEN)

            except Exception as e:
                print(traceback.format_exc())
                # LINE 메시지를 보내는 부분
                send_message(traceback.format_exc(), LINE_MESSAGE_TOKEN)
        else:
            send_message('전략 인스턴스 초기화가 잘못 되었습니다!!', LINE_MESSAGE_TOKEN)

            # if check_transaction_open_usa():

    def run_kiwoom(self):
        """실질적 수행 역할을 하는 함수"""
        while self.is_init_success:
            try:
                # (0)장중인지 확인
                if not check_transaction_open():
                    # TODO: 8시 59분에 돌리면 9시 4분까지 코드가 멈추므로, 9시부터 9시 4분까지 매매 기회를 놓치게 된다. 수정하고 싶으면 하자.
                    print("장시간이 아니므로 5분간 대기합니다.")
                    time.sleep(5 * 60)
                    continue

                for idx, code in enumerate(self.universe.keys()):
                    # 종목 명과 전체 유니버스 중 몇 번째인지 나타내는 코드
                    print('[{}/{}_{}]'.format(idx + 1, len(self.universe), self.universe[code]['code_name']))
                    time.sleep(0.5)

                    # (1)접수한 주문이 있는지 확인
                    if code in self.kiwoom.order.keys():
                        # (2)주문이 있음
                        print('접수 주문', self.kiwoom.order[code])

                        # (2.1) '미체결수량' 확인하여 미체결 종목인지 확인
                        if self.kiwoom.order[code]['미체결수량'] > 0:
                            pass

                    # (3)보유 종목인지 확인
                    elif code in self.kiwoom.balance.keys():
                        print('보유 종목', self.kiwoom.balance[code])
                        # (6)매도 대상 확인
                        if self.check_sell_signal(code):
                            # (7)매도 대상이면 매도 주문 접수
                            self.order_sell(code)

                    else:
                        # (4)접수 주문 및 보유 종목이 아니라면 매수대상인지 확인 후 주문접수
                        self.check_buy_signal_and_order(code)

            except Exception as e:
                print(traceback.format_exc())
                # LINE 메시지를 보내는 부분
                send_message(traceback.format_exc(), LINE_MESSAGE_TOKEN)

    def set_universe_real_time(self):
        """유니버스 실시간 체결정보 수신 등록하는 함수"""
        # 임의의 fid를 하나 전달하기 위한 코드(아무 값의 fid라도 하나 이상 전달해야 정보를 얻어올 수 있음)
        # '체결 시간'만 전달해도, 다른 값들 전부 받아올 수 있습니다.
        fids = get_fid("체결시간")

        # 장운영구분을 확인하고 싶으면 사용할 코드
        # self.kiwoom.set_real_reg("1000", "", get_fid("장운영구분"), "0")

        # universe 딕셔너리의 key값들은 종목코드들을 의미
        codes = self.universe.keys()

        # 종목코드들을 ';'을 기준으로 묶어주는 작업
        codes = ";".join(map(str, codes))

        # 화면번호 9999에 종목코드들의 실시간 체결정보 수신을 요청 -> self.kiwoom.universe_realtime_transaction_info 에 저장
        self.kiwoom.set_real_reg("9999", codes, fids, "0")

    def check_sell_signal(self, code):
        """매도대상인지 확인하는 함수"""
        universe_item = self.universe[code] # universe_item 는 딕셔너리이며, 'code_name' 과 'price_df' 라는 키를 가짐

        # (1)현재 체결정보가 존재하지 않는지 확인
        if code not in self.kiwoom.universe_realtime_transaction_info.keys():
            # 체결 정보가 없으면 더 이상 진행하지 않고 함수 종료
            print("매도대상 확인 과정에서 아직 체결정보가 없습니다.")
            return

        # (2)실시간 체결 정보가 존재하면 현시점의 시가 / 고가 / 저가 / 현재가 / 누적 거래량이 저장되어 있음
        open = self.kiwoom.universe_realtime_transaction_info[code]['시가']
        high = self.kiwoom.universe_realtime_transaction_info[code]['고가']
        low = self.kiwoom.universe_realtime_transaction_info[code]['저가']
        close = self.kiwoom.universe_realtime_transaction_info[code]['현재가']
        volume = self.kiwoom.universe_realtime_transaction_info[code]['누적거래량']
        df = universe_item['price_df'].copy()

        # 옿늘 가격 데이터 추가!
        # 오늘 가격 데이터를 과거 가격 데이터(DataFrame)의 행으로 추가하기 위해 리스트로 만듦
        today_price_data = [open, high, low, close, volume]
        # 과거 가격 데이터에 금일 날짜로 데이터 추가
        df.loc[datetime.now().strftime('%Y%m%d')] = today_price_data

        # RSI(N) 계산
        period = 2  # 기준일 설정
        date_index = df.index.astype('str')
        # df.diff를 통해 (기준일 종가 - 기준일 전일 종가)를 계산하여 0보다 크면 증가분을 넣고, 감소했으면 0을 넣어줌
        U = np.where(df['close'].diff(1) > 0, df['close'].diff(1), 0)
        # df.diff를 통해 (기준일 종가 - 기준일 전일 종가)를 계산하여 0보다 작으면 감소분을 넣고, 증가했으면 0을 넣어줌
        D = np.where(df['close'].diff(1) < 0, df['close'].diff(1) * (-1), 0)
        AU = pd.DataFrame(U, index=date_index).rolling(window=period).mean()  # AU, period=2일 동안의 U의 평균
        AD = pd.DataFrame(D, index=date_index).rolling(window=period).mean()  # AD, period=2일 동안의 D의 평균
        RSI = AU / (AD + AU) * 100  # 0부터 1로 표현되는 RSI에 100을 곱함
        df['RSI(2)'] = RSI

        # 보유 종목의 매입가격 조회
        purchase_price = self.kiwoom.balance[code]['매입가']
        # 금일의 RSI(2) 구하기
        rsi = df[-1:]['RSI(2)'].values[0]

        # 매도 조건 두 가지를 모두 만족하면 True
        if rsi > 80 and close > purchase_price:
            return True
        else:
            return False

    def order_sell(self, code):
        """매도 주문 접수 함수"""
        # 보유 수량 확인(전량 매도 방식으로 보유한 수량을 모두 매도함)
        quantity = self.kiwoom.balance[code]['보유수량']

        # 최우선 매도 호가 확인
        ask = self.kiwoom.universe_realtime_transaction_info[code]['(최우선)매도호가']

        order_result = self.kiwoom.send_order('send_sell_order', '1001', 2, code, quantity, ask, '00')

        # LINE 메시지를 보내는 부분
        message = "[{}]sell order is done! quantity:{}, ask:{}, order_result:{}".format(code, quantity, ask,
                                                                                        order_result)
        send_message(message, LINE_MESSAGE_TOKEN)

    def check_buy_signal_and_order(self, code):
        """매수 대상인지 확인하고 주문을 접수하는 함수"""
        # 매수 가능 시간 확인 -> 매수는 15시가 넘은 장 종료 시점부터 하려고 의도
        if not check_adjacent_transaction_closed():
            return False

        # (1)현재 체결정보가 존재하지 않는지 확인
        if code not in self.kiwoom.universe_realtime_transaction_info.keys():
            # 존재하지 않다면 더이상 진행하지 않고 함수 종료
            print("매수대상 확인 과정에서 아직 체결정보가 없습니다.")
            return

        # (2)실시간 체결 정보가 존재하면 현 시점의 시가 / 고가 / 저가 / 현재가 / 누적 거래량이 저장되어 있음
        open = self.kiwoom.universe_realtime_transaction_info[code]['시가']
        high = self.kiwoom.universe_realtime_transaction_info[code]['고가']
        low = self.kiwoom.universe_realtime_transaction_info[code]['저가']
        close = self.kiwoom.universe_realtime_transaction_info[code]['현재가']
        volume = self.kiwoom.universe_realtime_transaction_info[code]['누적거래량']

        # 오늘 가격 데이터를 과거 가격 데이터(DataFrame)의 행으로 추가하기 위해 리스트로 만듦
        today_price_data = [open, high, low, close, volume]

        universe_item = self.universe[code]
        df = universe_item['price_df'].copy()

        # 과거 가격 데이터에 금일 날짜로 데이터 추가
        df.loc[datetime.now().strftime('%Y%m%d')] = today_price_data

        # RSI(N) 계산
        period = 2  # 기준일 설정
        date_index = df.index.astype('str')
        # df.diff를 통해 (기준일 종가 - 기준일 전일 종가)를 계산하여 0보다 크면 증가분을 넣고, 감소했으면 0을 넣어줌
        U = np.where(df['close'].diff(1) > 0, df['close'].diff(1), 0)
        # df.diff를 통해 (기준일 종가 - 기준일 전일 종가)를 계산하여 0보다 작으면 감소분을 넣고, 증가했으면 0을 넣어줌
        D = np.where(df['close'].diff(1) < 0, df['close'].diff(1) * (-1), 0)
        AU = pd.DataFrame(U, index=date_index).rolling(window=period).mean()  # AU, period=2일 동안의 U의 평균
        AD = pd.DataFrame(D, index=date_index).rolling(window=period).mean()  # AD, period=2일 동안의 D의 평균
        RSI = AU / (AD + AU) * 100  # 0부터 1로 표현되는 RSI에 100을 곱함
        df['RSI(2)'] = RSI

        # 종가(close)를 기준으로 이동 평균 구하기
        df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['ma60'] = df['close'].rolling(window=60, min_periods=1).mean()

        rsi = df[-1:]['RSI(2)'].values[0]
        ma20 = df[-1:]['ma20'].values[0]
        ma60 = df[-1:]['ma60'].values[0]

        # 2 거래일 전 날짜(index)를 구함
        idx = df.index.get_loc(datetime.now().strftime('%Y%m%d')) - 2

        # 위 index로부터 2 거래일 전 종가를 얻어옴
        close_2days_ago = df.iloc[idx]['close']

        # 2 거래일 전 종가와 현재가를 비교함
        price_diff = (close - close_2days_ago) / close_2days_ago * 100

        # (3)매수 신호 확인(조건에 부합하면 주문 접수)
        if ma20 > ma60 and rsi < 5 and price_diff < -2:
            # (4)이미 보유한 종목, 매수 주문 접수한 종목의 합이 보유 가능 최대치(10개)라면 더 이상 매수 불가하므로 종료
            if (self.get_balance_count() + self.get_buy_order_count()) >= 10:
                return

            # (5)주문에 사용할 금액 계산(10은 최대 보유 종목 수로써 consts.py에 상수로 만들어 관리하는 것도 좋음)
            budget = self.deposit / (10 - (self.get_balance_count() + self.get_buy_order_count()))

            # 최우선 매도호가 확인
            bid = self.kiwoom.universe_realtime_transaction_info[code]['(최우선)매수호가']

            # (6)주문 수량 계산(소수점은 제거하기 위해 버림)
            quantity = math.floor(budget / bid)

            # (7)주문 주식 수량이 1 미만이라면 매수 불가하므로 체크
            if quantity < 1:
                return

            # (8)현재 예수금에서 수수료를 곱한 실제 투입금액(주문 수량 * 주문 가격)을 제외해서 계산
            amount = quantity * bid
            self.deposit = math.floor(self.deposit - amount * 1.00015)

            # (8)예수금이 0보다 작아질 정도로 주문할 수는 없으므로 체크
            if self.deposit < 0:
                return

            # (9)계산을 바탕으로 지정가 매수 주문 접수
            order_result = self.kiwoom.send_order('send_buy_order', '1001', 1, code, quantity, bid, '00')

            # _on_chejan_slot가 늦게 동작할 수도 있기 때문에 미리 약간의 정보를 넣어둠
            self.kiwoom.order[code] = {'주문구분': '매수', '미체결수량': quantity}

            # LINE 메시지를 보내는 부분
            message = "[{}]buy order is done! quantity:{}, bid:{}, order_result:{}, deposit:{}, get_balance_count:{}, get_buy_order_count:{}, balance_len:{}".format(
                code, quantity, bid, order_result, self.deposit, self.get_balance_count(), self.get_buy_order_count(),
                len(self.kiwoom.balance))
            send_message(message, LINE_MESSAGE_TOKEN)

        # 매수신호가 없다면 종료
        else:
            return

    def get_balance_count(self):
        """ 매도 주문이 접수되지 않은 보유 종목 수를 계산하는 함수"""
        balance_count = len(self.kiwoom.balance)
        # kiwoom balance에 존재하는 종목이 매도 주문 접수되었다면 보유 종목에서 제외시킴
        for code in self.kiwoom.order.keys():
            if code in self.kiwoom.balance and self.kiwoom.order[code]['주문구분'] == "매도" and self.kiwoom.order[code]['미체결수량'] == 0:
                balance_count = balance_count - 1
        return balance_count

    def get_buy_order_count(self):
        """매수 주문 종목 수를 계산하는 함수"""
        buy_order_count = 0
        # 아직 체결이 완료되지 않은 매수 주문
        for code in self.kiwoom.order.keys():
            if code not in self.kiwoom.balance and self.kiwoom.order[code]['주문구분'] == "매수":
                buy_order_count = buy_order_count + 1
        return buy_order_count

# class ComprehensiveDualMomentumSrategyWKiwoom(QThread, ComprehensiveDualMomentumSrategy):
#     from api.Kiwoom import Kiwoom
#     from api.KiwoomWorld import KiwoomWorld
#     def __init__(self):
#         QThread.__init__(self)
#         ComprehensiveDualMomentumSrategy.__init__(self, use_kiwoom=True)
