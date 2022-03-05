from bs4 import BeautifulSoup
import requests

# BASE_URL = 'https://finance.naver.com/sise/sise_market_sum.nhn?sosok='
BASE_KOREA_URL = 'https://finance.naver.com/item/main.naver?code='
BASE_ABROAD_URL = 'https://finance.yahoo.com/quote/'
# https://finance.yahoo.com/quote/EDV?p=EDV&.tsrc=fin-srch
BASE_URLS = [BASE_KOREA_URL, BASE_ABROAD_URL]
def get_realtime_price_korea(code):
    # TODO: 해외 데이터 크롤링
    URL = BASE_KOREA_URL + str(code)
    res = requests.get(URL)
    bs_obj = BeautifulSoup(res.content, "html.parser")
    no_today = bs_obj.find("p", {"class": "no_today"})
    blind = no_today.find("span", {"class": "blind"})
    now_price = blind.text
    now_price = now_price.replace(',', '')
    return now_price
