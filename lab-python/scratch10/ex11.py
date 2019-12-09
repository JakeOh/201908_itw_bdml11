"""
웹 주소(URI, URL)의 형식:
    프로토콜://서버주소[:포트번호]/경로?쿼리스트링
    http://www.naver.com
    https://comic.naver.com/webtoon/list.nhn?titleId=183559&weekday=mon
쿼리 스트링(query string): 클라이언트(브라우저)가 서버로 보내는 정보
    param이름=param값 형식으로 작성
    파라미터가 여러개일 경우에는 &로 파라미터들을 구분

다음에서 "머신 러닝"으로 검색한 기사 100개의 URL 주소와 기사 제목을 출력
다음에서 임의의 검색어(키워드)로 검색한 기사 100개의 URL 주소와 기사 제목을 출력하는
함수 작성하고 테스트
"""
import requests
from bs4 import BeautifulSoup


def daum_search(keyword):
    url = 'https://search.daum.net/search?w=news&DA=PGD&spacing=0'
    # 검색 결과는 1페이지부터 10페이지 까지
    for page in range(1, 11):
        print(f'=== Page {page} ===')
        req_params = {
            'q': keyword,  # 검색어(키워드)를 쿼리 스트링에 파라미터로 추가
            'p': page  # 검색 페이지 번호를 쿼리 스트링에 파라미터로 추가
        }
        response = requests.get(url, params=req_params)
        html = response.text.strip()

        soup = BeautifulSoup(html, 'html5lib')
        news_links = soup.select('.coll_cont ul li a.f_link_b')
        for link in news_links:
            news_url = link.get('href')
            news_title = link.text
            print(news_url, news_title)


if __name__ == '__main__':
    daum_search('시동')

if False:
    # url = 'https://search.daum.net/search?w=news&q=%EB%A8%B8%EC%8B%A0%20%EB%9F%AC%EB%8B%9D&DA=PGD&spacing=0&p='
    url = 'https://search.daum.net/search?w=news&q=%EB%A8%B8%EC%8B%A0%20%EB%9F%AC%EB%8B%9D&DA=PGD&spacing=0'
    for page in range(1, 11):
        print(f'=== Page {page} ===')
        # page_url = url + str(page)
        # print(page_url)

        # 해당 페이지 URL 주소로 GET 방식 요청(request)을 보내고,
        # 서버에서 보낸 응답(response)을 문자열로 처리
        # html = requests.get(page_url).text.strip()
        response = requests.get(url, params={'p': page})
        # requests.get(url, params={key: value}):
        #   params의 내용을 url의 query string의 파라미터로 추가해줌
        #   url?...&key=value
        html = response.text.strip()

        # BeautifulSoup 객체 생성
        soup = BeautifulSoup(html, 'html5lib')
        news_links = soup.select('.coll_cont ul li a.f_link_b')
        for link in news_links:
            # HTML 요소(element)의 href 속성(attribute) 값을 읽음.
            news_url = link.get('href')
            # HTML 요소가 가지고 있는 컨텐트 문자열
            news_title = link.text
            print(news_url, news_title)
