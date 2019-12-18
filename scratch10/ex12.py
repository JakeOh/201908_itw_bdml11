"""
한겨레 신문사 페이지에서, 특정 검색어 검색 결과 50개의
1) URL 주소, 기사 제목, 등록 시간 출력
2) URL 주소, 기사 제목, 등록 시간, 기사 내용 출력
3) URL 주소, 기사 제목, 등록 시간, 기사 내용을 파일에 저장
"""
import requests
from bs4 import BeautifulSoup


def hani_search(keyword):
    # 한겨레 신문사 검색 URL
    url = 'http://search.hani.co.kr/Search?command=query&media=news&sort=d&period=all'
    for page in range(5):
        print(f'\n=== page {page} ===')
        # 쿼리 스트링(query string, 질의 문자열)의 파라미터 설정
        req_params = {
            'keyword': keyword,  # 검색어
            'pageseq': page  # 페이지 번호
        }
        # 서버로 요청(request)를 보낸 후, 응답(response)를 받음.
        response = requests.get(url, params=req_params)
        # 응답에서 HTML 문서를 추출
        html = response.text.strip()
        # HTML 문서를 분석하기 위한 BeautifulSoup 객체 생성
        soup = BeautifulSoup(html, 'html5lib')
        results = soup.select('ul.search-result-list li dt a')
        for link in results:
            news_url = link.get('href')  # 검색 결과 뉴스 링크 URL
            news_title = link.text  # 검색 결과 뉴스 제목
            print(news_url, news_title)
            hani_article(news_url)  # 검색 결과 뉴스 링크를 새로 열기


def hani_article(url):
    response = requests.get(url)
    html = response.text.strip()
    soup = BeautifulSoup(html, 'html5lib')
    # article = soup.find('div', class_='text').text.strip()
    article = soup.select('div.article-text div.text')[0].text.strip()
    print(article)


if __name__ == '__main__':
    hani_search('머신 러닝')
