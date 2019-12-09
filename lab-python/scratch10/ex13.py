"""
icrawler 패키지를 사용해서, Google 이미지 검색 결과의 이미지들을 다운로드
> pip install icrawler
"""
import os
from icrawler.builtin import GoogleImageCrawler

# 이미지 저장 경로
save_dir = os.path.join('C:' + os.sep, 'dev', 'images')

# GoogleImageCrawler 객체 생성
google_crawler = GoogleImageCrawler(storage={'root_dir': save_dir})

# 검색 필터링(filter) 조건들
filters = {
    'size': 'large',
    'license': 'noncommercial,modify',
    'color': 'blackandwhite'
}

# 이미지 다운로드
google_crawler.crawl(keyword='cat', filters=filters, max_num=20)




