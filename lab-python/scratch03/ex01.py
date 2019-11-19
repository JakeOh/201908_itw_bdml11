"""
matplotlib.pyplot 모듈을 사용한 데이터 시각화
"""

# plotting을 위한 패키지 임포트
import matplotlib.pyplot as plt

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

# plot: 선 그래프 생성
plt.plot(years, gdp, color='green', marker='o')
plt.title('연도별 GDP', fontname='D2Coding')
plt.xlabel('Year')
plt.ylabel('GDP(Billions of $)')

# savefig: 그래프를 이미지 파일로 저장
plt.savefig('../image_output/year_gdp.png')

# show: 그래프를 화면에 보여줌
plt.show()


# 막대 그래프(bar chart)
# 영화 제목
movies = ['Annie Hall', 'Ben-Hur', 'Casablanca', 'Gandhi', 'West Side Story']
# 아카데미 시상식에서 받은 상의 갯수
num_oscars = [5, 11, 3, 8, 10]

plt.bar(movies, num_oscars)
font_name = {'fontname': 'D2Coding'}
plt.title('아카데미 수상작', font_name)
plt.ylabel('수상 갯수', font_name)
plt.show()


# 히스토그램(histogram): 정해진 구간에 해당하는 항목의 개수
grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]

from collections import Counter  # 파이썬 기본 패키지
# histogram = Counter(grades)
histogram = Counter(min(grade // 10 * 10, 90) for grade in grades)
print(histogram)  # 각 점수대별 시험 성적 개수
plt.bar(histogram.keys(), histogram.values())
plt.yticks([0, 1, 2, 3, 4, 5])
plt.show()

plt.hist(grades, edgecolor='black')
plt.yticks([0, 1, 2, 3, 4, 5])  # y축 눈금 표시
plt.show()

mentions = [500, 505]
years = [2013, 2014]
plt.bar(years, mentions, 0.5)
plt.xticks(years)  # x축 눈금 표시
plt.show()
