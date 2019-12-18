import pandas as pd

dataset = pd.read_csv('nb_test.csv', sep='\t')
print(dataset)

# 외출할 확률: P(go_out)
p_go_out = 5 / 10

# 집에 머물러 있을 확률: P(stay-home)
p_stay_home = 5 / 10  # P(stay-home) = 1 - P(go-out)

# 외출했을 때 날씨가 맑을 확률: P(sunny|go-out)
p_sunny_when_go_out = 4 / 5

# 외출했을 때 비가 올 확률: P(rainy|go-out)
p_rainy_when_go_out = 1 / 5

# 외출했을 때 자동차가 정상일 확률: P(working|go-out)
p_working_when_go_out = 4 / 5

# 외출했을 때 자동차가 고장일 확률: P(broken|go-out)
p_broken_when_go_out = 1 / 5

# 방콕일 때 맑을 확률: P(sunny|stay-home)
p_sunny_when_stay_home = 2 / 5

# 방콕일 때 비가 올 확률: P(rainy|stay-home)
p_rainy_when_stay_home = 3 / 5

# 방콕일 때 자동차가 정상인 경우: P(working|stay-home)
p_working_when_stay_home = 1 / 5

# 방콕일 때 자동차가 고장인 경우: P(broken|stay-home)
p_broken_when_stay_home = 4 / 5

# 위쪽까지는 과거의 데이터를 기반으로 이미 알고 있는 확률을 계산: 사전 확률
# 만약 날씨가 맑고, 자동차가 정상이면 외출할 확률? 방콕 확률?
# P(go-out|sunny,working) = ?, P(stay-home|sunny,working) = ?

# P(A|B) = P(A,B) / P(B): 조건부 확률의 정의
# P(B|A) = P(B,A) / P(A) => P(B,A) = P(B|A) * P(A)
# P(B,A) = P(A,B) 이기 때문에,
# P(A|B) = P(B|A) * P(A) / P(B): 베이즈 정리(Bayes' theorem)

# P(X|A,B) = P(A,B|X) * P(X) / P(A,B)
# naive(순진한) 가정: P(A,B|X) = P(A|X) * P(B|X)
# P(X|A,B) ~ P(A|X) * P(B|X) * P(X): Naive 베이즈 정리

# 날씨가 맑고 차가 정상일 때 외출할 확률:
# P(go-out|sunny,working) ~ P(sunny|go-out) * P(working|go-out) * P(go-out)
p_go_out_when_sunny_working = p_sunny_when_go_out * \
                              p_working_when_go_out * \
                              p_go_out
print('P(go-out|sunny,working) ~', p_go_out_when_sunny_working)

# 날씨가 맑고 차가 정상일 때 방콕 확률:
# P(stay-home|sunny,working) ~ P(sunny|stay-home) * P(working|stay-home) * P(stay-home)
p_stay_home_when_sunny_working = p_sunny_when_stay_home * \
                                 p_working_when_stay_home * \
                                 p_stay_home
print('P(stay-home|sunny,working) ~', p_stay_home_when_sunny_working)

# 날씨가 맑고 차가 정상일 때 외출/방콕 확률을 각각 계산하고,
# 둘 중 더 큰 값으로 예측을 한다 -> NB 머신 러닝

# 비가 오고 차가 고장일 때 외출/방콕 예측
# P(go-out|rainy,broken) = ?, P(stay-home|rainy,broken) = ?
p_go_out_when_rainy_broken = p_rainy_when_go_out * \
                             p_broken_when_go_out * \
                             p_go_out
print('P(go-out|rainy,broken) ~', p_go_out_when_rainy_broken)
p_stay_home_when_rainy_broken = p_rainy_when_stay_home * \
                                p_broken_when_stay_home * \
                                p_stay_home
print('P(stay-home|rainy,broken) ~', p_stay_home_when_rainy_broken)
