import pandas as pd

from scratch10.ex02 import peak_to_peak

if __name__ == '__main__':
    # tips.csv 파일을 읽어서 데이터 프레임 생성
    tips = pd.read_csv('tips.csv')

    # 앞 5개의 데이터 출력
    print(tips.iloc[0:5])
    print(tips.shape)

    # DataFrame에 tip_pct 컬럼 추가: 팁금액 / 총금액
    tips['tip_pct'] = tips['tip'] / tips['total_bill']
    print(tips.iloc[0:5])

    # day, smoker별 그룹을 지어서,
    grouped = tips.groupby(['day', 'smoker'])
    grouped_tip_pct = grouped['tip_pct']

    # tip_pct의 평균을 출력
    # print(grouped_tip_pct.mean())
    print(grouped_tip_pct.agg('mean'))

    # day, smoker별 그룹의 tip_pct 컬럼의 평균, 표준편차, 최대/최소 차이를 출력
    print(grouped_tip_pct.agg(['mean', 'std', peak_to_peak]))

    # day, smoker별 그룹의 tip_pct, total_bill 컬럼의 평균, 표준편차, 최대/최소 차이
    grouped_pct_bill = grouped[['tip_pct', 'total_bill']]
    print(grouped_pct_bill.agg([('average', 'mean'),
                                ('std-dev', 'std'),
                                ('range', peak_to_peak)]))

    # GroupBy 객체의 컬럼들마다 다른 함수를 agg로 적용할 때
    # agg({'col_name': [functions], ...})
    # 그룹핑된 데이터 프레임의 tip 컬럼에는 max() 함수를 aggregate하고,
    # size 컬럼에는 sum() 함수를 aggregate함.
    result = grouped.agg({'tip': 'max', 'size': 'max'})
    print(result)

    functions = [('mu', 'mean'), ('sigma', 'std'), ('range', peak_to_peak)]
    result = grouped.agg({
        'tip_pct': functions,
        'total_bill': functions
    })
    print(result)

    # grouping 컬럼들이 aggregate 결과에서 인덱스로 사용하지 않고자 할 때,
    grouped = tips.groupby(['day', 'smoker'], as_index=False)
    print(grouped['tip'].mean())

