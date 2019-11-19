import matplotlib.pyplot as plt

# 선 그래프(line chart)
x = [i for i in range(10)]
print(x)

y1 = [2 ** x for x in range(10)]
print(y1)

plt.plot(x, y1, 'go-', label='example 1')
# g: 선 색깔(green), o: 마커,  -: 선 스타일

y2 = [2 ** x for x in range(9, -1, -1)]
print(y2)
plt.plot(x, y2, 'r--', label='example 2')

y3 = [x + y for x, y in zip(y1, y2)]
print(y3)
plt.plot(x, y3, 'b:', label='example 3')

plt.legend(loc='best')  # plot의 label을 화면에 표시
plt.title('Line Chart Example')

plt.show()
