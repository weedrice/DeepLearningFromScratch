import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)

A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
print(A * B)

print(X[0])
print(X[0][1])

for row in X:
    print(row)

newX = X.flatten()
print(newX)

print(X > 15)

# 데이터 준비
x = np.arange(0, 6, 0.1)  # 0에서 6까지 0.1 간격으로 생
y = np.sin(x)

# 그래프 그리기
plt.plot(x, y)
plt.show()

# 데이터 준비
y1 = np.sin(x)
y2 = np.cos(x)

# 그래프 그리기
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos")  # cos 함수는 점선으로 그리기
plt.xlabel("x")  # x축 이름
plt.ylabel("y")  # y축 이름
plt.title('sin & cos')  # 제목
plt.legend()
plt.show()

# 이미지 그리기
img = imread('/Users/jiwon/Downloads/R800x0.png') #이미지 읽어오기

plt.imshow(img)
plt.show()