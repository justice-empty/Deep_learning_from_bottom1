import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# 단순한 그래프 그릭
## 데이터 준비
x = np.arange(0, 6, 0.1) # 0부터 6미만까지 0.1 간격으로 배열 생성
y = np.sin(x)
## 그래프 그리기
plt.plot(x, y)
plt.show()

# pyplot의 기능
## 데이터 준비
x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)
## 그래프 그리기
plt.plot(x, y1, label='sin')
plt.plot(x, y2, linestyle='--', label='cos') # cos함수는 점선으로 그리기
plt.xlabel("x") # x축 이름
plt.ylabel("y") # y축 이름
plt.title('sin & cos') # 그래프 제목
plt.legend() # 그래프 범례
plt.show()

# 이미지 표시하기
img = imread('C:/Users/user/Desktop/코딩/My_coding_study/밑바닥부터 시작하는 딥러닝1/chapter1_hello_python/lena.png')

plt.imshow(img)
plt.show()