import numpy as np
import matplotlib.pylab as plt

# 계단 함수 구현
def step_function(x):
    y = x > 0
    return y.astype(np.int) # 자료형 변환을 하지 않으면 y값이 불리언 값으로 생성됨

"""
# 계단 함수 그래프
def step_function(x):
    return np.array(x > 0, dtype=np.int64) # x가 0보다 크다면 숫자형으로 배열을 리턴

x = np.arange(-5.0, 5.0, 0.1) # -5부터 5미만 까지 0.1의 간격으로 배열 생성
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축 범위 지정
plt.show()
"""

# 시그모이드 함수 구현
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])

"""
# 시그모이드 함수 그래프
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축 범위 지정
plt.show()
"""

# ReLU 함수 구현
def relu(x):
    return np.maximum(0, x) # maximum은 두 입력 중 큰 값을 선택하여 반환하는 함수

"""
# ReLU 함수 그래프
x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-0.1, 5) # y축 범위 지정
plt.show()
"""

# 소프트맥스 함수 구현
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

"""
위의 구현은 오버플로의 문제가 발생할 수 있는 함수 구현이다.
지수 함수가 큰 값을 반환하여 큰 값끼리 나눗셈을 하게 되면 결과 수치가 불안정해진다.
"""

# 소프트맥스 함수 구현(개선)
def softmax(a):
    c = np.max(a) # a로 받게될 값 중 최댓값을 c로 할당
    exp_a = np.exp(a - c) # 최댓값에서 a값을 뺴주어 수의 크기를 줄여주어 오버플로 문제를 방지
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y