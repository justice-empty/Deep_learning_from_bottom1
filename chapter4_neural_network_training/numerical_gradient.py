import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x): # 수치 미분
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

# 수치 미분의 예시와 그래프

def function_1(x):
    return 0.01*x**2 + 0.1*x

print(numerical_diff(function_1, 5)) 
print(numerical_diff(function_1, 10)) # 해석적 미분과 같은 값이라고 할 수 있을만큼 작은 오차의 값

"""
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()
"""

# 편미분의 예시

def function_2(x):
    return x[0]**2 + x[1]**2 # 또는 np.sum(x**2) x는 넘파이 배열이 들어감

# x0=3, x1=4일 때, x0에 대한 편미분
def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

print(numerical_diff(function_tmp1, 3.0))

# x0=3, x1=4일 때, x1에 대한 편미분
def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

print(numerical_diff(function_tmp2, 4.0))

# 기울기 구현
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # x와 같은 형상의 0으로된 배열 생성

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원

    return grad

print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))

# 경사법(경사하강법) 구현
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    print(x)
    return x

# 경사법 문제 예시
init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)

# 학습률이 너무 큰 예
init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100)
# 너무 큰 값으로 발산함

# 학습률이 너무 작은 예
init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100)
# 학습이 채 되지 않은채로 끝남

print(__name__)