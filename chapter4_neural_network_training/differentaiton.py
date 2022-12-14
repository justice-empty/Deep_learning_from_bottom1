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