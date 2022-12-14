import sys, os
sys.path.append('C:/Users/user/Desktop/코딩/My_coding_study/밑바닥부터 시작하는 딥러닝1')
import numpy as np
from activation_function import softmax
from loss_function import cross_entropy_error
from numerical_gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # 정규분포로 초기화
    
    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss
    
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)