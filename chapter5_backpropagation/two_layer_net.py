import sys
sys.path.append('C:\\Users\\user\\Desktop\\coding\\My_coding_study\\Deep_learning_from_bottom1')
import numpy as np
from chapter4_neural_network_training.numerical_gradient import numerical_gradient
from activation_layer import Affine, Relu, SoftmaxWithLoss
from collections import OrderedDict

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01): # 초기화
        
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict() # 순서가 있는 딕셔너리
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

# 예측 수행
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x) # forward가 어떤 클래스의 메소드인지 잘 모르겠음
        
        return x

# 손실 함수 값 구하기
    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

# 정확도 구하기
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1) # axis=1 은 가로축 원소들끼리의 비교에서 최대값의 위치를 보여줌
        if t.ndim != 1 : t = np.argmax(t, axis=1) # t.ndim != 1 (t의 배열이 1차원이 아닌 경우)
        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

# 수치 미분으로 구하는 기울기
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t) #람다(lambda) 매개변수: 표현식 (W: self.loss)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

# 오차역전파법으로 구하는 기울기
    def gradient(self, x, t):
        # 순전파
        self.loss(x, t)

        # 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads