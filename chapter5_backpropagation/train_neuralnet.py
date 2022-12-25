import sys, os
sys.path.append('C:\\Users\\user\\Desktop\\coding\\My_coding_study\\Deep_learning_from_bottom1')
import numpy as np
from chapter3_neural_network.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 변수 설정
iters_num = 10000 # 반복 횟수
train_size = x_train.shape[0] # (60000, 784) 이므로 train_size는 60000
batch_size = 100 # 배치 사이즈
learning_rate = 0.1 # 학습률 0.1 (기울기를 갱신할 때, 학습률의 크기만큼 갱신함)

train_loss_list = [] 
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1) # 60000/100

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size) # 0부터 59999까지의 숫자를(train_size만큼) 100개(batch_size만큼) 뽑음
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)