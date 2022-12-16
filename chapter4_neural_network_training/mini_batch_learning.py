import sys, os
sys.path.append(r'C:\Users\user\Desktop\coding\My_coding_study\Deep_learning_from_bottom1') 
import numpy as np
from chapter3_neural_network.mnist import load_mnist
 
(x_train, t_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)

# 훈련 데이터에서 무작위로 10장 빼내기
train_size = x_train.shape[0] # 60000
batch_size = 10 # 배치 10장 지정
batch_mask = np.random.choice(train_size, batch_size) # 랜덤하게 0 이상 60000 미만의 수 중에서 10개를 고름
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]