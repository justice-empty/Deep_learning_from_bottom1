import sys, os
sys.path.append(os.pardir) # 부모 디렉토리의 파일을 가져올 수 있도록 설정
import numpy as np
from mnist import load_mnist # MNIST 데이터셋 불러오기
from PIL import Image

def img_show(img): # 넘파이로 저장된 이미지 데이터를 PIL용 데이터 객체로 변환
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

"""
# 각 데이터의 형상 출력
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
"""

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape) # 1차원 배열
img = img.reshape(28, 28) #flatten을 True로 설정해 읽어 들인 이미지는 1차원 배열이므로 이미지를 표시할 때는 원래 형상인 28x28 크기로 다시 변형해야함
print(img.shape) # 2차원 배열

img_show(img)