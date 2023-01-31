import sys, os
sys.path.append(os.pardir)
from im2col import im2col
import numpy as np

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
    
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out

if __name__ == '__main__':
    x1 = np.random.rand(1, 3, 7, 7) # (데이터 수, 채널 수, 높이, 너비)
    col1 = im2col(x1, 5, 5, stride=1, pad=0)
    print(col1.shape)

    x2 = np.random.rand(10, 3, 7, 7) # (데이터 수, 채널 수, 높이, 너비)
    col2 = im2col(x2, 5, 5, stride=1, pad=0)
    print(col2.shape)