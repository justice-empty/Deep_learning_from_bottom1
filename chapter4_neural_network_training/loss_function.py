import numpy as np

def sum_squares_error(y, t): # 오차제곱합
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error_single(y, t): # 교차 엔트리피 오차(데이터 하나씩 처리하는 구현)
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def cross_entropy_error(y, t): 
    if y.ndim == 1: # y가 1차원 이라면
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

"""
def cross_entropy_error(y, t): # 정답 레이블이 원-핫 인코딩이 아닐떄
    if y.ndim == 1: # y가 1차원 이라면
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
"""