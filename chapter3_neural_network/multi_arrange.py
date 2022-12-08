import numpy as np

# 다차원 배열의 계산

# 1차원 배열
A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A)) # ndim은 배열의 차원 수를 확인하는 메소드
print(A.shape) # 배열의 형태를 튜플로 반환
print(A.shape[0])

# 2차원 배열
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))
print(B.shape)

# 행렬의 곱
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(np.dot(A, B))

C = np.array([[1, 2], [3, 4], [5, 6]])
D = np.array([7, 8])
print(np.dot(C, D))
