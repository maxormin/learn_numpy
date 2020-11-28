# 线性代数学习及练习
-----
![](https://img.shields.io/badge/python-3.8-blue) ![](https://img.shields.io/badge/numpy-1.18.5-pink)<br>
## 学习
--------
numpy定义了`matrix`类型，使用该类型创建的是矩阵对象，它们的加减乘除运算缺省采用矩阵方式计算，因此用法与matlab类似。但是由于numpu中同时存在`ndarray`和`matrix`对象，因此用户很容易将两者混淆。这有违python的“显式优于隐式”的原则，因此官方并不推荐在程序中使用matrix。

### 矩阵和向量积
矩阵的定义、矩阵的加法、矩阵的数乘、矩阵的转置与二维数组完全一致，不再进行说明，但矩阵的乘法有不同的表示。
`numpy.dot(a, b[, out])`计算两个矩阵的乘积,而\*则是计算点积
``` python
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])
z = np.dot(x, y)
print(z)  # 70

x = np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
print(x)
# [[1 2 3]
#  [3 4 5]
#  [6 7 8]]

y = np.array([[5, 4, 2], [1, 7, 9], [0, 4, 5]])
print(y)
# [[5 4 2]
#  [1 7 9]
#  [0 4 5]]

print(np.dot(x, y))
# [[  7  30  35]
#  [ 19  60  67]
#  [ 37 105 115]]
print(x*y)
# [[ 5  8  6]
#  [ 3 28 45]
#  [ 0 28 40]]
```
注意：在线性代数里面讲的维数和数组的维数不同，如线代中提到的n维行向量在 Numpy 中是一维数组，而线性代数中的n维列向量在 Numpy 中是一个shape为(n, 1)的二维数组。
### 矩阵特征值与特征向量
* numpy.linalg.eig(a) 计算方阵的特征值和特征向量。
* numpy.linalg.eigvals(a) 计算方阵的特征值。
``` python
import numpy as np

x = np.diag((1, 2, 3))  
print(x)
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]

print(np.linalg.eigvals(x))
# [1. 2. 3.]

a, b = np.linalg.eig(x)  
# 特征值保存在a中，特征向量保存在b中
print(a)
# [1. 2. 3.]
print(b)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# 检验特征值与特征向量是否正确
for i in range(3): 
    if np.allclose(a[i] * b[:, i], np.dot(x, b[:, i])):
        print('Right')
    else:
        print('Error')
# Right
# Right
# Right
```
### 矩阵分解
---------
#### 奇异值分解
[奇异值分解原理](http://datawhale.club/ "悬停显示")<br>
u, s, v = numpy.linalg.svd(a, full_matrices=True, compute_uv=True, hermitian=False)奇异值分解
| 参数  | 解释|
| ---------- | -----------|
| a   | 一个形如(M,N)矩阵 |
| full_matrices   | 取值为False或True，默认值为True，这时u的大小为(M,M)，v的大小为(N,N)。否则u的大小为(M,K)，v的大小为(K,N) ，K=min(M,N)。 |
| compute_uv   | 取值为False或True，默认值为True，表示计算u,s,v。为False的时候只计算s。 |
* 总共有三个返回值u,s,v，u大小为(M,M)，s大小为(M,N)，v大小为(N,N)，a = u*s*v。
* 其中s是对矩阵a的奇异值分解。s除了对角元素不为0，其他元素都为0，并且对角元素从大到小排列。s中有n个奇异值，一般排在后面的比较接近0，所以仅保留比较大的r个奇异值。
#### QR分解

#### Cholesky分解

### 范数和其它数字
