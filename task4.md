# 线性代数学习及练习
-----
![](https://img.shields.io/badge/python-3.8-blue) ![](https://img.shields.io/badge/numpy-1.18.5-pink)<br>
* [学习](#学习)
  * [矩阵和向量积](#矩阵和向量积)<br>
  * [矩阵特征值与特征向量](#矩阵特征值与特征向量)<br>
  * [矩阵分解](#泊松分布)
    * [奇异值分解](#奇异值分解)<br>
    * [QR分解](#QR分解)<br>
    * [Cholesky分解](#Cholesky分解)
  * [范数和其它数字](#范数和其它数字)
    * [矩阵的范数](#矩阵的范数)
    * [方阵的行列式](#方阵的行列式)
    * [矩阵的秩](#矩阵的秩)
    * [矩阵的迹](#矩阵的迹)
  * [解方程和逆矩阵](#解方程和逆矩阵)
    * [逆矩阵](#逆矩阵)
    * [求解线性方程组](#求解线性方程组)
* [练习](#练习)
    
* [练习](#练习)
  * [创建数组](#创建一个形为5×3的二维数组，以包含5到10之间的随机数)
  * [画图](#生成相应的数据)
## 学习
--------
numpy定义了`matrix`类型，使用该类型创建的是矩阵对象，它们的加减乘除运算缺省采用矩阵方式计算，因此用法与matlab类似。但是由于numpu中同时存在`ndarray`和`matrix`对象，因此用户很容易将两者混淆。这有违python的“显式优于隐式”的原则，因此官方并不推荐在程序中使用matrix。

### 矩阵和向量积
矩阵的定义、矩阵的加法、矩阵的数乘、矩阵的转置与二维数组完全一致，但矩阵的乘法有不同的表示。
使用函数`numpy.dot(a, b[, out])`计算两个矩阵的乘积,而\*则是计算点积
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
* 使用函数`numpy.linalg.eig(a)`计算方阵的特征值和特征向量。
* 使用函数`numpy.linalg.eigvals(a)`计算方阵的特征值。
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
使用函数`numpy.linalg.svd(a, full_matrices=True, compute_uv=True, hermitian=False)`进行奇异值分解，其返回u,s,v三个值
| 参数  | 解释|
| ---------- | -----------|
| a   | 一个形如(M,N)矩阵 |
| full_matrices   | 取值为False或True，默认值为True，这时u的大小为(M,M)，v的大小为(N,N)。否则u的大小为(M,K)，v的大小为(K,N) ，K=min(M,N)。 |
| compute_uv   | 取值为False或True，默认值为True，表示计算u,s,v。为False的时候只计算s。 |
* 总共有三个返回值u,s,v，u大小为(M,M)，s大小为(M,N)，v大小为(N,N)，a = u*s*v。
* 其中s是对矩阵a的奇异值分解。s除了对角元素不为0，其他元素都为0，并且对角元素从大到小排列。s中有n个奇异值，一般排在后面的比较接近0，所以仅保留比较大的r个奇异值。
``` python
import numpy as np

A = np.array([[4, 11, 14], [8, 7, -2]])
print(A)
# [[ 4 11 14]
#  [ 8  7 -2]]

u, s, vh = np.linalg.svd(A, full_matrices=False)
print(u.shape)  # (2, 2)
print(u)
# [[-0.9486833  -0.31622777]
#  [-0.31622777  0.9486833 ]]

print(s.shape)  # (2,)
print(np.diag(s))
# [[18.97366596  0.        ]
#  [ 0.          9.48683298]]

print(vh.shape)  # (2, 3)
print(vh)
# [[-0.33333333 -0.66666667 -0.66666667]
#  [ 0.66666667  0.33333333 -0.66666667]]

a = np.dot(u, np.diag(s))
print(np.dot(a, vh))
# [[ 4. 11. 14.]
#  [ 8.  7. -2.]]
```
#### QR分解
使用函数`numpy.linalg.qr(a, mode='reduced')`计算矩阵a的QR分解,该函数返回q,r两个值。
 参数  | 解释|
| ---------- | -----------|
| a   | 一个形如(M,N)矩阵 |
| mode   | mode='reduced',返回(M, N)的列向量两两正交的矩阵q，和(N, N)的三角阵r（Reduced QR分解;
mode='reduced',返回(M, M)的正交矩阵q，和(M, N)的三角阵r（Full QR分解）|
``` python
import numpy as np

A = np.array([[2, -2, 3], [1, 1, 1], [1, 3, -1]])
print(A)
# [[ 2 -2  3]
#  [ 1  1  1]
#  [ 1  3 -1]]

q, r = np.linalg.qr(A)
print(q.shape)  # (3, 3)
print(q)
# [[-0.81649658  0.53452248  0.21821789]
#  [-0.40824829 -0.26726124 -0.87287156]
#  [-0.40824829 -0.80178373  0.43643578]]

print(r.shape)  # (3, 3)
print(r)
# [[-2.44948974  0.         -2.44948974]
#  [ 0.         -3.74165739  2.13808994]
#  [ 0.          0.         -0.65465367]]

print(np.dot(q, r))
# [[ 2. -2.  3.]
#  [ 1.  1.  1.]
#  [ 1.  3. -1.]]

a = np.allclose(np.dot(q.T, q), np.eye(3))
print(a)  # True
```
#### Cholesky分解
使用函数` numpy.linalg.cholesky(a)`对矩阵进行Cholesky分解
L = numpy.linalg.cholesky(a) 返回正定矩阵a的 Cholesky 分解a = L*L.T，其中L是下三角。
``` python
import numpy as np

A = np.array([[1, 1, 1, 1], [1, 3, 3, 3],
              [1, 3, 5, 5], [1, 3, 5, 7]])
print(A)
# [[1 1 1 1]
#  [1 3 3 3]
#  [1 3 5 5]
#  [1 3 5 7]]

print(np.linalg.eigvals(A))
# [13.13707118  1.6199144   0.51978306  0.72323135]

L = np.linalg.cholesky(A)
print(L)
# [[1.         0.         0.         0.        ]
#  [1.         1.41421356 0.         0.        ]
#  [1.         1.41421356 1.41421356 0.        ]
#  [1.         1.41421356 1.41421356 1.41421356]]

print(np.dot(L, L.T))
# [[1. 1. 1. 1.]
#  [1. 3. 3. 3.]
#  [1. 3. 5. 5.]
#  [1. 3. 5. 7.]]
```
### 范数和其它数字
---------
#### 矩阵的范数
使用函数`numpy.linalg.norm(x, ord=None, axis=None, keepdims=False)`计算向量或者矩阵的范数。
-----
求向量范数
``` python
import numpy as np
x = np.array([1, 2, 3, 4])

print(np.linalg.norm(x, ord=1)) 
# 10.0
print(np.sum(np.abs(x)))  
# 10

print(np.linalg.norm(x, ord=2))  
# 5.477225575051661
print(np.sum(np.abs(x) ** 2) ** 0.5)  
# 5.477225575051661

print(np.linalg.norm(x, ord=-np.inf))  
# 1.0
print(np.min(np.abs(x)))  
# 1

print(np.linalg.norm(x, ord=np.inf))  
# 4.0
print(np.max(np.abs(x)))  
# 4
```
----
求矩阵范数
``` python
A = np.array([[1, 2, 3, 4], [2, 3, 5, 8],
              [1, 3, 5, 7], [3, 4, 7, 11]])

print(A)
# [[ 1  2  3  4]
#  [ 2  3  5  8]
#  [ 1  3  5  7]
#  [ 3  4  7 11]]

print(np.linalg.norm(A, ord=1))  # 30.0
print(np.max(np.sum(A, axis=0)))  # 30

print(np.linalg.norm(A, ord=2))  
# 20.24345358700576
print(np.max(np.linalg.svd(A, compute_uv=False)))  
# 20.24345358700576

print(np.linalg.norm(A, ord=np.inf))  # 25.0
print(np.max(np.sum(A, axis=1)))  # 25

print(np.linalg.norm(A, ord='fro'))  
# 20.273134932713294
print(np.sqrt(np.trace(np.dot(A.T, A))))  
# 20.273134932713294
```
#### 方阵的行列式
使用函数`numpy.linalg.det(a)`计算行列式。
``` python
import numpy as np

x = np.array([[1, 2], [3, 4]])
print(x)
# [[1 2]
#  [3 4]]

print(np.linalg.det(x))
# -2.0000000000000004
```
#### 矩阵的秩
使用函数`numpy.linalg.matrix_rank(M, tol=None, hermitian=False)`计算矩阵的秩。
``` python
import numpy as np

I = np.eye(3)  # 先创建一个单位阵
print(I)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

r = np.linalg.matrix_rank(I)
print(r)  # 3

I[1, 1] = 0  # 将该元素置为0
print(I)
# [[1. 0. 0.]
#  [0. 0. 0.]
#  [0. 0. 1.]]

r = np.linalg.matrix_rank(I)  # 此时秩变成2
print(r)  # 2
```
#### 矩阵的迹
使用函数`numpy.trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None)` 计算矩阵的迹<br>
***方阵的迹是主对角元素之和****
``` python
import numpy as np

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

print(np.trace(x))
# 13
print(np.trace(np.transpose(x)))
# 13

print(np.trace(x + y))
# 30
print(np.trace(x) + np.trace(y))
# 30
```
### 解方程和逆矩阵
-------
#### 逆矩阵
逆矩阵定义:设 A 是数域上的一个 n 阶矩阵，若在相同数域上存在另一个 n 阶矩阵 B，使得：AB=BA=E，则 B 是 A 的逆矩阵，而 A 则被称为可逆矩阵。<br>
通过函数`numpy.linalg.inv(a)` 计算矩阵a的逆矩阵（矩阵可逆的充要条件：det(a) != 0，或者a满秩）。
``` python
import numpy as np

A = np.array([[1, -2, 1], [0, 2, -1], [1, 1, -2]])
print(A)
# [[ 1 -2  1]
#  [ 0  2 -1]
#  [ 1  1 -2]]

# 求A的行列式，不为零则存在逆矩阵
A_det = np.linalg.det(A)  
print(A_det)
# -2.9999999999999996

A_inverse = np.linalg.inv(A)  # 求A的逆矩阵
print(A_inverse)
# [[ 1.00000000e+00  1.00000000e+00 -1.11022302e-16]
#  [ 3.33333333e-01  1.00000000e+00 -3.33333333e-01]
#  [ 6.66666667e-01  1.00000000e+00 -6.66666667e-01]]

x = np.allclose(np.dot(A, A_inverse), np.eye(3))
print(x)  # True
x = np.allclose(np.dot(A_inverse, A), np.eye(3))
print(x)  # True

A_companion = A_inverse * A_det  # 求A的伴随矩阵
print(A_companion)
# [[-3.00000000e+00 -3.00000000e+00  3.33066907e-16]
#  [-1.00000000e+00 -3.00000000e+00  1.00000000e+00]
#  [-2.00000000e+00 -3.00000000e+00  2.00000000e+00]]
```
#### 求解线性方程组
使用函数`numpy.linalg.solve(a, b)`求解线性方程组或矩阵方程。
``` python
#  x + 2y +  z = 7
# 2x -  y + 3z = 7
# 3x +  y + 2z =18

import numpy as np

A = np.array([[1, 2, 1], [2, -1, 3], [3, 1, 2]])
b = np.array([7, 7, 18])
x = np.linalg.solve(A, b)
print(x)  # [ 7.  1. -2.]

x = np.linalg.inv(A).dot(b)
print(x)  # [ 7.  1. -2.]

y = np.allclose(np.dot(A, x), b)
print(y)  # True
```
--------
## 练习
--------
### 计算两个数组a和数组b之间的欧氏距离。
* a = np.array([1, 2, 3, 4, 5])
* b = np.array([4, 5, 6, 7, 8])
