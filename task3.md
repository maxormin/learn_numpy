统计相关学习及练习
-----
![](https://img.shields.io/badge/python-3.8-blue) ![](https://img.shields.io/badge/numpy-1.18.5-pink)<br>
* [学习](#学习)
  * [次序统计](#次序统计)
    * [计算最小值](#计算最小值)<br>
    * [计算最大值](#计算最大值)<br>
    * [计算极差](#计算极差)
  * [均值与方差](#均值与方差)
    * [计算中位数](#计算中位数)<br>
    * [计算平均值](#计算平均值)<br>
    * [计算加权平均值](#计算加权平均值)<br>
    * [计算标准差](#计算标准差)
  * [相关](#相关)
    * [计算协方差矩阵](#计算协方差矩阵)<br>
    * [计算相关系数](#计算相关系数)<br>
    * [直方图](#直方图)
* [练习](#练习)
  * [计算最大值](#计算给定数组中每行的最大值)
## 学习
------
### 次序统计
------
#### 计算最小值
使用numpy中`numpy.amin()`函数求数组中最小值，该函数接口为<br>
numpy.amin(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue, where=np._NoValue)
| 参数  | 输入 | 个人理解 |
| ---------- |-----------| -----------|
| a   | array_like | 输入的数据 |
| axis | None or int or tuple of ints, optional | 其运行的轴，简单而言，当axis=0且a为二维时，所求为每行最小值；当axis=1时，所求为每列最小值；当无参数时，所求为全局最小 |
| out  | ndarray, optional | 放置结果的备用输出数组 |
| initial  | scalar, optional | 当该参数有输入时，则输出元素的最大值不超过该参数，若超过，则将超过的元素替换为该值 |
| where  | array_like of bool, optional | 要比较的元素的最小值。 |
``` python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30]])
y = np.amin(x)
print(y)  # 11

y = np.amin(x, axis=0)
print(y)  # [11 12 13 14 15]

y = np.amin(x, axis=1)
print(y)  # [11 16 21 26]
```
#### 计算最大值
使用numpy中`numpy.amin()`函数求数组中最大值，该函数接口为<br>
numpy.amax(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue, where=np._NoValue)<br>
``` python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30]])
y = np.amax(x)
print(y)  # 30

y = np.amax(x, axis=0)
print(y)  # [26 27 28 29 30]

y = np.amax(x, axis=1)
print(y)  # [15 20 25 30]
```
#### 计算极差
使用numpy中`numpy.ptp()`函数求数组中最大值与最小值的差值，该函数接口为<br>
numpy.ptp(a, axis=None, out=None, keepdims=np._NoValue)<br>
``` python
import numpy as np

np.random.seed(2020)
x = np.random.randint(0, 20, size=[4, 5])
print(x)
# [[ 0  8  3  3  3]
#  [ 7 16  0 10  9]
#  [19 11 18  3  6]
#  [ 5 16  8  6  1]]

y = np.ptp(x)
print(y)  # 19

y = np.ptp(x, axis=0)
print(y)  # [19  8 18  7  8]

y = np.ptp(x, axis=1)
print(y)  # [ 8 16 16 15]
```
#### 计算分位数
使用numpy中`numpy.percentile()`函数求数组中的分位数，该函数接口为<br>
numpy.percentile(a, q, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
| 参数  | 输入 | 个人理解 |
| ---------- |-----------| -----------|
| a   | array_like | 用于计算分位数的对象，可以是多维数组 |
| q   | float | 用来计算时几分位的参数，如四分之一位就是25，如要算两个位置数就是[25,75]。 |
| axis | 0/1 | 多维时用该参数调整计算的维度方向 |
``` python
import numpy as np

np.random.seed(2020)
x = np.random.randint(0, 20, size=[4, 5])
print(x)
# [[ 0  8  3  3  3]
#  [ 7 16  0 10  9]
#  [19 11 18  3  6]
#  [ 5 16  8  6  1]]

print(np.percentile(x, [25, 50]))
# [3.  6.5]

print(np.percentile(x, [25, 50], axis=0))
# 6 = (7 + 5) / 2
# [[ 3.75 10.25  2.25  3.    2.5 ]
#  [ 6.   13.5   5.5   4.5   4.5 ]]

print(np.percentile(x, [25, 50], axis=1))
# [[ 3.  7.  6.  5.]
#  [ 3.  9. 11.  6.]]
```
## 均值与方差
------
### 计算中位数
使用numpy中`numpy.median()`函数求数组中位数，计算结果与50%分位数相同，该函数接口为<br>
numpy.median(a, axis=None, out=None, overwrite_input=False, keepdims=False)
``` python
import numpy as np

np.random.seed(2020)
x = np.random.randint(0, 20, size=[4, 5])
print(x)
# [[ 0  8  3  3  3]
#  [ 7 16  0 10  9]
#  [19 11 18  3  6]
#  [ 5 16  8  6  1]]

print(np.median(x))
# 6.5

print(np.median(x, axis=0))
# [ 6.  13.5  5.5  4.5  4.5]

print(np.median(x, axis=1))
# [ 3.  9. 11.  6.]
```
### 计算平均值
使用numpy中`numpy.mean()`函数求数组均值，该函数接口为<br>
numpy.mean(a[, axis=None, dtype=None, out=None, keepdims=np._NoValue)])
``` python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.mean(x)
print(y)  # 23.0

y = np.mean(x, axis=0)
print(y)  # [21. 22. 23. 24. 25.]

y = np.mean(x, axis=1)
print(y)  # [13. 18. 23. 28. 33.]
```
### 计算加权平均
mean和average都是计算均值的函数，在不指定权重的时候average和mean是一样的。指定权重后，average可以计算加权平均值。<br>
numpy.average(a[, axis=None, weights=None, returned=False])
``` python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

print(np.average(x))  # 23.0
print(np.mean(x))   #23.0

y = np.arange(1, 26).reshape([5, 5])
print(y)
# [[ 1  2  3  4  5]
#  [ 6  7  8  9 10]
#  [11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]]

print(np.mean(x))   #23.0
print(np.average(x, weights=y))  # 27.0
print(((y/y.sum())*x).sum()) #27.0

print(np.mean(x,axis=0))# [21. 22. 23. 24. 25.]
print(np.average(x, axis=0, weights=y))# [25.54545455 26.16666667 26.84615385 27.57142857 28.33333333]

print(np.average(x, axis=1, weights=y))
# [13.66666667 18.25       23.15384615 28.11111111 33.08695652]
```
### 计算方差
要注意方差和样本方差的无偏估计，方差公式中分母上是n；样本方差无偏估计公式中分母上是n-1（n为样本个数）。<br>
numpy.var(a[, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue])
``` python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.var(x)
print(y)  # 52.0
y = np.mean((x - np.mean(x)) ** 2)
print(y)  # 52.0

y = np.var(x, ddof=1)
print(y)  # 54.166666666666664
y = np.sum((x - np.mean(x)) ** 2) / (x.size - 1)
print(y)  # 54.166666666666664

y = np.var(x, axis=0)
print(y)  # [50. 50. 50. 50. 50.]

y = np.var(x, axis=1)
print(y)  # [2. 2. 2. 2. 2.]
```
### 计算标准差
标准差是一组数据平均值分散程度的一种度量，是方差的算术平方根。<br>
numpy.std(a[, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue])
``` python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.std(x)
print(y)  # 7.211102550927978
y = np.sqrt(np.var(x))
print(y)  # 7.211102550927978

y = np.std(x, axis=0)
print(y)
# [7.07106781 7.07106781 7.07106781 7.07106781 7.07106781]

y = np.std(x, axis=1)
print(y)
# [1.41421356 1.41421356 1.41421356 1.41421356 1.41421356]

y = np.var(x, axis=1)
print(y)  # [2. 2. 2. 2. 2.]
```
## 相关
---------
### 计算协方差矩阵
cov(X,Y) = E[XY] - E[X]E[Y]<br>
协方差表示的是两个变量总体误差的期望<br>
numpy.cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None,aweights=None)
``` python
import numpy as np

x = [1, 2, 3, 4, 6]
y = [0, 2, 5, 6, 7]
print(np.cov(x))  # 3.7   #样本方差
print(np.cov(y))  # 8.5   #样本方差
print(np.cov(x, y))
# [[3.7  5.25]
#  [5.25 8.5 ]]

print(np.var(x))  # 2.96    #方差
print(np.var(x, ddof=1))  # 3.7    #样本方差
print(np.var(y))  # 6.8    #方差
print(np.var(y, ddof=1))  # 8.5    #样本方差

#协方差
print(np.mean((x - np.mean(x)) * (y - np.mean(y))))  # 4.2

#样本协方差
print(np.sum((x - np.mean(x)) * (y - np.mean(y))) / (len(x) - 1))  # 5.25

#样本协方差
print(np.dot(x - np.mean(x), y - np.mean(y)) / (len(x) - 1))  # 5.25
```
### 计算相关系数
np.cov()描述的是两个向量协同变化的程度，它的取值可能非常大，也可能非常小，这就导致没法直观地衡量二者协同变化的程度。相关系数实际上是正则化的协方差，n个变量的相关系数形成一个n维方阵。<br>
numpy.corrcoef(x, y=None, rowvar=True, bias=np._NoValue, ddof=np._NoValue)
``` python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(2020)
x, y = np.random.randint(0, 20, size=(2, 4))

print(x)  # [0 8 3 3]
print(y)  # [ 3  7 16  0]

# 通过corrcoef计算
z = np.corrcoef(x, y)
print(z)
# [[1.         0.18793271]
#  [0.18793271 1.        ]]

#通过公式计算
a = np.dot(x - np.mean(x), y - np.mean(y))
b = np.sqrt(np.dot(x - np.mean(x), x - np.mean(x)))
c = np.sqrt(np.dot(y - np.mean(y), y - np.mean(y)))
print(a / (b * c))  # 0.18793271211851736
```
### 直方图
numpy.digitize(x, bins, right=False)
| 参数  | 输入 | 个人理解 |
| ---------- |-----------| -----------|
| x   | numpy | 传入数组 |
| bins   | 数组 | 用于划分的界限，因此必须单调。 |
| right | False/True | 是否包含右 |
| x | numpy | x在bins中的位置 |
``` python
import numpy as np

x = np.array([1.2, 10.0, 12.4, 15.5, 20.])
bins = np.array([0, 5, 10, 15, 20])
inds = np.digitize(x, bins, right=True)
print(inds)  # [1 2 3 4 4]

inds = np.digitize(x, bins, right=False)
print(inds)  # [1 3 3 4 5]
```
## 练习
-------
### 计算给定数组中每行的最大值
`a = np.random.randint(1, 10, [5, 3])`
``` python
import numpy as np

a = np.random.randint(1, 10, [5, 3])
print(a)
# [[4 8 7]
#  [4 2 4]
#  [2 3 9]
#  [1 8 3]
#  [5 9 8]]
print(np.amax(a,axis=1)) # [8 4 9 8 9]
```
[返回顶部](#readme)
