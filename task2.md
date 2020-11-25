numpy随机抽样学习及练习
-----
![](https://img.shields.io/badge/python-3.8-blue) ![](https://img.shields.io/badge/numpy-1.18.5-pink)<br>
* [学习](#学习)
  * [离散型随机变量](#离散型随机变量)
    * [二项分布](#二项分布)<br>
    * [泊松分布](#泊松分布)
  * [连续型随机变量](#连续型随机变量)
    * [均匀分布](#均匀分布)<br>
    * [正态分布](#正态分布)
* [练习](#练习)
------
## 学习
------
在生成numpy数组时，可使用`numpy.random`进行数组的随机生成，`numpy.random`是对python内置的random进行了补充，
增加了一些用于高效生成多种概率分布的样本值的函数，如正态分布、泊松分布等。<br>
* `numpy.random.seed(seed=None) `其中seed用于指定随机种子，若不指定，则根据时间来自己选择该值。
-------
### 离散型随机变量
------
#### 二项分布
用于只有两种结果的问题中，例如抛硬币问题<br>
**numpy中使用`numpy.random.binomial(n,p,size=None)`实现**
| 参数  | 解释|
| ---------- | -----------|
| n   | 做了n重伯努利试验 |
| p   | 成功的概率 |
| size   | 采用的次数 |
``` python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

np.random.seed(2020)
n = 2
p = 0.5
size = 50000
x = np.random.binomial(n, p, size)

print(np.sum(x == 0) / size)
print(np.sum(x == 1) / size)
print(np.sum(x == 2) / size)
#0.25232
#0.49972
#0.24796

plt.hist(x)
plt.xlabel('随机变量：硬币为正面次数')
plt.ylabel('50000个样本中出现的次数')
plt.show()

s = stats.binom.pmf(range(n + 1), n, p)
print(np.around(s, 3))
#[0.25 0.5  0.25]
```
![](https://github.com/maxormin/learn_numpy/blob/main/task2_img/%E4%BA%8C%E9%A1%B9%E5%88%86%E5%B8%83.png)
#### 泊松分布
主要用于估计某个时间段某事件发生的概率，例如假定某航空公司预定票处平均每小时接到42次订票电话，
那么10分钟内恰好接到6次电话的概率是多少。
**numpy中使用`numpy.random.poisson(lam=1.0, size=None)`实现**
| 参数  | 解释|
| ---------- | -----------|
| lam   | 一个单位内发生事件的平均值 |
| size   | 采用的次数 |
``` python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(2020)
# 平均值：平均每十分钟接到42/6次订票电话
lam = 42 / 6
size = 50000
x = np.random.poisson(lam, size)

print(np.sum(x == 6) / size)
#0.15022

plt.hist(x)
plt.xlabel('随机变量：每十分钟接到订票电话的次数')
plt.ylabel('50000个样本中出现的次数')
plt.show()

x = stats.poisson.pmf(6, lam)
print(x)  
# 0.14900277967433773
```
![](https://github.com/maxormin/learn_numpy/blob/main/task2_img/%E6%B3%8A%E6%9D%BE%E5%88%86%E5%B8%83.png)
### 连续型随机变量
------
#### 均匀分布
**numpy中使用`numpy.random.uniform(low=0.0, high=1.0, size=None)`实现**
| 参数  | 解释|
| ---------- | -----------|
| low   | 下限 |
| high   | 上限 |
| size   | 采用的次数 |
``` python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(2020)
a = 0
b = 100
size = 50000
x = np.random.uniform(a, b, size=size)
print(np.all(x >= 0))  
# True
print(np.all(x < 100))  
# True
y = (np.sum(x < 50) - np.sum(x < 10)) / size 
print(y)  
# 0.40006

plt.hist(x, bins=100)
plt.show()
```
![](https://github.com/maxormin/learn_numpy/blob/main/task2_img/%E5%9D%87%E5%8C%80%E5%88%86%E5%B8%83.png)

#### 正态分布
**numpy中使用`numpy.random.normal(loc=0.0, scale=1.0, size=None)`或`numpy.random.randn(size)`实现,其中由randn生成的是服从均值为0，标准差
为1的数组**
| 参数  | 解释|
| ---------- | -----------|
| loc   | 均值 |
| scale   | 方差 |
| size   | 采用的次数 |
``` python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

mu = 5#平均值
sigma = 0.5#标准差

np.random.seed(2020)
x = sigma * np.random.randn(2, 4) + mu
print(x)
# [[4.11557715 5.03777614 4.43468515 4.67428492]
#  [4.55344219 4.36294951 4.96942278 5.03225692]]

np.random.seed(2020)
x = np.random.normal(mu, sigma, (2, 4))
print(x)
# [[4.11557715 5.03777614 4.43468515 4.67428492]
#  [4.55344219 4.36294951 4.96942278 5.03225692]]

size = 50000
x = np.random.normal(mu, sigma, size)
plt.hist(x, bins=20)
plt.show()
```
![](https://github.com/maxormin/learn_numpy/blob/main/task2_img/%E6%AD%A3%E6%80%81%E5%88%86%E5%B8%83.png)
## 练习