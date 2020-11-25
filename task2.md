numpy随机抽样学习及练习
-----
![](https://img.shields.io/badge/python-3.8-blue) ![](https://img.shields.io/badge/numpy-1.18.5-pink)<br>
## 学习
------
在生成numpy数组时，可使用`numpy.random`进行数组的随机生成，`numpy.random`是对python内置的random进行了补充，
增加了一些用于高效生成多种概率分布的样本值的函数，如正态分布、泊松分布等。<br>
* `numpy.random.seed(seed=None) `其中seed用于指定随机种子，若不指定，则根据时间来自己选择该值。
### 离散型随机变量
------
#### 二项分布
抛硬币问题<br>
**核心函数`numpy.random.binomial(n,p,size=None)`二项随机变量,可视化地表现概率**
| 参数  | 解释|
| ---------- | -----------|
| n   | 一次试验的样本数n，并且相互不干扰(个人理解为有n种结果)   |
| p   | 事件发生的概率p，范围[0,1]   |
| size   | 表示实验size次，返回每次实验中事件发生的次数   |
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
