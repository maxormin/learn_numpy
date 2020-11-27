统计相关学习及练习
-----
![](https://img.shields.io/badge/python-3.8-blue) ![](https://img.shields.io/badge/numpy-1.18.5-pink)<br>

## 学习
------
### 次序统计
------
#### 计算最小值
使用numpy中`numpy.amin()`函数，该函数接口为<br>
numpy.amin(a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)
| 参数  | 输入 | 个人理解 |
| ---------- |-----------| -----------|
| a   | array_like | 输入的数据 |
| axis | None or int or tuple of ints, optional | 其运行的轴，简单而言，当axis=0且a为二维时，所求为每行最小值；当axis=1时，所求为每列最小值；当无参数时，所求为全局最小 |
| out  | ndarray, optional | 放置结果的备用输出数组 |
| initial  | scalar, optional | 当该参数有输入时，则输出元素的最大值不超过该参数，若超过，则将超过的元素替换为该值 |
| where  | array_like of bool, optional | 要比较的元素的最小值。 |
