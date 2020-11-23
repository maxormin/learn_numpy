numpy输入输出练习
-----
![](https://img.shields.io/badge/python-3.8-blue) ![](https://img.shields.io/badge/numpy-1.18.5-pink) <br>
[numpy数组只打印小数点后三位](#readme)
#### numpy数组只打印小数点后三位
任务详情:只打印或显示`numpy`数组的小数点后三位:
``` python
import numpy as np
rand_arr = np.random.random([5, 3])
print(rand_arr)
#[[0.50330202 0.22642223 0.08250267]
# [0.14615026 0.79519263 0.37899363]
# [0.12026668 0.04385484 0.70282621]
# [0.32450256 0.39103128 0.99140394]
# [0.98976714 0.09250011 0.45865713]]
```
其存在多种解决方法，以下举例两种:<br>
##### 1. 使用np.around
`numpy.around(a, decimals=0, out=None)`
<br>该方法是通过对numpy数组进行四舍五入保留3位小数而得到，在不赋值的情况下并不会对原数组造成影响
``` python
print(np.around(rand_arr,decimals = 3))
# [[0.503 0.226 0.083]
#  [0.146 0.795 0.379]
#  [0.12  0.044 0.703]
#  [0.325 0.391 0.991]
#  [0.99  0.093 0.459]]

print(rand_arr)
# [[0.50330202 0.22642223 0.08250267]
#  [0.14615026 0.79519263 0.37899363]
#  [0.12026668 0.04385484 0.70282621]
#  [0.32450256 0.39103128 0.99140394]
#  [0.98976714 0.09250011 0.45865713]]
```
##### 2. 使用np.set_printoptions()
`np.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, suppress=None, nanstr=None, infstr=None, formatter=None, sign=None, floatmode=None, *, legacy=None)`
<br>该方法是对numpy数组内所有元素进行四舍五入保留3位小数，会对原数组造成影响，并且新建立的numpy数组也会只打印小数点后3位
``` python
np.set_printoptions(precision=3)
print(rand_arr)
# [[0.503 0.226 0.083]
#  [0.146 0.795 0.379]
#  [0.12  0.044 0.703]
#  [0.325 0.391 0.991]
#  [0.99  0.093 0.459]]
print(np.random.random([5, 3]))
# [[0.952 0.375 0.93 ]
#  [0.727 0.376 0.316]
#  [0.655 0.242 0.123]
#  [0.722 0.289 0.642]
#  [0.327 0.66  0.866]]
```
