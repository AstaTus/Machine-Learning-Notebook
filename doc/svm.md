[TOC]

# Machine Learning Notebook

## Multiclass Support Vector Machine

### 线性分类函数
$$f(x) = W*X $$

### 激活函数

* sigmoid :   

* ReLU:          $ \max (0, h(x))$

* ELU

* tanh

* Maxout


### Linear Support Vector Machine代价函数

b是svm中的margin，正则项中的$\lambda$来控制正则项的权重，CS231N课件中正则化缺少$ \frac{1}{N}$项, Ng的Machine Learning视频中在第一项中乘以参数C，取消了正则项中的$\lambda$，作用一样，只是C越小容易欠拟合，C越大越容易过拟合。

$$J(W) = \frac{1}{N}\sum_{i = 1}^N\sum_{j \neq y_i}^M ( \max (0, f_{i,j}(x) - f_{i,y_i}(x) + b ) ) + \frac{1}{N} \lambda \sum_{i=1}^N\sum_{j=1}^M(W_{i,j}^2) $$



### 激活函数ReLU的导数

要对代价函数中的$w_j$和$w_{y_i}$分别求偏导
$$
\nabla_w J_{i,j}=\begin{cases}
\frac{\sigma J_i}{\sigma w_j}&\text{$j \neq y_i$ }\\
\frac{\sigma J_i}{\sigma w_{y_i}}&
\text{$j=y_i$}
\end{cases}\
$$

将$J(W)$带入,括号中为条件，当条件不成立时则为0，又由于有$\sum\sum$,所以对$w_{y_i}$求导时，$w_{y_i}$总共有j- 1个，所以要求和，对于$w_j$项求导时，则每一行只有1个$w_j$所以不需要求和，负号是因为$-w_{y_i} * x_i  $这项求导为负。得到的导数如下:
$$
\nabla_{w_j} J_i=\begin{cases}
1(w_j*x_i - w_{y_i} * x_i + b > 0)*x_i +  2\sum_{i=1}^nw_{i,j}&\text{$j \neq y_i$ }\\
\sum_{j \neq {y_i}}-1(w_j*x_i - w_{y_i} * x_i + b > 0)*x_i + 2\sum_{i=1}^nw_{i,j}&
\text{$j=y_i$}
\end{cases}\
$$

w参数第0列是否为1？？？如果为1则j=0时不需要加正则项

##Binary Support Vector Machine