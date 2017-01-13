[TOC]

# Softmax (Multinomial Logistic Regression)

### Logistic Regression

二分逻辑回归假设函数（hypothesis function）,该函数的取值范围在（0,1）之间，当h为（0，0.5）时则y=0，当h[0.5, 1)时则以y=1

$h_{\theta}(x)= \frac{1}{1+e^{(-\theta^Tx)}}$

### 代价函数
$$
J(\theta)=-\frac{1}{m}\sum_{i=1}^my^{(i)}logh_\theta(x^{(i)}) + (1-y^{(i)})logh_\theta(x^{(i)}))
$$

### Softmax 

##### 每个样本的得分

$scores =  f(X_i; W)$          

为了防止scores在代价函数中，数值溢出，一般scores计算出来后要进行归一化，减去每个样本向量的最大值

```python
  f = np.dot(X,W)
  f_max = np.reshape(np.max(f, axis=1), (m, 1))
  probabilitys = np.exp(f - f_max); 
```



##### 样本得分的概率

$P_k = \frac{e^{s_k}}{\sum_{i=0}^me^s_i}$



##### 代价函数

$L_i=-[\sum_{j=1}^k1\{y^{(i)}=y^{(label)}\}log\frac{e^{\theta_j^T x^{(i)}}}{\sum_{l=1}^k e^{\theta_j^Tx^{(i)}}}]$



##### 代价函数的导数

$$
\frac{\delta L_i}{\delta W_i} = \frac{\delta L_i}{\delta P_i} \frac{\delta P_i}{\delta scores}\frac{\delta socres }{\delta W_i}
$$

$\frac{\delta socres }{\delta W_i} = X_i$



$$ \frac{\delta P_i}{\delta scores} = \frac{(e^{s_k})'(\sum_{i=0}^me^s_i) - (e^{s_k})(\sum_{i=0}^me^s_i)'}{(\sum_{i=0}^me^s_i)^2} = \begin{cases}P_k(1 - P_k)&\text{$i = k$ }\\-P_kP_i&\text{$i \neq k$}\end{cases} $$





$$\frac{\delta L_i}{\delta P_i} = \begin{cases}- \frac{1}{P_i}&\text{${y^{(i)}=y^{(label)}}$ }\\0&\text{${y^{(i)} \neq y^{(label)}}$}\end{cases}  $$



满足条件：${y^{(i)}=y^{(label)}}$
$$
\frac{\delta L_i}{\delta W_i} = \begin{cases}X_i( P_k - 1)&\text{$i = k$ }\\X_iP_k&\text{$i \neq k$}\end{cases}
$$

否则：
$$
\frac{\delta L_i}{\delta W_i} = 0
$$
