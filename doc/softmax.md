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

多类逻辑回归假设函数（hypothesis function），

$$
softmax(X_i)=
\left\{\begin{array}{ll}
p(y^{(i)} = 1|x^{(i)};\theta) &\\
p(y^{(i)} = 2|x^{(i)};\theta)  &\\
p(y^{(i)} = 3|x^{(i)};\theta)  &\\
p(y^{(i)} = 4|x^{(i)};\theta) &\\
p(y^{(i)} = 5|x^{(i)};\theta)  &\\
...\\
\end{array}\right.
=
\frac{1}{\sum_j^Ne^{\theta^T x^{(j)}}}
\left\{\begin{array}{ll}
e^{\theta_1 x_i} &\\
e^{\theta_2 x_i} &\\
e^{\theta_3 x_i} &\\
e^{\theta_4 x_i} &\\
e^{\theta_5 x_i} &\\
...\\
\end{array}\right.
$$

### 代价函数

$$
L_i=-[\sum_{j=1}^k1\{y^{(i)}=y^{(label)}\}log\frac{e^{\theta_j^T x^{(i)}}}{\sum_{l=1}^k e^{\theta_j^Tx^{(i)}}}]
$$

### 代价函数的导数

以$e$为底关于$\theta$的导数是其本身，所以$log$项不变

$$\nabla_{\theta_j}J(\theta)=-\frac{1}{m}\sum_{i=1}^m[x^{(i)}(1\{y^{(i)}=j\}-\frac{e^{\theta_j^T x^{(i)}}}{\sum_{l=1}^k e^{\theta_j^Tx^{(i)}}})]$$