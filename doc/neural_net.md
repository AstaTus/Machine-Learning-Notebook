[TOC]

# Neural Network



### 两层神经网络

#### 正向推导

$h1 = max(0, f(X;W1))$



$scores = f(h1;W2)$



输出层不用添加RELU函数，但需要加一层softmax层（参考softmax）

$output = softmax(scores)$



```python
shift_scores = scores - np.max(scores, axis=1).reshape([m, 1]);       # [N,C]
exp_scores = np.exp(shift_scores)                                     # [N,C]
p_sum = np.sum(exp_scores,  axis=1, keepdims=True);                   # [N,]
props = exp_scores / p_sum;                                           # [N,C]
correct_logprobs = -np.log(props[range(m), y])                        # [N,1]
loss = np.sum(correct_logprobs) / m;
loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2));
```

##### 代价函数

$L_i = - \frac{1}{m}(\sum_{i=0}^m logP_i) + \frac{1}{2}\lambda(W1^2 + W2^2)$



#### 反向推导

##### 代价函数导数

矩阵的链式法则，是点乘而不是标量乘法

$$\frac{\delta L_i}{\delta W_2} = \frac{\delta softmax}{\delta scores} \frac{\delta scores}{\delta W_2}$$



$$\frac{\delta L_i}{\delta b_2} = \frac{\delta softmax}{\delta scores} \frac{\delta scores}{\delta b_2}$$ 



$$\frac{\delta L_i}{\delta W_1} = \frac{\delta softmax}{\delta scores} \frac{\delta scores}{\delta h_1}\frac{\delta h_1}{\delta W_1}$$



$$\frac{\delta L_i}{\delta b_1} = \frac{\delta softmax}{\delta scores} \frac{\delta scores}{\delta h_1}\frac{\delta h_1}{\delta b_1}$$



$$\frac{\delta softmax}{\delta scores} = \begin{cases}\frac{1}{N}(socres - 1)&\text{${y^{(i)}=y^{(label)}}$ }\\ \frac{1}{N}socres&\text{${y^{(i)} \neq y^{(label)}}$}\end{cases}$$



$$\frac{\delta scores}{\delta W_2}=h_1$$



$$\frac{\delta scores}{\delta b_2}=1 [C,1]$$



$$\frac{\delta scores}{\delta h_1} = \begin{cases}W_2&\text{$h_1 >0$ }\\ 0&\text{$h_1<=0$}\end{cases}$$



$$\frac{\delta h_1}{\delta W_1} = X$$



$$\frac{\delta h_1}{\delta b_1} = 1[H,1]$$



```python
    dscores = props;
    dscores[range(N), y] -= 1
    dscores = dscores / N;                               # [N,C]
    dW2 = np.dot(h1.T, dscores);                         # [N,H].T dot [N,C] = [H,C]
    db2 = np.sum(dscores, axis=0, keepdims=True)         # [1,C]
    dh1 = np.dot(dscores, W2.T)                          # [N,C] dot [H,C].T = (N,H)
    dh1[h1 < 0] = 0                                      # (N,H)  RELU
    dW1 = np.dot(X.T, dh1);                               # [N,D].T dot [N,C] = [D,C]
    db1 = np.sum(dh1, axis=0, keepdims=True)              #[1,H]

    dW2 += reg * W2
    dW1 += reg * W1
```

