---
title: "第二章 预备知识"
date: 2024-07-27T17:06:25+08:00
description: 动手深度学习 (Pytorch) 第二章
menu:
  sidebar:
    name: 第二章 预备知识
    identifier: dltorch-ch2
    parent: dltorch
    weight: 11
hero: /images/sunset.jpg
author:
  name: 赵浩翰
  image: /images/author/zhh.png
tags:
- Deep Learning
- Pytorch
math: true
---

接下来一段时间，我想自学深度学习，使用的教材为 **动手学深度学习 (Pytorch 版)**，该书有线上网址，且提供配套代码和相关 Python 包，详情可参见 [动手学深度学习](https://zh.d2l.ai/)。

第一章内容，介绍了深度学习的相关背景和应用场景，以及深度学习领域常见的术语和名词，有一定机器学习经验的人或许已比较熟悉，故不再赘述，我们直接从第二章开始。

## 1. Tensor 操作和数据预处理
深度学习中的数据以**张量** (tensor) 形式存储，支持 GPU 计算和 autograd 自动微分。

张量的创建、变形、运算 (按元素 / 矩阵)、广播机制、索引、切片等均与 `numpy.ndarray` 类似。

{{< alert type="success" >}}
节省内存：
`Y = X + Y` 不是原地操作，即：`id(Y = X + Y) != id(Y)`，会分配新的内存。
使用 `Y[:] = X + Y` 或 `Y += X` 进行原地操作以避免不必要的内存分配。
{{< /alert >}}

Tensor 可以与其他 Python 对象互相转换，如 `tensor.numpy()`。大小为 1 的张量可以转化为 Python 标量，使用 `tensor.item()` 或 `float(tensor)` 等。

数据需要经过预处理，如填充 `nan`，标准化等，可以借用其他 Python 包处理后再转化为 tensor。

## 2. 线性代数
1. 标量，以小写字母 $x,y,z$ 等表示。
2. 向量，以粗体小写字母 $\bold{x,y,z}$ 表示，向量的维度 (形状) 代表元素个数 (向量长度)，可以使用 `len(x), x.shape` 获取。以列向量为默认的向量方向，例如：
$$
\begin{equation*}
    x = \begin{bmatrix*}
        x_{1} \cr
        x_{2} \cr
        \vdots \cr
        x_{n}
    \end{bmatrix*}
\end{equation*}
$$
3. 矩阵，以粗体大写字母 $\bold{X,Y,Z}$ 表示，是具有两个轴的张量。
4. 张量 (此处指代数对象)，矩阵的拓展，一种具有更多轴的数据结构，使用特殊字体的大写字母 $X, Y, Z$ 表示。

张量的计算，与 `numpy.ndarray` 相同，普通的加减乘除、求和、平均、向量点积、矩阵 hadamard 积、矩阵-向量积、矩阵乘法、范数等。

## 3. 微积分
### 3.1 导数和微分
设 $y=f(x)$，其导数被定义为：
$$
f^{'}(x) = \lim_{h\rightarrow 0}\frac{f(x+h) - f(x)}{h}
$$

以下符号等价：
$$
f^{'}(x)=y^{'}=\frac{dy}{dx}=\frac{df}{dx}=\frac{d}{dx}f(x)=Df(x)=D_{x}f(x)
$$

自行回顾求导法则。

### 3.2 偏导数
将微分的思想拓展至多元函数上，设 $y=f(x_{1},\cdots,x_{n})$ 是一个 $n$ 元函数，其关于第 $i$ 个变量 $x_{i}$ 的偏导数为：
$$
\frac{\partial y}{\partial x_{i}} = \lim_{h\rightarrow 0}\frac{f(x_{1},\cdots,x_{i-1},x_{i}+h,x_{i+1},\cdots,x_{n}) - f(x_{1},\cdots,x_{i},\cdot,x_{n})}{h}
$$

以下表达等价：
$$
\frac{\partial y}{\partial x_{i}} = \frac{\partial f}{\partial x_{i}} = f_{x_{i}} = f_{i} = D_{i}f = D_{x_{i}}f
$$

### 3.3 梯度
函数的梯度（gradient）向量，即其对所有变量的偏导数。设函数 $f: \mathbb{R}^n \to \mathbb{R}$ 的输入是一个 $n$ 维向量 $\mathbf{x} = [x_1, x_2, \ldots, x_n]$，并且输出是一个标量。函数 $f(\mathbf{x})$ 相对于 $\mathbf{x}$ 的梯度是一个包含 $n$ 个偏导数的向量：
$$
\nabla_{\mathbf{x}} f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial f(\mathbf{x})}{\partial x_1} & \frac{\partial f(\mathbf{x})}{\partial x_2} & \cdots & \frac{\partial f(\mathbf{x})}{\partial x_n}
\end{bmatrix}^\top,
$$

假设 $\mathbf{x}$ 为 $n$ 维向量，在微分多元函数时经常使用以下规则：
- 对于所有 $\mathbf{A} \in \mathbb{R}^{n \times n}$，都有 $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x} = \mathbf{A}^\top$
- 对于所有 $\mathbf{A} \in \mathbb{R}^{n \times m}$，都有 $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} = \mathbf{A}$
- 对于所有 $\mathbf{A} \in \mathbb{R}^{n \times n}$，都有 $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x} = (\mathbf{A} + \mathbf{A}^\top) \mathbf{x}$
- $\nabla_{\mathbf{x}} \|\mathbf{x}\|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2 \mathbf{x}$

同样，对于任意矩阵 $\mathbf{X}$，都有 $\nabla_{\mathbf{X}} \|\mathbf{X}\|_F^2 = 2 \mathbf{X}$。

### 3.4 链式法则
考虑单变量函数 $y = f(u)$ 和 $u = g(x)$， 假设都是可微的，根据链式法则：
$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

当函数具有任意数量的变量时，假设可微分函数 $y$ 有变量 $u_1, u_2, \cdots, u_m$，其中每个可微分函数 $u_i$ 都有变量 $x_1, x_2, \cdots, x_n$。注意，$y$ 是 $x_1, x_2, \cdots, x_n$ 的函数。对于任意 $i = 1, 2, \cdots, n$，链式法则给出：

$$
\frac{\partial y}{\partial x_i} = \frac{\partial y}{\partial u_1} \frac{\partial u_1}{\partial x_i} + \frac{\partial y}{\partial u_2} \frac{\partial u_2}{\partial x_i} + \cdots + \frac{\partial y}{\partial u_m} \frac{\partial u_m}{\partial x_i}.
$$

## 4. 自动微分
Pytorch 使用自动微分 (automatic differentiation) 来加快求导。 实际中，根据设计好的模型，系统会构建一个计算图 (computational graph)，以跟踪计算是哪些数据通过哪些操作组合起来产生输出，并使用自动微分进行反向传播梯度。 这里，反向传播 (backpropagate) 意味着跟踪整个计算图，填充关于每个参数的偏导数。

当 $y$ 是标量时，可以通过链式法则反向求导输入参数的梯度，该梯度是一个与输入向量 $\bold{x}$ 形状相同的向量。

当 $y$ 不是标量时，向量 $\bold{y}$ 关于 $\bold{x}$ 的导数是一个矩阵，更高阶情况下是一个高阶张量。但当调用向量的反向传播计算时，通常会试图计算一批训练样本中每个组成部分的损失函数的导数。 这里、的目的不是计算微分矩阵，而是单独计算批量中每个样本的偏导数之和。如：
```python
x = torch.arange(4.0)
y = x * x
y.sum().backward() # y.backward(torch.ones(len(x)))
```

如果想将某些计算移到记录的计算图之外，例如，假设 $y$ 是作为 $x$ 的函数计算的，而 $z$ 则是作为 $y$ 和 $x$ 的函数计算的。比如，我们想计算 $z$ 关于 $x$ 的梯度，但由于某种原因，希望将 $y$ 视为一个常数，并且只考虑到 $x$ 在 $y$ 被计算后发挥的作用。

此时，可以使用 `detach()` 分离 $y$ 来返回一个新变量 $u$，该变量与 $y$ 具有相同的值，但丢弃计算图中如何计算 $y$ 的任何信息。换句话说，梯度不会向后流经 $u$ 到 $x$。因此，下面的反向传播函数计算 $z=u*x$ 关于 $x$ 的偏导数，$u$ 为常数。
```python
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
```

## 5. 概率
* 单个随机变量：概率分布
* 多个随机变量
  * 联合概率 (联合分布)
  * 条件概率 (条件分布)
  * 贝叶斯定理
  * 边际化 (边际概率、边际分布 -- 全概率公式) $$P(B)=\sum_{A}P(A,B)$$
  * 独立性
  * 期望和方差