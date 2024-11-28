---
title: "Questions of Coefficients"
date: 2024-11-16T11:45:25+08:00
description: Linear Regression Questions regarding Coefficients
menu:
  sidebar:
    name: Coefficients
    identifier: lr-quant-coefficients
    parent: lr-quant
    weight: 206
hero: /images/sky.jpg
tags:
- Linear Model
- Linear Regression
- Quant
categories:
- Basic
---

1. $y\sim(1,\boldsymbol{x})$, regress $y$ on $x$ with intercept.
2. $y\sim(\boldsymbol{x})$, regress $y$ on $x$ without intercept.
3. In the context of Statistics, SSE (Sum of Squares due to Error) and SSR (Sum of Squares due to Regression) are used more frequently. But in Economitrics, ESS (Explained Sum of Squares) and RSS (Residual Sum of Squares) are prefered.

### 0.1 Product of $\beta$
Denote $\beta_1$ as the least squares optimal solution of $y=\beta x+\epsilon$, $\beta_2$ as the least squares optimal solution of $x=\beta y+\epsilon$. Find the min and max values of $\beta_1\beta_2$.
$$
\beta_1 = \frac{Cov(X,Y)}{Var(X)},\quad \beta_2 = \frac{Cov(X,Y)}{Var(Y)}\Rightarrow \beta_1\beta_2 = \rho_{XY}^2 \in [-1,1]
$$

### 0.2 Load Memory

When performing linear regression, if the dataset is too large to fit into memory at once, how can this issue be resolved?

Suppose $n \gg p$, the extimator of OLS is:
$$
\hat{\boldsymbol{\beta}} = (\boldsymbol{X}^\top \boldsymbol{X})^{-1}\boldsymbol{X}^\top \boldsymbol{Y}
$$

Divide our data into $n$ pieces, every piece $(\boldsymbol{X}_i,\boldsymbol{Y}_i)$ can be loaded to memory:
$$
\boldsymbol{X} = 
\begin{pmatrix}
    \boldsymbol{X}_1 \cr \boldsymbol{X}_2 \cr \vdots \cr \boldsymbol{X}_n
\end{pmatrix}
\quad
\boldsymbol{Y} = 
\begin{pmatrix}
    \boldsymbol{Y}_1 \cr \boldsymbol{Y}_2 \cr \vdots \cr \boldsymbol{Y}_n
\end{pmatrix}
$$ 

So we can get:
$$
\begin{align*}
    \hat{\boldsymbol{\beta}} &= 
    \left(
        \begin{pmatrix} \boldsymbol{X}_1 \cr \boldsymbol{X}_2 \cr \vdots \cr \boldsymbol{X}_n \end{pmatrix}^\top\begin{pmatrix} \boldsymbol{X}_1 \cr \boldsymbol{X}_2 \cr \vdots \cr \boldsymbol{X}_n \end{pmatrix}
    \right)^{-1}\begin{pmatrix} \boldsymbol{X}_1 \cr \boldsymbol{X}_2 \cr \vdots \cr \boldsymbol{X}_n \end{pmatrix}^\top\begin{pmatrix} \boldsymbol{Y}_1 \cr \boldsymbol{Y}_2 \cr \vdots \cr \boldsymbol{Y}_n \end{pmatrix}
    \cr
    &= \left(\sum\_{i=1}^{n}\boldsymbol{X}\_i^\top \boldsymbol{X}\_i \right)^{-1} \left(\sum\_{i=1}^{n}\boldsymbol{X}\_i^\top\boldsymbol{Y}\_i \right)
\end{align*}
$$

So we only need to conpute $\sum\_{i=1}^{n}\boldsymbol{X}\_i^\top \boldsymbol{X}\_i$ and $\sum\_{i=1}^{n}\boldsymbol{X}\_i^\top\boldsymbol{Y}\_i$.

### 0.3 $y\sim(1,x)$ and $x\sim(1,y)$
Denote $\beta_1,\beta_2$ as the coefficients of these two regressions. We have:
$$
\beta_1 = \frac{Cov(X,Y)}{Var(X)},\quad \beta_2 = \frac{Cov(X,Y)}{Var(Y)}\Rightarrow \beta_1\beta_2 = \rho_{XY}^2
$$

### 0.4 $y\sim(x)$ and $x\sim(y)$
If the coefficient of $y\sim(x)$ is 1, what condition does the coefficient of $x\sim(y)$ satisfy? If the coefficient of $y\sim(x)$ is $k$, what about the corresponding coefficient of $x\sim(y)$?
$$
y = kx \Rightarrow x = \frac{1}{k}y
$$

### 0.5 Classes of Sample
Assume that in a dataset, $n_1$ samples belong to class $a$, and $n_2$ samples belong to class $b$, where $n_1 = 2n_2$. If linear regression is applied directly, the regression results will be biased towards the linear relationship of class $a$. How can this issue be resolved?

The imbalance in the number of samples from classes $a$ and $b$ will lead to a bias in the regression coefficients, as class $a$ has twice as many samples as class $b$, resulting in it having more influence on the regression line. Here’s how this can be addressed:

- Reweight the Samples:
   - Assign **weights** to each sample such that the contribution of each class to the loss function is balanced. For example:
     $$
     w_a = \frac{1}{n_1}, \quad w_b = \frac{1}{n_2}
     $$
     Multiply the loss function by these weights to ensure that both classes have equal influence.
   - For example, if using ordinary least squares (OLS), minimize the weighted sum of squared residuals:
    $$
    \text{Loss} = \sum\_{i \in a} w_a \cdot \text{residual}_i^2 + \sum\_{j \in b} w_b \cdot \text{residual}_j^2
    $$
- Downsample Class $a$:
   - Randomly sample $n_2$ instances from class $a$ so that the number of samples in both classes is equal. This reduces the bias but may result in losing information from the discarded samples.
- Oversample Class $b$:
   - Duplicate or synthetically generate samples in class $b$ until the number of samples matches class $a$. Techniques like SMOTE (Synthetic Minority Oversampling Technique) can be used for this purpose.
- Add Class Balancing Terms:
   - Include a penalty term in the loss function to account for the imbalance. For example, use stratified approaches in weighted regression models to penalize the majority class more heavily.
- Use Regularized Regression Models:
   - Use models like Ridge or LASSO regression with regularization to prevent overfitting towards the majority class.

### 0.6 Angle between $Y$ and $X_i$
For $y\sim(x_1,x_2,\cdots,x_p)$, supposing $||x_i||=1$, if the angles between $\hat{Y}=X\hat{\beta}$ and any $x_i$ are equal, what's the value of $\hat{\beta}$?
$$
\cos(\theta_i) = \frac{<x_i, \hat{Y}>}{||X_i||\cdot||\hat{Y}||} = \frac{<x_i, X\hat{\beta}>}{||X_i||\cdot||\hat{Y}||} = \frac{x_i^\top X\hat{\beta}}{||\hat{Y}||}
$$

So "All angles are equal" is equivalent to:
$$
X^\top X\beta=\lambda\boldsymbol{1},\lambda\in R,\boldsymbol{1}=(1,1,\cdots,1)^\top\in\mathbb{R}^p
$$

So we can transform the original question to this convex optimization question:
$$
\min_{\beta,\lambda}||Y-X\beta||^2 \qquad s.t. X^\top X\beta = \lambda\boldsymbol{1}
$$

Lagrange Function:
$$
L(\beta,\lambda,\mu)=||Y-X\beta||^2-\mu^\top(X^\top X\beta - \lambda\boldsymbol{1})
$$

The first order conditions are:
$$
\begin{equation*}
    \begin{cases}
        \frac{\partial L}{\partial \beta} = -2X^\top Y + X^\top X(2\beta-\mu) \cr
        \frac{\partial L}{\partial \lambda} = \mu^\top\boldsymbol{1} \cr
        \frac{\partial L}{\partial \mu} = -(X^\top X\beta - \lambda\boldsymbol{1})
    \end{cases}
\end{equation*}
$$

Then we can get:
$$
\lambda = \frac{\boldsymbol{1}^\top(X^\top X)^{-1}X^\top Y}{\boldsymbol{1}^\top(X^\top X)^{-1}\boldsymbol{1}}
\qquad
\beta = \frac{\boldsymbol{1}^\top(X^\top X)^{-1}X^\top Y(X^\top X)^{-1}\boldsymbol{1}}{\boldsymbol{1}^\top(X^\top X)^{-1}\boldsymbol{1}}
$$