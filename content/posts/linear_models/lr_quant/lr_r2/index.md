---
title: "Questions of R Square"
date: 2024-11-26T14:00:25+08:00
description: Linear Regression Questions regarding R Square
menu:
  sidebar:
    name: R Square
    identifier: lr-quant-r2
    parent: lr-quant
    weight: 207
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

### 0.1 Definition
$$
R^2 = \frac{SSR}{SST} = \frac{||\hat{Y} - \overline{Y}||^2}{||Y - \overline{Y}||^2} = 1 - \frac{SSE}{SST} = 1 - \frac{||\hat{\epsilon}||^2}{||Y - \overline{Y}||^2}
$$

### 0.2 Replicate Data
If the sample is doubled, how does $R^2$ change?

Denote the original sample is $(X,Y)$, then after replication:
$$
\left(
\begin{pmatrix}
    X \cr X
\end{pmatrix},
\begin{pmatrix}
    Y \cr Y
\end{pmatrix}
\right)
$$

So we can get $\hat{\beta}$, which is as same as its original value:
$$
\begin{align*}
    \hat{\beta} &= \left(\begin{pmatrix} X \cr X\end{pmatrix}^\top \begin{pmatrix} X \cr X\end{pmatrix}\right)^{-1}\begin{pmatrix} X \cr X\end{pmatrix}^\top \begin{pmatrix} Y \cr Y\end{pmatrix} \cr
    &= (2X^\top X)^{-1}(2X^\top Y) \cr
    &= (X^\top X)^{-1}X^\top Y
\end{align*}
$$

Then we can compute new $R^2$:
$$
\begin{align*}
    R^2 &= \frac{SSR}{SST} \cr
    &= \frac{||(\hat{Y}, \hat{Y})^\top - (\overline{Y}, \overline{Y})^\top||^2}{||(Y, Y)^\top - (\overline{Y}, \overline{Y})^\top||^2} \cr
    &= \frac{2||\hat{Y} - \overline{Y}||^2}{2||Y - \overline{Y}||^2} \cr
    &= \frac{||\hat{Y} - \overline{Y}||^2}{||Y - \overline{Y}||^2}
\end{align*}
$$

So $R^2$ will not change while we duplicate the sample.

### 0.3 Expectation of $R^2$
If $X,Y\sim N(0,1)$, $\{X\_i\}\_{i=1}^{100}$ and $\{Y\_i\}\_{i=1}^{100}$ are 100 samples sampled from $X,Y$. Perform regression $y\sim(x)$ using these 100 samples, and compute $E[R^2]$.

We have proved that $R^2 = \rho_{XY}^2$.

**Understanding the Distribution of $r_{XY}$ When $\rho = 0$**

When the population correlation $\rho = 0$, the sampling distribution of the sample correlation coefficient $r_{XY}$ follows a specific distribution that depends on the sample size $n$. Specifically, $r_{XY}$ is distributed such that $r_{XY}^2$ follows a Beta distribution with parameters $\alpha = \frac{1}{2}$ and $\beta = \frac{n - 2}{2}$:
$$
r_{XY}^2 \sim \text{Beta}\left( \alpha = \frac{1}{2}, \beta = \frac{n - 2}{2} \right)
$$

The Beta distribution is a continuous probability distribution defined on the interval $[0, 1]$ and is parameterized by two positive shape parameters $\alpha$ and $\beta$.

**Calculating the Expected Value of $r_{XY}^2$**

The expected value of a Beta-distributed random variable $Z \sim \text{Beta}(\alpha, \beta)$ is:
$$
\mathbb{E}[Z] = \frac{\alpha}{\alpha + \beta}
$$

Applying this to $r_{XY}^2$:
$$
\mathbb{E}[r_{XY}^2] = \frac{\alpha}{\alpha + \beta} = \frac{\frac{1}{2}}{\frac{1}{2} + \frac{n - 2}{2}} = \frac{\frac{1}{2}}{\frac{n - 1}{2}} = \frac{1}{n - 1}
$$

Thus, the expected value of $r_{XY}^2$ is:
$$
\mathbb{E}[r_{XY}^2] = \frac{1}{n - 1}
$$

**Relating Variance to Expected Value of $r_{XY}^2$**

Since $\mathbb{E}[r_{XY}] = 0$ when $\rho = 0$, the variance of $r_{XY}$ is:
$$
\operatorname{Var}(r_{XY}) = \mathbb{E}[r_{XY}^2] - (\mathbb{E}[r_{XY}])^2 = \mathbb{E}[r_{XY}^2]
$$

### 0.4 Outsample $R^2$
Will outsample $R^2<0$? (Hint: SSR + SSE + SST)

$$
R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST} = 1 - \frac{||\hat{\epsilon}||^2}{||Y - \overline{Y}||^2}
$$

If the prediction errors are big enough so as to make $||\hat{\epsilon}||^2 > ||Y - \overline{Y}||^2$, the outsample $R^2$ will be negative.