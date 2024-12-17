---
title: "Questions of R Square and Corr Coef"
date: 2024-11-26T17:00:25+08:00
description: Linear Regression Questions regarding R Square and corr coef
menu:
  sidebar:
    name: R Square and Corr
    identifier: lr-quant-r2rho
    parent: lr-quant
    weight: 208
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

### 0.1 Bivariate Regression
Denote the $R^2$ of $y\sim(1,x_1)$ as $R_1^2$, $y\sim(1,x_2)$ as $R_2^2$, $y\sim(1,x_1,x_2)$ as $R_3^2$. And we have $corr(x_1,x_2)=\rho$.

Find the relationships between $R_1^2,R_2^2,R_3^2,\rho$. Specifically, if $\rho=1$ or $\rho=0$, the relationships between $R_1^2,R_2^2,R_3^2$.

First, we must keep this in mind: $R^2 = \rho_{xy}^2$ for $y\sim(1,x)$.

So we have:
$$
R_1^2 = \rho_{x_1,y}^2\quad R_2^2 = \rho_{x_1,y}^2
$$

For multiple linear regression, the coefficient-of-determination ($R^2$) can be written in terms of the pairwise correlations for the variables using the quadratic form:
$$
R^2 = \boldsymbol{r_{y,x}}^\top\boldsymbol{r_{x,x}}^{-1}\boldsymbol{r_{y,x}}
$$
where $\boldsymbol{r_{y,x}}$ is the vector of correlations between the response vector and each of the explanatory vectors, and $\boldsymbol{r_{x,x}}$ is the matrix of correlations between the explanatory vectors. In the case of a bivariate regression:
$$
\begin{align*}
    R_3^2 &= \begin{bmatrix} r_{Y,X_1} \cr r_{Y,X_2}\end{bmatrix}^\top \begin{bmatrix} 1 & r_{X_1,X_2} \cr r_{X_1,X_2} & 1 \end{bmatrix}^{-1} \begin{bmatrix} r_{Y,X_1} \cr r_{Y,X_2} \end{bmatrix} \cr
    &= \frac{1}{1-r_{X_1,X_2}^2} \begin{bmatrix} r_{Y,X_1} \cr r_{Y,X_2} \end{bmatrix}^\top \begin{bmatrix} 1 & -r_{X_1,X_2} \cr - r_{X_1,X_2} & 1 \end{bmatrix} \begin{bmatrix} r_{Y,X_1} \cr r_{Y,X_2} \end{bmatrix} \cr
    &= \frac{1}{1-r_{X_1,X_2}^2} (r_{Y,X_1}^2 + r_{Y,X_2}^2 - 2r_{X_1,X_2}r_{Y,X_1}r_{Y,X_2})
\end{align*}
$$

Denote $S = sign(r_{Y,X_1}) \times sign(r_{Y,X_2})\in \\{-1,1\\}$, we can get:
$$
R_3^2 = \frac{R_1^2 + R_2^2 - 2\rho \cdot S \cdot R_1 \cdot R_2}{1-\rho^2}
$$

So $R_3^2 \in [\max(R_1^2,R_2^2), 1]$, where:
* $R_3^2 = \max(R_1^2,R_2^2)$ while $\rho=1$
* $R_3^2=R_1^2+R_2^2$ while $\rho = 0$
* $R_3^2>R_1^2+R_2^2$ while $\rho < 0$ (maybe need other conditions)
* $R_3^2=1$ in special cases

### 0.2 Corr Coef and $R^2$
The relation between the $R^2$ and $\beta$ of $y\sim (1,x)$?

$$
R^2 = \frac{\sum\_{i-1}^n(\hat{y}_i-\overline{y})^2}{\sum\_{i=1}^n(y_i-\overline{y})^2}
\quad
\beta = \frac{\operatorname{Cov}(X,Y)}{\operatorname{Var}(X)}
\quad
\alpha = \overline{Y} - \beta\overline{X}
\quad
\hat{Y} = \overline{Y} + \beta(X - \overline{X})
$$

So we have:
$$
\begin{align*}
    R^2 &= \frac{\sum\_{i-1}^n(\hat{y}\_i-\overline{y})^2}{\sum\_{i=1}^n(y\_i-\overline{y})^2} \cr
    &= \beta^2\frac{\sum\_{i=1}^n(x_i - \overline{x})^2}{\sum\_{i=1}^n(y_i-\overline{y})^2} \cr
    &= \frac{\operatorname{Cov}^2(X,Y)}{\operatorname{Var}^2(X)} \frac{\operatorname{Var}(X)}{\operatorname{Var}(Y)} \cr
    &= \rho_{X,Y}^2
\end{align*}
$$

### 0.3 Corr Coef with $\hat{y}$
Perfome regression $y\sim(1,x_1,x_2)$ and get $R^2$, and we also have $Corr(y,x_1)=\rho_1, Corr(y,x_2)=\rho_2$, try to compute $Corr(\hat{y},x_1)$.

An important thing we need to prove: $Cov(y-\hat{y})=0$.
$$
\hat{\boldsymbol{Y}}=\boldsymbol{X}\hat{\boldsymbol{\beta}} = \boldsymbol{X}(\boldsymbol{X}^\top\boldsymbol{X})^{-1}\boldsymbol{X}^\top \boldsymbol{Y}
$$

Multiple $\boldsymbol{X}^\top$ on the left:
$$
\boldsymbol{X}^\top\hat{\boldsymbol{Y}}=\boldsymbol{X}^\top \boldsymbol{Y} \Rightarrow \boldsymbol{X}^\top (\boldsymbol{Y} - \hat{\boldsymbol{Y}}) = 0
$$

Then we can compute:
$$
\begin{align*}
    Corr(\hat{y},x_1) &= \frac{Cov(\hat{y},x_1)}{\sqrt{Var(\hat{y})Var(x_1)}} \cr
    &= \frac{Cov(\hat{y}-y,x_1) + Cov(y,x_1)}{\sqrt{Var(\hat{y})Var(x_1)}} \cr
    &= \frac{Cov(y,x_1)}{\sqrt{Var(y)Var(x_1)}} \sqrt{\frac{Var(y)}{Var(\hat{y})}} \cr
    &= \frac{\rho_1}{R} \quad (R^2 = \frac{Var(\hat{y})}{Var(y)}, \because \overline{\hat{y}} = \overline{y}\text{ while we doing regression with intercept})
\end{align*}
$$

### 0.4 Corr Coef with $\hat{y}$ - 2
Find the relation between the $R^2$ of $y\sim(x)$ and $Corr(\hat{y},y)$.

If we change the linear regression to any other models $y\sim f(x)$, what will the relation be?

$$
\Corr()
$$