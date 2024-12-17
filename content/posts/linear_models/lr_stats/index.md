---
title: "Linear Regression and Stats"
date: 2024-11-17T15:45:25+08:00
description: Linear Regression and Stats
menu:
  sidebar:
    name: Linear Regression and Stats
    identifier: lr-stats
    parent: linear-models
    weight: 202
hero: /images/sky.jpg
tags:
- Linear Model
- Linear Regression
categories:
- Basic
---

This post focuses on **Ordinary Linear Regression**.

## 0.1 Simple Linear Regression

The most basic version of a linear model is **Simple Linear Regression**, which can be expressed by this formular:
$$
y = \alpha + \beta \times x + \epsilon
$$
where $\alpha$ is called **intercept**, $\beta$ is called **slope**, and $\epsilon$ is called **residual**.

The coefficients of **Simple Linear Regression** can be solved using **Least Squres Method**, by minimizing $\sum_{i=1}^{n}(y_i-\hat{y}_i)^2$.

$$
\beta = \frac{\sum_{i=1}^{n} (x_{i} - \overline{x}) (y_{i} - \overline{y})} {\sum_{i=1}^{n} (x_{i} - \overline{x})^2}, \quad \alpha = \overline{y} - \beta\times\overline{x}
$$

{{< alert type="info" >}}
An important feature of linear regression: point $(\overline{x}, \overline{y})$ must locates on the regression line.
{{< /alert >}}

### 0.1.1 Important Features of SLR
Here I want to illustrate the relationships between $\beta,\rho, Var, Cov, R^2$.
From the solution of $\beta$, we can also get:
$$
\beta = \frac{\sum_{i=1}^{n} (x_{i} - \overline{x}) (y_{i} - \overline{y})} {\sum_{i=1}^{n} (x_{i} - \overline{x})^2} = \frac{Cov(x,y)}{Var(x)}
$$

So if we exchange the position of $x,y$ in **SLR**, $x=\alpha_1 + \beta_1\times y + \epsilon_1$, we can get:
$$
\beta_1 = \frac{Cov(x,y)}{Var(y)}
$$

And we know: 
$$
\rho_{xy} = \frac{Cov(x,y)}{\sqrt{Var(x)Var(y)}}
$$

So we have:
$$
\beta = \rho_{xy}\frac{\sqrt{Var(y)}}{\sqrt{Var(x)}}, \quad \beta_1 = \rho_{xy}\frac{\sqrt{Var(x)}}{\sqrt{Var(y)}}, \quad\beta\times\beta_1=\rho_{xy}^2
$$

For $R^2$, we need to know what are **TSS**, **ESS** and **RSS**.

**TSS** (Total Sum of Squares): $TSS = \sum_{i=1}^{n}(y-\overline{y})^2$

**ESS** (Explained Sum of Squares): $ESS = \sum_{i=1}^{n}(\hat{y}-\overline{y})^2$

**RSS** (Residual Sum of Squares): $RSS = \sum_{i=1}^{n}(y-\hat{y})^2$

$$
R^2 = \frac{ESS}{TSS} = \frac{\sum_{i=1}^{n}(\hat{y}-\overline{y})^2}{\sum_{i=1}^{n}(y-\overline{y})^2}
$$

Because $\hat{y} = \alpha + \beta\times x = \overline{y} - \beta\times\overline{x} + \beta\times x = \overline{y}  + \beta\times(x - \overline{x})$, so:
$$
R^2 = \frac{\beta^2\sum_{i=1}^{n}(x-\overline{x})^2}{\sum_{i=1}^{n}(y-\overline{y})^2} = \rho_{xy}^2\frac{Var(y)}{Var(x)}\times\frac{Var(x)}{Var(y)} = \rho_{xy}^{2}
$$

## 0.2 Bivariate Regression
$$
y = \alpha + \beta_1 x_1 + \beta_2 x_2 + \epsilon 
$$


## 0.3 Multivariate Regression
$$y_{i}=\beta_{0}+\beta_{1}\times x_{i1}+\cdots+\beta_{p}\times x_{ip}+\epsilon_{i},\quad i=1,2,\cdots,n$$
$$
\begin{align*}
\boldsymbol{Y}&=(y_{1},y_{2},\cdots,y_{n})^\top \cr
\boldsymbol{X}&=\begin{bmatrix}1 & x_{11} & x_{12} & \cdots & x_{1p} \cr 1 & x_{21} & x_{22} & \cdots & x_{2p} \cr \vdots & \vdots & \vdots & \vdots & \vdots \cr 1 & x_{n1} & x_{n2} & \cdots & x_{np} \end{bmatrix} \cr
\boldsymbol{\beta}&=(\beta_{0},\beta_{1},\cdots,\beta_{p})^\top \cr
\boldsymbol{\epsilon}&=(\epsilon_{1}, \epsilon_{2},\cdots,\epsilon_{n})^\top 
\end{align*}
$$

We have:
$$
\boldsymbol{Y} = \boldsymbol{X\beta} + \boldsymbol{\epsilon}
$$

ormally, the OLS estimator of $\beta$ is defined by the minimizer of the **residual sum of squares (RSS)**:
$$
\hat{\boldsymbol{\beta}}=arg\ min_{\beta}\ S(\boldsymbol{\beta})
$$
$$
S(\boldsymbol{\beta})=(\boldsymbol{Y}-\boldsymbol{X\beta})^\top(\boldsymbol{Y}-\boldsymbol{X\beta})=\sum\limits_{i=1}^{n}(y_{i}-\beta_{0}-\beta_{1}\times x_{i1}-\cdots-\beta_{p}\times x_{ip})^{2}
$$

Derive it we can get:
$$
\hat{\boldsymbol{\beta}}=(\boldsymbol{X^\top X})^{-1}\boldsymbol{X^\top Y}
$$