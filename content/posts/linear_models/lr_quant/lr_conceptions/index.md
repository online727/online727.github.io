---
title: "Questions of Conceptions"
date: 2024-11-16T11:45:25+08:00
description: Linear Regression Questions regarding Conceptions
menu:
  sidebar:
    name: Conceptions
    identifier: lr-quant-conceptions
    parent: lr-quant
    weight: 204
hero: /images/sky.jpg
tags:
- Linear Model
- Linear Regression
- Quant
categories:
- Basic
---

With reference to [donggua](https://zhuanlan.zhihu.com/p/443658898).

I complete the answers of these questions.

## 0.1 Notations
1. $y\sim(1,\boldsymbol{x})$, regress $y$ on $x$ with intercept.
2. $y\sim(\boldsymbol{x})$, regress $y$ on $x$ without intercept.
3. In the context of Statistics, SSE (Sum of Squares due to Error) and SSR (Sum of Squares due to Regression) are used more frequently. But in Economitrics, ESS (Explained Sum of Squares) and RSS (Residual Sum of Squares) are prefered.

## 0.2 Conceptions and Basic Definitions
##### 0.2.1. The assumptions of LR
**Gauss-Markov Theory**: Under the assumptions of classical linear regression, the ordinary least squares (OLS) estimator is the linear unviased estimator with the minimum variance. **(BLUE)**
1. **Linear in Parameters**: The model in the population can be written as:
$$
y=\beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_k x_k + u
$$
where $\beta_1,\beta_2,\cdots,\beta_k$ are the unknown parameters (constants) of interest and $u$ is an unobservable random error or disturbance term.
2. **Random Sampling**: We have a random sample of $N$ observations, $\{(x_{i1}, x_{i2}, \cdots, x_{ik}, y_i)\}_{i=1}^N$ following the population model defined in assumption 1.
3. **No Perfect Collinearity**: In the sample (and therefore in the population), none of the independent variables is constant, and there are no exact linear relationships among the independent variables.
4. **Zero Conditional Mean**: The error $u$ has an expected value of zero given any value of the independent variables. In the other words, $E(u|x_1,x_2,\cdots,x_k)=0$.
5. **Homoskedasticity**: The error $u_i$ has the same variance given any values of the explanatory variables. In other words, $E(u_i^2|x_1,x_2,\cdots,x_k)=\sigma^2$.
6. **Normality**: The population error $u$ is independent of the explanatory variables $x_1,x_2,\cdots,x_k$ and is normally distributed with zero mean and constant variance: $u\sim N(0,\mu)$.

The properties of OLS estimators: **Best Linear Unbiased Estimators (BLUEs)**: they have smallest variance among all linear unbiased estimators. These properties only need the first 5 assumptions to be proved (Unbiased only needs the first 4 assumptions). The 6.th assumption is only utilized when conducting hypothesis testing on the parameters.

##### 0.2.2 The loss function of LR and why we choose it?
**MSE (Mean Square Error)** is the most frequently used loss function (optimization objection) of OLS. What's more, **MAE (Mean Absolute Error)** is also used is some situations.
$$
MSE = \frac{\sum\_{i=1}^n(y\_i-\hat{y}\_i)^2}{n}, \quad MAE = \frac{\sum\_{i=1}^n|y\_i-\hat{y}\_i|}{n}
$$

Compared to MAE, MSE has many good properties:
* **Unique optimal solution**: For the least squares method, as long as the variables are not multicollinear, the solution is unique. However, for the least absolute deviation method, the solution is not fixed. For example, if there are no independent variables (only the dependent variable $y$), the least squares method predicts the **mean** of $y$, while the least absolute deviation method predicts the **median** of $y$. The mean and median are generally not the same. For example, if the data values are 0 and 2, the least squares method predicts 1 as the forecast value, while the least absolute deviation method predicts a random number between 0~2 as the forecast value.
* **Ease of Derivation**: For the least squares method, since the optimization problem is to minimize the sum of squared errors, the problem can be solved by linear algebra. However, for the least absolute deviation method, the target is to minimize the sum of absolute errors, which is not differentiable at certain points, making the derivation process more complicated. Also, because the absolute values can introduce a subgradient, the solution process becomes even more challenging.
* **Excellent Statistical Properties**: If the error term satisfies a normal distribution, the least squares method can maximize the likelihood function (MLE), meaning it is the best linear unbiased estimator (BLUE).

However, the primary reason why the least absolute deviation method is not widely used is due to a significant disadvantage of the least squares method: **Vulnerability to Outliers**.

The least squares method is easily influenced by outliers. Therefore, when comparing predicted values with real-world data that contains outliers, the least squares method often deviates significantly. Adjusting the weights of outliers can alleviate the issue, but in extreme cases, the least absolute deviation method is still more robust in dealing with outliers.

In summary, while the least squares method produces accurate and reliable predictions in most scenarios, it may perform poorly when there are outliers in the data. **In cases where outliers are present, the least squares method is not an ideal choice.**

##### 0.2.3 Bayesian LR and OLS
The relationship between Bayesian linear regression and the ordinary least squares (OLS) method for solving linear regression?
* If we assume a non-informative prior for $\beta$ and use the posterior mode as the point estimate, the result will be identical to the OLS regression outcome.
* If the prior for $\beta$ follows a Laplace distribution, the estimation result corresponds to the LASSO regression, where the regularization coefficient is related to the parameters of the Laplace distribution.
* If the prior for $\beta$ follows a normal distribution, the estimation result corresponds to Ridge regression, where the regularization coefficient is related to the parameters of the normal distribution.

For a non-informative prior on the parameters, it essentially means we impose no restrictions on the parameters, resulting in the same outcome as OLS.

However, if we assume that the parameters follow a certain distribution, it is equivalent to imposing some restrictions on them, and the results obtained in this case are akin to being adjusted through certain regularization techniques.

With reference to [here](https://zhuanlan.zhihu.com/p/86009986)

##### 0.2.4 When will $\overline{y} = \overline{\hat{y}}$?
Rewrite the condition of the question, we can get: $\frac{1}{n}\boldsymbol{1}^\top Y=\frac{1}{n}\boldsymbol{1}^\top\hat{Y}$, which is equivalent to $\boldsymbol{1}^\top(Y-\hat{Y})=0$.

We can prove that $\boldsymbol{X}^\top(Y-\hat{Y})=0$:
$$
\hat{\boldsymbol{Y}}=\boldsymbol{X}\hat{\boldsymbol{\beta}} = \boldsymbol{X}(\boldsymbol{X}^\top\boldsymbol{X})^{-1}\boldsymbol{X}^\top \boldsymbol{Y}
$$

Multiple $\boldsymbol{X}^\top$ on the left:
$$
\boldsymbol{X}^\top\hat{\boldsymbol{Y}}=\boldsymbol{X}^\top \boldsymbol{Y} \Rightarrow \boldsymbol{X}^\top (\boldsymbol{Y} - \hat{\boldsymbol{Y}}) = 0
$$

So if $\boldsymbol{X}$ contains a colume vector like $\boldsymbol{1}$, an intercept term, $\overline{y} = \overline{\hat{y}}$.

While sometime LR does not contain an intercept term, so the answer is $rank(\boldsymbol{X},\boldsymbol{1})=rank(\boldsymbol{X})$ (assume $n>p$).
