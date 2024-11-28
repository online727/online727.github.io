---
title: "Questions of Assumptions"
date: 2024-11-26T12:00:25+08:00
description: Linear Regression Questions regarding Assumptions
menu:
  sidebar:
    name: Assumptions
    identifier: lr-quant-assumptions
    parent: lr-quant
    weight: 205
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

### 0.1 Heteroskedasticity and Autocorrelation
If the residuals ($\epsilon$) in a linear regression model exhibit heteroskedasticity (non-constant variance) or autocorrelation (correlation between residuals across observations), how will it impact the estimation and inference of $\beta$? How to test and solve these problems?

I've discusses this question in [Linear Regression](../../lr/index.html).

#### 0.1.1 Impact on $\beta$ Estimation
1. **Heteroskedasticity**:
   - The OLS estimator of $\beta$ remains **unbiased** and **consistent**, but it is no longer **efficient** (i.e., it does not have the minimum variance among linear unbiased estimators).
   - The usual standard errors are incorrect, leading to unreliable hypothesis tests and confidence intervals.

2. **Autocorrelation**:
   - Similar to heteroskedasticity, the OLS estimator remains **unbiased** and **consistent**, but it is **inefficient** when autocorrelation is present.
   - Autocorrelation inflates or deflates the standard errors, resulting in incorrect $t$-statistics and $p$-values for hypothesis testing.

#### 0.1.2 Diagnostic Tests
1. **For Heteroskedasticity**:
   - **Breusch-Pagan Test**: Tests whether the residual variance is dependent on the values of independent variables.
   - **White’s Test**: A more general test that detects any form of heteroskedasticity, including non-linear relationships.
   - **Residual Plot**: Plot residuals against fitted values. If the spread increases or decreases systematically, it suggests heteroskedasticity.

2. **For Autocorrelation**:
   - **Durbin-Watson Test**: Tests for the presence of first-order autocorrelation.
   - **Breusch-Godfrey Test**: Tests for higher-order autocorrelation.
   - **Residual Plot**: Plot residuals against time or another ordering variable to visually inspect patterns.

#### 0.1.3 Solutions
1. **For Heteroskedasticity**:
   - Use **robust standard errors** (e.g., White or HC standard errors) to correct for heteroskedasticity in inference.
   - Apply **Weighted Least Squares (WLS)**: Reweight observations based on the inverse of the variance of the residuals.
   - Transform the data: Log or square root transformations can stabilize variance in some cases.

2. **For Autocorrelation**:
   - Use **Newey-West standard errors** to adjust for autocorrelation and heteroskedasticity.
   - Apply **Generalized Least Squares (GLS)**: Models the structure of autocorrelation and heteroskedasticity explicitly.
   - Add lagged dependent variables or other time-series-specific features to the model to capture the autocorrelation structure.

#### 0.1.4 Summary Table

| Issue              | Effect on $\beta$ Estimator | Key Tests                | Solutions                         |
|--------------------|-------------------------------|--------------------------|-----------------------------------|
| Heteroskedasticity | Unbiased, inconsistent SEs   | Breusch-Pagan, White     | Robust SEs, WLS, transformations |
| Autocorrelation    | Unbiased, inconsistent SEs   | Durbin-Watson, Breusch-Godfrey | Newey-West SEs, GLS, model adjustments |

### 0.2 Multicollinearity

* Bad influences
  * Larger values of OLS estamators' variance and standard error.
  * Wider confidence interval.
  * Not significant t-value.
  * Higher $R^{2}$ but not all t values are significant.
  * Not robust, sensitive to the small change of data.
* Diagnoses
  * Higher $R^{2}$ but the number of significant t values is small.
  * High correlation between variables.
  * Partial correlation coefficient.
  * Subsidiary or auxiliary regression, and test $R^{2}_{i}$.
  * Variance inflation factor VIF: $VIF=\frac{1}{1-R^{2}_{i}}$.
* Solutions
  * Delete some variables.
  * More new data.
  * Reset model.
  * Variable transformation.
  * Factor analysis / principal component analysis / ridge regression / LASSO

### 0.3 Heavy Tail of $y$

#### 0.3.1 Issues
- **Sensitivity in Parameter Estimation**
   - **Problem**: Linear regression typically uses Ordinary Least Squares (OLS) to estimate coefficients, which minimizes the sum of squared residuals. Heavy-tailed distributions often include extreme values (outliers), which disproportionately influence the regression estimates.
   - **Cause**: OLS is highly sensitive to large residuals because the squared term amplifies their impact.
   - **Consequence**: The regression coefficients can deviate significantly from their true values, reducing the model's reliability in capturing the overall trend.
- **Reduced Robustness of the Model**
   - **Problem**: With heavy-tailed $y$, the model might overfit to extreme values, losing its ability to represent the central trend of the data.
   - **Cause**: The extreme observations exert an outsized influence on the regression line, potentially leading to distorted predictions.
   - **Consequence**: The model may perform well on training data but fail to generalize effectively to new data.
- **Increased Prediction Error**
   - **Problem**: Heavy tails in $y$ imply a higher probability of extreme values. The regression model struggles to predict these values accurately.
   - **Cause**: Linear regression assumes normally distributed errors, which is incompatible with heavy-tailed distributions, leading to larger prediction deviations.
   - **Consequence**: Metrics like Mean Squared Error (MSE) may increase, and the model's practical utility can diminish.
- **Invalid Diagnostic Tools**
   - **Problem**: Many diagnostic tools for linear regression, such as $R^2$ or residual normality checks, rely on the assumption of normally distributed residuals. Heavy-tailed $y$ may violate these assumptions.
   - **Cause**: The non-normal distribution of residuals undermines standard hypothesis tests (e.g., $t$-tests, $F$-tests) and model validation methods.
   - **Consequence**: Misleading results in assessing model significance or goodness-of-fit.
- **Decreased Interpretability**
   - **Problem**: Heavy-tailed $y$ often leads to inflated or deflated regression coefficients, making it harder to interpret their relationship with predictors.
   - **Cause**: Extreme values have high leverage in OLS, skewing coefficient estimates away from their representative values.
   - **Consequence**: The economic or statistical interpretation of the model becomes unreliable.

#### 0.3.2 Solutions
- **Transform the Target Variable**:
   - Apply transformations like log-transform or Box-Cox to reduce the impact of heavy tails.
- **Use Robust Regression Methods**:
   - Replace OLS with **Huber Regression**, **Quantile Regression**, or **Theil-Sen Regression**, which are less sensitive to outliers.
- **Detect and Address Outliers**:
   - Preprocess the data by identifying and handling extreme values through trimming, winsorizing, or imputation.
- **Adopt Models with Weaker Distributional Assumptions**:
   - Use non-parametric methods such as Gradient Boosting Machines (GBMs) or Random Forests, which do not rely on specific error distributions.
- **Leverage Extreme Value Theory (EVT)**:
   - For cases where heavy tails are intrinsic to the data, use EVT to explicitly model extreme values.

### 0.4 Heavy Tail of $x$

#### 0.4.1 Issues
- **Sensitivity to Leverage Points**
   - **Problem**: In linear regression, extreme values in $x$ (caused by heavy tails) can act as high-leverage points, disproportionately influencing the regression line.
   - **Cause**: Observations with extreme $x$-values have more "leverage" in determining the slope and intercept of the regression line.
   - **Consequence**: This can lead to biased or unstable estimates of regression coefficients, especially when the number of such points is small.
- **Increased Variance in Coefficient Estimates**
   - **Problem**: Heavy-tailed $x$ increases the variability in the estimation of regression coefficients.
   - **Cause**: The large variability in $x$ inflates the denominator in variance calculations for the OLS estimates, leading to higher uncertainty.
   - **Consequence**: Confidence intervals for the coefficients may widen, and statistical tests for significance (e.g., $t$-tests) become less reliable.
- **Violated Assumptions**
   - **Problem**: The heavy-tailed distribution of $x$ may violate linear regression assumptions, particularly those related to the design matrix.
   - **Cause**:
     - The heavy-tailed $x$ may not adequately represent the predictor space, violating the assumption of a well-conditioned matrix.
     - Extreme values in $x$ can create multicollinearity problems when combined with other predictors.
   - **Consequence**: Regression estimates may become unstable or non-invertible (e.g., if the design matrix becomes ill-conditioned).
- **Overemphasis on Extreme Values**
   - **Problem**: The model may overfit to observations with extreme $x$-values.
   - **Cause**: Since OLS minimizes the residual sum of squares, extreme $x$-values amplify the influence of corresponding residuals.
   - **Consequence**: The regression line might be skewed toward capturing trends in extreme $x$-values rather than the majority of the data.
- **Interpretability Challenges**
   - **Problem**: Heavy-tailed $x$ can distort the interpretability of regression coefficients.
   - **Cause**: Extreme $x$-values may represent atypical or non-representative scenarios, leading to coefficients that do not reflect the central tendency of the data.
   - **Consequence**: The model's explanatory power for the bulk of the data may decrease, and coefficients may lose practical significance.

#### 0.4.2 Solutions
- **Transform $x$**:
   - Apply transformations like log-transform or Winsorization to reduce the impact of heavy tails in $x$.
- **Robust Regression**:
   - Use robust regression methods, such as Huber Regression or Ridge Regression, which can mitigate the influence of extreme $x $-values.
- **Regularization**:
   - Techniques like Lasso or Ridge Regression can help stabilize coefficient estimates by adding penalties that prevent overemphasis on extreme $x$-values.
- **Identify and Handle Leverage Points**:
   - Diagnose high-leverage points using metrics like Cook's Distance or leverage statistics, and consider removing or down-weighting these points.
- **Alternative Models**:
   - Use models that are less sensitive to extreme $x $-values, such as decision trees, random forests, or gradient boosting machines.
