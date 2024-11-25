---
title: "Basics"
date: 2024-11-25T20:10:25+08:00
description: Basics
menu:
  sidebar:
    name: Basics
    identifier: basic-content
    parent: matrix-basic
    weight: 303
hero: /images/volcano.jpg
tags:
- Linear Algebra
- Matrix
categories:
- Basic
---

## 0.1 Basic Content
$$
\begin{align}
    (\boldsymbol{AB})^{-1} &= \boldsymbol{B}^{-1}\boldsymbol{A}^{-1} \cr
    (\boldsymbol{ABC\cdots})^{-1} &= \cdots\boldsymbol{C}^{-1}\boldsymbol{B}^{-1}\boldsymbol{A}^{-1} \cr
    (\boldsymbol{A}^\top)^{-1} &= (A^{-1})^\top \cr
    (\boldsymbol{A} + \boldsymbol{B})^\top &= \boldsymbol{A}^\top + \boldsymbol{B}^\top \cr
    (\boldsymbol{AB})^\top &= \boldsymbol{B}^\top\boldsymbol{A}^\top \cr
    (\boldsymbol{ABC\cdots})^\top &= \cdots\boldsymbol{C}^\top\boldsymbol{B}^\top\boldsymbol{A}^\top \cr
    (\boldsymbol{A}^{H})^{-1} &= (\boldsymbol{A}^{-1})^{H} \cr
    (\boldsymbol{A} + \boldsymbol{B})^H &= \boldsymbol{A}^H + \boldsymbol{B}^H \cr
    (\boldsymbol{AB})^H &= \boldsymbol{B}^H\boldsymbol{A}^H \cr
    (\boldsymbol{ABC\cdots})^H &= \cdots\boldsymbol{C}^H\boldsymbol{B}^H\boldsymbol{A}^H
\end{align}
$$

## 0.2 Trace
$$
\begin{align}
    \text{TR}(\boldsymbol{A}) &= \sum_{i}A_{ii} \cr
    \text{TR}(\boldsymbol{A}) &= \sum_{i}\lambda_{i}, \quad \lambda_{i}=\text{eig}(\boldsymbol{A})_{i} \cr
    \text{TR}(\boldsymbol{A}) &= \text{TR}(\boldsymbol{A}^\top) \cr
    \text{TR}(\boldsymbol{AB}) &= \text{TR}(\boldsymbol{BA}) \cr
    \text{TR}(\boldsymbol{A+B}) &= \text{TR}(\boldsymbol{A}) + \text{TR}(\boldsymbol{B}) \cr
    \text{TR}(\boldsymbol{ABC}) &= \text{TR}(\boldsymbol{BCA}) = \text{TR}(\boldsymbol{CAB}) \cr
    \boldsymbol{a}^\top\boldsymbol{a} &= \text{Tr}(\boldsymbol{aa}^\top)
\end{align}
$$

## 0.3 Determinant
Let $\boldsymbol{A}$ is a $n\times n$ matrix.
$$
\begin{align}
    \det(\boldsymbol{A}) &= \prod_i \lambda_i \quad \lambda_i=\text{eig}(\boldsymbol{A})_i \cr
    \det(c\boldsymbol{A}) &= c^n \det(\boldsymbol{A})\quad \text{if }\boldsymbol{A}\in\mathbb{R}^{n\times n} \cr
    \det(\boldsymbol{A}^\top) &= \det(\boldsymbol{A}) \cr
    \det(\boldsymbol{AB}) &= \det(\boldsymbol{A})\det(\boldsymbol{B}) \cr
    \det(\boldsymbol{A}^{-1}) &= 1/\det(\boldsymbol{A}) \cr
    \det(\boldsymbol{A}^n) &= \det(\boldsymbol{A})^n \cr
    \det(\boldsymbol{I}+\boldsymbol{uv}^\top) &= 1 + \boldsymbol{u}^\top\boldsymbol{v}
\end{align}
$$

For $n=2$:
$$
\begin{equation}
    \det(\boldsymbol{I}+\boldsymbol{A}) = 1 + \det(\boldsymbol{A}) + \text{Tr}(\boldsymbol{A})
\end{equation}
$$

For $n=3$:
$$
\begin{equation}
    \det(\boldsymbol{I}+\boldsymbol{A}) = 1 + \det(\boldsymbol{A}) + \text{Tr}(\boldsymbol{A}) + \frac{1}{2}\text{Tr}(\boldsymbol{A})^2 - \frac{1}{2}\text{Tr}(\boldsymbol{A}^2)
\end{equation}
$$

For $n=4$:
$$
\begin{align}
    \det(\boldsymbol{I}+\boldsymbol{A}) = &1 + \det(\boldsymbol{A}) + \text{Tr}(\boldsymbol{A}) + \frac{1}{2} \notag\cr
    &+ \text{Tr}(\boldsymbol{A})^2 - \text{Tr}(\boldsymbol{A}^2) \notag\cr
    &+ \frac{1}{6}\text{Tr}(\boldsymbol{A})^3 - \frac{1}{2}\text{Tr}(\boldsymbol{A})\text{Tr}(\boldsymbol{A}^2) + \frac{1}{3}\text{Tr}(\boldsymbol{A}^3)
\end{align}
$$

For small $\epsilon$, the following approximation holds:
$$
\begin{equation}
    \det(\boldsymbol{I}+\boldsymbol{A}) = 1 + \det(\boldsymbol{A}) + \epsilon\text{Tr}(\boldsymbol{A}) + \frac{1}{2}\epsilon^2\text{Tr}(\boldsymbol{A})^2-\frac{1}{2}\epsilon^2\text{Tr}(\boldsymbol{A}^2)
\end{equation}
$$

## 0.4 The Special Case $2\times 2$
Consider the matrix $\boldsymbol{A}$:
$$
A = 
\begin{bmatrix}
    A_{11} & A_{12} \cr
    A_{21} & A_{22}
\end{bmatrix}
$$

Determinant and trace:
$$
\begin{align}
    \det(\boldsymbol{A}) &= A_{11}A_{22} - A_{12}A_{21} \cr
    \text{Tr}(\boldsymbol{A}) &= A_{11} + A_{22}
\end{align}
$$

Eigenvalues:
$$
\lambda^2 - \lambda\cdot\text{Tr}(\boldsymbol{A})+\det(\boldsymbol{A})=0
$$
$$
\lambda_1=\frac{\text{Tr}(\boldsymbol{A}) + \sqrt{\text{Tr}(\boldsymbol{A})^2 - 4\det(\boldsymbol{A})}}{2},\quad \lambda_1=\frac{\text{Tr}(\boldsymbol{A}) - \sqrt{\text{Tr}(\boldsymbol{A})^2 - 4\det(\boldsymbol{A})}}{2}
$$
$$
\lambda_1+\lambda_2=\text{Tr}(\boldsymbol{A}) \quad \lambda_1\lambda_2=\det(\boldsymbol{A})
$$

Eigenvectors:
$$
\boldsymbol{v}_1\propto \[A\_{12},\lambda_1-A\_{11}\]^\top \qquad \boldsymbol{v}_1\propto \[A\_{12},\lambda_2-A\_{11}\]^\top
$$

Inverse:
$$
\begin{equation}
    \boldsymbol{A}^{-1} = \frac{1}{\det(\boldsymbol{A})}\begin{bmatrix}
        A_{22} & -A_{12} \cr
        -A_{21} & A_{11}
    \end{bmatrix}
\end{equation}
$$