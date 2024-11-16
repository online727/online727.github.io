---
title: Discrete Random Variable
weight: 22
menu:
  notes:
    name: Discrete Random Variable
    identifier: notes-randomvariable-discrete
    parent: notes-randomvariable
    weight: 22
---

<!-- Bernoulli -->
{{< note title="Bernoulli" >}}
If a random variable $X$ can only take $0$ or $1$ as values, and its probability distribution is given by:
$$
P(X=0)=p, P(X=1)=q, p+q=1
$$

such a random variable $X$ is said to follow **Bernoulli distribution**, denoted as $X\sim B(1,p)$ or $X\sim b(1,p)$
{{< /note >}}

<!-- Binomial -->
{{< note title="Binomial" >}}
If a random variable's probability distribution is given by:
$$
P(X=k)=C_{n}^{k}p^{k}q^{n-k},k=0,1,2,\cdots,n,\quad (pq>0, p+q=1)
$$

such a random variable $X$ is said to follow **Binomial distribution**, denoted as $X\sim B(n,p)$.

$C_{n}^{k}p^{k}q^{n-k}$ is the $k+1_{th}$ item of $(p+q)^{n}\sum\limits_{k=0}^{n}C_{n}^{k}p^{k}q^{n-k}$.

Here is the translation of the given text into English:

**Background of the Binomial Distribution:**  
Suppose the probability of success in an experiment $S$ is $p$, and the experiment $S$ is repeated $n$ times. Let $X$ represent the number of successes. Then $X\sim B(n,p)$.
{{< /note >}}


<!-- Poisson -->
{{< note title="Poisson" >}}
If a random variable's probability distribution is given by:
$$
P(X=k)=\frac{\lambda^{k}}{k!}e^{-\lambda},k=0,1,\cdots
$$

such a random variable $X$ is said to follow **Poisson distribution** with parameter $\lambda$, denoted as $X\sim Poisson(\lambda)$, where $\lambda$ is a positive number.

**Classical Problem**: The number of red lights encountered by a taxi.
{{< /note >}}


<!-- Hyper Geometric -->
{{< note title="Hyper Geometric" >}}
If the probability distribution of $X$ is:
$$
P(X=m)=\frac{C_M^mC_{N-M}^{n-m}}{C_N^n},m=0,1,\cdots,M
$$

such a random variable $X$ is said to follow **Hyper Geometric distribution**, denoted as $X\sim H(n,M,N)$.

The meaning of Hyper Geometric distribution: 

Out of $N$ products, exactly $M$ are defective. If $n$ products are randomly selected, and let $X$ represent the number of defective products among the $n$ selected, then $X$ follows a hypergeometric distribution.

If the number of products is sufficiently large, there is essentially no difference between sampling without replacement and sampling with replacement:
$$
P(X=m)=\frac{C_M^mC_{N-M}^{n-m}}{C_N^n}\approx C_n^m p^m (1-p)^{n-m}, (p=\frac{M}{N})
$$
{{< /note >}}


<!-- Geometric -->
{{< note title="Geometric" >}}
If the probability distribution of $X$ is:
$$
P(X=m)=q^{k-1}p,k=1,2\cdots,\quad pq>0,p+q=1
$$

such a random variable $X$ is said to follow **Geometric distribution** with parameter $p$, denoted as $X\sim Geometric(p)$.

The meaning of Geometric distribution: 

Suppose the probability of success in an experiment is $p$, and the experiment is independently repeated until the first success occurs. Then the number of trials needed for the first success follows a geometric distribution with parameter $p$.
{{< /note >}}