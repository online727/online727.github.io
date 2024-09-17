---
title: Basic Definitions
weight: 131
menu:
  notes:
    name: Basic Definitions
    identifier: quant-greenbook-prob-1
    parent: quant-greenbook-prob
    weight: 131
---

<!-- Coin Toss Game -->
{{< note title="Coin Toss Game" >}}
### Problem
Two gamblers are playing a coin toss game. Gambler $A$ has $(n+1)$ fair coins; $B$ has $n$ fair coins. What is the probability that $A$ will have more heads than $B$ if both flip all their coins?

### Solution
First, let's remove the last coin of $A$, then the number of heads in the first $n$ coins of $A$ and $B$ has three outcomes:

$E_1$: $A$'s $n$ coins have more heads than $B$'s $n$ coins.

$E_2$: $A$'s $n$ coins have equal number of heads as $B$'s $n$ coins.

$E_3$: $A$'s $n$ coins have fewer heads than $B$'s $n$ coins.

Now, denote $P(A)$ as the probability of $A$ win the game after fliping $A$'s $(n+1)_{th}$ coin, we have:
$$
\begin{align*}
    P(A) &= P(A|E_1) + P(A|E_2) + P(A|E_3) \cr
    &= P(E_1) * 1 + P(E_2) * \frac{1}{2} + P(E_3) * 0 \cr
    &= \frac{1}{2} \cr
    (P(E_1) &+ P(E_2) + P(E_3) = 1\quad and\quad P(E_1)=P(E_3))
\end{align*}
$$
{{< /note >}}


<!-- Card Game -->
{{< note title="Card Game" >}}
### Problem
A casino offers a simple card game. There are 52 cards in a deck with 4 cards for each value 2,3,4,5,6,7,8,9,10,J,Q,K,A. Each time the cards are thoroughly shuffled (so each card has equal probability of being selected). You pick up a card from the deck and the dealer picks another one without replacement. IF you have larger number, you win; if the numbers are equal or yours is smaller, the house win -- as in all other casinos, the house always has better odds of winning. What is your probability of winning?

### Solution 1
Similarly, this problem can be solved as the former **Coin Toss Game**, it also has three outcomes:

$E_1$: your number is larger than the dealer's.

$E_2$: your number is equal to the dealer's.

$E_3$: your number is smaller than the dealer's.

Now, $P(E_1)$ is the probability we want to get. By symmetry, $P(E_1)=P(E_3)$. And we also have $P(E_1) + P(E_2) + P(E_3) = 1$.

$P(E_2)$ denotes the two cards have equal numbers. If you have selected one card randomly, then there are only 3 cards withe the same value in the remaining 51 cards, so $P(E_2)=3/51$.

Then we can get:
$$
P(E_1)=\frac{1-P(E_2)}{2}=\frac{8}{17}
$$

### Solution 2
If you select a card first, there are 13 outcomes. Each outcome has a corresponding probability of winning the game.
$$
\begin{align*}
    P(Win) &= P(Win|2) + P(Win|3) + \cdots + P(Win|A) \cr
    &= \frac{1}{13} \times (0 + \frac{4}{51} + \frac{8}{51} + \cdots + \frac{48}{51}) \cr
    &= \frac{4}{13} \times \frac{6\times 13}{51} = \frac{8}{17}
\end{align*}
$$
{{< /note >}}


<!-- Drunk Passenger -->
{{< note title="Drunk Passenger" >}}
### Problem
A line of 100 airline passengers are waiting to board a plane. They each hold a ticket to one of the 100 seats on that flight. For convenience, let’s say that the $n_{th}$ passenger in line has a ticket for the seat number $n$. Being drunk, the first person in line picks a random seat (equally likely for each seat). All of the other passengers are sober, and will go to their proper seats unless it is already occupied; In that case, they will randomly choose a free seat. You’re person number 100. What is the probability that you end up in your seat (i.e., seat #100) ?

### Solution
Let's consider seats #1 and #100. There are two possible outcomes:

$E_1$: Sear #1 is taken before #100;

$E_2$: Sear #100 is taken before #1.

By symmetry, $P(E_1) = P(E_2) = \frac{1}{2}$.

If $E_1$ happens, no matter which passenger takes #1, all passengers after him(her) will take their own seats, so you will end up in your own seat.

If $E_2$ happends, you will definited noe end up in your own seat.
{{< /note >}}


<!-- N points on a circle -->
{{< note title="N points on a circle" >}}
### Problem
Given $N$ points drawn randomly on the circumference of a circle, what is the probability that they are all within a semicircle?

### Solution
Denote $E_i, i\in \{1,2,\cdots,N\}$ as starting at point $i$, all the other $N-1$ points are in the clockwise semicircle. We can verify that these events are mutually exclusive. 

For example, if we, starting at point $i$ and proceeding clockwise along the circle, sequentially encounters points $i+1,i+2,\cdots,N,1,\cdots,i-1$ in a half circle, then starting at any other point $j$, we cannot encounter all other points within a clockwise semicircle.

Hence, all these events are mutually exclusive. 

For any $E_i$, if point $i$ is determined, the clockwise semicircle is also determined, every point is located on that semicircle with probability $\frac{1}{2}$, so $P(E_i)=\frac{1}{2^{N-1}}$.

So we have:
$$
P(\cup_{i=1}^{N}E_i) = \sum_{i=1}^{N}P(E_i) = \frac{N}{2^{N-1}}
$$
{{< /note >}}