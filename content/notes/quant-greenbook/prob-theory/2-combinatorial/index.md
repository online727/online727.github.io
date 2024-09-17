---
title: Combinatorial Analysis
weight: 132
menu:
  notes:
    name: Combinatorial Analysis
    identifier: quant-greenbook-prob-2
    parent: quant-greenbook-prob
    weight: 132
---

<!-- Poker Hands -->
{{< note title="Poker Hands" >}}
### Problem
Pocker is a card game in which each player gets a hand of 5 cards. There are 52 cards in a deck. Each has a value and belongs to a suit. There are 13 values, 2,3,4,5,6,7,8,9,10,J,Q,K,A, and four suits, spade, club, heart and diamond.

What are the probability of getting hands with four-of-a-kind (four of the five cards with the same value)? Hands with a full house (three cards of one values and two cards another value)? Hands with two pairs?

### Solution
The total number of selecting 5 cards from 52 cards randomly is $\binom{52}{5}$.

**Hands with a four-of-a-hand**
$$
N = \binom{13}{1}\times \binom{48}{1} = 13\times 48
$$

**Hands with a full house**
$$
N = \binom{13}{1}\times \binom{4}{3}\times \binom{12}{1}\times \binom{4}{2} = 13\times 4\times 12 \times 6
$$

**Hands with two pairs**
$$
N = \binom{13}{2}\times \binom{4}{2}\times \binom{4}{2}\times \binom{44}{1} = 78\times 6\times 6\times 44
$$
{{< /note >}}


<!-- Hoppoing Rabbit -->
{{< note title="Hoppoing Rabbit" >}}
### Problem
A rabbit sits at the bottom of a staircase with $n$ stairs. The rabbit can hop up only one or two stairs at a time. How many different ways are there for the rabbit to ascend to the top of the stairs?

### Solution
Denote $f(n)$ as the number of ways for the rabbit to ascend to the top when there are $n$ stairs.

If $n=1$, obviously $f(1)=1$.

Similarly, if $n=2,f(2)=2$ (one 2-stair hop or two 1-stair hops).

For any $n>2$, there are always two possibilities for the last hop: either it's a 1-stair hop or a 2-stair hop. In the former case, the rabbit is at $(n-1)$ before reaching $n$, and it has $f(n-1)$ ways to reach $(n-1)$. In the latter case, the rabbit is at $(n-2)$ before reaching $n$, and it has $f(n-2)$ ways to reach $(n-2)$.

So we have $f(n)=f(n-1)+f(n-2)$. Then we can get all $f(n)$ based on $f(1)=1$ and $f(2)=2$.
{{< /note >}}