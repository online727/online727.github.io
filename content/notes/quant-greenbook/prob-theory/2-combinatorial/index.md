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


<!-- Screwy pirates 2 -->
{{< note title="Screwy pirates 2" >}}
### Problem
Having peacefully divided the loot (in chapter 2), the pirate team goes on for more looting and expands the group to 11 pirates. To protect their hard-won treasure, they gather together to put all the loot in a safe. Still being a democratic bunch, they decide that only a majority - any majority - of them ($\geq6$) together can open the safe. So they ask a locksmith to put a certain number of locks on the safe. To access the treasure, every lock needs to be opened. Each lock can have multiple keys; but each key only opens one lock. The locksmith can give more than one key to each pirate.

What is the smallest number of locks needed? And how many keys must each pirate carry?

**Hint**: every subgroup of 6 pirates should have the same key to a unique lock that the other 5 pirates do not have.

### Solution

Let's randomly select 5 pirates, they can't open the safe. But once the $6_{th}$ pirate joins, who holds a unique key to a unique lock, they can open the safe. So every subgroup of 5 pirates corresponds to 1 unique lock, so there are $\binom{11}{5}$ locks.

Every lock should have 6 keys so that anyone in a subgroup of 6 pirates joins the selected 5 pirates can make the safe open. So every pirate should have $\frac{\binom{11}{5}\times 6}{11}$ keys.
{{< /note >}}


<!-- Chess Tournament -->
{{< note title="Chess Tournament" >}}
### Problem
A chess tournament has $2^n$ players with skills $1>2>\cdots>2^n$. It is organized as a knockout tournament, so that after each round only the winner proceeds to the next round. Except for the final, opponents in each round are drawn at random. Let's also assume that when two players meet in a game, the player with better skills always wins.

What's the probability that players 1 and 2 will meet in the final?

**Hint**: Consider separating the players to two $2^{n-1}$ subgroups. What will happen if player 1 and 2 in the same group? Or not in the same group?

### Solution 1

Suppose we have finished a complete game, player 1 definitely won the game no matter how the game proceeded. But let's focus on the two player participating in the final round, we can divide the left $2^n - 2$ players into two subgroups according to which one defeated them. So we get two $2^{n-1}$ subgroups.

If player 2 was also a participant of final round, he/she must be in the subgroup different from player 1.

Since any of the remaining players in $2,3,\cdots,2^n$ are likely to be one of the $(2^{n-1}-1)$ players in the same subgroup as player 1 or one of the $2^{n-1}$ players in the subgroup different with player 1, the probability that player 2 is in a different subgroup from player 1 is simply $\frac{2^{n-1}}{2^n-1}$

### Solution 2
The game has $n$ rounds for there are $2^n$ players.

Define $E_i, i=1,2,\cdots,n-1$ as the event that player 1 and 2 do not meet in round $1,2,\cdots,i$.

At the first round, every player except 1 has the same probability $\frac{1}{2^{n}-1}$ to be the rival of player 1. So $E_1=\frac{2^{n}-2}{2^n-1}=\frac{2(2^{n-1}-1)}{2^n-1}$.

Similarly:
$$
\begin{align*}
    E_2 &= \frac{2^{n-1}-2}{2^{n-1}-1} = \frac{2(2^{n-2}-1)}{2^{n-1}-1} \cr
    &\vdots \cr
    E_i &= \frac{2^{n-i+1}-2}{2^{n-i+1}-1} = \frac{2(2^{n-i}-1)}{2^{n-i+1}-1} \cr
    &\vdots \cr
    E_{n-1} &= \frac{2^2-2}{2^2-1} = \frac{2(2-1)}{2^2-1}
\end{align*}
$$

Then we can get:
$$
P(\text{player 1 and 2 meet in the final round}) = P(E_1)\times P(E_2|E_1)\times\cdot\times P(E_{n-1}|E_1E_2\cdots E_{n-2}) = \frac{2^{n-1}}{2^n-1}
$$
{{< /note >}}


<!-- Application Letters -->
{{< note title="Application Letters" >}}
### Problem
You're sending job applications to 5 firms: Morgan Stanley, Lehman Brothers, UBS, Goldman Sachs, and Merrill Lynch. You have 5 envelopes on the table neatly typed with names and addresses of people at these 5 firms. You even have 5 cover letters personalized to each of these firms. Your 3-year-old tried to be helpful and stuffed each cover letter into each of the envelopes for you. Unfortunately she randomly put letters into envelopes without realizing that the letters are personalized. What is the probability that all 5 cover letters are mailed to the wrong firms?

**Hint**: The complement is that at least one letter is mailed to the correct firm.

### Solution

Denote $E_I, i=1,2,\cdots,5$ as the event that the $i_{th}$ letter has the correct envelope. Then $P(\cup_{i=1}^{5}E_i)$ is the probability that at least one letter has the correct envelope and $1-P(\cup_{i=1}^{5}E_i)$  is the probability that all letters have the wrong envelopes.

$P(\cap_{i=1}^{5}E_i)$ can be calculated using the **Inclusion-Exclusion Principle**:
$$P(\cup_{i=1}^{5}E_i)=\sum_{i=1}^5P(E_i) - \sum_{i_1<i_2}P(E_{i_1}E_{i_2}) + \cdots + (-1)^6P(E_{i_1}E_{i_2}\cdots E_{i_5})$$

It's obvious that $P(E_i)=\frac{1}{5},i=1,2,\cdots,5$ so that $\sum_{i=1}^5P(E_i)=1$.

$P(E_{i_1}E_{i_2})=P(E_{i_1})P(E_{i_2}|E_{i_1})=\frac{1}{5}\cdot\frac{1}{4}=\frac{1}{20}$ so that $\sum_{i_1<i_2}P(E_{i_1}E_{i_2})=\binom{5}{2}\frac{1}{20}=\frac{1}{2!}$

Similarly we have $\sum_{i_1<i_2<i_3}P(E_{i_1}E_{i_2}E_{i_3})=\frac{1}{3!}$, $\sum_{i_1<i_2<i_3<i_4}P(E_{i_1}E_{i_2}E_{i_3}E_{i_4})=\frac{1}{4!}$ and $P(E_{i_1}E_{i_2}\cdots E_{i_5})=\frac{1}{5!}$.

So $1-P(\cup_{i=1}^{5}E_i) = 1 - \frac{1}{2} + \frac{1}{3!} - \frac{1}{4!} + \frac{1}{5!} = \frac{11}{30}$.
{{< /note >}}


<!-- Birthday Problem -->
{{< note title="Birthday Problem" >}}
### Problem
How many people do we need in a class to make the probability that two people have the same birthday more than 1/2? (For simplicity, assume 365 days a year.)

### Solution

Suppose there are $n$ people in the class, so there are $365^n$ possible combinations. The probability of their birthday are all different is:
$$
P = \frac{365\times(365-1)\times(365-2)\times\cdots\times(365-n+1)}{365^n}
$$

Solve:
$$
\begin{align*}
    &\min\quad n \cr
    &s.t.\quad 1 - P > \frac{1}{2}
\end{align*}
$$

The smallest such $n$ is 23.
{{< /note >}}


<!-- 100th Digit -->
{{< note title="100th Digit" >}}
### Problem
What is the $1OO_{th}$ digit to the right of the decimal point in the decimal representation of $(1 + \sqrt{2})^{3000}$?

**Hint**: $(1+\sqrt{2})^2+(1-\sqrt{2})^2=6$. What will happen to $(1-\sqrt{2})^{2n}$ as $n$ becomes large?

### Solution

Applying the binomial theorem for $(x+y)^n$, we have:
$$
(1+\sqrt{2})^n = \sum_{k=0}^n\binom{n}{k}1^{n-k}\sqrt{2}^k = \sum_{k=2j, 0\leq j\leq\frac{n}{2}}^n\binom{n}{k}1^{n-k}\sqrt{2}^k + \sum_{k=2j+1, 0\leq j<\frac{n}{2}}^n\binom{n}{k}1^{n-k}\sqrt{2}^k
$$

and:
$$
(1-\sqrt{2})^n = \sum_{k=0}^n\binom{n}{k}1^{n-k}(-\sqrt{2})^k = \sum_{k=2j, 0\leq j\leq\frac{n}{2}}^n\binom{n}{k}1^{n-k}\sqrt{2}^k - \sum_{k=2j+1, 0\leq j<\frac{n}{2}}^n\binom{n}{k}1^{n-k}\sqrt{2}^k
$$

So we have:
$$
(1+\sqrt{2})^n + (1-\sqrt{2})^n = 2\times \sum_{k=2j, 0\leq j\leq\frac{n}{2}}^n\binom{n}{k}1^{n-k}\sqrt{2}^k,
$$
which is always an integer.

It's easy to see that $0<(1-\sqrt{2})^n<<10^{-100}$. So the $100_{th}$ digit of $(1 + \sqrt{2})^n$ must be 9.
{{< /note >}}


<!-- Cubic of Integer -->
{{< note title="Cubic of Integer" >}}
### Problem
Let $x$ be an integer between 1 and $10^{12}$, what is the probability that the cubic of $x$ endswith 11?

**Hint**: The last two digits of $x^3$ only depend on the last two digits of $x$.

### Solution

All integers can be expressed as $x=a+10b$, where $a$ is the last digit of $x$. So we have $x^3=(a+10b)^3=a^3+30a^b+300ab^2+1000b^3$.

The unit digit of $x^3$ only depends on $a^3$, so $a=1$.

The tenth digit only depends on $30a^2b$, so $b=7$.

Consequently, the last two digits of $x$ should be $71$, which has a probability of $1$% for integers between $1$ and $10^{12}$.
{{< /note >}}