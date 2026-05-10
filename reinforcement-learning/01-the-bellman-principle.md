---

title: "The Bellman Principle"
subtitle: "State, action, reward, return, and the Bellman equations as a recursive decomposition of value"
---


The problem of acting wisely in a world that responds to your actions is ancient. Philosophers called it practical reason; economists called it decision theory; control engineers called it optimal control. The term *reinforcement learning* came later, borrowed from behavioral psychology, where Thorndike and Skinner had shown that organisms strengthen behaviors producing rewards and weaken those producing punishment. But the mathematical foundation came not from psychology but from a single insight, published in 1957 by Richard Bellman in a book simply titled *Dynamic Programming*.

Bellman was working on optimal control problems at RAND Corporation — problems like: given an aircraft at a particular altitude and velocity, what sequence of control inputs minimizes fuel consumption while reaching the destination? These are not single-shot decisions but sequential ones: each control input changes the state of the aircraft, and the quality of each subsequent decision depends on the state produced by earlier ones. The challenge was to reason backward from goals to actions across this chain of dependencies. Bellman found the key — and the key was recursion.

## The Agent-Environment Interface

Before the equations, the setup. An **agent** — a learner or decision-maker — interacts with an **environment** over discrete time steps. At each step $t$, the agent observes the current **state** $S_t \in \mathcal{S}$ of the environment, selects an **action** $A_t \in \mathcal{A}$, and receives a **reward** $R_{t+1} \in \mathbb{R}$ while the environment transitions to a new state $S_{t+1}$.

The rewards encode what we want: positive signals when things go well, negative when they go poorly. The agent's goal is not to maximize the reward at any single step but to maximize the total reward accumulated over time. This is the distinction that makes reinforcement learning hard. Maximizing immediate reward often sacrifices future reward, and the long-term consequences of an action may not become apparent until many steps later. A chess-playing agent might sacrifice a pawn for a positional advantage that yields a win fifty moves hence — or it might not. Without a way to evaluate long-term consequences, no good policy is learnable.

## Markov Decision Processes

The formal framework is the **Markov Decision Process** (MDP), refined by Howard, Shapley, and others throughout the late 1950s and 1960s. An MDP is a tuple $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$:

- $\mathcal{S}$ is the **state space**
- $\mathcal{A}$ is the **action space**
- $p(s' \mid s, a)$ is the **transition kernel**: the probability of reaching state $s'$ by taking action $a$ in state $s$
- $r(s, a)$ is the **reward function**: the expected immediate reward for taking action $a$ in state $s$
- $\gamma \in [0, 1)$ is the **discount factor**

The defining property of an MDP is the **Markov property**: the distribution over the next state depends only on the current state and action, not on the full history of how that state was reached:

$$P(S_{t+1} = s' \mid S_t, A_t, S_{t-1}, A_{t-1}, \ldots) = P(S_{t+1} = s' \mid S_t, A_t).$$

The current state is a **sufficient statistic** for the history. This is an assumption rather than a fact about the world — many real environments are not truly Markovian, since the observable state may not encode all relevant information about the future. But it is an assumption that makes the problem tractable, and for a large class of problems it is either exactly satisfied (board games with full state visibility) or can be made approximately true through careful state design. Its depth as a simplifying assumption is comparable to the Markov assumption in language modeling, and it proves equally fruitful.

## The Return and the Discount Factor

The agent's objective is to maximize the **return** $G_t$: the total discounted reward from time $t$ onward,

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+1+k}.$$

The discount factor $\gamma$ has two interpretations. The mathematical one: discounting ensures the return is finite even for infinite-horizon problems, since for bounded rewards $|R_t| \leq R_{\max}$, the geometric series converges:

$$|G_t| \leq R_{\max} \sum_{k=0}^{\infty} \gamma^k = \frac{R_{\max}}{1 - \gamma}.$$

The economic one: future rewards are worth less than immediate ones, by a factor of $\gamma$ per time step. A dollar received next period is worth $\gamma$ dollars now. When $\gamma = 0$, the agent is purely myopic — it cares only about the next reward. When $\gamma \to 1$, the agent places nearly equal weight on rewards arbitrarily far in the future. In practice, $\gamma \in \{0.95, 0.99\}$ is typical; the choice encodes a prior about how far ahead an agent must plan to behave well.

The return admits a **recursive decomposition** that is the seed of everything that follows:

$$G_t = R_{t+1} + \gamma G_{t+1}.$$

The return from step $t$ equals the immediate reward plus the discounted return from the next step. This one identity is, in a sense, the entire field of reinforcement learning.

## Policies

A **policy** $\pi$ specifies how the agent selects actions. A stochastic policy maps states to distributions over actions:

$$\pi(a \mid s) = P(A_t = a \mid S_t = s).$$

A deterministic policy is the special case $\pi: \mathcal{S} \to \mathcal{A}$ assigning a single action to each state. Given a policy and the environment's dynamics, the agent's future trajectory is a probability distribution over sequences of states, actions, and rewards. The goal of reinforcement learning is to find the policy that maximizes expected return.

## Value Functions

The **state-value function** $V^\pi(s)$ of a policy $\pi$ is the expected return when starting from state $s$ and following $\pi$ thereafter:

$$V^\pi(s) = \mathbb{E}_\pi\!\left[G_t \mid S_t = s\right] = \mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty} \gamma^k R_{t+1+k} \;\Bigg|\; S_t = s\right].$$

This encodes how good it is to *be* in state $s$ under $\pi$ — not the immediate reward, but the long-run expected return from that state forward.

The **action-value function** $Q^\pi(s, a)$ is the expected return when starting from state $s$, taking action $a$, and then following $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi\!\left[G_t \mid S_t = s,\; A_t = a\right].$$

The state-value is the expectation of the action-value under the policy's action distribution:

$$V^\pi(s) = \sum_a \pi(a \mid s)\, Q^\pi(s, a).$$

Both functions represent the same underlying quantity — long-run expected return — from different starting conditions. The distinction matters because $V^\pi$ suffices for evaluating a policy, while $Q^\pi$ is needed for improving it: you can only choose a better action than $\pi$ prescribes if you can evaluate each action's consequences, which is exactly what $Q^\pi$ provides.

## The Bellman Expectation Equations

Now the recursion. Starting from the definition of $V^\pi$ and using $G_t = R_{t+1} + \gamma G_{t+1}$:

$$V^\pi(s) = \mathbb{E}_\pi\!\left[R_{t+1} + \gamma G_{t+1} \mid S_t = s\right].$$

Expanding the expectation over the policy's action choice, the environment's stochastic transition, and the subsequent value:

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} p(s' \mid s, a) \bigl[r(s, a) + \gamma V^\pi(s')\bigr].$$

This is the **Bellman expectation equation for $V^\pi$**. The value of state $s$ under $\pi$ equals the expected immediate reward plus the expected discounted value of the next state, where both expectations are taken jointly over the policy's action distribution and the environment's transition dynamics.

An identical derivation gives the action-value Bellman equation:

$$Q^\pi(s, a) = \sum_{s'} p(s' \mid s, a) \!\left[r(s, a) + \gamma \sum_{a'} \pi(a' \mid s')\, Q^\pi(s', a')\right].$$

These are not approximations. They are exact identities satisfied by any value function of any policy on any MDP — consequences of the Markov property and the recursive structure of the return. For a finite state space, they constitute a system of $|\mathcal{S}|$ linear equations in $|\mathcal{S}|$ unknowns, which can be written in matrix form as:

$$\mathbf{v}^\pi = \mathbf{r}^\pi + \gamma\, \mathbf{P}^\pi\, \mathbf{v}^\pi,$$

with unique solution $\mathbf{v}^\pi = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{r}^\pi$. The matrix $\mathbf{I} - \gamma \mathbf{P}^\pi$ is invertible because $\gamma < 1$ ensures all eigenvalues of $\gamma \mathbf{P}^\pi$ lie strictly inside the unit circle.

## The Bellman Optimality Equations

The Bellman expectation equations describe the value of a *given* policy. The more important question is: what is the value of the *best* policy?

Define the **optimal state-value function**:

$$V^*(s) = \max_\pi V^\pi(s).$$

Bellman's **principle of optimality** states that an optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision. This sounds obvious. Its consequence is not: it says that a globally optimal policy is also *locally* optimal at every state it visits. Optimality is consistent under the recursive decomposition of the return.

The principle allows the derivation of the **Bellman optimality equations**. The optimal value function satisfies:

$$V^*(s) = \max_a \sum_{s'} p(s' \mid s, a) \bigl[r(s, a) + \gamma V^*(s')\bigr].$$

The expectation over $\pi(a \mid s)$ is replaced by a maximization: the optimal action is the one that maximizes the expected sum of immediate reward and discounted next-state value. For the optimal action-value function:

$$Q^*(s, a) = \sum_{s'} p(s' \mid s, a) \!\left[r(s, a) + \gamma \max_{a'} Q^*(s', a')\right].$$

Given $Q^*$, the optimal policy is simply greedy with respect to it:

$$\pi^*(s) = \arg\max_a\, Q^*(s, a).$$

No lookahead into future states is required at execution time. The optimal policy acts myopically with respect to the optimal action-value function — it selects the action with the highest Q-value at each step, and this is globally optimal precisely because $Q^*$ already accounts for all future consequences.

## A Contraction Mapping

The Bellman optimality equations are nonlinear — the max operator prevents a direct matrix solution — but they admit a clean analytical structure. Define the **Bellman optimality operator** $\mathcal{T}^*$ acting on a value function $V$:

$$(\mathcal{T}^* V)(s) = \max_a \sum_{s'} p(s' \mid s, a) \bigl[r(s, a) + \gamma V(s')\bigr].$$

One can verify that $\mathcal{T}^*$ is a **contraction mapping** with modulus $\gamma$ under the supremum norm:

$$\|\mathcal{T}^* V - \mathcal{T}^* U\|_\infty \leq \gamma \|V - U\|_\infty.$$

By the Banach fixed-point theorem, starting from any bounded initial function $V_0$, the sequence $V_{k+1} = \mathcal{T}^* V_k$ converges geometrically to the unique fixed point $V^*$. The error after $k$ iterations is bounded by $\gamma^k \|V_1 - V_0\|_\infty / (1 - \gamma)$. This convergent iteration — initialize a value function, repeatedly apply the Bellman operator, converge to optimality — is the prototype for every value-based reinforcement learning algorithm that followed.

## What the Principle of Optimality Buys

The principle of optimality is the reason dynamic programming is computationally tractable when brute-force search is not. Consider the alternative: searching directly over the space of all policies. For a finite MDP with $|\mathcal{A}|$ actions per state and $|\mathcal{S}|$ states, there are $|\mathcal{A}|^{|\mathcal{S}|}$ deterministic policies — a combinatorial explosion that is infeasible even for modest problems. A grid world with $100$ cells and $4$ actions has $4^{100} \approx 10^{60}$ deterministic policies.

The Bellman equations reduce this to finding $|\mathcal{S}|$ real numbers — the optimal value function — after which the optimal policy is obtained for free by greedy selection. The subproblems (evaluating states) are solved once and reused. This is the hallmark structure of dynamic programming: **overlapping subproblems with optimal substructure**, solved by memoization rather than repeated recomputation.

## The Problem of Scale

The Bellman equations are exact and complete. They characterize the optimal policy for *any* finite MDP given full knowledge of the transition kernel and reward function. The problem is computational scale.

A tabular representation of $V^*$ or $Q^*$ requires storing one number per state (or per state-action pair). For a $100 \times 100$ grid world, this is $10{,}000$ entries — manageable. For a game of backgammon, the state space is approximately $10^{20}$ — infeasible to store. For chess, an estimated $10^{43}$ positions. For a language model generating tokens sequentially, the number of possible context sequences is effectively unbounded. Bellman himself was well aware of this: he coined the phrase *curse of dimensionality* to describe how the state space of a problem grows exponentially with the number of state variables, making exact dynamic programming algorithms impractical at scale.

Two ideas resolve this, and together they define the history of reinforcement learning after the classical era. The first is **sampling**: rather than sweeping over the entire state space in each iteration, update value estimates based on individual transitions observed through interaction with the environment. This transforms an exact but intractable computation into an approximate but scalable one. The second is **function approximation**: represent the value function not as a lookup table but as a parameterized function — a linear model, a decision tree, or eventually a neural network — that generalizes across similar states.

The first idea leads directly to temporal difference learning and Q-learning. The second leads to deep Q-networks and the modern era of deep reinforcement learning. Both are grounded in the Bellman equations: they are algorithms for approximately satisfying the Bellman identities using samples from the environment rather than the true transition kernel. The equations that Bellman wrote in 1957 remain, beneath all subsequent elaboration, the theoretical spine of the field.

## What Bellman Left Behind

The n-gram era of language modeling ended not because the framework was wrong, but because it had exhausted what counting could do without learned representations. The classical era of reinforcement learning ended for exactly the analogous reason. The Bellman equations were right. The tools for solving them at scale had not yet been invented.

What the subsequent decades produced was not a replacement for Bellman's framework but a series of approximations to it: first sampling-based approximations, then parameterized approximations, then deep approximations. Each introduced new problems — variance, instability, reward hacking — that required new solutions. The field's entire technical vocabulary — temporal difference errors, actor-critic architectures, trust regions, KL penalties — is the vocabulary of dealing with approximation while preserving the logic that Bellman made precise.

The next chapter turns to the first of these approximations: dynamic programming algorithms that solve the Bellman equations exactly in the tabular setting, and the reasons why this is simultaneously possible in principle and impossible in practice.

---

*The foundational reference is Bellman (1957), Dynamic Programming. The MDP formalism as a general framework for sequential decision making was developed by Howard (1960) in Dynamic Programming and Markov Processes. The contraction mapping analysis of value iteration is treated rigorously in Puterman (1994), Markov Decision Processes: Discrete Stochastic Dynamic Programming. The standard modern textbook covering all material in this chapter is Sutton and Barto (1998, revised 2018), Reinforcement Learning: An Introduction.*
