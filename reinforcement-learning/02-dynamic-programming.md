---
title: "Dynamic Programming"
subtitle: "Policy evaluation, policy iteration, value iteration, and their exact-but-intractable cost"
---


Dynamic programming arrived at the reinforcement learning problem with a mandate: given everything about the environment — the transition probabilities, the reward function, the full state and action spaces — compute the optimal policy exactly. This is a different problem from learning, and it has a cleaner solution. Bellman's 1957 book established the theoretical framework; Howard's 1960 monograph *Dynamic Programming and Markov Processes* worked out the computational algorithms. The result was a pair of procedures — policy iteration and value iteration — that provably converge to the optimal policy in finite time, require no approximation, and suffer from a single decisive flaw: they scale catastrophically with the size of the state space.

That flaw is important to understand not merely because it limits the direct applicability of dynamic programming, but because the algorithms that follow — temporal difference learning, Q-learning, deep Q-networks, policy gradients — are all attempts to preserve the logical structure of DP while avoiding its computational pathologies. The right way to understand TD learning is as policy evaluation without sweeping all states. The right way to understand Q-learning is as value iteration without knowing the transition kernel. These connections are lost if the DP algorithms are treated as a historical curiosity rather than the theoretical spine they actually are.

---

## The Two Problems

> [!info]
> **The prediction problem** (policy evaluation): given a policy $\pi$, compute $V^\pi$. This tells us how good the policy is at every starting state — not the immediate reward, but the full long-run expected return.
>
> **The control problem**: find the optimal policy $\pi^*$ and its value function $V^*$.
>
> Both assume **full model knowledge** — the transition kernel $p(s' \mid s, a)$ and reward function $r(s, a)$ are given. This is the **planning** setting, distinct from the reinforcement learning setting where the model is unknown and must be discovered through interaction.

The separation between prediction and control is not cosmetic. It reflects a genuine division of labor that survives into every modern algorithm. Before you can improve a policy, you must be able to measure it. The critic in an actor-critic architecture is solving the prediction problem; the actor is solving the control problem. Howard's 1960 formulation drew exactly this line.

---

## Iterative Policy Evaluation

> [!question]
> Given a policy $\pi$, how do we compute its value function $V^\pi$ exactly?

The Bellman expectation equation for $\pi$ is a fixed-point equation in $V^\pi$. Define the **Bellman expectation operator** $\mathcal{T}^\pi$:

$$(\mathcal{T}^\pi V)(s) = \sum_a \pi(a \mid s) \sum_{s'} p(s' \mid s, a) \bigl[r(s, a) + \gamma V(s')\bigr].$$

The function we want is the unique fixed point $V^\pi = \mathcal{T}^\pi V^\pi$. Since $\mathcal{T}^\pi$ is a contraction with modulus $\gamma$ (established in Chapter 01), Banach's fixed-point theorem guarantees that the iteration $V_{k+1} = \mathcal{T}^\pi V_k$ converges to $V^\pi$ from any initialization.

**Iterative policy evaluation** initializes $V_0(s) = 0$ for all $s$ and repeatedly sweeps:

$$V_{k+1}(s) \leftarrow \sum_a \pi(a \mid s) \sum_{s'} p(s' \mid s, a) \bigl[r(s, a) + \gamma V_k(s')\bigr], \quad \forall s \in \mathcal{S}.$$

> [!note]
> Each sweep is **synchronous**: all states are updated in parallel using values from the previous iteration. The convergence bound follows directly from the contraction:
> $$\|V_{k} - V^\pi\|_\infty \leq \gamma^k \|V_0 - V^\pi\|_\infty.$$
> In practice, iteration stops when $\max_s |V_{k+1}(s) - V_k(s)| < \varepsilon$, which guarantees residual error $\|V_k - V^\pi\|_\infty \leq 2\varepsilon\gamma/(1-\gamma)$.
> Cost per sweep: $O(|\mathcal{S}|^2 |\mathcal{A}|)$ — for each state, an expectation over actions, each requiring a sum over successor states.

---

## The Policy Improvement Theorem

> [!question]
> Given an accurate estimate of $V^\pi$, is there always a better policy — and how do we find it?

Suppose we have computed $V^\pi$ for some policy $\pi$. Consider a modified policy $\pi'$ that at some state $s$ takes a different action $a$ — one for which $Q^\pi(s, a) > V^\pi(s)$ — and thereafter follows $\pi$ exactly. The **policy improvement theorem** states:

> If $Q^\pi(s, \pi'(s)) \geq V^\pi(s)$ for all $s$, then $V^{\pi'}(s) \geq V^\pi(s)$ for all $s$.

The proof is a clean unrolling of the Bellman equations. Since $V^\pi(s) \leq Q^\pi(s, \pi'(s))$:

$$V^\pi(s) \leq Q^\pi(s, \pi'(s)) = \mathbb{E}\!\left[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s,\, A_t = \pi'(s)\right].$$

Applying the same inequality at $S_{t+1}$ and iterating forward:

$$V^\pi(s) \leq \mathbb{E}_{\pi'}\!\left[\sum_{k=0}^{\infty} \gamma^k R_{t+1+k} \;\middle|\; S_t = s\right] = V^{\pi'}(s).$$

The **greedy improvement policy** takes the argument to its conclusion:

$$\pi'(s) = \arg\max_a \sum_{s'} p(s' \mid s, a) \bigl[r(s, a) + \gamma V^\pi(s')\bigr].$$

> [!tip]
> **The policy improvement theorem is the engine of the entire field.** Its logic — evaluate the current policy, act greedily on the evaluation — reappears in every subsequent algorithm. The specific form of "greedy" changes: in deep RL, gradient steps replace exact maximization. But the underlying principle is invariant. If the greedy policy $\pi'$ agrees with $\pi$ everywhere, then $V^\pi$ already satisfies the Bellman optimality equations and $\pi$ is optimal. Greedy improvement either strictly improves or certifies nothing can be improved.

---

## Policy Iteration

**Policy iteration**, introduced by Howard (1960), alternates between evaluation and improvement:

> [!note]
> **Algorithm: Policy Iteration**
> 1. Initialize $\pi_0$ arbitrarily.
> 2. **Evaluate**: compute $V^{\pi_k}$ by iterative policy evaluation to convergence.
> 3. **Improve**: set $\pi_{k+1}(s) = \arg\max_a \sum_{s'} p(s' \mid s, a) [r(s, a) + \gamma V^{\pi_k}(s')]$ for all $s$.
> 4. If $\pi_{k+1} = \pi_k$, terminate — the policy is optimal. Otherwise, return to step 2.

The sequence $\pi_0, \pi_1, \pi_2, \ldots$ is monotonically non-decreasing in value. Since each step is either an improvement or a termination, and there are finitely many deterministic policies ($|\mathcal{A}|^{|\mathcal{S}|}$ of them), convergence is guaranteed in finite time.

> [!success]
> In practice, policy iteration converges in a handful of iterations regardless of state space size — far fewer than the $|\mathcal{A}|^{|\mathcal{S}|}$ worst-case bound. Howard observed this empirically in 1960 and it has been confirmed across countless domains since. Greedy improvement jumps across policy space in large discrete leaps, skipping thousands of intermediate policies in a single step.

The evaluation phase can be performed either by many sweeps of iterative policy evaluation or, for small enough problems, by direct matrix inversion:

$$\mathbf{v}^{\pi_k} = (\mathbf{I} - \gamma \mathbf{P}^{\pi_k})^{-1} \mathbf{r}^{\pi_k},$$

at cost $O(|\mathcal{S}|^3)$ per iteration. Both are exact. The choice depends on problem size.

---

## Value Iteration

> [!question]
> Do we have to run evaluation to full convergence before improving? What if we interleave evaluation and improvement more aggressively?

**Value iteration** collapses the separation between evaluation and improvement entirely, iterating the Bellman optimality operator $\mathcal{T}^*$ directly:

$$V_{k+1}(s) \leftarrow \max_a \sum_{s'} p(s' \mid s, a) \bigl[r(s, a) + \gamma V_k(s')\bigr], \quad \forall s \in \mathcal{S}.$$

Since $\mathcal{T}^*$ is a contraction with modulus $\gamma$, the iterates $V_k$ converge to $V^*$ geometrically from any initialization. Once $V_k$ is sufficiently close to $V^*$, the optimal policy is extracted by a single greedy step.

> [!tip]
> Value iteration is policy iteration in which each evaluation phase is truncated to a single sweep. The total computational cost is comparable in practice. The conceptual payoff: value iteration needs no notion of a "current policy" — it manipulates value functions directly, treating the max operator as a combined evaluation and improvement step. **The $\max$ is simultaneously doing policy evaluation and policy improvement in one pass.**

The practical difference between the two algorithms is mainly structural. Policy iteration maintains a policy $\pi_k$ and a value function $V^{\pi_k}$ as separate objects. Value iteration maintains only a value function and updates it toward its own greedy improvement. Both terminate at $V^*$.

---

## Asynchronous Updates and Prioritized Sweeping

Both algorithms as stated perform synchronous sweeps: every state is updated simultaneously. This is analytically clean but computationally wasteful — many states may be irrelevant to the optimal policy, or may have already converged.

> [!note]
> **In-place dynamic programming** replaces the two-array synchronous update with a single-array update that uses the most recent value of each state immediately:
> $$V(s) \leftarrow \max_a \sum_{s'} p(s' \mid s, a) \bigl[r(s, a) + \gamma V(s')\bigr].$$
> The order of updates now affects convergence speed (but not correctness). Updating states in an order that respects information flow — from states with known values toward states that depend on them — speeds convergence substantially.

**Prioritized sweeping** (Moore and Atkeson, 1993) goes further: maintain a priority queue ordered by Bellman error $|V(s) - (\mathcal{T}^* V)(s)|$, and always update the highest-priority state. States with large Bellman errors are the states where the current estimate is most wrong; concentrating compute there is more efficient than sweeping uniformly.

> [!success]
> Prioritized sweeping solves small MDPs much faster than uniform sweeping, often by an order of magnitude. This heuristic anticipates, in the tabular setting, the experience replay and prioritized sampling ideas that would appear decades later in deep RL. The logic is the same: not all updates are equally informative, and spending compute on the most informative ones is better than spending it uniformly.

---

## The Complexity Wall

> [!warning]
> **The curse of dimensionality.** Dynamic programming works on a $10 \times 10$ grid (10,000 entries, exact, milliseconds). Add one dimension and the state space multiplies by $10$. Discretize a robot joint angle into $k$ bins: the state space grows by a factor of $k$ per joint. A six-joint arm with 100 positions per joint has $100^6 = 10^{12}$ states; a single DP sweep requires $10^{24}$ operations. Backgammon: $\approx 10^{20}$ states. Chess: $\approx 10^{43}$. The state space grows exponentially in the number of state variables, and the cost of tabular DP grows with the state space.

The curse is not an artifact of poor algorithm design. It is a statement about the **representation**. A tabular value function assigns an independent number to each state, with no structure connecting similar states. Two physically adjacent, geometrically similar, or semantically related states share no information in a lookup table. All generalization must come from the Bellman backup — one step at a time — providing only local generalization. The value at robot configuration $(45°, 30°, \ldots)$ conveys nothing to the estimate at $(46°, 30°, \ldots)$ except through an explicit DP update.

---

## What DP Establishes for Everything That Follows

The computational intractability of DP does not make it unimportant. Its algorithms establish the reference points against which all subsequent methods are measured.

> [!tip]
> Three properties of DP algorithms carry forward, in approximate form, into every algorithm in the field:
>
> 1. **Convergence from any initialization** — because the Bellman operators are contractions. Tabular Q-learning inherits this property directly. Deep Q-networks lose it when function approximation violates the contraction conditions, and the failure modes are precisely the ways those conditions break.
>
> 2. **Monotone policy improvement** — the guarantee that no update makes the policy worse. TRPO constructs a lower bound on this improvement; PPO approximates it with a clipped objective. Both are attempts to preserve the discrete-jump guarantee of policy iteration in the continuous world of gradient steps.
>
> 3. **The prediction-control separation** — survives intact in actor-critic architectures: the critic solves the prediction problem (iterative policy evaluation with function approximation), the actor solves the control problem (policy improvement via gradient ascent).

---

## From Computation to Learning

> [!question]
> Dynamic programming requires the transition kernel $p(s' \mid s, a)$ at every sweep. What happens when $p$ is unknown — which is the case in almost every real problem of interest?

Every DP sweep applies the expectation $\sum_{s'} p(s' \mid s, a) [\cdots]$, which requires knowing $p$. In most real problems — robot dynamics, game engines, financial markets, molecular simulations — the transition kernel is either too complex to model exactly or unobservable. The agent must discover how the environment works through experience.

> [!tip]
> The solution, identified by Sutton (1988), is to replace the expectation over next states with a **sample**. Instead of computing $\sum_{s'} p(s' \mid s, a) [r(s,a) + \gamma V(s')]$ exactly, observe a single transition $(s, a, r, s')$ and use $r + \gamma V(s')$ as a noisy but unbiased estimate. Averaging over many such samples recovers the true expectation in the limit.
>
> This replacement — **sampling instead of sweeping** — is the birth of temporal difference learning. Bootstrapping (using current value estimates to update other value estimates, rather than waiting for a full return) is what makes it feasible. And the Bellman equations are what make bootstrapping *valid*: they say exactly what the value of a state should be in terms of the value of its successors.

The logic runs from Bellman (1957) through Howard (1960) through Sutton (1988) in a direct line. Each step is forced by the limitations of the previous one: the need for computation forced DP; the intractability of exact computation forced sampling; the combination of sampling and function approximation would eventually force the deep RL methods of the next decade. The Bellman equations are the fixed point toward which all of these algorithms are converging, by different approximations and at different speeds.

---

*Howard (1960), Dynamic Programming and Markov Processes, introduced policy iteration and the policy improvement theorem. The relationship between policy iteration and value iteration via modified policy iteration is analyzed in Puterman and Shin (1978). Prioritized sweeping was introduced in Moore and Atkeson (1993), Prioritized Sweeping: Reinforcement Learning with Less Data and Less Time. The full complexity theory of dynamic programming for MDPs is in Puterman (1994), Markov Decision Processes. The connection between policy iteration and policy gradient methods — monotone improvement as a design constraint — is analyzed in Kakade and Langford (2002), Approximately Optimal Approximate Reinforcement Learning.*
