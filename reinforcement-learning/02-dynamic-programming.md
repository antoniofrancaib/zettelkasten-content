---

title: "Dynamic Programming"
subtitle: "Policy evaluation, policy iteration, value iteration, and their exact-but-intractable cost"
---


Dynamic programming arrived at the reinforcement learning problem with a mandate: given everything about the environment — the transition probabilities, the reward function, the full state and action spaces — compute the optimal policy exactly. This is a different problem from learning, and it has a cleaner solution. Bellman's 1957 book established the theoretical framework; Howard's 1960 monograph *Dynamic Programming and Markov Processes* worked out the computational algorithms. The result was a pair of procedures — policy iteration and value iteration — that provably converge to the optimal policy in finite time, require no approximation, and suffer from a single decisive flaw: they scale catastrophically with the size of the state space.

That flaw is important to understand not merely because it limits the direct applicability of dynamic programming, but because the algorithms that follow — temporal difference learning, Q-learning, deep Q-networks, policy gradients — are all attempts to preserve the logical structure of DP while avoiding its computational pathologies. The right way to understand TD learning is as policy evaluation without sweeping all states. The right way to understand Q-learning is as value iteration without knowing the transition kernel. These connections are lost if the DP algorithms are treated as a historical curiosity rather than the theoretical spine they actually are.

## The Two Problems

Before describing algorithms, it is worth separating the two problems they solve.

The **prediction problem** asks: given a policy $\pi$, compute $V^\pi$. This is also called *policy evaluation*. Its solution tells us how good the policy is at every starting state — not the immediate reward, but the full long-run expected return. Policy evaluation is a subproblem of everything that follows, and an important task in its own right: before you can improve a policy, you have to measure it.

The **control problem** asks: find the optimal policy $\pi^*$ and its value function $V^*$. This is harder, since $\pi^*$ is unknown. Policy iteration and value iteration are the two classical algorithms for the control problem; both use policy evaluation as a subroutine or implicit building block.

Both problems assume **full model knowledge**: the transition kernel $p(s' \mid s, a)$ and reward function $r(s, a)$ are given exactly. This is the **planning** setting — an agent that knows how the world works and needs only to compute what to do. When the model is unknown and must be discovered through interaction, we are in the reinforcement learning setting proper, and the algorithms of this chapter must be adapted.

## Iterative Policy Evaluation

The Bellman expectation equation for a policy $\pi$ is a fixed-point equation in $V^\pi$. Define the **Bellman expectation operator** $\mathcal{T}^\pi$:

$$(\mathcal{T}^\pi V)(s) = \sum_a \pi(a \mid s) \sum_{s'} p(s' \mid s, a) \bigl[r(s, a) + \gamma V(s')\bigr].$$

The function we want is the unique fixed point $V^\pi = \mathcal{T}^\pi V^\pi$. As established in the previous chapter, $\mathcal{T}^\pi$ is a contraction with modulus $\gamma$ under the supremum norm. Banach's fixed-point theorem guarantees that the iteration $V_{k+1} = \mathcal{T}^\pi V_k$ converges to $V^\pi$ from any initialization, at a geometric rate.

The practical algorithm, **iterative policy evaluation**, initializes $V_0(s) = 0$ for all $s$ and repeatedly sweeps:

$$V_{k+1}(s) \leftarrow \sum_a \pi(a \mid s) \sum_{s'} p(s' \mid s, a) \bigl[r(s, a) + \gamma V_k(s')\bigr], \quad \forall s \in \mathcal{S}.$$

Each sweep is **synchronous**: all states are updated in parallel using values from the previous iteration. The convergence bound is direct from the contraction property:

$$\|V_{k} - V^\pi\|_\infty \leq \gamma^k \|V_0 - V^\pi\|_\infty.$$

In practice, iteration is stopped when the maximum single-sweep change falls below a threshold $\varepsilon$: $\max_s |V_{k+1}(s) - V_k(s)| < \varepsilon$. This guarantees that the residual error satisfies $\|V_k - V^\pi\|_\infty \leq 2\varepsilon\gamma/(1-\gamma)$, which can be made arbitrarily small.

The cost per sweep is $O(|\mathcal{S}|^2 |\mathcal{A}|)$: for each of $|\mathcal{S}|$ states, compute an expectation over $|\mathcal{A}|$ actions, each requiring a sum over $|\mathcal{S}|$ next states. The algorithm terminates in a number of sweeps proportional to $\log(1/\varepsilon) / \log(1/\gamma)$ — modest for typical $\gamma$ values, but multiplied against the quadratic state-space cost at each sweep.

## The Policy Improvement Theorem

Given an accurate estimate of $V^\pi$, a natural question arises: is there a better policy? The answer is almost always yes, and the mechanism for finding one is the **policy improvement theorem**.

Suppose we have computed $V^\pi$ for some policy $\pi$. Consider a modified policy $\pi'$ that at some state $s$ takes a different action $a$ — one for which $Q^\pi(s, a) > V^\pi(s)$ — and thereafter follows $\pi$ exactly. The policy improvement theorem states that this modification makes $\pi'$ at least as good as $\pi$ at every state: $V^{\pi'}(s') \geq V^\pi(s')$ for all $s'$, with strict inequality at $s$.

The proof is a straightforward unrolling of the Bellman equations. Since $V^\pi(s) \leq Q^\pi(s, \pi'(s))$, we can write:

$$V^\pi(s) \leq Q^\pi(s, \pi'(s)) = \mathbb{E}\bigl[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s,\, A_t = \pi'(s)\bigr].$$

Applying the same inequality at $S_{t+1}$ and iterating the expansion forward through time:

$$V^\pi(s) \leq \mathbb{E}_{\pi'}\!\left[\sum_{k=0}^{\infty} \gamma^k R_{t+1+k} \;\middle|\; S_t = s\right] = V^{\pi'}(s).$$

The extension to all states simultaneously is immediate: the **greedy improvement** policy

$$\pi'(s) = \arg\max_a \sum_{s'} p(s' \mid s, a) \bigl[r(s, a) + \gamma V^\pi(s')\bigr]$$

satisfies $V^{\pi'}(s) \geq V^\pi(s)$ for every $s \in \mathcal{S}$. Moreover, if $\pi' = \pi$ — if the greedy policy agrees with the current policy everywhere — then $V^\pi$ already satisfies the Bellman optimality equations, and $\pi$ is optimal. Greedy improvement either strictly improves the policy or certifies that no improvement is possible.

This theorem is the engine of the entire field. Its logic reappears in every policy optimization algorithm: take a policy, evaluate it, find a better one by acting greedily on the evaluation. The specific form of "greedy" changes — in deep RL, gradient steps replace exact maximization — but the underlying principle does not.

## Policy Iteration

The **policy iteration** algorithm, introduced by Howard (1960), alternates between evaluation and improvement:

1. Initialize $\pi_0$ arbitrarily (e.g., uniformly random).
2. **Evaluate**: compute $V^{\pi_k}$ by running iterative policy evaluation to convergence.
3. **Improve**: set $\pi_{k+1}(s) = \arg\max_a \sum_{s'} p(s' \mid s, a) [r(s, a) + \gamma V^{\pi_k}(s')]$ for all $s$.
4. If $\pi_{k+1} = \pi_k$, terminate — the policy is optimal. Otherwise, return to step 2.

The sequence of policies $\pi_0, \pi_1, \pi_2, \ldots$ is monotonically non-decreasing in value. Since every step is either an improvement or a termination, and there are at most $|\mathcal{A}|^{|\mathcal{S}|}$ distinct deterministic policies, the algorithm terminates in finite time. In practice, convergence is far faster than this worst-case bound: Howard observed that policy iteration typically converges in a handful of iterations regardless of state space size. The reason is that greedy improvement makes large leaps across policy space — updating every state simultaneously can skip over thousands of intermediate policies in a single step.

The main cost is the evaluation phase, which requires either many sweeps of iterative policy evaluation or, for small enough problems, a direct matrix inversion:

$$\mathbf{v}^{\pi_k} = (\mathbf{I} - \gamma \mathbf{P}^{\pi_k})^{-1} \mathbf{r}^{\pi_k},$$

at cost $O(|\mathcal{S}|^3)$ per iteration. Both approaches are exact; the choice between them depends on state space size.

## Value Iteration

An alternative approach collapses the separation between evaluation and improvement. **Value iteration** directly iterates the Bellman optimality operator $\mathcal{T}^*$:

$$V_{k+1}(s) \leftarrow \max_a \sum_{s'} p(s' \mid s, a) \bigl[r(s, a) + \gamma V_k(s')\bigr], \quad \forall s \in \mathcal{S}.$$

Since $\mathcal{T}^*$ is a contraction with modulus $\gamma$, the iterates $V_k$ converge to $V^*$ geometrically from any initialization. The convergence rate is identical to iterative policy evaluation: $\|V_{k+1} - V^*\|_\infty \leq \gamma \|V_k - V^*\|_\infty$. Once $V_k$ is sufficiently close to $V^*$, the optimal policy is extracted by a single final greedy step.

Value iteration can be understood as policy iteration in which each evaluation phase is truncated to a single sweep rather than run to convergence. This is less work per outer iteration, but requires more outer iterations to compensate. The total computational cost is comparable in practice. The conceptual payoff is that value iteration requires no notion of a "current policy" to evaluate — it manipulates value functions directly, treating the max operator as a combined evaluation and improvement step.

The practical difference between the two algorithms is mainly one of implementation structure. Policy iteration maintains a policy $\pi_k$ and a value function $V^{\pi_k}$ as separate objects, alternating between updating one and using it to update the other. Value iteration maintains only a value function $V_k$ and updates it toward its own greedy one-step improvement. Both terminate at $V^*$.

## Asynchronous Updates

Both algorithms as stated perform synchronous sweeps: every state is updated simultaneously, using values from the previous iteration. This is analytically clean but computationally wasteful. In a $10{,}000$-state MDP, a single synchronous sweep updates every state — including states that will never be visited by any reasonable policy, and states whose values converged long ago.

**In-place dynamic programming** replaces the two-array synchronous update with a single-array update that uses the most recent value of each state immediately:

$$V(s) \leftarrow \max_a \sum_{s'} p(s' \mid s, a) \bigl[r(s, a) + \gamma V(s')\bigr].$$

The order in which states are updated now matters for the speed of convergence (though not for its eventual correctness). Updating states in an order that respects the flow of information — from states with known values toward states that depend on them — speeds convergence substantially.

**Prioritized sweeping** (Moore and Atkeson, 1993) takes this further: maintain a priority queue of states ordered by the magnitude of their Bellman error, $|V(s) - (\mathcal{T}^* V)(s)|$, and always update the highest-priority state. States with large Bellman errors are the states where the current value estimate is most wrong; prioritizing them concentrates computation where it is most needed. Empirically, prioritized sweeping solves small MDPs much faster than uniform sweeping, often by an order of magnitude.

This heuristic anticipates, in the tabular setting, the experience replay and prioritization ideas that would appear decades later in deep RL. The logic is the same: not all updates are equally informative, and spending compute on the most informative ones is better than spending it uniformly.

## The Complexity Wall

Dynamic programming works. On a $10 \times 10$ grid world, policy iteration converges in under twenty iterations, each requiring milliseconds. The optimal policy is exact.

The problem arrives immediately thereafter.

Add one dimension: a $10 \times 10 \times 10$ grid has $10^3$ states, and the cost per sweep grows by a factor of ten. Discretize a single continuous variable — say, a robot joint angle in $[0, 2\pi)$ — into $k$ bins: the state space grows by a factor of $k$ for each joint. A robot arm with six joints, each discretized into 100 positions, has $100^6 = 10^{12}$ states. A single DP sweep would require $10^{24}$ operations. A game of backgammon has approximately $10^{20}$ states; chess has approximately $10^{43}$. Bellman named this the **curse of dimensionality**: the state space grows exponentially in the number of state variables, and the cost of tabular dynamic programming grows with the state space.

The curse is not an artifact of poor algorithm design. It is a statement about the representation. A tabular value function assigns an independent number to each state, with no structure connecting similar states. Two states that are physically adjacent, geometrically similar, or semantically related carry no shared information in a lookup table. All generalization must come from the Bellman backup — from the fact that a state's value is computed from the values of its neighbors — and this provides only local generalization, one step at a time. For the robot arm problem, the value of a configuration at angle $(45°, 30°, \ldots)$ conveys nothing about the value at $(46°, 30°, \ldots)$ except through an explicit DP update connecting the two.

## What the Algorithms Establish

The computational intractability of DP does not make it unimportant. Its algorithms establish the reference points against which all subsequent methods are measured, and its theoretical properties are the properties that subsequent methods must approximate.

**Convergence from any initialization** is a property that carries forward. Tabular Q-learning converges to $Q^*$ under mild conditions because it is performing stochastic approximation of value iteration, inheriting the contraction property that makes value iteration converge. Deep Q-networks converge less reliably, and the reasons are precisely the ways in which function approximation violates the conditions that make the tabular contraction argument work.

**Monotone policy improvement** is the property that policy gradient methods try to preserve. TRPO constructs a lower bound on the policy improvement guaranteed by a policy update; PPO approximates this bound with a simpler clipped objective. Both are attempts to ensure that gradient steps in policy space produce improvements analogous to the discrete jumps of policy iteration — steps that never make the policy worse, and often make it much better.

**The prediction-control separation** survives intact. Actor-critic architectures split the RL algorithm along exactly this line: a critic that solves the prediction problem (estimating the value of the current policy), and an actor that solves the control problem (improving the policy using the critic's estimates). The critic is iterative policy evaluation implemented with function approximation; the actor is policy improvement implemented with gradient ascent. The separation is Howard's, even if the implementation is entirely different.

## From Computation to Learning

Dynamic programming requires the transition kernel. Every sweep applies the expectation $\sum_{s'} p(s' \mid s, a) [\cdots]$, which requires knowing $p$. In most real problems, $p$ is unknown: the dynamics of a robot arm, a game engine, a financial market, or a protein-folding simulation are either too complex to model exactly or not observable at all. The agent must discover how the environment works through experience.

The solution, identified by Sutton in 1988, is to replace the expectation over next states with a sample. Instead of computing $\sum_{s'} p(s' \mid s, a) [r(s,a) + \gamma V(s')]$ exactly, observe a single transition $(s, a, r, s')$ and use the sample $r + \gamma V(s')$ as an estimate. This sample is noisy — different transitions from the same $(s, a)$ produce different $s'$ — but it is unbiased, and averaging over many such samples recovers the true expectation in the limit.

The resulting algorithm is **temporal difference learning**: policy evaluation by sampling rather than sweeping, one transition at a time, converging to the same fixed point as iterative policy evaluation in the limit of infinite data. Bootstrapping — using current value estimates to update other value estimates, rather than waiting for a full return — is what makes it feasible. And the Bellman equations are what make bootstrapping valid: they say exactly what the value of a state should be in terms of the value of its successors, which is the quantity being estimated one sample at a time.

The logic runs from Bellman (1957) through Howard (1960) through Sutton (1988) in a direct line. Each step is forced by the limitations of the previous one: the need for computation forced DP; the intractability of exact computation forced sampling; and the combination of sampling and function approximation would eventually force the deep RL methods of the next decade. The Bellman equations are the fixed point toward which all of these algorithms are converging, by different approximations and at different speeds.

---

*Howard (1960), Dynamic Programming and Markov Processes, introduced policy iteration and the policy improvement theorem. The relationship between policy iteration and value iteration via modified policy iteration is analyzed in Puterman and Shin (1978). Prioritized sweeping was introduced in Moore and Atkeson (1993), Prioritized Sweeping: Reinforcement Learning with Less Data and Less Time. The full complexity theory of dynamic programming for MDPs is in Puterman (1994), Markov Decision Processes. The connection between policy iteration and policy gradient methods — and the monotone improvement property as a design constraint — is analyzed in Kakade and Langford (2002), Approximately Optimal Approximate Reinforcement Learning.*
