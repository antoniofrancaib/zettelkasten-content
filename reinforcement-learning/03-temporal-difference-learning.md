---
title: "Temporal Difference Learning"
subtitle: "Bootstrapping, TD(0), the n-step spectrum, eligibility traces, SARSA, and Q-learning"
---


The algorithms of the previous chapter assumed something that is almost never true in practice: the transition kernel is known. Every sweep of iterative policy evaluation computes $\sum_{s'} p(s' \mid s, a)[\cdots]$ — an expectation that requires knowing the probability of landing in each next state. In a board game with explicit rules, this can be computed analytically. In robotics, financial markets, protein folding, or language generation, the dynamics are either too complex to model exactly or simply unobservable. The agent must learn the value function from **experience** — from sequences of observed states, actions, and rewards — without ever accessing $p$ directly.

Two approaches suggest themselves. The first is Monte Carlo estimation: run complete episodes, compute the actual return $G_t = R_{t+1} + \gamma R_{t+2} + \cdots$, and average. No model, no bootstrap, no approximation — just the empirical average of observed returns. The second is less obvious and more powerful: do not wait for the episode to end. After observing a single transition $(S_t, A_t, R_{t+1}, S_{t+1})$, use the current estimate of $V(S_{t+1})$ as a proxy for the future return and update immediately. This is **temporal difference learning** — the bootstrapping insight identified by Sutton in 1988.

The difference between these two approaches is not merely algorithmic. It reflects two different views of what evidence should update a value estimate, and the tension between them shapes every algorithm in the field.

---

## Monte Carlo: The Unbiased Baseline

> [!question]
> What is the simplest model-free approach to policy evaluation, and where does it break down?

The **Monte Carlo prediction** algorithm is the most direct model-free approach: run the policy $\pi$ for complete episodes, record the actual return $G_t$ from each visited state, and update:

$$V(S_t) \leftarrow V(S_t) + \alpha\bigl[G_t - V(S_t)\bigr].$$

Since $G_t = \sum_{k=0}^\infty \gamma^k R_{t+1+k}$ is an unbiased sample of $V^\pi(S_t)$, the empirical average converges to the true value function by the law of large numbers. No Bellman equation, no model, no approximation. The estimate is statistically clean.

> [!warning]
> **Monte Carlo's two decisive limitations:**
>
> 1. **Requires complete episodes.** $G_t$ cannot be computed until the episode terminates, because it depends on all future rewards. Many environments have no natural termination — continuous control tasks, physical processes, and most real-time systems operate indefinitely.
>
> 2. **High variance.** $G_t$ is a sum of many random rewards, each contributing independent noise. Variance grows with the horizon. For long episodes in stochastic environments, thousands of samples may be needed for reliable estimation.
>
> Both limitations share the same root cause: Monte Carlo treats each state's value as an independent quantity to be estimated from scratch. The Bellman equations say state values are *not* independent — they are constrained by $V^\pi(s) = \mathbb{E}[R_{t+1} + \gamma V^\pi(S_{t+1})]$. Monte Carlo refuses to exploit this structure.

---

## The Bootstrapping Idea and TD(0)

> [!question]
> Can we exploit the recursive structure of the Bellman equations to update value estimates after a single transition, without waiting for an episode to end?

The Bellman expectation equation for $\pi$ states:

$$V^\pi(S_t) = \mathbb{E}_\pi\bigl[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t\bigr].$$

This says that $R_{t+1} + \gamma V^\pi(S_{t+1})$ is an unbiased estimator of $V^\pi(S_t)$ — exactly as $G_t$ is, but requiring only one step of observed reward. If we had the true $V^\pi$, using this one-step target would be strictly better than Monte Carlo: same bias (zero), lower variance.

We do not have the true $V^\pi$. But we have an estimate $V$, and we can substitute it:

$$V(S_t) \leftarrow V(S_t) + \alpha \underbrace{\bigl[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\bigr]}_{\delta_t}.$$

> [!info]
> This is **TD(0)**. The quantity $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the **TD error**: the difference between what the Bellman equation predicts $V(S_t)$ should be (given the observed transition) and what the current estimate says it is. A positive TD error means $S_t$ was better than expected; negative means it was worse. The update nudges $V(S_t)$ toward consistency with the observed transition.

> [!note]
> **Bootstrapping** refers to the circularity here: the current estimate $V(S_{t+1})$ is being used to generate a target for updating $V(S_t)$. A value estimate is used to improve another value estimate. This is not sound in general inference — it introduces a bias that Monte Carlo avoids. But it works because the Bellman equations constrain what a self-consistent set of value estimates must look like. The contraction property of $\mathcal{T}^\pi$ ensures that the fixed point of bootstrapped iteration is the unique true value function $V^\pi$.

TD(0) updates after every **single transition**, without waiting for an episode to end. It is fully online: estimates improve continuously as experience arrives. This alone makes it qualitatively different from Monte Carlo.

---

## Stochastic Approximation and Convergence

> [!info]
> TD(0) is a special case of **stochastic approximation**: a framework for finding the fixed point of an operator using noisy estimates of its output. The target $R_{t+1} + \gamma V(S_{t+1})$ is a single-sample estimate of $(\mathcal{T}^\pi V)(S_t)$. The TD(0) update moves $V(S_t)$ toward this noisy target with step size $\alpha$.

> [!success]
> **Convergence theorem (Sutton 1988, formalized by Tsitsiklis 1994).** Under the **Robbins-Monro conditions** on step sizes:
> $$\sum_t \alpha_t = \infty \quad \text{and} \quad \sum_t \alpha_t^2 < \infty,$$
> and assuming every state is visited infinitely often, TD(0) converges with probability 1 to $V^\pi$.
>
> The first condition ensures enough total learning to reach the fixed point; the second ensures noise diminishes fast enough for estimates to stabilize. The Banach contraction property of $\mathcal{T}^\pi$ is the key analytical ingredient: it guarantees that the expected update at each step points toward $V^\pi$.

In practice, a **constant step size** $\alpha$ is often preferred over a decaying schedule. It converges to a neighborhood of $V^\pi$ rather than the exact fixed point, but it tracks non-stationarity — useful in continuing tasks where the environment or policy evolves over time.

---

## The $n$-Step Spectrum

> [!question]
> TD(0) bootstraps after one step; Monte Carlo waits for the full return. Is there a principled way to interpolate between them?

The **$n$-step return** uses $n$ steps of observed reward before bootstrapping:

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n}).$$

For $n = 1$: the TD(0) target. As $n \to \infty$: the Monte Carlo return $G_t$.

> [!tip]
> **The bias-variance tradeoff made explicit.** Larger $n$ reduces the bias introduced by bootstrapping — the target depends less on the current (potentially inaccurate) estimates — at the cost of higher variance from summing more observed rewards.
>
> - *Early in training*, when $V$ is far from $V^\pi$: larger $n$ is often better, because bootstrapped values introduce substantial bias.
> - *Late in training*, when $V \approx V^\pi$: smaller $n$ is often better, because bias is small and shorter returns propagate information faster with lower variance.

This tradeoff is not merely a curiosity of the tabular setting. It recurs in a different form in the **generalized advantage estimation (GAE)** of actor-critic methods — where $\lambda$-weighted averaging of $n$-step advantages plays the same role that TD($\lambda$) plays here.

---

## TD($\lambda$) and Eligibility Traces

Rather than committing to a single $n$, **TD($\lambda$)** averages across all $n$-step returns with geometrically decaying weights:

$$G_t^\lambda = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}.$$

When $\lambda = 0$: reduces to TD(0). When $\lambda = 1$: equivalent to Monte Carlo. For $0 < \lambda < 1$: a smooth interpolation that avoids the binary choice.

> [!note]
> The **forward view** of TD($\lambda$) updates using $G_t^\lambda$ as the target, but requires knowing all future rewards first — not online.
>
> The **backward view** provides an equivalent online algorithm using **eligibility traces**: a per-state memory vector $\mathbf{z}_t$ that accumulates recent visits with geometric decay:
> $$z_t(s) = \begin{cases} \gamma\lambda\, z_{t-1}(s) + 1 & \text{if } s = S_t, \\ \gamma\lambda\, z_{t-1}(s) & \text{otherwise.} \end{cases}$$
> At each step, the TD error $\delta_t$ is propagated backward through all recently visited states proportional to their eligibility:
> $$V(s) \leftarrow V(s) + \alpha\,\delta_t\,z_t(s), \quad \forall s.$$
> States visited recently and frequently have high eligibility and receive large updates. The forward and backward views are equivalent in expectation — they produce the same expected parameter updates, step by step, even though they compute them in opposite temporal directions.

> [!tip]
> Eligibility traces are the first mechanism in this progression that explicitly connects **past states to present errors** — the credit assignment problem in embryonic form. In neuroscience, the TD error has been identified with dopaminergic prediction error signals; eligibility traces correspond to synaptic eligibility — the biological mechanism by which neurons that fired recently remain eligible for potentiation. The mathematical structure and the biological implementation converge on the same solution to the same problem.

---

## From Prediction to Control: SARSA

> [!question]
> TD(0) solves the prediction problem. How do we extend it to the control problem — finding the optimal policy — without a model?

Improving a policy given $V^\pi$ requires $\arg\max_a \sum_{s'} p(s' \mid s, a)[r(s,a) + \gamma V^\pi(s')]$, which needs $p$. The model-free resolution: estimate **action-value functions** $Q^\pi(s, a)$ directly. With $Q^\pi$ in hand, policy improvement requires no model: $\pi'(s) = \arg\max_a Q^\pi(s, a)$.

**SARSA** — named for the quintuple $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ used in each update — applies TD prediction to $Q$:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\bigl[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\bigr].$$

> [!info]
> The target uses $A_{t+1}$, the action *actually selected* by the current policy. This makes SARSA **on-policy**: the Q-function estimate is consistent with the policy generating the experience, including its exploratory deviations. Interleaving SARSA updates with $\varepsilon$-greedy action selection produces an online generalized policy iteration that converges to the optimal policy as $\varepsilon \to 0$.

The exploration parameter $\varepsilon$ resolves the **exploration-exploitation tension**: an always-greedy agent can trap itself in a locally good policy, never visiting regions where current estimates are wrong. Occasional random actions ensure all relevant regions of state-action space are eventually visited. This is the simplest instantiation of a problem that pervades RL and has no clean general solution.

---

## Q-Learning: Off-Policy Control

> [!question]
> SARSA evaluates the policy that is actually being executed, including its exploratory noise. Can we instead directly estimate the value of the optimal greedy policy, regardless of how experience is collected?

Watkins' 1989 PhD thesis introduced the modification. Replace the action-value of the next *selected* action in the SARSA target with the *maximum* action-value over all next actions:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\bigl[\underbrace{R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')}_{\text{Bellman optimality target}} - Q(S_t, A_t)\bigr].$$

> [!tip]
> The target $R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')$ is a sample estimate of the right-hand side of the Bellman optimality equation:
> $$Q^*(s, a) = \sum_{s'} p(s' \mid s, a)\bigl[r(s, a) + \gamma \max_{a'} Q^*(s', a')\bigr].$$
> **Q-learning is stochastic approximation of value iteration applied to action-value functions.** Because the target uses $\max_{a'}$ — the greedy action, not the action taken — Q-learning is **off-policy**: the target policy (greedy w.r.t. $Q$) differs from the behavior policy generating experience ($\varepsilon$-greedy). The behavior policy needs only to ensure sufficient exploration; the target policy is always implicitly greedy.

> [!note]
> **Off-policy character: strength and eventual instability.**
>
> *Strength*: experience from any exploratory policy can be used to learn about the greedy policy. The behavior policy could be $\varepsilon$-greedy, uniformly random, or replayed from a buffer of old transitions — making Q-learning flexible about data collection.
>
> *Instability*: the asymmetry between the behavior policy generating bootstrapped targets and the target policy being evaluated can, with function approximation, cause Q-estimates to diverge. This is the *deadly triad* — the subject of the next chapter.

---

## SARSA vs. Q-Learning: A Concrete Distinction

The difference between on-policy and off-policy control is not merely technical — in environments where exploration is costly or dangerous it produces qualitatively different behavior.

> [!success]
> **The cliff-walking example.** A grid world with a cliff along one edge: the optimal path runs close to the cliff (direct), a conservative path runs far from it.
>
> - **SARSA** evaluates the $\varepsilon$-greedy policy *including* random deviations. With small probability, a random step near the cliff leads to falling, and this risk is absorbed into the Q-values. SARSA learns the **conservative path**.
> - **Q-learning** always uses the greedy target, which never deliberately steps toward the cliff. Q-values reflect the optimal cost, not the $\varepsilon$-greedy cost. Q-learning learns the **optimal (risky) path**.
>
> Both converge to the same optimal policy as $\varepsilon \to 0$. But at any finite $\varepsilon$ — which is always the case during learning — Q-learning is more aggressive. This is not always the right tradeoff, particularly in systems with real-world consequences for exploration failures.

The choice between on-policy and off-policy estimation is one of the first genuinely practical design decisions in reinforcement learning. It does not disappear in the deep RL era: PPO is on-policy by design (importance ratios correct for the small distribution shift within an epoch); Q-learning with replay buffers is aggressively off-policy.

---

## The Frontier of the Tabular Era

> [!success]
> By the mid-1990s, the tabular TD landscape was complete. TD(0) converged to $V^\pi$. SARSA converged to $Q^*$ with appropriately decaying $\varepsilon$. Q-learning converged to $Q^*$ regardless of the behavior policy, provided all state-action pairs were visited infinitely often. Given enough experience, all three algorithms produced the correct answer — the same answer as the exact DP algorithms of Chapter 02, without ever requiring the transition kernel.

> [!warning]
> **The remaining wall: state space size.** Every convergence proof relies on visiting every state-action pair infinitely often. Every Q-value requires an entry in a lookup table. For Atari with raw pixel inputs, the Q-table would require one entry per possible pixel configuration — a table with $256^{33600}$ entries, of which any single game visits an infinitesimal fraction. The Q-table cannot be stored; even if it could, no individual state would be visited enough for a reliable estimate.
>
> The solution, proposed by Mnih et al. (2013): replace the Q-table with a **neural network** that takes a state as input and outputs Q-values for all actions. The resulting algorithm — the Deep Q-Network — achieves superhuman performance across dozens of Atari games. But the convergence guarantees of tabular Q-learning do not carry over automatically, and new engineering innovations are required to stabilize neural Q-learning. Those innovations, and the reasons they were necessary, are the subject of the next chapter.

---

*Temporal difference learning was introduced in Sutton (1988), Learning to Predict by the Methods of Temporal Differences. Q-learning was developed by Watkins in his 1989 PhD thesis and published as Watkins and Dayan (1992), Q-learning. Rigorous convergence proofs for TD(0) using the ODE method are in Tsitsiklis (1994), Asynchronous Stochastic Approximation and Q-Learning. SARSA was described by Rummery and Niranjan (1994) and named by Sutton (1996). TD($\lambda$) and the forward-backward equivalence of eligibility traces are developed in Sutton and Barto (2018), chapters 6, 7, and 12. The connection between TD errors and dopaminergic prediction error signals was established by Schultz, Dayan, and Montague (1997), A Neural Substrate of Prediction and Reward.*
