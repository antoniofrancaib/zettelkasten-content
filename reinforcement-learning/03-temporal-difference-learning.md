---

title: "Temporal Difference Learning"
subtitle: "TD(0), TD(λ), the bootstrapping idea, SARSA and Q-learning"
---


The algorithms of the previous chapter assumed something that is never true in most real applications: the transition kernel is known. Every sweep of iterative policy evaluation computes $\sum_{s'} p(s' \mid s, a)[\cdots]$ — an expectation that requires knowing the probability of landing in each next state. In a board game with explicit rules, this can be computed analytically. In robotics, financial markets, protein folding, or language generation, the dynamics are either too complex to model or simply unobservable. The agent must learn the value function from experience — from sequences of observed states, actions, and rewards — without ever accessing $p$ directly.

Two approaches suggest themselves. The first is Monte Carlo estimation: run complete episodes, compute the actual return $G_t = R_{t+1} + \gamma R_{t+2} + \cdots$, and average over many episodes. No model, no bootstrap, just the empirical average of observed returns. The second approach is less obvious and more powerful: do not wait for the episode to end. After observing a single transition $(S_t, A_t, R_{t+1}, S_{t+1})$, use the current estimate of $V(S_{t+1})$ as a proxy for the future return, and update immediately. This is **temporal difference learning** — the bootstrapping insight identified by Sutton in 1988 in a paper titled "Learning to Predict by the Methods of Temporal Differences."

The difference between these two approaches is not merely algorithmic. It reflects two different views of what a value estimate represents and what evidence should update it.

## Monte Carlo: The Unbiased Baseline

Before developing temporal difference methods, it helps to understand precisely what they improve upon. The **Monte Carlo prediction** algorithm is the simplest model-free approach to policy evaluation: run the policy $\pi$ for complete episodes, record the actual return $G_t$ from each visited state, and update:

$$V(S_t) \leftarrow V(S_t) + \alpha\bigl[G_t - V(S_t)\bigr].$$

Since $G_t = \sum_{k=0}^\infty \gamma^k R_{t+1+k}$ is an unbiased sample of $V^\pi(S_t)$, the empirical average converges to the true value function by the law of large numbers. No Bellman equation, no transition kernel, and no approximation are involved. The estimate is statistically correct in a clean and verifiable sense.

The limitations follow directly from this cleanliness. First, Monte Carlo requires **complete episodes**: $G_t$ cannot be computed until the episode terminates, because it depends on all future rewards. Many environments have no natural termination — continuous control tasks, real-time systems, and most physical processes operate indefinitely. Second, Monte Carlo estimates have **high variance**: $G_t$ is a sum of many random rewards, and each contributes independent noise. The variance of an $n$-step return grows with $n$, and for long episodes in stochastic environments, the variance of a single-episode return can be large enough to require thousands of samples for reliable estimation.

Both limitations have the same root cause: Monte Carlo refuses to use any information about the relationship between state values. It treats each state's value as an independent quantity to be estimated from scratch. The Bellman equations say that state values are not independent — they are related by the recursive identity $V^\pi(s) = \mathbb{E}[R_{t+1} + \gamma V^\pi(S_{t+1})]$ — and this relationship can be exploited.

## The Bootstrapping Idea and TD(0)

The Bellman expectation equation for a fixed policy $\pi$ is:

$$V^\pi(S_t) = \mathbb{E}_\pi\bigl[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t\bigr].$$

This says that $R_{t+1} + \gamma V^\pi(S_{t+1})$ is an unbiased estimator of $V^\pi(S_t)$ — exactly as $G_t$ is, but requiring only one step of observed reward rather than an entire trajectory. If we had the true $V^\pi$, we could use this one-step target directly and achieve lower variance than Monte Carlo, since it depends on only a single random reward rather than a sum of many.

We do not have the true $V^\pi$. But we have an estimate $V$, and we can substitute it:

$$V(S_t) \leftarrow V(S_t) + \alpha \underbrace{\bigl[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\bigr]}_{\delta_t}.$$

This is the **TD(0)** update. The quantity $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the **TD error**: the difference between what the Bellman equation predicts the value of $S_t$ should be, given the observed transition, and what the current estimate says it is. A positive TD error means $S_t$ was better than expected; a negative error means it was worse. The update nudges $V(S_t)$ toward consistency with the observed transition.

**Bootstrapping** refers to the circularity in this update: we are using the current estimate $V(S_{t+1})$ to improve the current estimate $V(S_t)$. A value estimate is being used to generate a target for updating another value estimate. This is not, in general, a sound inference procedure — and indeed it introduces a bias that Monte Carlo avoids — but it works because the Bellman equations constrain what a self-consistent set of value estimates must look like. The contraction property of $\mathcal{T}^\pi$ ensures that the fixed point of bootstrapped iteration is the unique true value function $V^\pi$.

The practical advantage is significant. TD(0) updates after every **single transition**, without waiting for an episode to end. It is fully online: estimates improve continuously as experience arrives, rather than in batch at the end of episodes.

## Stochastic Approximation and Convergence

TD(0) is a special case of **stochastic approximation**: a general framework for finding the fixed point of an operator using noisy estimates of its output. The target $R_{t+1} + \gamma V(S_{t+1})$ is a single-sample estimate of the Bellman expectation $(\mathcal{T}^\pi V)(S_t)$. The TD(0) update moves $V(S_t)$ toward this noisy target with step size $\alpha$.

The classical convergence result — due to Sutton (1988) and formalized using the ODE method by Tsitsiklis (1994) — states that under the **Robbins-Monro conditions** on step sizes,

$$\sum_t \alpha_t = \infty \quad \text{and} \quad \sum_t \alpha_t^2 < \infty,$$

and assuming every state is visited infinitely often, TD(0) converges with probability 1 to $V^\pi$. The first condition ensures enough total learning to reach the fixed point; the second ensures the noise diminishes fast enough for the estimates to stabilize. The Banach contraction property of $\mathcal{T}^\pi$ provides the key analytical ingredient: it guarantees that the expected update at each step points toward $V^\pi$, so the stochastic iterates cannot wander arbitrarily far.

In practice, a **constant step size** $\alpha$ is often preferred over a decaying schedule: it converges not to a fixed point but to a neighborhood of one, tracking non-stationarity if the environment or policy changes over time. This is often the desired behavior in continuing (non-episodic) tasks.

## The $n$-Step Spectrum

TD(0) and Monte Carlo are the extremes of a continuous spectrum of algorithms, parameterized by how many steps of observed reward are used before bootstrapping. The **$n$-step return** is:

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n}).$$

For $n = 1$, this is the TD(0) target $R_{t+1} + \gamma V(S_{t+1})$. As $n \to \infty$, this approaches the Monte Carlo return $G_t$. The corresponding $n$-step TD update replaces the one-step target with $G_t^{(n)}$:

$$V(S_t) \leftarrow V(S_t) + \alpha\bigl[G_t^{(n)} - V(S_t)\bigr].$$

The tradeoff is explicit. Larger $n$ reduces the bias introduced by bootstrapping — the target depends less on the potentially inaccurate current estimates — at the cost of higher variance from summing more observed rewards. Early in training, when $V$ is far from $V^\pi$, larger $n$ is often better: the bootstrapped value introduces substantial bias, and longer observed returns are more informative. Late in training, when $V \approx V^\pi$, smaller $n$ is often better: the bias is small, and shorter returns have lower variance and propagate information faster.

## TD($\lambda$) and Eligibility Traces

Rather than committing to a single $n$, **TD($\lambda$)** averages across all $n$-step returns with geometrically decaying weights parameterized by $\lambda \in [0, 1]$:

$$G_t^\lambda = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}.$$

The factor $(1-\lambda)$ normalizes the weights to sum to one. When $\lambda = 0$, only the one-step return receives weight: TD($\lambda$) reduces to TD(0). When $\lambda = 1$, the geometric series assigns all weight to the infinite-horizon return and TD(1) is equivalent to Monte Carlo. For $0 < \lambda < 1$, the $\lambda$-return is dominated by short returns when $\lambda$ is small, and by longer returns when $\lambda$ is large — a smooth interpolation that avoids the binary choice between the two extremes.

The **forward view** of TD($\lambda$) updates using $G_t^\lambda$ as the target, but requires knowing all future rewards before computing the weighted sum — not online. The **backward view** provides an equivalent online algorithm using **eligibility traces**: a per-state memory vector $\mathbf{z}_t \in \mathbb{R}^{|\mathcal{S}|}$ that accumulates recent visits, decaying geometrically:

$$z_t(s) = \begin{cases} \gamma\lambda\, z_{t-1}(s) + 1 & \text{if } s = S_t, \\ \gamma\lambda\, z_{t-1}(s) & \text{otherwise.} \end{cases}$$

At each step, the TD error $\delta_t$ is propagated backward through all recently visited states in proportion to their eligibility:

$$V(s) \leftarrow V(s) + \alpha\,\delta_t\,z_t(s), \quad \forall s.$$

States visited recently and frequently have high eligibility and receive large updates; states visited long ago have low eligibility and receive small ones. The trace vector encodes a decaying memory of which states were responsible for the current TD error — hence the name. Sutton's original paper proved that the forward and backward views are equivalent in expectation: they produce the same expected parameter updates, step by step, even though they compute them in opposite temporal directions.

Eligibility traces are the first mechanism in this progression that explicitly connects past states to present errors. This is relevant beyond the algorithmic efficiency argument: in neuroscience, the TD error has been identified with dopaminergic prediction error signals, and eligibility traces correspond to synaptic eligibility — the biological mechanism by which neurons that fired recently remain eligible for Hebbian potentiation. The mathematical structure and the biological mechanism converge on the same solution to the credit assignment problem.

## From Prediction to Control: SARSA

TD(0) solves the prediction problem. For the control problem — finding the optimal policy — we need to not only evaluate but also improve. The complication is that $V^\pi$ alone is not sufficient for improvement without the model: selecting the greedy action requires $\arg\max_a \sum_{s'} p(s' \mid s, a)[r(s,a) + \gamma V^\pi(s')]$, which requires knowing $p$.

The model-free resolution is to estimate **action-value functions** $Q^\pi(s, a)$ directly. With $Q^\pi$ in hand, policy improvement requires no model: $\pi'(s) = \arg\max_a Q^\pi(s, a)$. Applying TD prediction to $Q$ functions yields the control algorithm.

**SARSA** — named for the quintuple $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ used in each update — is the on-policy version:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\bigl[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\bigr].$$

The target $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$ uses the action $A_{t+1}$ that was actually selected by the current policy. This makes SARSA **on-policy**: the Q-function estimate is consistent with the policy generating the experience, including its exploratory deviations. Interleaving SARSA updates with $\varepsilon$-greedy action selection — act greedily with probability $1 - \varepsilon$ and uniformly at random with probability $\varepsilon$ — produces an online generalized policy iteration that converges to the optimal policy as $\varepsilon \to 0$.

The exploration parameter $\varepsilon$ resolves the **exploration-exploitation tension**: an agent that always acts greedily can trap itself in a locally good policy, never visiting states where the current estimates are wrong. Occasional random actions ensure all relevant regions of state-action space are eventually visited. This is the simplest instantiation of a problem that pervades reinforcement learning and does not have a clean general solution.

## Q-Learning: Off-Policy Control

Sutton's 1988 paper established TD prediction; Watkins' 1989 PhD thesis introduced the model-free control algorithm that would ultimately prove most influential. The modification is deceptively small: replace the action-value of the next selected action in the SARSA target with the maximum action-value over all next actions.

The **Q-learning** update is:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\bigl[R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)\bigr].$$

The target $R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')$ is a sample estimate of the right-hand side of the Bellman optimality equation:

$$Q^*(s, a) = \sum_{s'} p(s' \mid s, a)\bigl[r(s, a) + \gamma \max_{a'} Q^*(s', a')\bigr].$$

Q-learning is therefore stochastic approximation of value iteration applied to action-value functions. Because the target uses $\max_{a'}$ — the greedy action, not the action actually taken — Q-learning is **off-policy**: the target policy (greedy with respect to $Q$) differs from the behavior policy generating experience (e.g., $\varepsilon$-greedy). The behavior policy needs only to ensure sufficient exploration; the target policy is always implicitly greedy.

This off-policy character is both Q-learning's strength and, eventually, its source of instability. The strength: experience from any exploratory policy can be used to learn about the greedy policy, making Q-learning flexible about data collection. The behavior policy could be $\varepsilon$-greedy, uniformly random, or even replayed from a buffer of old transitions. The instability: the asymmetry between the behavior policy generating bootstrapped targets and the target policy being evaluated can, with function approximation, cause the Q-estimates to diverge — a problem the next chapter will confront directly.

## SARSA vs. Q-Learning: A Concrete Distinction

The difference between SARSA and Q-learning is not merely technical. In environments where exploration is costly or dangerous, it produces qualitatively different behavior.

The standard example is a grid world with a cliff along one edge. The globally optimal path runs close to the cliff — direct and fast. A conservative path runs far from it. SARSA, which evaluates the $\varepsilon$-greedy policy including its random deviations, learns the conservative path: with small probability, a random step near the cliff leads to falling, and this risk is incorporated into the Q-values. Q-learning learns the optimal path: its target is always the greedy policy, which never deliberately steps toward the cliff, so the Q-values reflect the optimal cost rather than the $\varepsilon$-greedy cost.

At the limit $\varepsilon \to 0$, both converge to the same optimal policy. But at any finite $\varepsilon$ — which is always the case during learning — Q-learning is more aggressive than SARSA. This is not always the right tradeoff, particularly in systems with real-world consequences for exploration failures. The choice between on-policy and off-policy estimation is one of the first genuinely practical design decisions in reinforcement learning.

## The Frontier of the Tabular Era

By the mid-1990s, the tabular TD landscape was fully mapped. TD(0) converged to $V^\pi$. SARSA converged to $Q^*$ with an appropriately decaying $\varepsilon$. Q-learning converged to $Q^*$ regardless of the behavior policy, provided all state-action pairs were visited infinitely often. These convergence results were clean and complete. Given enough experience, all three algorithms produced the correct answer — the same answer as the exact DP algorithms of Chapter 02, without ever requiring the transition kernel.

The remaining limitation was the same one that constrained dynamic programming: the state space must be small enough to represent explicitly. Every state-action pair needs an entry in the Q-table; every convergence proof relies on visiting every entry infinitely often. For the problems that motivated reinforcement learning — controlling physical systems, playing complex games, making decisions in language — the state space is too large to enumerate, and a lookup table cannot generalize across unseen states.

Applying Q-learning to a game of Atari with raw pixel inputs, for instance, requires a Q-value for every possible pixel configuration — a table with $256^{33600}$ entries, of which any single game will visit an infinitesimal fraction. The Q-table cannot be stored; even if it could, no individual state would be visited enough times for a reliable estimate.

The solution was proposed by Mnih et al. in 2013: replace the Q-table with a neural network that takes a state as input and outputs Q-values for each action. The resulting algorithm — the **Deep Q-Network** — achieved superhuman performance on dozens of Atari games. But the convergence guarantees of tabular Q-learning did not carry over automatically, and new engineering innovations were required to make neural Q-learning stable. Those innovations, and the reasons they were necessary, are the subject of the next chapter.

---

*Temporal difference learning was introduced in Sutton (1988), Learning to Predict by the Methods of Temporal Differences. Q-learning was developed by Watkins in his 1989 PhD thesis and published as Watkins and Dayan (1992), Q-learning. Rigorous convergence proofs for TD(0) using the ODE method are in Tsitsiklis (1994), Asynchronous Stochastic Approximation and Q-Learning. SARSA was described by Rummery and Niranjan (1994) and named by Sutton (1996). TD($\lambda$) and the forward-backward equivalence of eligibility traces are developed in Sutton and Barto (2018), chapters 6, 7, and 12. The connection between TD errors and dopaminergic prediction error signals was established by Schultz, Dayan, and Montague (1997), A Neural Substrate of Prediction and Reward.*
