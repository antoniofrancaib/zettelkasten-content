---
title: "Actor-Critic and RLOO"
subtitle: "Bootstrapped advantage estimation, GAE, the two-network architecture, and critic-free leave-one-out baselines"
---


The policy gradient theorem established the correct form of the gradient. REINFORCE provided an unbiased estimator of it. But unbiasedness does not mean usability. The Monte Carlo return $G_t = \sum_{k=0}^{T-t} \gamma^k R_{t+k+1}$ is a sum of many random future rewards, and its variance scales with the horizon. On tasks where rewards are sparse or delayed — which is most tasks of interest — REINFORCE updates are dominated by noise. An agent learning to navigate a maze may spend thousands of episodes receiving zero reward at every step. The Monte Carlo return for every step in those empty episodes is zero, which provides no learning signal at all; the first successful trajectory produces a gradient so large and temporally correlated that it can destabilize the policy as easily as improve it.

There is a more fundamental objection. REINFORCE requires episodes to terminate before any update can be computed: $G_t$ is defined only with access to $R_T$, the final-step reward. For continuing tasks — robotic control, dialogue, any problem that runs indefinitely — the Monte Carlo return is undefined. The algorithm cannot be applied.

The resolution to both problems is **bootstrapping**: replace the tail of the Monte Carlo sum with a learned estimate of the value function. Instead of waiting to observe the true return from $S_t$, use a neural network $\hat{V}_w(S_t)$ as a proxy for future rewards. This introduces bias — the estimate is wrong, especially early in training — but controls variance: the estimate has only the noise of a single network evaluation, not the accumulated noise of hundreds of future rewards. The actor-critic framework is, at its core, a family of methods for navigating this tradeoff.

---

## The TD Error as Advantage Estimate

> [!question]
> How precisely does the one-step TD error relate to the advantage function, and when is it a good substitute?

The true advantage is $A^{\pi}(S_t, A_t) = Q^{\pi}(S_t, A_t) - V^{\pi}(S_t)$. By definition of the Q-function:

$$Q^{\pi}(S_t, A_t) = \mathbb{E}\bigl[R_{t+1} + \gamma V^{\pi}(S_{t+1}) \mid S_t, A_t\bigr].$$

Substituting:

$$A^{\pi}(S_t, A_t) = \mathbb{E}\bigl[R_{t+1} + \gamma V^{\pi}(S_{t+1}) - V^{\pi}(S_t) \mid S_t, A_t\bigr] = \mathbb{E}[\delta_t^{\pi} \mid S_t, A_t],$$

where $\delta_t^{\pi} = R_{t+1} + \gamma V^{\pi}(S_{t+1}) - V^{\pi}(S_t)$ is the TD error computed with the **true** value function.

> [!tip]
> If we had $V^{\pi}$ exactly, each realized TD error would be an **unbiased estimate of the advantage** — and a much lower-variance one than the Monte Carlo return, since it depends on only a single random reward rather than a sum of many. In practice we have $\hat{V}_w \approx V^{\pi}$, so the realized TD error is a *biased but low-variance* estimate. The bias enters through two channels: approximation error $\hat{V}_w(s) - V^{\pi}(s)$, and the replacement of the expectation by a single sample. Both diminish as the critic improves.

The actor and critic updates after a single transition $(S_t, A_t, R_{t+1}, S_{t+1})$:

$$\theta \leftarrow \theta + \alpha \, \delta_t \, \nabla_\theta \log \pi_\theta(A_t \mid S_t), \qquad w \leftarrow w + \alpha_w \, \delta_t \, \nabla_w \hat{V}_w(S_t).$$

Both can be computed after a single step, without waiting for an episode to end. The actor learns to select better actions; the critic learns to predict cumulative reward; together they bootstrap each other toward a stable policy.

---

## The $n$-Step Spectrum

One-step bootstrapping is an extreme. At the opposite extreme sits the full Monte Carlo return — no bootstrapping at all. Between them lies a family of **$n$-step returns**:

$$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n \hat{V}_w(S_{t+n}).$$

For $n=1$: the one-step TD target. For $n \to \infty$: the Monte Carlo return. Every intermediate $n$ represents a different point on the bias-variance frontier.

> [!note]
> Larger $n$ reduces bias — more actual rewards replace the approximated future — at the cost of higher variance from summing more random rewards. Smaller $n$ reduces variance at the cost of more bias from the value estimate. The one-step return has minimum variance, maximum bias; the Monte Carlo return has zero bias, maximum variance. In practice, $n$ between 5 and 20 is common.

---

## Generalized Advantage Estimation (GAE)

> [!question]
> Choosing a fixed $n$ is arbitrary — different parts of the state space may benefit from different horizons. Is there a principled way to avoid this commitment?

Schulman, Moritz, Levine, Jordan, and Abbeel (2016) observed that choosing a fixed $n$ is itself an arbitrary choice and proposed a weighted average over all horizons simultaneously: the **generalized advantage estimator** GAE($\lambda$).

Define the sequence of one-step TD errors: $\delta_t = R_{t+1} + \gamma \hat{V}_w(S_{t+1}) - \hat{V}_w(S_t)$.

The GAE estimate is a geometrically weighted sum:

$$\hat{A}_t^{\text{GAE}(\lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \, \delta_{t+l} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \hat{A}_t^{(n)}.$$

> [!tip]
> **The $\lambda$ dial:**
>
> | $\lambda$ | Behavior |
> |-----------|----------|
> | $\lambda = 0$ | $\hat{A}_t = \delta_t$ — one-step TD error, minimum variance, maximum critic bias |
> | $\lambda = 1$ | $\hat{A}_t = G_t - \hat{V}_w(S_t)$ — full Monte Carlo advantage, zero bias, maximum variance |
> | $\lambda \in (0,1)$ | Geometrically decays long-horizon errors, giving more weight to shorter, lower-variance estimates while retaining some unbiasedness |
>
> **Practical computation** — a single backward pass over the trajectory in $O(T)$:
> $$\hat{A}_T = \delta_T, \qquad \hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1}.$$

> [!success]
> $\lambda = 0.95$ with $\gamma = 0.99$ (effective horizon $\approx 17$ steps) has proven remarkably robust across continuous locomotion, dexterous manipulation, and multi-task domains. It became the near-universal default adopted by PPO, A2C, and most subsequent actor-critic methods. The effective decay parameter $\gamma\lambda = 0.9405$ achieves what the $n$-step return achieves with a hard cutoff, but softly and without committing to a single $n$.

---

## The Two-Network Architecture

> [!note]
> The actor-critic can be implemented with two fully separate networks — one for $\pi_\theta$ and one for $\hat{V}_w$ — or, more commonly, with a **shared backbone** $f_\phi$ and two output heads:
> - **Policy head**: $\pi_\theta(a \mid s) = \pi(f_\phi(s))$
> - **Value head**: $\hat{V}_w(s) = V(f_\phi(s))$
>
> The total loss combines actor, critic, and entropy terms:
> $$\mathcal{L} = \underbrace{-\mathbb{E}\bigl[\hat{A}_t \log \pi_\theta(A_t \mid S_t)\bigr]}_{\text{actor}} + \underbrace{c_1 \|\hat{V}_w(S_t) - G_t^{\text{target}}\|^2}_{\text{critic}} - \underbrace{c_2 H(\pi_\theta(\cdot \mid S_t))}_{\text{entropy bonus}}.$$

Sharing lower layers is not cosmetic — the low-level features useful for predicting value (object positions, velocities, environment structure) are the same features useful for selecting actions. The critic's dense gradient signal (from every step) improves the shared feature extractor used by the actor; the policy gradient shapes the representations used by the critic.

The **entropy bonus** $H(\pi_\theta) = -\sum_a \pi_\theta(a \mid s) \log \pi_\theta(a \mid s)$ penalizes premature determinism: without it, a policy gradient method will collapse to a deterministic policy once it identifies actions that reliably produce positive advantages, stopping exploration before discovering better strategies. The entropy coefficient $c_2 \approx 0.01$ is small but critical early in training, when critic estimates are inaccurate and the apparent best action may not be the true best.

---

## A3C and A2C: Scaling Actor-Critic with Parallelism

> [!note]
> **A3C** (Mnih et al., 2016) addressed the correlation problem that DQN solved with experience replay, without needing a replay buffer. Rather than mixing past transitions, it runs many workers **in parallel**, each maintaining its own environment copy, collecting short $n$-step trajectories asynchronously, computing gradients locally, and pushing them to a central parameter server. Workers in different environment states produce decorrelated gradients — achieving the diversity that replay achieves through historical mixing, without off-policy corrections.
>
> **A2C** (OpenAI, 2017) removes the asynchrony: $N$ workers step in lockstep, all gradients come from the same parameter version, and a single synchronized update is applied. Equivalent performance to A3C, simpler to reason about, and efficiently batched on GPU: $N$ environments stepped simultaneously produce a single batch for one forward-backward pass.

> [!success]
> Before A3C, policy gradient methods had not demonstrated competitive performance on Atari, which DQN had owned since 2013. After A3C, the gap closed substantially. A3C's 3D navigation results — solving labyrinthine first-person environments — demonstrated that actor-critic methods could learn effectively where the value function structure was genuinely complex and long-horizon.

---

## The Step-Size Problem

> [!warning]
> What neither A3C nor A2C addressed is the **step-size problem**. Policy gradients can take destructively large steps: a single bad update can collapse the policy to a degenerate solution — all probability mass on one action — from which recovery is slow. The gradient estimator, however good, does not constrain how far the parameters should move. A large step in parameter space can produce catastrophic changes in behavior in a non-convex policy landscape.
>
> Variance is now controlled — bootstrapping, shared architectures, and GAE solve REINFORCE's original pathology. The step-size problem remains. Chapter 07 addresses it using the geometry of policy space itself: the Fisher information metric and KL divergence constrain how much the policy is allowed to change per update, guaranteeing that each gradient step is a genuine improvement.

---

## RLOO: A Critic-Free Baseline from Multiple Rollouts

> [!question]
> The actor-critic architecture requires training and maintaining a learned value network alongside the policy. If instead we can afford to sample multiple responses to the same input, can we construct a good advantage estimate without a critic at all?

The baseline subtraction identity from Chapter 05 allows any state-dependent function $b(s)$ to be subtracted without biasing the gradient. The standard choice is the learned critic $\hat{V}_w(s)$. But there is another route: estimate $V^\pi(s)$ empirically, by sampling multiple actions from the current policy at the same state and averaging their returns.

**REINFORCE Leave-One-Out (RLOO)** formalizes this idea. For each prompt $x$, sample $G$ independent responses $y_1, \ldots, y_G$ from the current policy, with rewards $r_1, \ldots, r_G$. The leave-one-out advantage for response $i$ is:

$$\hat{A}_i = r_i - \frac{1}{G-1} \sum_{j \neq i} r_j.$$

> [!tip]
> **Why leave-one-out is unbiased.** The baseline $\frac{1}{G-1}\sum_{j \neq i} r_j$ does not depend on $y_i$ — it is computed from all other responses. Since the baseline is independent of the action being evaluated, the zero-mean property from Chapter 05 applies: $\mathbb{E}_{y_i \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(y_i \mid x) \cdot b] = 0$. The advantage estimate is therefore an unbiased estimate of $A^\pi(x, y_i) = Q^\pi(x, y_i) - V^\pi(x)$, for exactly the same reason that the REINFORCE baseline is unbiased.
>
> As $G \to \infty$, the leave-one-out mean converges to $V^\pi(x)$ — the true value — and the advantage estimate converges to the true advantage. The variance of the baseline is $O(1/(G-1))$, decreasing with group size. The bias of the baseline relative to a perfectly trained critic is zero in expectation.

> [!note]
> **Relationship to group baselines.** The leave-one-out estimator averages $G-1$ rewards; the group mean (used in GRPO, Chapter 12) averages all $G$ including the current response:
> $$\bar{r} = \frac{1}{G}\sum_{j=1}^G r_j.$$
> The difference is small when $G$ is large (the current response has weight $1/G \to 0$) but non-negligible for small $G$. RLOO's leave-one-out structure strictly preserves the unbiasedness property; the group mean introduces a small self-correlation that the leave-one-out baseline avoids by design. In practice, $G = 4$–$16$ makes the difference modest.

> [!success]
> **What RLOO found about PPO-style clipping.** When RLOO was applied to language model post-training (Kool, Haffner, and Wierstra, 2019; Ahmadian et al., 2024), researchers found that PPO's importance-ratio clipping was active in fewer than 5% of update steps — suggesting that, when the policy is updated smoothly and rollouts are on-policy, the trust-region machinery is largely redundant. RLOO drops clipping entirely, returning to pure REINFORCE-style updates weighted by leave-one-out advantages. The simplification loses nothing measurable relative to PPO on the tasks studied, while eliminating the memory cost of the importance-ratio computation and the complexity of clipping schedules.

> [!warning]
> **RLOO's binding constraint.** Like all group-baseline methods, RLOO requires generating $G$ complete responses per prompt before any gradient update can be computed. The rollout compute scales linearly with $G$. For tasks where generating responses is expensive — long chain-of-thought reasoning, multimodal generation, physical robot interaction — the $G$-fold cost may be prohibitive. The learned critic, by contrast, adds a memory cost but not a rollout cost: it produces its estimate from a single forward pass per state, without requiring additional environment interaction.
>
> The practical recommendation: RLOO is best suited to settings where rollout is cheap and the reward is binary or low-noise — exactly the verifiable reward settings (mathematics, code, formal verification) where GRPO also excels. For tasks with dense, continuous, expensively-computed rewards, the learned critic remains the right tool.

---

## The Bias-Variance Landscape: A Summary

> [!tip]
> The actor-critic family of methods can be understood as a single design space parameterized by two choices:
>
> 1. **How to estimate the baseline**: learned critic $\hat{V}_w$ (trained online) vs. sample-based estimate (leave-one-out, group mean) vs. no baseline (pure REINFORCE)
> 2. **How many steps before bootstrapping**: $\lambda$ in GAE, or equivalently $n$ in $n$-step returns — from one-step TD ($\lambda = 0$) to full Monte Carlo ($\lambda = 1$)
>
> These two choices are largely orthogonal. RLOO occupies the "sample-based baseline, full Monte Carlo" corner: it uses observed returns with no bootstrapping and derives its baseline from peer rollouts rather than a trained critic. Standard actor-critic with GAE occupies the "learned critic, intermediate $\lambda$" region. Both are valid; the optimal choice depends on rollout cost, reward noise, and episode length.

---

*The actor-critic architecture in its early form appears in Barto, Sutton, and Anderson (1983), Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems. The formal policy gradient framework is in Sutton, McAllester, Singh, and Mansour (2000). A3C is introduced in Mnih, Puigdomènech Badia, Mirza, Graves, Lillicrap, Harley, Silver, and Kavukcuoglu (2016), Asynchronous Methods for Deep Reinforcement Learning. Generalized Advantage Estimation is from Schulman, Moritz, Levine, Jordan, and Abbeel (2016), High-Dimensional Continuous Control Using Generalized Advantage Estimation. The RLOO estimator in the RL context originates in Kool, van Hoof, and Welling (2019), Buy 4 REINFORCE Samples, Get a Baseline for Free! Its application to language model post-training and the finding that clipping is rarely active are in Ahmadian, Cremer, Gallé, Kreutzer, Luketina, Neue, Rokos, Shaibanov, Sherborn, Yiu, and Ulmer (2024), Back to Basics: Revisiting REINFORCE-Style Optimization for Language Models.*
