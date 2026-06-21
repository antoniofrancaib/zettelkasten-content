---
title: "The Policy Gradient Theorem"
subtitle: "Parameterizing policies directly, the log-derivative trick, REINFORCE, baseline subtraction, and the advantage function"
---


The deep Q-network resolved, by engineering rather than theory, the instability that the deadly triad predicted. But it resolved it only for a constrained class of problems. Q-learning operates by computing $\arg\max_a Q(s, a; \theta)$ at every step — an operation that requires enumerating all actions. In Atari, the action space has 18 elements, enumerable in microseconds. For a robot arm that must choose a continuous joint torque from an infinite set, or a language model that must select the next token from a vocabulary of $10^5$, the operation is either intractable or ill-defined.

The alternative, developed through the 1990s in a line of work culminating in Sutton, McAllester, Singh, and Mansour (2000), is to abandon the detour through value functions entirely. Instead of learning a value function and deriving a policy from it, parameterize the policy directly and optimize its parameters by following the gradient of the expected return. This is the **policy gradient** approach, and it operates in a fundamentally different geometry: where Q-learning searches over value functions in a space of lookup tables or neural networks, policy gradients search directly over policies in a parameter space, using gradient ascent on the performance objective itself.

---

## Parameterizing Policies Directly

> [!question]
> What does it mean to parameterize a policy, and what properties must the parameterization have for gradient-based optimization to be possible?

A **parameterized policy** $\pi_\theta$ is any differentiable mapping from states and actions to probabilities (or, in the continuous case, to the parameters of a probability distribution over actions). The two canonical forms are:

For **discrete action spaces**, the softmax policy:

$$\pi_\theta(a \mid s) = \frac{\exp\bigl(h(s, a; \theta)\bigr)}{\sum_{a'} \exp\bigl(h(s, a'; \theta)\bigr)},$$

where $h(s, a; \theta)$ is a scalar preference computed by a neural network. Softmax policies are everywhere-positive — they assign nonzero probability to every action at every state — which automatically satisfies the GLIE exploration condition without any explicit $\varepsilon$-greedy schedule. Exploration is built into the structure rather than bolted on.

For **continuous action spaces**, the Gaussian policy:

$$\pi_\theta(a \mid s) = \mathcal{N}\!\bigl(\mu_\theta(s),\, \sigma_\theta(s)^2\bigr),$$

where $\mu_\theta(s)$ and $\sigma_\theta(s)$ are neural network outputs. Continuous control — robotic locomotion, manipulation, molecular design — is handled naturally, with no discretization required.

> [!tip]
> What both forms share is **differentiability**: $\partial \log \pi_\theta(a \mid s) / \partial \theta$ exists and can be computed by backpropagation. This quantity — the score function — is exactly what the policy gradient theorem will require. Everything else is commentary.

---

## The Performance Objective

> [!info]
> The **performance objective** $J(\theta)$ is the expected return from the start state under policy $\pi_\theta$:
> $$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \gamma^t R_{t+1}\right] = \mathbb{E}_{S_0 \sim \mu_0}\bigl[V^{\pi_\theta}(S_0)\bigr],$$
> where $\tau = (S_0, A_0, R_1, S_1, \ldots)$ is a trajectory sampled by running $\pi_\theta$, and $\mu_0$ is the initial state distribution. The goal is $\theta^* = \arg\max_\theta J(\theta)$, pursued by gradient ascent: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$.

The challenge is computing $\nabla_\theta J(\theta)$. The expected return depends on $\theta$ in two ways: through the policy distribution (which actions are chosen) and through the state visitation distribution (which states are visited, since that depends on which actions were taken). The first dependence is accessible — $\pi_\theta$ is differentiable. The second is not — the transition kernel $p(s' \mid s, a)$ is part of the environment and does not depend on $\theta$.

A naive attempt at differentiation runs into this:

$$\nabla_\theta J(\theta) = \nabla_\theta \int_\tau p(\tau; \theta) G(\tau) \, d\tau,$$

where $p(\tau; \theta) = \mu_0(S_0) \prod_t \pi_\theta(A_t \mid S_t) p(S_{t+1} \mid S_t, A_t)$ is the trajectory probability. Differentiating $p(\tau; \theta)$ appears to require differentiating $p(S_{t+1} \mid S_t, A_t)$ — the unknown dynamics. The policy gradient theorem shows how to avoid this entirely.

---

## The Log-Derivative Trick

> [!tip]
> The **log-derivative trick** (also called the score function estimator or REINFORCE trick) rewrites the gradient of an expectation as an expectation of a gradient:
> $$\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[f(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\!\bigl[f(\tau) \nabla_\theta \log p(\tau; \theta)\bigr].$$
>
> The proof is a single line. Since $\nabla_\theta p(\tau; \theta) = p(\tau; \theta) \nabla_\theta \log p(\tau; \theta)$ by the chain rule:
> $$\nabla_\theta \int_\tau p(\tau; \theta) f(\tau) \, d\tau = \int_\tau p(\tau; \theta) \nabla_\theta \log p(\tau; \theta) f(\tau) \, d\tau.$$

Now the environmental dynamics drop out cleanly:

$$\nabla_\theta \log p(\tau; \theta) = \underbrace{\nabla_\theta \log \mu_0(S_0)}_{= 0} + \sum_t \nabla_\theta \log \pi_\theta(A_t \mid S_t) + \underbrace{\sum_t \nabla_\theta \log p(S_{t+1} \mid S_t, A_t)}_{= 0}.$$

The first and third terms vanish — neither the initial distribution nor the transition kernel depends on $\theta$. What remains is $\sum_t \nabla_\theta \log \pi_\theta(A_t \mid S_t)$: a sum of score functions, each computable from the neural network alone, with no knowledge of the environment dynamics required.

---

## The Policy Gradient Theorem

> [!success]
> **Theorem** (Sutton, McAllester, Singh, and Mansour, 2000). For any differentiable policy $\pi_\theta$ and discount factor $\gamma \in [0, 1)$:
> $$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(A_t \mid S_t) \cdot Q^{\pi_\theta}(S_t, A_t)\right],$$
> where $Q^{\pi_\theta}(s, a) = \mathbb{E}_{\pi_\theta}\!\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s, A_t = a\right]$.
>
> The gradient of the performance objective — a function of the state visitation distribution, which itself depends on the unknown dynamics — equals an expectation that involves only the policy gradient and the action-value function. **No model is required.**

> [!tip]
> The intuition behind the theorem: $\nabla_\theta \log \pi_\theta(A_t \mid S_t)$ is the direction in parameter space that most rapidly increases the log-probability of the action actually taken. Multiplying by $Q^{\pi_\theta}(S_t, A_t)$ weights the update: actions that led to high long-run return are reinforced; actions that led to low return are suppressed. The Bellman equations are still present, but only through the action-value function, averaged over under the trajectory distribution.

---

## REINFORCE

The policy gradient theorem immediately suggests an algorithm. If the exact $Q^{\pi_\theta}$ is unknown, replace it with a Monte Carlo estimate: run an episode to completion, collect the return $G_t = \sum_{k=0}^{T-t} \gamma^k R_{t+k+1}$, and use it as an unbiased estimate of $Q^{\pi_\theta}(S_t, A_t)$:

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(A_t \mid S_t) \cdot G_t.$$

> [!info]
> This is **REINFORCE** (Williams, 1992). It is unbiased — $G_t$ is an unbiased estimate of $Q^{\pi_\theta}(S_t, A_t)$ — and requires no model of the environment. The only inputs are the trajectory of states, actions, and rewards collected by running the current policy.
>
> The update points in the direction that most increases log-probability of high-return trajectories. Actions that contributed to positive returns have their probabilities nudged upward; actions that contributed to negative returns have theirs nudged downward.

> [!note]
> **REINFORCE is essentially a weighted form of supervised learning.** Each action taken is treated as a training label; the return $G_t$ scales the gradient update — large positive returns produce large reinforcing updates, near-zero returns produce near-zero updates. This connection becomes explicit when we consider REINFORCE applied to language model post-training: the log-probability gradient $\nabla_\theta \log \pi_\theta(A_t \mid S_t)$ is identical in form to the supervised cross-entropy gradient, with the return playing the role of a per-token weight.

---

## The Variance Problem

> [!warning]
> **REINFORCE's decisive practical weakness: high variance.**
>
> The Monte Carlo return $G_t$ is a sum of many future rewards, each multiplied by a discount factor, each random due to both environment stochasticity and policy stochasticity. Variance grows with the horizon. In a task with $T = 1000$ steps and moderate reward noise, the variance of $G_t$ can easily exceed $10^3$–$10^4$, even when the expected return is modest.
>
> High-variance gradient estimates mean the gradient signal is dominated by noise: the agent repeatedly moves its parameters in directions that reflect the randomness of a single trajectory rather than the systematic structure of $J(\theta)$. Convergence — guaranteed in theory — can require an impractical number of episodes. **Variance reduction is not an optional refinement; it is the core engineering challenge of policy gradient methods.**

---

## Baseline Subtraction

> [!question]
> Can we reduce the variance of the REINFORCE gradient estimator without introducing bias?

The key observation: for any function $b(s)$ that depends only on the state (not on the action taken), the following identity holds:

$$\mathbb{E}_{\pi_\theta}\!\bigl[\nabla_\theta \log \pi_\theta(A_t \mid S_t) \cdot b(S_t)\bigr] = 0.$$

> [!note]
> **Proof.** Conditioning on $S_t$:
> $$\mathbb{E}_{A_t \sim \pi_\theta(\cdot \mid S_t)}\!\bigl[\nabla_\theta \log \pi_\theta(A_t \mid S_t)\bigr] = \nabla_\theta \sum_a \pi_\theta(a \mid S_t) = \nabla_\theta 1 = 0.$$
> Subtracting a state-dependent baseline does not change the expected gradient — but it can dramatically reduce its variance.

Replacing $G_t$ with $(G_t - b(S_t))$ gives an unbiased estimator with potentially much lower variance:

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(A_t \mid S_t) \cdot \bigl(G_t - b(S_t)\bigr).$$

> [!tip]
> If $b$ is a good predictor of the return from $S_t$, the centered quantity $(G_t - b(S_t))$ has much lower variance than $G_t$ alone. Returns above the baseline represent actions better than expected and are reinforced; returns below represent actions worse than expected and are suppressed. The baseline **centers the signal**, removing variance due to state-value variation rather than action-quality variation.
>
> The optimal baseline (minimizing variance) is a weighted average of returns with weights given by the squared norm of the score function. In practice, the standard choice is $b(s) = V^{\pi_\theta}(s)$, the state-value function of the current policy.

---

## The Advantage Function

> [!info]
> When $b(s) = V^{\pi_\theta}(s)$, the centered quantity is the **advantage function**:
> $$A^{\pi_\theta}(s, a) = Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s).$$
> It measures how much better action $a$ is than the average action in state $s$ under the current policy. The policy gradient becomes:
> $$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(A_t \mid S_t) \cdot A^{\pi_\theta}(S_t, A_t)\right].$$

> [!tip]
> The advantage function is not just a variance reduction trick — it is the **right quantity to use**. The action-value $Q^{\pi_\theta}(s, a)$ measures total expected return, which includes a state-dependent component $V^{\pi_\theta}(s)$ that has nothing to do with the action taken. Using $Q^{\pi_\theta}$ directly to weight updates attributes to the action credit that actually belongs to the state. The advantage strips that component away, leaving only the action's contribution *above and beyond* what the agent would have received on average.
>
> Note that $\mathbb{E}_{\pi_\theta}[A^{\pi_\theta}(S_t, A_t)] = 0$ for every state by construction — the advantage is centered. The gradient signal pushes probabilities of above-average actions up and below-average actions down, exactly as intended.

---

## The Geometry of Policy Space

> [!question]
> What is the policy gradient update actually doing in the space of policies?

The policy $\pi_\theta$ lives in a manifold of probability distributions — one distribution over actions for each state — parameterized by $\theta$. The performance objective $J(\theta)$ is a scalar function on this manifold.

The score function $\nabla_\theta \log \pi_\theta(A_t \mid S_t)$ is the direction in parameter space that most rapidly increases the log-probability of $A_t$ at $S_t$ — the local tangent direction corresponding to "take this action more often." The policy gradient theorem says to follow the combination of these directions, weighted by the advantage: increase the probability of advantageous actions, decrease the probability of disadvantageous ones, proportionally to how strongly the parameters can change those probabilities.

> [!note]
> This is gradient ascent in parameter space, but it corresponds to a meaningful operation in policy space: the update moves $\pi_\theta$ toward the policy that concentrates more mass on better actions. The step size $\alpha$ controls how far to move. The policy gradient theorem guarantees that small enough steps are never harmful — the gradient always points uphill in $J$.
>
> **What REINFORCE does not guarantee**: convergence to a *global* optimum. $J(\theta)$ is non-convex in $\theta$ for neural network parameterizations — multiple local maxima, saddle points, and flat regions exist. Gradient ascent finds a local maximum. In practice, this is often acceptable in high-dimensional overparameterized networks, but there is no theorem guaranteeing it.

---

## The Bridge to Actor-Critic

REINFORCE uses the Monte Carlo return $G_t$ to estimate $Q^{\pi_\theta}(S_t, A_t)$ — unbiased but high-variance. An alternative: use a second neural network — a **critic** — to estimate the value function directly, and use the critic's output in place of the Monte Carlo return.

> [!tip]
> **Actor-critic architecture**: the actor is $\pi_\theta$, updated by policy gradients; the critic is $\hat{V}_w(s)$, updated by TD learning to track $V^{\pi_\theta}$. The advantage estimate becomes:
> $$A^{\pi_\theta}(S_t, A_t) \approx R_{t+1} + \gamma \hat{V}_w(S_{t+1}) - \hat{V}_w(S_t),$$
> the TD error. This requires neither a full episode nor a lookup table: it can be computed after every single step, enabling online learning. The variance is far lower than the Monte Carlo return, at the cost of a small bias introduced by the critic's approximation error.
>
> The division of labor is exact: the actor solves the control problem (better actions via policy gradients); the critic solves the prediction problem (evaluating the current policy via TD learning). This is Howard's policy evaluation and policy improvement, re-expressed in stochastic gradient methods and function approximation, operating continuously without model access.

The bias-variance tradeoff between Monte Carlo returns and TD-bootstrapped returns in the advantage estimator is the central design decision of the next generation of policy gradient methods. The $n$-step tradeoff from Chapter 03 reappears here: GAE (generalized advantage estimation) applies the same $\lambda$-weighted averaging to advantage estimates that TD($\lambda$) applied to value estimates, providing a tunable dial between high-variance unbiased Monte Carlo advantages and low-variance biased one-step TD advantages.

---

*The policy gradient theorem in its general form is proved in Sutton, McAllester, Singh, and Mansour (2000), Policy Gradient Methods for Reinforcement Learning with Function Approximation. The REINFORCE algorithm and the score function estimator are from Williams (1992), Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. The baseline subtraction result and the zero-mean property of the centered gradient estimator appear in Williams (1992) and are elaborated in Weaver and Tao (2001), The Optimal Reward Baseline for Gradient-Based Reinforcement Learning. The advantage function and its role in variance reduction are analyzed in Greensmith, Bartlett, and Baxter (2004), Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning.*
