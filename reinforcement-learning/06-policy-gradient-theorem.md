---

title: "The Policy Gradient Theorem"
subtitle: "REINFORCE, score function estimators, baseline subtraction, and the geometry of policy space"
---


The deep Q-network resolved, by engineering rather than theory, the instability that the deadly triad had predicted. But it resolved it only for a constrained class of problems. Q-learning operates by computing $\arg\max_a Q(s, a; \theta)$ at every step; this operation requires enumerating all actions and selecting the largest. In Atari, the action space has 18 elements — enumerable in microseconds. For a robot arm that must choose a continuous joint torque from an infinite set, or a language model that must select the next token from a vocabulary of $10^5$, the operation is either intractable or ill-defined. The Q-learning architecture has no natural extension to these settings.

The alternative, developed through the 1990s in a line of work culminating in Sutton, McAllester, Singh, and Mansour (2000), is to give up the detour through value functions entirely. Instead of learning a value function and deriving a policy from it, parameterize the policy directly, and optimize the parameters by following the gradient of the expected return. This is the **policy gradient** approach, and it operates in a fundamentally different geometry: where Q-learning searches over value functions in a space of lookup tables or neural networks, policy gradients search directly over policies in a parameter space, using gradient ascent on the performance objective itself.

## Parameterizing Policies Directly

A **parameterized policy** $\pi_\theta$ is any differentiable mapping from states and actions to probabilities (or, in the continuous case, to the parameters of a probability distribution over actions). The simplest and most widely used forms are:

For discrete action spaces, the **softmax policy**:

$$\pi_\theta(a \mid s) = \frac{\exp\bigl(h(s, a; \theta)\bigr)}{\sum_{a'} \exp\bigl(h(s, a'; \theta)\bigr)},$$

where $h(s, a; \theta)$ is a scalar preference for action $a$ in state $s$, computed by a neural network or linear features. Softmax policies are everywhere-positive — they assign nonzero probability to every action at every state — which automatically satisfies the GLIE exploration condition required for convergence, without any explicit $\varepsilon$-greedy schedule. Exploration is built into the structure rather than bolted on.

For continuous action spaces, the **Gaussian policy**:

$$\pi_\theta(a \mid s) = \mathcal{N}\!\bigl(\mu_\theta(s),\, \sigma_\theta(s)^2\bigr),$$

where $\mu_\theta(s)$ and $\sigma_\theta(s)$ are neural network outputs parameterizing the mean and standard deviation of the action distribution. The agent samples an action from this Gaussian, applies it to the environment, and receives a reward signal. Gradient-based learning adjusts $\mu_\theta$ and $\sigma_\theta$ to shift the distribution toward high-reward regions. Continuous control — robotic locomotion, manipulation, flight — is handled naturally, with no discretization required.

What both forms share is the property of **differentiability**: $\partial \log \pi_\theta(a \mid s) / \partial \theta$ exists and can be computed. This is the quantity the policy gradient theorem will require.

## The Performance Objective

The goal is to maximize the expected return from the start state. Define the **performance objective** $J(\theta)$:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \gamma^t R_{t+1}\right] = \mathbb{E}_{S_0 \sim \mu_0}\bigl[V^{\pi_\theta}(S_0)\bigr],$$

where $\tau = (S_0, A_0, R_1, S_1, A_1, R_2, \ldots)$ is a trajectory sampled by running $\pi_\theta$ in the environment, and $\mu_0$ is the initial state distribution. The goal is to find $\theta^* = \arg\max_\theta J(\theta)$ by gradient ascent: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$.

The challenge is computing $\nabla_\theta J(\theta)$. The expected return depends on $\theta$ in two ways: through the policy distribution (which actions are chosen) and through the environment dynamics (which states are visited, since the distribution over states depends on which actions were taken in the past). The first dependence is accessible — $\pi_\theta$ is differentiable. The second is not — the transition kernel $p(s' \mid s, a)$ is part of the environment and does not depend on $\theta$.

A naive attempt at differentiation runs into this immediately:

$$\nabla_\theta J(\theta) = \nabla_\theta \int_\tau p(\tau; \theta) G(\tau) \, d\tau,$$

where $p(\tau; \theta) = \mu_0(S_0) \prod_t \pi_\theta(A_t \mid S_t) p(S_{t+1} \mid S_t, A_t)$ is the probability of trajectory $\tau$ under $\pi_\theta$, and $G(\tau)$ is its return. Moving the gradient inside the integral appears to require differentiating $p(\tau; \theta)$, which involves $p(S_{t+1} \mid S_t, A_t)$ — the unknown dynamics. The policy gradient theorem shows how to avoid this entirely.

## The Log-Derivative Trick

The key identity is the **log-derivative trick** (also called the score function estimator or REINFORCE trick), which rewrites the gradient of an expectation as an expectation of a gradient:

$$\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[f(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\!\bigl[f(\tau) \nabla_\theta \log p(\tau; \theta)\bigr].$$

The proof is a single line. Since $\nabla_\theta p(\tau; \theta) = p(\tau; \theta) \nabla_\theta \log p(\tau; \theta)$ (by the chain rule), we have:

$$\nabla_\theta \int_\tau p(\tau; \theta) f(\tau) \, d\tau = \int_\tau \nabla_\theta p(\tau; \theta) f(\tau) \, d\tau = \int_\tau p(\tau; \theta) \nabla_\theta \log p(\tau; \theta) f(\tau) \, d\tau.$$

Now the environmental dynamics drop out cleanly:

$$\nabla_\theta \log p(\tau; \theta) = \nabla_\theta \log \mu_0(S_0) + \sum_t \nabla_\theta \log \pi_\theta(A_t \mid S_t) + \sum_t \nabla_\theta \log p(S_{t+1} \mid S_t, A_t).$$

The first and third terms are zero — neither the initial distribution nor the transition kernel depends on $\theta$. What remains is $\sum_t \nabla_\theta \log \pi_\theta(A_t \mid S_t)$: a sum of log-policy gradients, each computable from the neural network alone.

## The Policy Gradient Theorem

**Theorem** (Sutton et al., 2000). For any differentiable policy $\pi_\theta$ and any discount factor $\gamma \in [0, 1)$:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(A_t \mid S_t) \cdot Q^{\pi_\theta}(S_t, A_t)\right],$$

where $Q^{\pi_\theta}(s, a) = \mathbb{E}_{\pi_\theta}\!\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s, A_t = a\right]$ is the action-value function of the current policy.

This is a profound result. The gradient of the performance objective — a function of the state visitation distribution, which itself depends on the unknown dynamics — is equal to an expectation that involves only the policy gradient and the action-value function. No model is required.

The intuition: $\nabla_\theta \log \pi_\theta(A_t \mid S_t)$ is the direction in parameter space that increases the probability of the action actually taken. Multiplying by $Q^{\pi_\theta}(S_t, A_t)$ — the long-run value of that action — weights the update: actions that led to high return are reinforced; actions that led to low return are suppressed. The Bellman equations are still present, but only through the value function, which is averaged over under the trajectory distribution.

## REINFORCE

The policy gradient theorem immediately suggests an algorithm. If the exact $Q^{\pi_\theta}$ is unknown, replace it with a **Monte Carlo estimate**: run an episode to completion, collect the return $G_t = \sum_{k=0}^{T-t} \gamma^k R_{t+k+1}$, and use it as an unbiased estimate of $Q^{\pi_\theta}(S_t, A_t)$:

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(A_t \mid S_t) \cdot G_t.$$

This is the **REINFORCE** algorithm, introduced by Williams (1992) under the name "Monte Carlo policy gradient." It is unbiased — because $G_t$ is an unbiased estimate of $Q^{\pi_\theta}(S_t, A_t)$ — and requires no model of the environment. The only inputs are the trajectory of states, actions, and rewards collected by running the current policy.

REINFORCE inherits the full structure of the policy gradient theorem: the gradient points in the direction that most increases the log-probability of high-return trajectories. Actions that contributed to positive returns have their probabilities nudged upward; actions that contributed to negative returns have theirs nudged downward. Summed over many trajectories, the parameter update is an unbiased estimate of $\nabla_\theta J(\theta)$, and gradient ascent converges (under appropriate step size conditions) to a local maximum of $J$.

## The Variance Problem

Despite its theoretical cleanliness, REINFORCE has a decisive practical weakness: its gradient estimates have high variance.

The Monte Carlo return $G_t$ is a sum of hundreds or thousands of future rewards, each multiplied by a discount factor and random due to both environment stochasticity and policy stochasticity. The variance of $G_t$ grows with the horizon. In a task with $T = 1000$ steps and moderate reward noise, the variance of the Monte Carlo return can easily exceed $10^3$ or $10^4$, even when the expected return is modest. High-variance gradient estimates mean that the gradient signal is dominated by noise: the agent repeatedly moves its parameters in directions that reflect the randomness of a single trajectory rather than the systematic structure of $J(\theta)$.

The practical consequence is slow, unstable learning. The parameter updates fluctuate wildly from episode to episode, and convergence — while guaranteed in theory — can require an impractical number of episodes in any environment where rewards are delayed or stochastic. Variance reduction is not an optional refinement; it is the core engineering challenge of policy gradient methods.

## Baseline Subtraction

The most important variance reduction technique is the **control variate** or **baseline**. The key observation is that, for any function $b(s)$ that depends only on the state (not on the action taken), the following identity holds:

$$\mathbb{E}_{\pi_\theta}\!\bigl[\nabla_\theta \log \pi_\theta(A_t \mid S_t) \cdot b(S_t)\bigr] = 0.$$

The proof is immediate: conditioning on $S_t$,

$$\mathbb{E}_{A_t \sim \pi_\theta(\cdot \mid S_t)}\!\bigl[\nabla_\theta \log \pi_\theta(A_t \mid S_t)\bigr] = \nabla_\theta \sum_a \pi_\theta(a \mid S_t) = \nabla_\theta 1 = 0.$$

Subtracting a baseline does not change the expected gradient — but it can dramatically reduce its variance. Replacing $G_t$ with $(G_t - b(S_t))$ gives:

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(A_t \mid S_t) \cdot \bigl(G_t - b(S_t)\bigr).$$

The quantity $(G_t - b(S_t))$ has the same expectation as $G_t$ but, if $b$ is a good predictor of the return from $S_t$, its variance is much smaller. Returns above the baseline represent actions that were better than expected and are reinforced; returns below it represent actions that were worse than expected and are suppressed. The baseline centers the signal, removing the component of variance that is due to variation in state value rather than variation in action quality.

## The Optimal Baseline

The baseline that minimizes variance can be derived explicitly. The variance of the gradient estimator for a single time step is (ignoring correlations across time):

$$\text{Var}\!\bigl[\nabla_\theta \log \pi_\theta(A_t \mid S_t) (G_t - b)\bigr] = \mathbb{E}\!\bigl[\|\nabla_\theta \log \pi_\theta\|^2 (G_t - b)^2\bigr] - \bigl(\mathbb{E}\!\bigl[\|\nabla_\theta \log \pi_\theta\| (G_t - b)\bigr]\bigr)^2.$$

Setting the derivative with respect to $b$ to zero and solving gives the optimal constant baseline:

$$b^* = \frac{\mathbb{E}\!\bigl[\|\nabla_\theta \log \pi_\theta(A_t \mid S_t)\|^2 G_t\bigr]}{\mathbb{E}\!\bigl[\|\nabla_\theta \log \pi_\theta(A_t \mid S_t)\|^2\bigr]},$$

a weighted average of the return, with weights given by the squared norm of the score function. In practice, this optimal baseline is rarely used directly — it depends on quantities that are expensive to estimate. The standard choice is to use $b(s) = V^{\pi_\theta}(s)$, the **state value function** of the current policy.

## The Advantage Function

When $b(s) = V^{\pi_\theta}(s)$, the centered quantity has a name. The **advantage function** is:

$$A^{\pi_\theta}(s, a) = Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s).$$

It measures how much better action $a$ is than the average action in state $s$ under the current policy. $A^{\pi_\theta}(s, a) > 0$ means action $a$ is above average; $A^{\pi_\theta}(s, a) < 0$ means it is below average. The policy gradient becomes:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(A_t \mid S_t) \cdot A^{\pi_\theta}(S_t, A_t)\right].$$

The advantage function is not just a variance reduction trick. It is the right quantity to use: the action-value function $Q^{\pi_\theta}(s, a)$ measures total expected return, which includes a state-dependent component $V^{\pi_\theta}(s)$ that has nothing to do with the action taken. Using $Q^{\pi_\theta}$ to weight policy updates attributes to the action credit that actually belongs to the state. The advantage function strips that component away, leaving only the action's contribution above and beyond what the agent would have gotten on average.

Note that $\mathbb{E}_{\pi_\theta}[A^{\pi_\theta}(S_t, A_t)] = 0$ for every state, since $A^{\pi_\theta}(s, \cdot)$ is centered by construction. The gradient signal pushes probabilities of above-average actions up and below-average actions down, exactly as intended.

## The Gradient as a Direction in Policy Space

A geometric perspective clarifies what policy gradient methods are doing. The policy $\pi_\theta$ lives in a manifold of probability distributions — one distribution over actions for each state — parameterized by $\theta$. The performance objective $J(\theta)$ is a scalar function on this manifold.

The score function $\nabla_\theta \log \pi_\theta(A_t \mid S_t)$ is the direction in parameter space that most rapidly increases the log-probability of $A_t$ at $S_t$ — equivalently, the local tangent direction corresponding to "take this action more often." The policy gradient theorem says to follow the combination of these directions, weighted by the advantage: increase the probability of advantageous actions and decrease the probability of disadvantageous ones, proportionally to how advantageous or disadvantageous they are, in proportion to how strongly the parameters can change those probabilities.

This is gradient ascent in parameter space, but it corresponds to a meaningful operation in policy space: the update moves $\pi_\theta$ toward the policy that concentrates more mass on better actions. The step size $\alpha$ controls how far to move in this direction, and the policy gradient theorem guarantees that small enough steps are never harmful — the gradient always points uphill in $J$.

## What REINFORCE Does Not Guarantee

The policy gradient theorem guarantees that $\nabla_\theta J(\theta) = 0$ at any stationary point of $J$, and that gradient ascent converges to such a point under standard conditions. It does not guarantee convergence to a **global** optimum. $J(\theta)$ is typically non-convex in $\theta$: a policy parameterized by a neural network creates a loss landscape with multiple local maxima, saddle points, and flat regions. Gradient ascent finds a local maximum.

In practice, this is often acceptable. Local maxima in high-dimensional, highly overparameterized networks can be near-globally optimal — a phenomenon analogous to observations in supervised deep learning. But there is no theorem guaranteeing this, and the quality of the local maximum reached depends heavily on initialization, step size, and the structure of the environment's reward landscape.

REINFORCE also converges slowly relative to its theoretical potential. The Monte Carlo return has variance that grows with the horizon, even after baseline subtraction. Using a learned critic instead of a Monte Carlo return to estimate the advantage — and bootstrapping rather than waiting for episode ends — dramatically reduces variance while introducing a small bias. This tradeoff between variance and bias in the advantage estimator is the central design decision of the next generation of policy gradient methods.

## The Bridge to Actor-Critic

REINFORCE uses the Monte Carlo return $G_t$ to estimate $Q^{\pi_\theta}(S_t, A_t)$. This is unbiased but high-variance. An alternative is to use a second neural network — a **critic** — to estimate the value function directly, and use the critic's output in place of the Monte Carlo return. This is the **actor-critic** architecture: the actor is $\pi_\theta$, updated by policy gradients; the critic is $\hat{V}_w(s)$ (parameterized by $w$), updated by TD learning to track $V^{\pi_\theta}$.

The advantage estimate becomes $A^{\pi_\theta}(S_t, A_t) \approx R_{t+1} + \gamma \hat{V}_w(S_{t+1}) - \hat{V}_w(S_t)$ — the TD error. This requires neither a full episode nor a lookup table: it can be computed after every single step, enabling online, continuous learning. The variance is far lower than the Monte Carlo return, at the cost of a small bias introduced by the critic's approximation error.

The division of labor is precise: the actor solves the control problem (finding better actions via policy gradients), and the critic solves the prediction problem (evaluating the current policy via TD learning). This is Howard's policy evaluation and policy improvement, re-expressed in the language of stochastic gradient methods and function approximation — adapted to operate continuously, without model access, in the same pass through a trajectory.

---

*The policy gradient theorem in its general form is proved in Sutton, McAllester, Singh, and Mansour (2000), Policy Gradient Methods for Reinforcement Learning with Function Approximation. The REINFORCE algorithm and the score function estimator are from Williams (1992), Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. The baseline subtraction result and the zero-mean property of the centered gradient estimator appear in Williams (1992) and are elaborated in Weaver and Tao (2001), The Optimal Reward Baseline for Gradient-Based Reinforcement Learning. The advantage function and its role in variance reduction are analyzed in Greensmith, Bartlett, and Baxter (2004), Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning.*
