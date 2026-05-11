---

title: "Actor-Critic and Advantage Estimation"
subtitle: "Bootstrapped baselines, asynchronous workers, and the generalized advantage estimator"

---


The policy gradient theorem established the correct form of the gradient, and REINFORCE provided an unbiased estimator of it. But unbiasedness does not mean usability. The Monte Carlo return $G_t = \sum_{k=0}^{T-t} \gamma^k R_{t+k+1}$ is a sum of hundreds or thousands of random future rewards, and its variance scales with the horizon. On any task where rewards are sparse or delayed — which is most tasks of interest — REINFORCE updates are dominated by noise. An agent learning to navigate a maze may spend thousands of episodes receiving zero reward at every step until it stumbles, by chance, on the exit. The Monte Carlo return for every step in those empty episodes is zero, which provides no learning signal at all, and the first successful trajectory, when it arrives, produces a gradient that is so large and so correlated across time steps that it can destabilize the policy as easily as improve it.

There is a more fundamental objection. REINFORCE requires episodes to terminate before any update can be computed: $G_t$ is defined only with access to $R_{T}$, the reward at the final step. This is not merely inconvenient. For continuing tasks — robotic control, financial trading, dialogue systems, any problem that runs indefinitely rather than in discrete episodes — the Monte Carlo return is undefined. The algorithm cannot be applied.

The resolution to both problems is bootstrapping: replace the tail of the Monte Carlo sum with a learned estimate of the value function. Instead of waiting to observe the true return from state $S_t$, use a neural network $\hat{V}_w(S_t)$ as a stand-in for future rewards. This introduces bias — the estimated value is wrong, especially early in training — but controls variance: the estimate has only the noise of a single network evaluation, not the accumulated noise of hundreds of future rewards. The tradeoff is exact: more bootstrapping means less variance and more bias; less bootstrapping means more variance and less bias. The actor-critic framework is, in its essence, a family of methods for navigating this tradeoff.

## The TD Error as Advantage Estimate

Chapter 6 ended with the observation that the one-step TD error $\delta_t = R_{t+1} + \gamma \hat{V}_w(S_{t+1}) - \hat{V}_w(S_t)$ approximates the advantage. The derivation is worth making precise.

The true advantage is $A^{\pi}(S_t, A_t) = Q^{\pi}(S_t, A_t) - V^{\pi}(S_t)$. By definition of the Q-function:

$$Q^{\pi}(S_t, A_t) = \mathbb{E}\bigl[R_{t+1} + \gamma V^{\pi}(S_{t+1}) \mid S_t, A_t\bigr].$$

Substituting:

$$A^{\pi}(S_t, A_t) = \mathbb{E}\bigl[R_{t+1} + \gamma V^{\pi}(S_{t+1}) - V^{\pi}(S_t) \mid S_t, A_t\bigr] = \mathbb{E}[\delta_t^{\pi} \mid S_t, A_t],$$

where $\delta_t^{\pi} = R_{t+1} + \gamma V^{\pi}(S_{t+1}) - V^{\pi}(S_t)$ is the TD error computed with the **true** value function. If we had $V^{\pi}$ exactly, each realized TD error $\delta_t$ would be an unbiased estimate of the advantage. In practice, we have $\hat{V}_w \approx V^{\pi}$, so the realized TD error is a biased but low-variance estimate of the advantage. The bias enters through two channels: the approximation error $\hat{V}_w(s) - V^{\pi}(s)$, and the fact that the expectation is replaced by a single sample. Both diminish as the critic improves.

The policy gradient update with this one-step advantage estimate is:

$$\theta \leftarrow \theta + \alpha \, \delta_t \, \nabla_\theta \log \pi_\theta(A_t \mid S_t),$$

and the critic update is a TD step:

$$w \leftarrow w + \alpha_w \, \delta_t \, \nabla_w \hat{V}_w(S_t).$$

Both updates can be computed after a single step in the environment, without waiting for an episode to end. The actor learns to select better actions; the critic learns to predict cumulative reward; together they bootstrap each other toward a stable policy.

## The n-step Return

One-step bootstrapping represents an extreme on a spectrum. At the opposite extreme sits the full Monte Carlo return: no bootstrapping at all. Between them lies a family of **n-step returns**, which take $n$ actual rewards before substituting the value estimate:

$$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n \hat{V}_w(S_{t+n}).$$

For $n=1$, this is the one-step TD target. For $n \to \infty$, the value estimate contributes only at the end of a complete trajectory, approaching the Monte Carlo return. Every intermediate $n$ represents a different point on the bias-variance frontier: larger $n$ reduces bias (the actual rewards replace more of the approximated future) at the cost of higher variance (more random rewards are summed); smaller $n$ reduces variance at the cost of more bias (the value estimate dominates). The one-step return has minimum variance but maximum bias; the Monte Carlo return has zero bias but maximum variance.

In practice, the n-step return is often computed for $n$ between 5 and 20. For a step $t$ within $n$ steps of an episode boundary, the sum is truncated and the discount applied to the terminal value if the episode ends, or to zero if it does not. This technicality is important for correct implementation: conflating episode termination with the edge of the n-step window introduces a subtle bias that can be difficult to diagnose.

The n-step advantage estimate is $G_t^{(n)} - \hat{V}_w(S_t)$, and the actor update becomes:

$$\theta \leftarrow \theta + \alpha \bigl(G_t^{(n)} - \hat{V}_w(S_t)\bigr) \nabla_\theta \log \pi_\theta(A_t \mid S_t).$$

The critic can be updated by regressing toward $G_t^{(n)}$ as a target: $\mathcal{L}_{\text{critic}} = \bigl(\hat{V}_w(S_t) - G_t^{(n)}\bigr)^2$.

## Generalized Advantage Estimation

Schulman, Moritz, Levine, Jordan, and Abbeel (2016) identified that choosing a fixed $n$ is itself an arbitrary choice — different parts of the state space may benefit from different horizons, and the optimal $n$ changes as the critic improves. Their solution was to take a weighted average over all horizons simultaneously: the **generalized advantage estimator** GAE($\lambda$).

Define the sequence of one-step TD errors:

$$\delta_t = R_{t+1} + \gamma \hat{V}_w(S_{t+1}) - \hat{V}_w(S_t).$$

The GAE estimate is a geometrically weighted sum of these errors:

$$\hat{A}_t^{\text{GAE}(\lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \, \delta_{t+l}.$$

This expression has a compact interpretation. Define the $n$-step advantage $\hat{A}_t^{(n)} = G_t^{(n)} - \hat{V}_w(S_t)$. Then GAE($\lambda$) is the $\lambda$-weighted average:

$$\hat{A}_t^{\text{GAE}(\lambda)} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \hat{A}_t^{(n)}.$$

The parameter $\lambda \in [0, 1]$ interpolates the bias-variance spectrum in a principled way. At $\lambda = 0$, only the one-step TD error contributes: $\hat{A}_t^{\text{GAE}(0)} = \delta_t$, which is low-variance and high-bias. At $\lambda = 1$, the geometric sum telescopes exactly to the full Monte Carlo advantage: $\hat{A}_t^{\text{GAE}(1)} = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} = G_t - \hat{V}_w(S_t)$, which is unbiased but high-variance. Intermediate $\lambda$ blends both: the geometric weights discount long-horizon error accumulations, giving more weight to shorter, lower-variance estimates while retaining some of the unbiasedness of longer ones.

The practical computation is efficient. If the trajectory terminates at step $T$, the GAE can be computed backward in a single pass:

$$\hat{A}_T = \delta_T, \qquad \hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1}.$$

No inner loops are required; the entire sequence of advantage estimates for a trajectory is computed in $O(T)$ time.

The value of $\lambda = 0.95$, combined with $\gamma = 0.99$, has proven remarkably robust across a wide range of tasks — continuous locomotion, dexterous manipulation, multi-task domains. The combination became the de facto standard, adopted with minimal tuning in PPO, A2C, and most subsequent actor-critic methods. The geometric decay parameter $\gamma\lambda = 0.9405$ effectively limits the horizon of the advantage estimate to roughly $1/(1 - \gamma\lambda) \approx 17$ steps, beyond which earlier TD errors contribute negligibly. This adaptive truncation is precisely what the n-step return achieves with a hard cutoff, but GAE achieves it softly and without requiring a choice of $n$.

## The Two-Network Architecture

The actor-critic architecture can be implemented with two entirely separate neural networks — one for $\pi_\theta$ and one for $\hat{V}_w$ — with distinct parameter sets and independent optimization. In practice, sharing the lower layers of both networks substantially improves performance. The intuition: the low-level features useful for predicting value — object positions, velocities, the dynamic structure of the environment — are the same features useful for selecting actions. Training both heads on shared representations allows the gradient signal from the critic (strong, dense, from every step) to improve the feature extractor used by the actor, and the policy gradient from the actor to shape the representations used by the critic.

The standard architecture uses a shared backbone $f_\phi$ with two output heads: a policy head $\pi_\theta(a \mid s) = \pi(f_\phi(s))$ and a value head $\hat{V}_w(s) = V(f_\phi(s))$. The total loss is:

$$\mathcal{L}(\phi, \theta, w) = -\mathbb{E}\bigl[\hat{A}_t \nabla_\theta \log \pi_\theta(A_t \mid S_t)\bigr] + c_1 \mathcal{L}_{\text{VF}} - c_2 H(\pi_\theta(\cdot \mid S_t)),$$

where $\mathcal{L}_{\text{VF}} = \|\hat{V}_w(S_t) - G_t^{\text{target}}\|^2$ is the critic loss, $H(\pi_\theta(\cdot \mid S_t))$ is the policy entropy, and $c_1, c_2$ are scalar coefficients. The entropy term is discussed below. The gradient of this composite loss with respect to $\phi$ incorporates signal from both the actor and the critic, training the shared layers to serve both objectives simultaneously.

## A3C: Asynchronous Advantage Actor-Critic

In 2016, Mnih, Puigdomènech Badia, Mirza, Graves, Lillicrap, Harley, Silver, and Kavukcuoglu introduced **A3C** — the Asynchronous Advantage Actor-Critic. The method addressed two problems simultaneously: the sample efficiency cost of on-policy learning, and the correlation problem that DQN had solved with experience replay.

Experience replay solves the correlation problem by mixing transitions from many different time points and many past policies into each minibatch. But mixing past and current policy data requires the updates to be off-policy — the correction for the mismatch between the data distribution and the current policy. For value-based methods like Q-learning this is straightforward, since the target policy is always greedy regardless of how data was collected. For policy gradient methods, off-policy corrections require importance sampling ratios $\pi_\theta(A_t \mid S_t) / \pi_{\text{old}}(A_t \mid S_t)$, which become extremely variable when the two policies diverge, re-introducing high variance through a different channel.

A3C solves the correlation problem differently: rather than mixing past transitions in a buffer, it runs many workers **in parallel**, each maintaining its own copy of the environment and its own local copy of the actor-critic network. Workers collect short trajectories asynchronously, compute actor and critic gradient updates locally, and push those gradients to a central parameter server. The global parameters are updated with each incoming gradient, and workers periodically pull the updated global parameters to reset their local copies.

The decorrelation mechanism is: because workers run independently, they will be in different states of the environment at the same time step — different levels of a game, different positions in a maze, different phases of a locomotion task. The gradient updates arriving at the server come from different parts of the state space in rapid succession, producing a gradient signal with the diversity that experience replay achieves through historical mixing. No replay buffer is needed, and no off-policy corrections are required: each worker's trajectory was collected by a recent version of the policy.

The asynchrony itself provides a form of implicit variance reduction. When one worker collects a trajectory under a policy that has just been updated by another worker's gradient, the resulting trajectory is slightly off-policy — but only slightly, since updates are small and frequent. This mild off-policy-ness is accepted as a manageable approximation. Experiments showed it to be stable in practice: A3C solved a broad suite of Atari games and 3D maze navigation tasks in far less wall-clock time than DQN, using CPU-only computation across many threads rather than GPU-accelerated experience replay.

The advantage estimates in A3C are computed over short $n$-step windows — typically 5 to 20 steps — before the workers synchronize with the server. Each worker accumulates $n$ transitions, computes the $n$-step advantage estimates from that window, and pushes the gradients. This online, short-trajectory regime is fundamentally different from the episode-length Monte Carlo returns of REINFORCE: the agent is always learning, even in the middle of an episode, from arbitrarily short windows of experience.

## A2C: The Synchronous Variant

A3C's asynchrony, while effective, introduced implementation complexity and a subtle theoretical inconsistency. Workers pushing gradients from different parameter versions to the same server means the gradient updates are computed on parameters that are, by the time they arrive, already stale. The effect is a kind of noisy momentum: each update pushes the global parameters in a slightly out-of-date direction. Mnih et al. argued this was beneficial — similar to the stability provided by target networks in DQN — but it made analysis difficult and performance sensitive to the number of workers and their relative speeds.

The **A2C** variant (the synchronous version of A3C, described and popularized by OpenAI in 2017) removes the asynchrony entirely. Rather than having workers push gradients as they become available, A2C runs $N$ workers in lockstep: all workers collect an $n$-step trajectory simultaneously, all advantage estimates are computed, and a single gradient update is applied to the global parameters before all workers reset. The update is equivalent to computing the mean gradient over a large batch — $N \cdot n$ transitions — in a single synchronized step.

The theoretical advantage: all gradient contributions come from the same parameter version. There is no staleness, the gradient estimator is consistent with its nominal variance, and the algorithm is simpler to reason about. The practical tradeoff: by forcing synchronization, slow workers bottleneck the entire system — a single worker that lags due to environment complexity or machine variability holds up the entire update. In practice, A2C matches or slightly exceeds A3C performance on standard benchmarks, and its simplicity made it the preferred implementation basis for subsequent work.

The batch structure of A2C also enables efficient use of GPU hardware. $N$ parallel environments can be stepped simultaneously, producing a batch of observations that passes through the shared network in one forward pass, yielding both policy logits and value estimates for all $N$ states in a single compute step. The critic and actor losses are computed over the full batch and combined; a single backward pass produces the parameter update. For large $N$ — 16, 32, or even 128 workers — the compute efficiency is substantially higher than the sequential single-sample updates of online TD learning.

## Entropy Regularization

Both A3C and A2C include an **entropy bonus** in the actor loss: an additional term that rewards the policy for maintaining high entropy — spreading probability mass over many actions rather than concentrating it on a single one. The augmented actor objective is:

$$\mathcal{L}_{\text{actor}} = \mathbb{E}\bigl[\hat{A}_t \log \pi_\theta(A_t \mid S_t)\bigr] + c_H \, H\!\bigl(\pi_\theta(\cdot \mid S_t)\bigr),$$

where $H(\pi) = -\sum_a \pi(a \mid s) \log \pi(a \mid s)$ is the entropy and $c_H > 0$ is a small coefficient, typically 0.01.

The motivation is exploration. Without regularization, a policy gradient method will converge to a deterministic policy once it identifies a set of actions that reliably produce positive advantages. This premature determinism is catastrophic: the agent stops exploring, fixes on a locally optimal strategy, and can no longer discover improvements that require temporarily suboptimal behavior. The entropy bonus prevents this by penalizing certainty: the policy is pushed to maintain uncertainty about its own actions, distributing probability across alternatives even when one action appears best. This is particularly important early in training, when the critic's estimates are inaccurate and the apparent best action may not be the true best action.

The entropy bonus has a deeper connection to statistical mechanics and information theory, which will be made precise in the chapter on maximum entropy RL. For now it suffices to observe that it is not merely a heuristic: it is a gradient-accessible mechanism for encouraging exploration, analogous in effect to the $\varepsilon$-greedy schedule in value-based methods but differentiable and adaptive. As training progresses, the policy's advantage estimates become more accurate, and the entropy naturally tends to decrease toward the right level; the coefficient $c_H$ scales how strongly the entropy term resists this collapse.

## The Bias-Variance Landscape

The actor-critic framework exposes the bias-variance tradeoff in a form that is navigable rather than incidental. REINFORCE fixes one extreme: zero bias, maximum variance. One-step TD actor-critic fixes the other: minimum variance, maximum bias. GAE($\lambda$) places a knob between them, and the practice of tuning $\lambda$ to 0.95–0.98 reflects an empirical consensus: near the Monte Carlo end of the spectrum is usually better, because critic approximation errors accumulate less destructively when only a small fraction of the advantage estimate comes from the learned value function.

The critic's quality determines where this consensus holds. Early in training, when $\hat{V}_w$ is poorly initialized and has high approximation error, lower $\lambda$ would reduce bias — but the bootstrap bias from a bad critic is often worse than the variance from longer returns, so practitioners typically hold $\lambda$ fixed and let the critic improve. The alternative — $\lambda$-scheduling, raising $\lambda$ from a low initial value as the critic improves — has been studied but offers modest gains relative to its implementation complexity. In practice, $\lambda = 0.95$ with $\gamma = 0.99$ works well enough that it has become a near-universal default.

There is a subtler consideration: the advantage estimate and the value target for the critic are not independent. If $\hat{A}_t = G_t^{(n)} - \hat{V}_w(S_t)$ and the critic is regressed toward $G_t^{(n)}$, the critic is chasing the same n-step return that defines the advantage. When the n-step return is low-variance (small $n$), the critic trains on low-variance targets and converges faster, but the resulting value function has higher bias, which feeds back into biased advantage estimates. The design of the shared target — for both the actor's gradient and the critic's regression — is thus not separable from the choice of $\lambda$.

## What A3C and GAE Established

Before A3C, policy gradient methods had not demonstrated competitive performance on the Atari benchmark, which DQN had owned since 2013. After A3C, the gap between value-based and policy gradient methods closed substantially. A3C's 3D navigation results — solving labyrinthine first-person environments that DQN could not handle — demonstrated that actor-critic methods could learn effectively from observations where the value function structure was genuinely complex and long-horizon.

GAE's contribution was orthogonal: it provided a principled, computationally efficient way to estimate advantages that became the default estimator for all subsequent actor-critic methods. PPO, TRPO, and essentially every on-policy deep RL algorithm published after 2016 uses GAE or a closely related estimator. The advantage estimate is not the bottleneck in those methods; it was solved here.

What neither A3C nor A2C addressed was the problem of update step size. Policy gradients can take destructively large steps: a single bad update can collapse the policy to a degenerate solution — all probability mass on one action — from which recovery is slow. The gradient estimator, however good, does not constrain how far the parameters should move in a single step. A large gradient step in a non-convex landscape is as likely to jump over a good region as to enter it, and in the policy space of a neural network, large steps in parameter space can produce catastrophic changes in behavior.

The variance problem — why REINFORCE was unusable — has been resolved by bootstrapping, shared architectures, and GAE. The step-size problem remains. The next chapter addresses it directly: using the geometry of the policy space itself, via the Fisher information metric and Kullback-Leibler divergence, to constrain how much the policy is allowed to change per update and thereby guarantee that each gradient step is a genuine improvement.

---

*The actor-critic architecture in its early form appears in Barto, Sutton, and Anderson (1983), Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems. The formal policy gradient framework connecting actor-critic to the policy gradient theorem is in Sutton, McAllester, Singh, and Mansour (2000). The Asynchronous Advantage Actor-Critic (A3C) is introduced in Mnih, Puigdomènech Badia, Mirza, Graves, Lillicrap, Harley, Silver, and Kavukcuoglu (2016), Asynchronous Methods for Deep Reinforcement Learning. The A2C synchronous variant is described in the OpenAI Baselines release (Dhariwal et al., 2017). Generalized Advantage Estimation is from Schulman, Moritz, Levine, Jordan, and Abbeel (2016), High-Dimensional Continuous Control Using Generalized Advantage Estimation. The entropy regularization technique and its role in exploration in deep actor-critic methods is analyzed in Mnih et al. (2016) and elaborated in Williams and Peng (1991), Function Optimization Using Connectionist Reinforcement Learning Algorithms.*
