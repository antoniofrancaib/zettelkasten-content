---

title: "Maximum Entropy RL and SAC"
subtitle: "The entropy-augmented objective, soft Bellman equations, and off-policy continuous control"

---


The policy gradient methods of the preceding chapters — REINFORCE, A3C, TRPO, PPO — share a common objective: maximize expected cumulative reward. Entropy appears in these methods as an optional regularizer, added as a small bonus term to prevent premature determinism. It is useful but secondary, a correction applied to an objective whose primary meaning is reward. The maximum entropy framework inverts this priority. It treats entropy not as a regularizer bolted onto reward maximization but as a co-equal objective in its own right, part of what the agent is supposed to be doing. The shift is philosophical before it is algorithmic, and it produces a family of methods with properties — off-policy stability, robustness to reward misspecification, natural multimodality — that on-policy methods do not share.

The roots of the framework lie in work on inverse reinforcement learning and planning under uncertainty rather than in direct policy optimization. Ziebart, Maas, Bagnell, and Dey (2008) introduced maximum entropy IRL, which modeled human behavior as approximately optimal under a stochastic policy that maximized a reward plus an entropy term. The trajectory distribution that maximizes this augmented objective is a Boltzmann distribution over rewards — the same distribution that appears in statistical mechanics as the canonical ensemble. Ziebart (2010) extended this to the sequential setting, deriving soft Bellman equations that carry the entropy term through the dynamic programming recursion. The connection to statistical mechanics was precise: the soft value function is a free energy, and the optimal policy is a Gibbs distribution.

The connection remained largely theoretical until Haarnoja, Tang, Abbeel, and Levine (2017, 2018) built a practical deep RL algorithm around it: the **Soft Actor-Critic**, or SAC. SAC achieved state-of-the-art performance on continuous control benchmarks while requiring dramatically fewer environment interactions than PPO — and unlike PPO, it learned off-policy, reusing every collected transition many times via a replay buffer. The combination of sample efficiency, stability, and strong empirical performance made SAC the dominant algorithm for continuous control from 2018 onward.

## The Maximum Entropy Objective

Standard RL maximizes the expected return:

$$J_{\text{RL}}(\pi) = \mathbb{E}_{\tau \sim \pi}\!\left[\sum_{t=0}^{T} \gamma^t R(S_t, A_t)\right].$$

The **maximum entropy** (MaxEnt) objective augments every time step's reward with the entropy of the policy at that state:

$$J_{\text{MaxEnt}}(\pi) = \mathbb{E}_{\tau \sim \pi}\!\left[\sum_{t=0}^{T} \gamma^t \bigl(R(S_t, A_t) + \alpha \, H\!\bigl(\pi(\cdot \mid S_t)\bigr)\bigr)\right],$$

where $H(\pi(\cdot \mid s)) = -\mathbb{E}_{A \sim \pi(\cdot \mid s)}[\log \pi(A \mid s)]$ is the entropy of the policy at state $s$, and $\alpha > 0$ is a **temperature parameter** that sets the relative weight of entropy versus reward. At $\alpha = 0$, the objective reduces to standard RL. As $\alpha \to \infty$, the objective is dominated by entropy and the optimal policy is uniform over actions regardless of reward.

The entropy bonus incentivizes the agent to maintain uncertainty about its actions — to spread probability mass across actions that are comparably good rather than concentrating on a single best action. This is not merely exploration in the REINFORCE sense, where entropy decays as the policy converges. In the MaxEnt framework, entropy is a perpetual objective: the optimal policy is genuinely stochastic, with entropy calibrated by $\alpha$ to reflect the scale of reward differences between actions. When two actions have identical Q-values, the optimal MaxEnt policy assigns them equal probability. When one action is substantially better, its probability is higher, but not by more than $\alpha$ times the log-probability ratio — the temperature moderates how sharply the policy concentrates on the best action.

## The Soft Bellman Equations

The MaxEnt objective changes the Bellman recursion. Define the **soft Q-function** as the expected return under the MaxEnt objective when action $a$ is taken in state $s$ and the agent follows the optimal policy thereafter:

$$Q_{\text{soft}}(s, a) = R(s, a) + \gamma \, \mathbb{E}_{S' \sim p(\cdot \mid s,a)}\!\bigl[V_{\text{soft}}(S')\bigr],$$

where the **soft value function** is:

$$V_{\text{soft}}(s) = \mathbb{E}_{A \sim \pi(\cdot \mid s)}\!\bigl[Q_{\text{soft}}(s, A) - \alpha \log \pi(A \mid s)\bigr].$$

The soft value function is the expected soft Q-value under the policy, minus the expected log-probability — which is just the expected Q-value plus $\alpha$ times the entropy:

$$V_{\text{soft}}(s) = \mathbb{E}_{A \sim \pi}\!\bigl[Q_{\text{soft}}(s, A)\bigr] + \alpha H\!\bigl(\pi(\cdot \mid s)\bigr).$$

The entropy term is absorbed into the value function rather than treated as a separate bonus. Every step of the recursion carries the entropy forward, compounding it through the discounted sum in the same way reward compounds. A policy that is highly stochastic for many steps accumulates entropy as value, which is reflected in higher soft Q-values for states from which stochastic behavior is possible.

For the optimal policy, the soft value function has a closed form. Setting the derivative of $V_\text{soft}(s)$ with respect to $\pi(\cdot \mid s)$ to zero, subject to the constraint that $\pi$ is a valid distribution, gives:

$$V_{\text{soft}}(s) = \alpha \log \int_{\mathcal{A}} \exp\!\left(\frac{Q_{\text{soft}}(s, a)}{\alpha}\right) da.$$

This is the log-sum-exp (or log-partition function) of $Q_{\text{soft}} / \alpha$ over actions — the free energy of a Boltzmann distribution with inverse temperature $1/\alpha$. The integral replaces the hard maximum of standard Q-learning with a soft maximum: a smooth, everywhere-differentiable analog that approaches $\max_a Q(s, a)$ as $\alpha \to 0$.

## The Optimal Soft Policy

The optimal policy under the MaxEnt objective is the distribution that achieves the maximum in the soft value function. Substituting the closed form for $V_\text{soft}(s)$ and solving, the optimal policy is a **Boltzmann distribution** over Q-values:

$$\pi^*(a \mid s) = \frac{\exp\bigl(Q_{\text{soft}}(s, a) / \alpha\bigr)}{\int_{\mathcal{A}} \exp\bigl(Q_{\text{soft}}(s, a') / \alpha\bigr)\, da'} = \exp\!\left(\frac{Q_{\text{soft}}(s, a) - V_{\text{soft}}(s)}{\alpha}\right).$$

This is the canonical distribution of statistical mechanics. The Q-values play the role of negative energies: actions with high Q-values receive high probability, with the sharpness of the distribution controlled by $\alpha$. At low temperature ($\alpha \to 0$), the distribution concentrates on the single highest-Q action — recovering greedy behavior. At high temperature ($\alpha \to \infty$), the distribution flattens to uniform — recovering maximum entropy regardless of reward.

The optimal soft policy can be re-expressed as a KL minimization. For any policy $\pi$, define the information-theoretic divergence from the Boltzmann distribution:

$$\text{KL}\!\left[\pi(\cdot \mid s) \,\Big\|\, \frac{\exp(Q_\text{soft}(s,\cdot)/\alpha)}{Z(s)}\right],$$

where $Z(s)$ is the normalizing partition function. The optimal soft policy minimizes this KL divergence to zero. Policy optimization in the MaxEnt framework is projection of the policy distribution onto the Boltzmann manifold — the manifold of distributions proportional to $\exp(Q / \alpha)$. This information-geometric view will recur when we reach RLHF: the KL regularization used to prevent language models from drifting too far from a reference policy is precisely this structure, with $Q$ replaced by a learned reward and $\alpha$ replaced by the KL coefficient.

## The Soft Actor-Critic Architecture

SAC maintains three function approximators: two **soft Q-networks** $Q_{\phi_1}(s, a)$ and $Q_{\phi_2}(s, a)$, and a **policy network** $\pi_\psi(a \mid s)$ — specifically, a neural network that outputs the parameters of a squashed Gaussian distribution over actions. A replay buffer $\mathcal{D}$ stores all previously collected transitions.

The two Q-networks address overestimation bias. As in Double DQN, using a single Q-network to both select and evaluate actions produces inflated targets. In the continuous case, the max operator is replaced by the expectation under the policy, but the bias mechanism is the same: a single Q-network's positive noise inflates the bootstrap target, which inflates the Q-values, which further inflates subsequent targets in a compounding cascade. SAC uses the **minimum** of the two Q-networks to form the Bellman target:

$$y = R + \gamma \, \mathbb{E}_{A' \sim \pi_\psi(\cdot \mid S')}\!\Bigl[\min\bigl(Q_{\phi_1}(S', A'),\, Q_{\phi_2}(S', A')\bigr) - \alpha \log \pi_\psi(A' \mid S')\Bigr].$$

The $-\alpha \log \pi_\psi(A' \mid S')$ term is the entropy contribution of the next state: it enters the target as the negative log-probability of the sampled action, computing the per-sample entropy. The two Q-networks are trained independently by minimizing:

$$\mathcal{L}_{Q}(\phi_i) = \mathbb{E}_{(S, A, R, S') \sim \mathcal{D}}\!\bigl[\bigl(Q_{\phi_i}(S, A) - y\bigr)^2\bigr], \quad i = 1, 2.$$

## The Reparameterization Trick for Policy Gradients

The policy update in SAC maximizes the expected soft Q-value minus the expected log-probability:

$$\mathcal{L}_\pi(\psi) = \mathbb{E}_{S \sim \mathcal{D}}\!\Bigl[\mathbb{E}_{A \sim \pi_\psi(\cdot \mid S)}\!\bigl[\alpha \log \pi_\psi(A \mid S) - Q_{\phi}(S, A)\bigr]\Bigr].$$

Computing the gradient of this with respect to $\psi$ requires differentiating through the expectation over actions. The REINFORCE approach — multiply by the log-derivative — introduces high variance, and SAC avoids it entirely via the **reparameterization trick**. The policy is expressed as a deterministic function of a noise variable:

$$A = f_\psi(\varepsilon;\, S), \quad \varepsilon \sim \mathcal{N}(0, I),$$

where $f_\psi$ is the neural network and $\varepsilon$ is sampled independently of $\psi$. For a Gaussian policy, $f_\psi(\varepsilon; s) = \mu_\psi(s) + \sigma_\psi(s) \odot \varepsilon$. Actions in continuous spaces must be bounded, so SAC applies a $\tanh$ squashing function: $\tilde{A} = \tanh(f_\psi(\varepsilon; S))$, with the log-probability corrected for the change of variables.

The reparameterization makes the gradient of the expectation over actions exact rather than estimated: since $\varepsilon$ does not depend on $\psi$, the gradient passes directly through $f_\psi$ via backpropagation:

$$\nabla_\psi \mathbb{E}_{A \sim \pi_\psi}\bigl[\alpha \log \pi_\psi(A \mid S) - Q_\phi(S, A)\bigr] = \mathbb{E}_{\varepsilon \sim \mathcal{N}}\!\Bigl[\nabla_\psi\bigl(\alpha \log \pi_\psi(f_\psi(\varepsilon; S) \mid S) - Q_\phi(S, f_\psi(\varepsilon; S))\bigr)\Bigr].$$

The gradient is low-variance because it requires no importance weights and no Monte Carlo return: one sampled action per state is sufficient for an accurate gradient estimate. This is the practical reason SAC learns so efficiently — the policy gradient has the variance of a supervised gradient rather than the variance of a policy gradient estimator.

## Automatic Temperature Tuning

The temperature $\alpha$ controls the entropy-reward tradeoff and, in the original SAC formulation, required manual tuning for each task. Haarnoja, Zhou, Hartikainen, Tucker, Ha, Tan, Kumar, Zhu, Gupta, Abbeel, and Levine (2018) derived an automatic procedure by formulating temperature selection as a constrained optimization problem:

$$\max_\pi J_{\text{MaxEnt}}(\pi) \quad \text{subject to} \quad \mathbb{E}_{(S, A) \sim \rho_\pi}\!\bigl[-\log \pi(A \mid S)\bigr] \geq \mathcal{H}_{\text{target}},$$

where $\mathcal{H}_{\text{target}}$ is a minimum entropy constraint — a target entropy level that the policy must maintain. A natural choice for continuous actions with $|\mathcal{A}|$-dimensional action spaces is $\mathcal{H}_{\text{target}} = -|\mathcal{A}|$: require at least one bit of entropy per action dimension.

The Lagrangian relaxation of this constrained problem gives the following update for the dual variable $\alpha$ (the temperature):

$$\mathcal{L}(\alpha) = \mathbb{E}_{A \sim \pi}\!\bigl[-\alpha \log \pi(A \mid S) - \alpha \mathcal{H}_{\text{target}}\bigr].$$

Gradient descent on $\alpha$ increases the temperature when the policy's entropy falls below $\mathcal{H}_{\text{target}}$ and decreases it when the entropy exceeds the target. The policy, Q-functions, and temperature are all updated simultaneously with separate Adam optimizers, and the temperature adapts online throughout training.

Automatic temperature tuning transformed SAC from a method that required task-specific hyperparameter search into one that could be applied across continuous control tasks with a single fixed setting of $\mathcal{H}_{\text{target}}$. Together with the replay buffer (which provides sample efficiency) and the double Q-network (which controls overestimation), automatic $\alpha$ makes SAC a largely self-configuring algorithm for continuous action spaces.

## Off-Policy Learning and Sample Efficiency

The most practically significant difference between SAC and PPO is the treatment of past experience. PPO is on-policy: each batch of transitions is used for a single round of updates and discarded. SAC is off-policy: every transition is stored in a replay buffer of capacity $10^6$ or more, and each gradient update samples a minibatch of 256 transitions uniformly from the full buffer. A transition collected early in training — under a worse policy, in an unexplored part of the state space — contributes to gradient updates thousands of times before eventually being overwritten.

This reuse is possible because the soft Q-function update is a Bellman residual minimization, which is valid for any distribution of transitions, not just the current policy's distribution. The importance of the action taken matters only to the extent that the next-state value in the Bellman target is computed under the current policy — which is handled exactly by sampling $A' \sim \pi_\psi(\cdot \mid S')$ at update time, not by requiring that $A'$ was collected by the current policy. The entropy term in the target is likewise computed under the current policy at update time. Past transitions from different policies are therefore valid data for current Q-function updates, without importance correction.

The practical consequence: SAC on the HalfCheetah locomotion task reaches the performance level that PPO achieves after three million environment steps using fewer than three hundred thousand steps — roughly a 10× improvement in sample efficiency. In environments where interactions are expensive — physical robots, wet-lab experiments — this gap is the difference between a method that is practical and one that is not.

## SAC and PPO as Complementary Algorithms

SAC and PPO are not competing answers to the same question — they are answers to different questions, and the choice between them depends on the setting.

PPO is on-policy, which means its gradient estimates are always consistent with the current policy: no importance weighting, no distribution shift, no stale data. This makes it robust and stable in settings where the environment is cheap to simulate and many parallel workers can collect fresh data continuously: large-scale game-playing, Atari, language model fine-tuning with a fast reward model. PPO's multiple-epoch minibatch updates extract more gradient signal per rollout than single-step on-policy methods, but the fundamental unit of data — the on-policy rollout — is discarded after each round.

SAC is off-policy, which means it can learn from data collected under arbitrarily different policies — including the random policy used at initialization, before any learning has occurred. The replay buffer provides a dense gradient signal from the very first transition, and every collected transition contributes to Q-function improvement indefinitely. This makes SAC preferable whenever environment interaction is expensive: continuous robotics control, tasks with long episode length, settings where parallel simulation is unavailable or costly.

The deeper difference is the policy representation. PPO produces a deterministic policy in the limit — the entropy bonus decays as the policy improves — and uses the entropy term only to slow this convergence. SAC produces a genuinely stochastic optimal policy, with entropy calibrated by the temperature to reflect the reward geometry of the task. For tasks where multiple behaviors are comparably optimal — grasping objects with different grip configurations, locomotion with different gaits — the MaxEnt policy represents this multimodality explicitly. The PPO policy will collapse to one mode.

## The Energy-Based Perspective

The soft Q-function $Q_{\text{soft}}(s, a) / \alpha$ plays the role of a negative energy function, and the optimal policy $\pi^*(a \mid s) \propto \exp(Q_{\text{soft}}(s, a) / \alpha)$ is a Boltzmann distribution over actions. This is not a metaphor: the soft Bellman equations are formally equivalent to the partition function recursion in statistical mechanics, with the discount factor $\gamma$ playing the role of a spatial decay and the entropy bonus playing the role of the temperature's contribution to free energy.

This connection has practical implications. Energy-based models in machine learning — which define distributions proportional to $\exp(-E(x)/T)$ for a learned energy function $E$ — can be trained and sampled using the same infrastructure as SAC's Q-networks. The framework encompasses policy optimization as a special case of energy-based density estimation. And it provides a principled answer to the question of how stochastic a policy should be: not because stochasticity is useful for exploration, but because the optimal answer to the MaxEnt objective is genuinely stochastic when multiple actions are comparably valued, and the temperature controls how comparable they need to be to share probability mass.

This perspective will recur immediately in the next part of the arc. The KL divergence that appears in the optimal soft policy — the policy is the projection onto $\exp(Q/\alpha) / Z$ — is the same KL divergence that appears in RLHF's reward-model objective and in the DPO derivation. Maximum entropy RL is not a detour; it is the theoretical foundation for the alignment methods that follow.

---

*The maximum entropy framework for reinforcement learning and its connection to the Boltzmann distribution is developed in Ziebart, Maas, Bagnell, and Dey (2008), Maximum Entropy Inverse Reinforcement Learning, and extended to the sequential soft Bellman setting in Ziebart (2010), Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy. The energy-based policy formulation is introduced in Haarnoja, Tang, Abbeel, and Levine (2017), Reinforcement Learning with Deep Energy-Based Policies. The Soft Actor-Critic algorithm is introduced in Haarnoja, Zhou, Abbeel, and Levine (2018), Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. Automatic temperature tuning is derived in Haarnoja, Zhou, Hartikainen, Tucker, Ha, Tan, Kumar, Zhu, Gupta, Abbeel, and Levine (2018), Soft Actor-Critic Algorithms and Applications. The double Q-network technique for controlling overestimation in continuous control is adapted from van Hasselt (2010) and applied to the soft setting by Haarnoja et al. (2018). The reparameterization trick for low-variance policy gradients in continuous action spaces is described in Kingma and Welling (2013), Auto-Encoding Variational Bayes, and applied to policy optimization in Heess, Wayne, Silver, Lillicrap, Tassa, and Erez (2015), Learning Continuous Control Policies by Stochastic Value Gradients.*
