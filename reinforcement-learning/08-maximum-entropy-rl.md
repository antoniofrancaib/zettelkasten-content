---
title: "Maximum Entropy Reinforcement Learning"
subtitle: "Entropy as a co-equal objective, soft Bellman equations, SAC, and the Boltzmann connection to RLHF"
---


The policy gradient methods of the preceding chapters — REINFORCE, actor-critic, TRPO, PPO — share a common objective: maximize expected cumulative reward. Entropy appears in these methods as an optional regularizer, added as a small bonus term to prevent premature determinism. It is useful but secondary: a correction applied to an objective whose primary meaning is reward.

The **maximum entropy** framework inverts this priority. It treats entropy not as a regularizer bolted onto reward maximization but as a co-equal objective in its own right, part of what the agent is supposed to be doing. The shift is philosophical before it is algorithmic, and it produces a family of methods with properties — off-policy stability, robustness to reward misspecification, natural multimodality, and a precise connection to the alignment methods of the next four chapters — that standard on-policy methods do not share.

The roots lie in Ziebart, Maas, Bagnell, and Dey (2008), who modeled human behavior as approximately optimal under a stochastic policy maximizing reward plus entropy. The trajectory distribution that maximizes this augmented objective is a **Boltzmann distribution** over rewards — the same distribution that appears in statistical mechanics as the canonical ensemble. Ziebart (2010) extended this to the sequential setting, deriving soft Bellman equations that carry the entropy term through the dynamic programming recursion. The practical algorithm, Soft Actor-Critic (SAC), arrived in 2018 and became the dominant method for continuous control.

---

## The Maximum Entropy Objective

> [!question]
> Standard RL maximizes expected reward. What does it mean to treat entropy as an objective rather than a regularizer, and what behavior does this produce?

Standard RL maximizes:

$$J_{\text{RL}}(\pi) = \mathbb{E}_{\tau \sim \pi}\!\left[\sum_{t=0}^{T} \gamma^t R(S_t, A_t)\right].$$

The **MaxEnt objective** augments every time step's reward with the policy entropy at that state:

$$J_{\text{MaxEnt}}(\pi) = \mathbb{E}_{\tau \sim \pi}\!\left[\sum_{t=0}^{T} \gamma^t \bigl(R(S_t, A_t) + \alpha \, H\!\bigl(\pi(\cdot \mid S_t)\bigr)\bigr)\right],$$

where $H(\pi(\cdot \mid s)) = -\mathbb{E}_{A \sim \pi(\cdot \mid s)}[\log \pi(A \mid s)]$ is the policy entropy at state $s$, and $\alpha > 0$ is the **temperature** governing the entropy-reward tradeoff.

> [!tip]
> **What entropy as objective means in behavior.** The entropy bonus does not decay as the policy improves — it is a perpetual objective. The optimal MaxEnt policy is genuinely stochastic, with stochasticity calibrated to the reward geometry of the task:
>
> - When two actions have *identical* Q-values, the optimal MaxEnt policy assigns them equal probability.
> - When one action is substantially better, its probability is higher — but only by as much as the temperature $\alpha$ permits.
> - At $\alpha \to 0$: standard greedy behavior, probability concentrates on the best action.
> - At $\alpha \to \infty$: uniform distribution regardless of rewards.
>
> The temperature thus controls not just exploration rate but how much reward advantage is required for one action to dominate another. This is fundamentally different from the entropy bonus in PPO, which is a heuristic nudge toward exploration — in MaxEnt, entropy is part of what being optimal means.

---

## The Soft Bellman Equations

> [!question]
> How does the entropy term propagate through the Bellman recursion? Does adding it per-step change the fixed-point equations?

Define the **soft Q-function** — expected return under the MaxEnt objective when action $a$ is taken in state $s$ and the optimal policy is followed thereafter:

$$Q_{\text{soft}}(s, a) = R(s, a) + \gamma \, \mathbb{E}_{S' \sim p(\cdot \mid s,a)}\!\bigl[V_{\text{soft}}(S')\bigr].$$

The **soft value function** absorbs the entropy term:

$$V_{\text{soft}}(s) = \mathbb{E}_{A \sim \pi}\!\bigl[Q_{\text{soft}}(s, A)\bigr] + \alpha H\!\bigl(\pi(\cdot \mid s)\bigr) = \mathbb{E}_{A \sim \pi}\!\bigl[Q_{\text{soft}}(s, A) - \alpha \log \pi(A \mid s)\bigr].$$

> [!note]
> The entropy compounds through the discounted sum exactly as reward compounds. Every step of the recursion carries it forward. A policy that maintains high entropy for many steps accumulates entropy as value, reflected in higher soft Q-values for states from which stochastic behavior remains possible.
>
> For the optimal policy, the soft value function has a closed form. Setting the derivative with respect to $\pi(\cdot \mid s)$ to zero subject to normalization gives:
> $$V_{\text{soft}}^*(s) = \alpha \log \int_{\mathcal{A}} \exp\!\left(\frac{Q_{\text{soft}}(s, a)}{\alpha}\right) da.$$
> This is the **log-sum-exp** (log-partition function) of $Q_{\text{soft}} / \alpha$ over actions — the free energy of a Boltzmann distribution with inverse temperature $1/\alpha$. The hard $\max_a Q(s,a)$ of standard RL is replaced by a **soft maximum**: a smooth, everywhere-differentiable function that approaches $\max_a Q(s,a)$ as $\alpha \to 0$.

---

## The Optimal Soft Policy: A Boltzmann Distribution

> [!tip]
> The optimal policy under the MaxEnt objective is a **Boltzmann distribution** over Q-values:
> $$\pi^*(a \mid s) = \frac{\exp\bigl(Q_{\text{soft}}(s, a) / \alpha\bigr)}{\int_{\mathcal{A}} \exp\bigl(Q_{\text{soft}}(s, a') / \alpha\bigr) da'} = \exp\!\left(\frac{Q_{\text{soft}}(s, a) - V_{\text{soft}}(s)}{\alpha}\right).$$
>
> This is the canonical distribution of statistical mechanics: Q-values play the role of negative energies, high-Q actions receive high probability, and $\alpha$ controls the sharpness of the distribution.

The optimal soft policy can be re-expressed as a KL minimization. For any policy $\pi$, define:

$$\text{KL}\!\left[\pi(\cdot \mid s) \;\Big\|\; \frac{\exp(Q_\text{soft}(s,\cdot)/\alpha)}{Z(s)}\right].$$

The optimal soft policy minimizes this KL divergence to zero — it is the projection of the policy distribution onto the Boltzmann manifold.

> [!tip]
> **This structure is the theoretical foundation for RLHF.** The KL-regularized RLHF objective from Chapter 09:
> $$\max_\pi \; \mathbb{E}[r_\phi(x, y)] - \beta\, \text{KL}[\pi \,\|\, \pi_{\text{ref}}]$$
> has the same optimal solution form:
> $$\pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x) \exp\bigl(r_\phi(x, y) / \beta\bigr).$$
> This is precisely the Boltzmann distribution of the MaxEnt framework, with the reward model $r_\phi$ playing the role of the Q-function and the reference policy $\pi_{\text{ref}}$ playing the role of a prior. **Language model alignment is maximum entropy RL with the energy function defined by a learned reward model.** The theoretical framework and the alignment methods converge on the same fixed point.

---

## The Soft Actor-Critic Architecture

> [!info]
> **SAC** maintains three function approximators:
> - Two **soft Q-networks** $Q_{\phi_1}(s, a)$ and $Q_{\phi_2}(s, a)$, trained independently
> - A **policy network** $\pi_\psi(a \mid s)$ outputting a squashed Gaussian distribution over actions
> - A **replay buffer** $\mathcal{D}$ storing all collected transitions
>
> The two Q-networks address overestimation bias (Chapter 04): a single Q-network's positive noise inflates bootstrap targets, which inflates Q-values, which further inflates subsequent targets. SAC uses the **minimum** of both networks to form the Bellman target:
> $$y = R + \gamma \, \mathbb{E}_{A' \sim \pi_\psi}\!\Bigl[\min\bigl(Q_{\phi_1}(S', A'),\, Q_{\phi_2}(S', A')\bigr) - \alpha \log \pi_\psi(A' \mid S')\Bigr].$$
> The $-\alpha \log \pi_\psi$ term is the per-sample entropy contribution of the next state, entering the target as the negative log-probability of the sampled next action.

### The Reparameterization Trick

The policy update maximizes the expected soft Q-value minus the expected log-probability:

$$\mathcal{L}_\pi(\psi) = \mathbb{E}_{S \sim \mathcal{D}}\!\Bigl[\mathbb{E}_{A \sim \pi_\psi(\cdot \mid S)}\!\bigl[\alpha \log \pi_\psi(A \mid S) - Q_{\phi}(S, A)\bigr]\Bigr].$$

Computing $\nabla_\psi$ of this naively via the log-derivative trick (REINFORCE) introduces high variance. SAC avoids this entirely.

> [!note]
> **The reparameterization trick**: express the policy as a deterministic function of a noise variable:
> $$A = f_\psi(\varepsilon;\, S) = \tanh\!\bigl(\mu_\psi(S) + \sigma_\psi(S) \odot \varepsilon\bigr), \quad \varepsilon \sim \mathcal{N}(0, I).$$
> Since $\varepsilon$ does not depend on $\psi$, the gradient of the expectation over actions passes directly through $f_\psi$ via backpropagation:
> $$\nabla_\psi \mathbb{E}_{A \sim \pi_\psi}[\cdots] = \mathbb{E}_{\varepsilon \sim \mathcal{N}}\!\Bigl[\nabla_\psi\bigl(\alpha \log \pi_\psi(f_\psi(\varepsilon; S) \mid S) - Q_\phi(S, f_\psi(\varepsilon; S))\bigr)\Bigr].$$
> One sampled action per state suffices for an accurate gradient estimate — no importance weights, no Monte Carlo return. The policy gradient has the variance of a supervised gradient rather than a stochastic gradient estimator. This is the practical reason SAC learns so efficiently.

### Automatic Temperature Tuning

> [!note]
> Choosing $\alpha$ manually requires task-specific tuning. Haarnoja et al. (2018) derived an automatic procedure: formulate temperature selection as a constrained optimization problem — maximize $J_{\text{MaxEnt}}(\pi)$ subject to $\mathbb{E}[-\log \pi(A \mid S)] \geq \mathcal{H}_{\text{target}}$, where $\mathcal{H}_{\text{target}} = -|\mathcal{A}|$ (one bit of entropy per action dimension) is a natural target.
>
> The Lagrangian dual variable for this constraint is $\alpha$ itself. The dual update:
> $$\mathcal{L}(\alpha) = \mathbb{E}_{A \sim \pi}\!\bigl[-\alpha \log \pi(A \mid S) - \alpha \mathcal{H}_{\text{target}}\bigr]$$
> increases $\alpha$ when policy entropy falls below target and decreases it when entropy exceeds target. Policy, Q-functions, and temperature are all updated simultaneously via separate Adam optimizers. SAC becomes largely self-configuring: a single fixed $\mathcal{H}_{\text{target}}$ works across a wide range of continuous control tasks.

---

## Off-Policy Learning and Sample Efficiency

> [!question]
> PPO discards every batch of transitions after a round of updates. Can MaxEnt RL justify reusing past transitions — and does this produce meaningful efficiency gains?

The soft Q-function update is a Bellman residual minimization valid for *any* distribution of transitions, not just the current policy's. The next-state value in the Bellman target is computed under the *current* policy at update time — by sampling $A' \sim \pi_\psi(\cdot \mid S')$ — not by requiring that $A'$ was collected by the current policy. Past transitions from arbitrarily different policies are valid data for Q-function updates, without importance correction.

> [!success]
> **The sample efficiency gap.** SAC on HalfCheetah locomotion reaches the performance level that PPO achieves after 3 million environment steps using fewer than 300,000 — roughly a $10\times$ improvement. In environments where interactions are expensive — physical robots, wet-lab experiments — this gap is the difference between a practical method and an impractical one.
>
> The replay buffer provides a dense gradient signal from the very first transition: every collected transition contributes to Q-function improvement indefinitely, whereas PPO's on-policy rollout is discarded after a single round of updates.

---

## SAC and PPO as Complementary Algorithms

> [!tip]
> SAC and PPO are not competing answers to the same question — they are answers to different questions:
>
> | | PPO | SAC |
> |-|-----|-----|
> | **Data regime** | On-policy (fresh rollouts) | Off-policy (replay buffer) |
> | **Environment cost** | Cheap simulation, many workers | Expensive interaction |
> | **Policy in the limit** | Near-deterministic (entropy decays) | Genuinely stochastic (calibrated by $\alpha$) |
> | **Multimodality** | Collapses to one mode | Represents multiple comparable modes |
> | **Gradient variance** | Log-derivative trick (higher) | Reparameterization trick (lower) |
> | **Dominant setting** | Game-playing, LLM post-training | Continuous robotics, real-world control |

> [!note]
> The deeper difference is the **policy representation**. PPO produces a near-deterministic policy in the limit — the entropy bonus merely slows the collapse. SAC produces a genuinely stochastic optimal policy: for tasks where multiple behaviors are comparably optimal (grasping with different grip configurations, locomotion with different gaits), the MaxEnt policy represents this multimodality explicitly and does not arbitrarily collapse to a single mode. Whether multimodality in the policy is desirable depends on the task — for single-query response generation, collapsing to the best mode is often correct; for exploration-intensive robotics, preserving multimodality is important.

---

## The Energy-Based Perspective

> [!info]
> The soft Q-function $Q_{\text{soft}}(s, a) / \alpha$ is formally a **negative energy function**, and the optimal policy $\pi^*(a \mid s) \propto \exp(Q_{\text{soft}}(s, a) / \alpha)$ is a Gibbs distribution. The soft Bellman equations are the partition function recursion of statistical mechanics — with discount factor $\gamma$ playing the role of spatial decay and the entropy bonus playing the role of the temperature's free-energy contribution.
>
> The log-partition function $V^*_{\text{soft}}(s) = \alpha \log Z(s)$ is the free energy of the action distribution. Minimizing KL to the Boltzmann manifold is the variational free energy principle. Policy optimization in the MaxEnt framework is thermodynamic inference.

> [!tip]
> This perspective unifies several concepts that have appeared across this document:
> - The entropy bonus in A3C/PPO — a heuristic in those frameworks — is here derived as the necessary consequence of including entropy in the objective.
> - The KL regularization in RLHF — introduced in Chapter 09 as a pragmatic guard against reward hacking — is here shown to be the Lagrange multiplier enforcing the MaxEnt constraint.
> - The DPO derivation (Chapter 11) collapses the reward model and the reference policy into a single reparameterization of the Boltzmann optimal policy. The collapse is possible because the Boltzmann structure is known from MaxEnt theory.
>
> Maximum entropy RL is not a detour from the main arc — it is the theoretical substrate that the alignment methods of Chapters 09–12 are all built on top of.

---

*The maximum entropy framework for RL and its connection to the Boltzmann distribution is developed in Ziebart, Maas, Bagnell, and Dey (2008), Maximum Entropy Inverse Reinforcement Learning, and extended to the sequential soft Bellman setting in Ziebart (2010), Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy. The energy-based policy formulation is in Haarnoja, Tang, Abbeel, and Levine (2017), Reinforcement Learning with Deep Energy-Based Policies. The Soft Actor-Critic algorithm is in Haarnoja, Zhou, Abbeel, and Levine (2018), Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. Automatic temperature tuning is derived in Haarnoja, Zhou, Hartikainen, Tucker, Ha, Tan, Kumar, Zhu, Gupta, Abbeel, and Levine (2018), Soft Actor-Critic Algorithms and Applications. The reparameterization trick is from Kingma and Welling (2013), Auto-Encoding Variational Bayes, applied to policy gradients in Heess, Wayne, Silver, Lillicrap, Tassa, and Erez (2015), Learning Continuous Control Policies by Stochastic Value Gradients.*
