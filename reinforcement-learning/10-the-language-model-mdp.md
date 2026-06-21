---
title: "The Language Model MDP"
subtitle: "Token-level sequential decisions, per-token KL shaping, GAE in text generation, and the four-model memory wall"
---


Chapter 09 established the objective — maximize a KL-regularized reward toward the Boltzmann optimal policy — and the three-stage pipeline that pursues it. This chapter opens the black box of Stage 3. Applying PPO to a language model is not a straightforward substitution. The MDP structure of text generation is unusual in ways that require non-trivial adaptations to the algorithm, and the engineering choices made in those adaptations — how to distribute the reward signal across tokens, how to parameterize the value function, how to whiten advantages, which models must be resident in memory simultaneously — determine whether RLHF produces an aligned policy or incoherent, reward-hacked outputs.

The core challenge is a mismatch of granularities. PPO was designed for MDPs where the agent takes one action per state transition and may receive a reward at each step. In a language model, generating a response involves taking hundreds of actions — choosing one token at a time from a vocabulary of $10^4$ to $10^5$ — while the reward model evaluates only the complete response. The reward signal is episodic and terminal; the MDP dynamics are dense and sequential. Bridging these two timescales requires a specific set of choices that have become standard practice but are not forced by the theory.

---

## The Token-Level MDP

> [!info]
> A language model generating a response to a prompt is cast as an MDP at the level of individual tokens:
>
> - **State**: $S_t = (x, y_1, y_2, \ldots, y_t)$ — the prompt $x$ concatenated with all tokens generated so far.
> - **Action**: $A_t = y_{t+1}$ — the next token, drawn from the vocabulary $\mathcal{V}$ with $|\mathcal{V}| \approx 32{,}000$–$100{,}000$.
> - **Transition**: deterministic. Taking action $y_{t+1}$ in state $S_t$ always produces $S_{t+1} = (x, y_1, \ldots, y_{t+1})$. The only randomness comes from the policy's sampling of the next token.
> - **Episode**: ends when the model generates a special end-of-sequence token, or when a maximum response length $T$ is reached.
> - **Policy**: $\pi_\theta(y_{t+1} \mid S_t)$ — the language model itself. The conditional distribution over the next token given all preceding context. Policy gradient updates modify the model weights to increase the probability of token sequences leading to high reward.

> [!note]
> The state $S_t$ grows with each token generated — the sequence is always fully observed, so the MDP is Markovian in the exact sense. This is unusual: in robotics, the state is a fixed-dimensional vector; here, the state space has variable and growing dimensionality. The language model handles this via its context window: the transformer architecture encodes the full history $S_t$ as a fixed-size internal representation. The Markov property holds because the transformer's attention mechanism conditions on the entire prefix — the distribution over the next token depends only on the current state $S_t$, which contains all context.

---

## Terminal Reward and the Credit Assignment Problem

> [!question]
> The reward model evaluates only the complete response. Every intermediate token contributes zero intermediate reward. What does this imply for learning, and how is it addressed?

A reward of zero for all intermediate steps and $r_\phi(x, y)$ only at the end is technically valid — the Bellman equations still hold — but practically catastrophic. The advantage estimator must propagate a single terminal signal backward through hundreds of token positions. With a horizon of $T = 500$ tokens and a single reward at the end, the variance of the Monte Carlo return for early tokens is enormous: the return depends on the entire subsequent trajectory, which is a long sum of random transitions under a stochastic policy.

> [!tip]
> **The per-token KL penalty** simultaneously resolves two problems: it densifies the reward signal, providing gradient information at every token step, and it enforces the proximity constraint from Chapter 09 as a natural consequence of reward shaping.
>
> At each step $t$, the per-token KL contribution is:
> $$\text{KL}_t = \log \frac{\pi_\theta(y_t \mid S_{t-1})}{\pi_{\text{ref}}(y_t \mid S_{t-1})}.$$
> This is the log-ratio for the specific token $y_t$ that was sampled — a **single-sample, unbiased estimate** of the KL divergence contribution at step $t$, requiring only one forward pass through both $\pi_\theta$ and $\pi_{\text{ref}}$.
>
> The **shaped reward** at each step:
> $$r_t = \begin{cases} -\beta \, \text{KL}_t & t < T \\ r_\phi(x, y) - \beta \, \text{KL}_T & t = T \end{cases}$$
> distributes the KL cost across the sequence: every token contributes a small negative reward proportional to how far the new policy has moved from the reference. Steps where $\pi_\theta$ and $\pi_{\text{ref}}$ agree receive near-zero intermediate reward; steps where they diverge incur a penalty. The terminal reward model score is added only at the final token.

> [!success]
> **The decomposition is exact, not approximate.** The expected per-token KL penalties sum to the full sequence-level KL divergence:
> $$\mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=1}^T \text{KL}_t\right] = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\log \frac{\pi_\theta(y_{1:T} \mid x)}{\pi_{\text{ref}}(y_{1:T} \mid x)}\right] = \text{KL}\!\bigl[\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\bigr],$$
> where the second equality uses the chain rule of KL over a sequence of conditionals. The shaped reward is equivalent to the KL-penalized objective of Chapter 09 — it distributes it across timesteps but preserves the total penalty exactly.

---

## The Value Function in the Language Setting

> [!question]
> PPO requires a value function $V_w(S_t)$ estimating the expected future reward from each state. In text generation, what must this function predict, and how is it parameterized?

The value function $V_w(S_t)$ must estimate the expected future shaped reward from state $S_t = (x, y_{1:t})$ onward — the expected sum of remaining per-token KL penalties plus the terminal reward model score. This is a function that maps a partial response to a scalar representing how much total reward the episode is expected to yield from that point.

> [!note]
> The value function is typically implemented as a separate neural network initialized from the SFT model, with a scalar output head replacing the vocabulary projection. An alternative is a shared backbone with the policy and a separate value head — the actor-critic architecture of Chapter 06. In the language model setting:
>
> - **Separate networks**: cleaner gradient separation — the value loss cannot corrupt the policy's representations. Doubles memory cost.
> - **Shared backbone**: the dense critic gradient signal (from every token at every step) improves the shared feature extractor, benefiting the policy. Requires careful coefficient tuning ($c_1 \approx 0.5$) to prevent the value loss from dominating.
>
> In practice, large-scale implementations (7B+ models) typically use separate networks, accepting the memory cost in exchange for training stability.

> [!warning]
> **The value function faces an unusually hard prediction problem.** In robotics, the value function estimates future discounted reward from a state that changes meaningfully at every step. In the token-level MDP, since the only non-trivial reward is the terminal reward model score, the value at position $t$ is essentially the expected final reward given the tokens generated so far — a function that is hard to estimate accurately from prefixes, because response quality is not in general predictable from its first half. A value network trained on terminal rewards learns a slow, noisy approximation, and the advantage estimates it produces carry this noise through every token in every response.
>
> This is one of the motivations for GRPO: by eliminating the value function entirely and computing advantages from group rollouts, the noisy approximation error of the critic disappears from the gradient estimator.

---

## Advantage Estimation and Whitening

The per-token shaped rewards and value estimates combine to produce token-level advantage estimates via GAE:

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}, \qquad \delta_t = r_t + \gamma V_w(S_{t+1}) - V_w(S_t).$$

> [!note]
> **Key adaptation for language generation**: $\gamma = 1$ is standard. Future tokens in the same response are not discounted, since episodes are short (hundreds of tokens at most) and there is no meaningful temporal discounting within a single generation. The $\lambda$ parameter for GAE follows the same values as in continuous control — typically $\lambda = 0.95$ — providing the same bias-variance balance discussed in Chapter 06.
>
> One additional step nearly universal in language model RL: **advantage whitening**. Normalize advantages within a minibatch to have zero mean and unit variance before computing the policy gradient:
> $$\hat{A}_t \leftarrow \frac{\hat{A}_t - \mu_{\hat{A}}}{\sigma_{\hat{A}} + \varepsilon}.$$
> Whitening stabilizes training by preventing gradient magnitude from varying dramatically as the reward scale and KL penalty evolve during training. It also acts as a soft baseline: after whitening, advantages above zero represent above-average tokens and below zero represent below-average tokens, regardless of the absolute reward scale. Included by default in essentially all language model RL implementations.

---

## The PPO Objective at Token Level

The importance ratio at token $t$:

$$r_t(\theta) = \frac{\pi_\theta(y_t \mid S_{t-1})}{\pi_{\theta_{\text{old}}}(y_t \mid S_{t-1})},$$

where $\theta_{\text{old}}$ are the parameters at the start of the current PPO epoch. The clipped objective, summed over all token positions in the minibatch:

$$\mathcal{L}_{\text{policy}}(\theta) = \mathbb{E}_{t}\!\Bigl[\min\bigl(r_t(\theta)\,\hat{A}_t,\; \text{clip}(r_t(\theta),\, 1 - \varepsilon,\, 1 + \varepsilon)\,\hat{A}_t\Bigr)\Bigr].$$

> [!tip]
> **The KL penalty and the clip operate at different timescales — both are necessary:**
> - The **clip** prevents large policy changes *between PPO epochs* within a single training step (timescale: minutes).
> - The **KL penalty** embedded in the shaped reward prevents the policy from drifting far from $\pi_{\text{ref}}$ *over the course of all training* (timescale: days).
>
> Omitting the clip allows a single rollout batch to destabilize the policy. Omitting the KL penalty allows the policy to exploit the reward model over many training steps. Neither alone is sufficient.

---

## The Four-Model Memory Wall

> [!warning]
> **The central computational constraint of PPO-based RLHF**: at any given point during training, four large neural networks must be accessible:
>
> | Model | Role | Updated? | Memory at 7B params (bf16) |
> |-------|------|----------|---------------------------|
> | $\pi_\theta$ — online policy | generates rollouts, receives gradient | Yes | ~14 GB weights + gradients + optimizer |
> | $\pi_{\theta_{\text{old}}}$ — rollout policy | denominator of importance ratio | Frozen per epoch | ~14 GB |
> | $\pi_{\text{ref}}$ — reference policy | KL penalty denominator | Never | ~14 GB (quantized: ~4–7 GB) |
> | $V_w$ — value model | advantage estimation | Yes | ~14 GB weights + gradients |
>
> For a 7B model in bf16: the four models together consume **$\approx$56 GB** of GPU memory in weights alone, before accounting for activations, gradients, and optimizer states. At 70B scale, this is infeasible on any single node. Practical implementations use:
> - 4-bit or 8-bit quantization for $\pi_{\text{ref}}$ (forward passes only, no gradient required)
> - LoRA for $\pi_\theta$ (reduces trainable parameters to $\ll 1\%$ of total, reducing optimizer state from $\sim$3$\times$ weight size to negligible)
> - Gradient checkpointing for the value model (trades compute for memory)
>
> The algorithmic content is unchanged by these optimizations, but the infrastructure required to execute it at scale is substantial. This memory pressure is the single most direct engineering motivation for GRPO, which eliminates $V_w$ entirely, and for DPO, which eliminates the entire RL loop and the $\pi_{\theta_{\text{old}}}$ requirement.

---

## The Rollout Phase and the Learning Phase

> [!note]
> **Rollout phase**: the current policy $\pi_\theta$ generates complete responses. For each prompt $x$ in a batch, sample $y \sim \pi_\theta(\cdot \mid x)$ autoregressively, compute $r_\phi(x, y)$, and compute per-token log-probabilities under both $\pi_\theta$ and $\pi_{\text{ref}}$. Store the full trajectory — tokens, log-probabilities, shaped rewards, value estimates — in a rollout buffer.
>
> **Learning phase**: run $M$ epochs of minibatch PPO updates on the rollout buffer. For each minibatch, recompute the current policy's log-probabilities (fresh importance ratios), compute advantages via GAE, apply the clipped objective, update $\theta$ and $w$.
>
> The critical constraint: the rollout phase requires $\pi_\theta$, $\pi_{\text{ref}}$, and $V_w$ simultaneously (for log-probabilities and value estimates). The learning phase requires $\pi_\theta$ and $V_w$ for gradients, plus $\pi_{\theta_{\text{old}}}$ for importance ratios. Different phases have different memory profiles, and implementations typically offload or quantize inactive models between phases.

---

## The Alignment Tax

> [!warning]
> A consistent empirical finding across RLHF implementations: **the alignment tax**. Improving performance on the alignment objective — instruction following, refusing harmful requests, matching human preferences — tends to degrade performance on capability benchmarks not represented in the preference dataset. A model fine-tuned with RLHF to be more helpful in conversation may score lower on coding, mathematics, or factual recall that it handled well before fine-tuning.
>
> The mechanism is parameter drift: gradient updates improving the RLHF objective shift parameters away from their SFT values, and those values encoded capabilities the RLHF objective did not reward. The KL penalty against $\pi_{\text{ref}}$ mitigates but does not eliminate this — a model can remain close to $\pi_{\text{ref}}$ in KL while shifting probability mass in ways that affect specific capabilities.
>
> InstructGPT reported the largest degradations on tasks requiring precise factual recall — suggesting RLHF shifts probability mass toward more fluent but occasionally less accurate alternatives. The severity depends on $\beta$ and the diversity of the preference dataset; high $\beta$ reduces the tax at the cost of limiting alignment improvement.

---

## Practical Diagnostics

> [!note]
> Standard monitoring signals for language model RL training:
> - **Mean KL divergence** $\mathbb{E}[\sum_t \text{KL}_t]$: should increase gradually. Values above 10–20 nats typically signal reward hacking.
> - **Reward model score**: should increase and track human evaluation improvements.
> - **Policy entropy**: should decrease slightly — increasing confidence — but not collapse. Entropy collapse indicates reward hacking into a few stereotyped response patterns.
> - **Response length distribution**: unusual drift toward very long or very short responses can indicate optimization artifacts.
>
> Stopping criteria are heuristic. Training is typically halted when: the mean KL reaches a predetermined budget; a held-out human evaluation shows no further improvement; or the reward model score increases while human evaluations plateau — the signature that the policy has begun exploiting $r_\phi$ rather than genuinely improving.

---

## What Token-Level PPO Gets Right — and What It Motivates Next

> [!success]
> The adaptation of PPO to the token-level MDP is well-matched to the structure of language generation:
> - The **clipped objective** prevents single high-reward responses from dominating the gradient.
> - The **per-token KL penalty** provides dense gradient signal and prevents reward hacking via implausible token sequences.
> - The **GAE advantage estimates**, computed token by token, identify which parts of a response contributed to high or low reward — imperfectly, but enough to drive meaningful learning.
> - **Multiple epochs** on the same rollout batch extract far more gradient signal per environment interaction than single-step methods.

> [!warning]
> **What PPO-based RLHF does not address:**
>
> 1. **The value function quality ceiling.** The critic predicts terminal reward from partial responses — a hard prediction problem. Poor critic estimates introduce bias into advantage estimates that persists throughout training.
>
> 2. **The four-model memory cost.** At large model scale, the infrastructure required is substantial, and the value model contributes roughly 25% of the total memory footprint without a corresponding 25% contribution to learning quality.
>
> 3. **The on-policy constraint.** Every rollout must come from the current policy and is discarded after a single round of updates. There is no mechanism for reusing past rollouts as in SAC's replay buffer.
>
> These three limitations together motivate the algorithms of the next three chapters. **GRPO** (Chapter 12) eliminates the value function and computes advantages analytically from groups of rollouts, removing limitation 1 and partially addressing limitation 2. **DPO** (Chapter 11) bypasses the RL loop entirely by reparameterizing the optimal policy directly in terms of the preference data, eliminating all four models in favor of a single supervised objective. **Dr. GRPO** and related methods (Chapter 12) correct the normalization biases that arise when group-relative advantages replace the critic.

---

*The token-level MDP formulation and per-token KL penalty are introduced in Ziegler, Stiennon, Wu, Brown, Radford, Amodei, Christiano, and Irving (2019), Fine-Tuning Language Models from Human Preferences, and refined in Stiennon et al. (2020), Learning to Summarize with Human Feedback. The full InstructGPT system, including the engineering of PPO at large model scale and the alignment tax observations, is in Ouyang et al. (2022), Training Language Models to Follow Instructions with Human Feedback. Advantage whitening as a stabilization technique is standard in open implementations including TRL (von Werra et al., 2020) and trlX (Havrilla et al., 2023). The reward-KL overoptimization curve and stopping criteria are characterized in Gao, Hilton, Mićo, Askell, Ritter, Leike, Schulman, and Christiano (2022), Scaling Laws for Reward Model Overoptimization. LoRA for parameter-efficient fine-tuning is from Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, and Chen (2022), LoRA: Low-Rank Adaptation of Large Language Models.*
