---

title: "PPO for Token Generation"
subtitle: "The token-level MDP, per-token KL penalties, value function design, and the engineering of language model alignment"

---


The RLHF pipeline of the previous chapter described the three-stage recipe in broad strokes: supervised fine-tuning, reward model training, RL optimization with PPO. The third stage was treated as a black box — PPO maximizes the KL-penalized reward, producing the aligned policy. This chapter opens that box. Applying PPO to a language model is not a straightforward substitution. The MDP structure of text generation is unusual in ways that require non-trivial adaptations to the algorithm, and the engineering choices made in those adaptations — how to distribute the reward signal across tokens, how to parameterize the value function, how to whiten advantages, when to stop training — determine whether RLHF works or produces incoherent, reward-hacked outputs.

The core challenge is a mismatch of granularities. PPO was designed for MDPs where the agent takes one action per state transition and receives a reward at each step. In a language model, generating a response involves taking hundreds of actions — choosing one token at a time from a vocabulary of $10^4$ to $10^5$ — while the reward model evaluates only the complete response. The reward signal is episodic and terminal, but the MDP dynamics are dense and sequential. Bridging these two timescales requires a specific set of choices that have become standard practice but are not forced by the theory.

## The Token-Level MDP

A language model generating a response to a prompt can be cast as an MDP at the level of individual tokens. The **state** at step $t$ is the concatenation of the prompt $x$ and all tokens generated so far:

$$S_t = (x, y_1, y_2, \ldots, y_t),$$

where $y_1, \ldots, y_t$ are the tokens generated in the first $t$ steps. The **action** $A_t = y_{t+1}$ is the next token, drawn from the vocabulary $\mathcal{V}$ with $|\mathcal{V}|$ typically between 32,000 and 100,000. The **transition** is deterministic: taking action $y_{t+1}$ in state $S_t$ always produces state $S_{t+1} = (x, y_1, \ldots, y_{t+1})$. There is no environmental stochasticity — the only randomness comes from the policy's sampling of the next token.

The **episode** ends when the model generates a special end-of-sequence token, or when a maximum response length $T$ is reached. The reward model $r_\phi(x, y)$ evaluates the complete response $y = (y_1, \ldots, y_T)$ and produces a scalar — but this scalar is assigned only at the terminal step. At every intermediate step, the reward is zero (before subtracting the per-token KL penalty, discussed below). The episode is a single trajectory from the blank slate to the complete response, and the return is dominated by the single terminal reward signal.

The **policy** $\pi_\theta(y_{t+1} \mid S_t)$ is the language model itself: the conditional probability distribution over the next token given all preceding context. The policy parameters $\theta$ are the language model weights. Policy gradient updates will modify these weights to increase the probability of token sequences that lead to high reward model scores.

## Per-Token KL Penalties

A reward of zero for all intermediate steps and $r_\phi(x, y)$ only at the end is technically valid but practically problematic: the credit assignment problem over a sequence of hundreds of steps with a single terminal reward produces extremely high-variance advantage estimates. The per-token KL penalty both addresses this problem and enforces the proximity constraint from Chapter 11.

At each token step $t$, the policy generates token $y_t$ with probability $\pi_\theta(y_t \mid S_{t-1})$ while the reference policy assigns probability $\pi_{\text{ref}}(y_t \mid S_{t-1})$. The **per-token KL contribution** is:

$$\text{KL}_t = \log \frac{\pi_\theta(y_t \mid S_{t-1})}{\pi_{\text{ref}}(y_t \mid S_{t-1})}.$$

Note that this is the log-ratio for the specific token $y_t$ that was sampled — a single-sample estimate of the KL divergence at step $t$, not the full KL $\sum_{v \in \mathcal{V}} \pi_\theta(v \mid S_{t-1}) \log(\pi_\theta(v \mid S_{t-1}) / \pi_{\text{ref}}(v \mid S_{t-1}))$ over all tokens in the vocabulary. The single-sample estimate is unbiased and cheap to compute: it requires one forward pass through both $\pi_\theta$ and $\pi_{\text{ref}}$ to obtain the log-probabilities of the sampled token.

The **shaped reward** at each step combines the terminal reward with the per-token KL:

$$r_t = \begin{cases} -\beta \, \text{KL}_t & t < T \\ r_\phi(x, y) - \beta \, \text{KL}_T & t = T. \end{cases}$$

The per-token KL penalty converts the purely terminal reward signal into a dense signal: every token contributes a small negative reward proportional to how far the new policy has moved from the reference. Steps where $\pi_\theta$ has moved far from $\pi_{\text{ref}}$ incur a larger penalty; steps where the two policies agree receive near-zero intermediate reward. This distributes the KL cost across the sequence, providing gradient signal at every token step rather than only at the end, and reducing the variance of the advantage estimates substantially.

The shaped reward is equivalent to the KL-penalized objective of Chapter 11 — $\mathbb{E}[r_\phi(x, y)] - \beta \, \text{KL}[\pi_\theta \| \pi_{\text{ref}}]$ — because the expected per-token KL penalties sum to the expected total KL divergence across the sequence:

$$\mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=1}^T \text{KL}_t\right] = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\log \frac{\pi_\theta(y_{1:T} \mid x)}{\pi_{\text{ref}}(y_{1:T} \mid x)}\right] = \text{KL}\!\bigl[\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\bigr],$$

where the second equality uses the chain rule of the KL divergence over a sequence of conditionals. The decomposition into per-token penalties is not an approximation; it is exact.

## The Value Function

PPO requires both a policy and a value function. In the language model setting, the **value function** $V_w(S_t)$ must estimate the expected future shaped reward from state $S_t = (x, y_{1:t})$ onward — the expected sum of remaining per-token KL penalties plus the terminal reward model score.

The value function is typically implemented as a separate neural network initialized from the SFT model (or the reward model), with a scalar output head replacing the vocabulary projection head. An alternative is a shared backbone with the policy, with a separate value head — the same architecture choice discussed in Chapter 7 for actor-critic methods. In the language model setting, the shared-backbone approach requires careful coefficient tuning to prevent the value function loss from distorting the representations used by the policy; separate networks avoid this coupling at the cost of doubling memory.

During a rollout, the value network produces a scalar $V_w(S_t)$ for every prefix $(x, y_{1:t})$. The value at the final step, $V_w(S_T)$, is the expected continuation value after the episode ends — which is zero for a terminal state. In practice, when responses are truncated at a maximum length rather than terminated by an end-of-sequence token, the terminal value is not zero: the episode was artificially ended and the incomplete response may have some continuation value. Handling truncation correctly — bootstrapping from $V_w(S_T)$ rather than using zero — requires distinguishing true termination from truncation in the advantage calculation.

## Advantage Estimation and Whitening

The per-token shaped rewards and the value estimates combine to produce advantage estimates via GAE, exactly as in Chapter 7:

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V_w(S_{t+1}) - V_w(S_t).$$

For language model training, $\gamma = 1$ is standard: future tokens in the same response are not discounted, since the episode length is short (hundreds of tokens) and there is no meaningful temporal discounting within a single generation. The $\lambda$ parameter for GAE follows the same values as in Chapter 7 — typically 0.95.

One additional step that is nearly universal in language model RL but less common in robotics is **advantage whitening**: normalizing the advantage estimates within a minibatch to have zero mean and unit variance before computing the policy gradient:

$$\hat{A}_t \leftarrow \frac{\hat{A}_t - \mu_{\hat{A}}}{\sigma_{\hat{A}} + \varepsilon},$$

where $\mu_{\hat{A}}$ and $\sigma_{\hat{A}}$ are the mean and standard deviation of advantages in the current minibatch. Whitening stabilizes training by preventing the gradient magnitude from varying dramatically across updates as the reward scale and KL penalty evolve during training. It also acts as a soft baseline: after whitening, advantages above zero represent above-average tokens and advantages below zero represent below-average tokens, regardless of the absolute reward scale. The practice is simple and consistently helpful; it is included by default in essentially all language model RL implementations.

## The PPO Objective in the Language Setting

With shaped rewards, GAE advantages, and whitened estimates in hand, the PPO-Clip objective applies with one token-level adaptation. The importance ratio at token $t$ is:

$$r_t(\theta) = \frac{\pi_\theta(y_t \mid S_{t-1})}{\pi_{\theta_{\text{old}}}(y_t \mid S_{t-1})},$$

where $\theta_{\text{old}}$ are the parameters at the start of the current PPO epoch (before any minibatch updates within the epoch). The clipped policy objective is:

$$\mathcal{L}_{\text{policy}}(\theta) = \mathbb{E}_{t}\!\Bigl[\min\bigl(r_t(\theta)\,\hat{A}_t,\; \text{clip}(r_t(\theta),\, 1 - \varepsilon,\, 1 + \varepsilon)\,\hat{A}_t\bigr)\Bigr],$$

summed over all token positions in the minibatch. The value function is trained by minimizing:

$$\mathcal{L}_{\text{value}}(w) = \mathbb{E}_t\!\Bigl[\bigl(V_w(S_t) - V_t^{\text{target}}\bigr)^2\Bigr],$$

where $V_t^{\text{target}} = \hat{A}_t + V_{w_{\text{old}}}(S_t)$ is the GAE return. The combined loss is $\mathcal{L} = -\mathcal{L}_{\text{policy}} + c_1 \mathcal{L}_{\text{value}}$.

A key difference from the robotics PPO setting: the KL penalty is not included in the PPO loss function directly — it is already embedded in the shaped reward. The PPO objective optimizes the shaped reward (which includes the per-token KL) rather than the bare reward model score. This means the clipping mechanism and the KL penalty act simultaneously: the clip prevents large policy changes between PPO epochs, and the KL penalty prevents the policy from drifting far from the reference over the course of all training. They operate at different timescales — epochs versus full training runs — and both are necessary.

## The Rollout Phase and the Learning Phase

The training loop alternates between two distinct phases. In the **rollout phase**, the current policy $\pi_\theta$ generates complete responses: for each prompt $x$ in a batch, sample a response $y \sim \pi_\theta(\cdot \mid x)$ autoregressively, compute the reward model score $r_\phi(x, y)$, and compute the per-token log-probabilities under both $\pi_\theta$ and $\pi_{\text{ref}}$. Store the full trajectory — tokens, log-probabilities, rewards — in a rollout buffer.

In the **learning phase**, run $M$ epochs of minibatch PPO updates on the rollout buffer. For each minibatch, recompute the current policy's log-probabilities (to obtain the fresh importance ratios $r_t(\theta)$), compute advantages via GAE using the stored rewards and value estimates, apply the clipped objective, and update $\theta$ and $w$.

The key computational constraint: both $\pi_\theta$ and $\pi_{\text{ref}}$ must be resident in memory simultaneously during the rollout phase, and both policy and value model must be available during the learning phase. For large language models — 7B, 13B, 70B parameters — this memory pressure is severe. Practical implementations use 4-bit or 8-bit quantization for $\pi_{\text{ref}}$ (which is never updated and only needs forward passes), LoRA for $\pi_\theta$ (reducing the number of trainable parameters to a small fraction of the total), and gradient checkpointing for the value model. The algorithmic content is unchanged, but the infrastructure required to execute it at scale is substantial.

## The Alignment Tax

A consistent empirical finding across RLHF implementations is the **alignment tax**: improving performance on the alignment objective — following instructions, refusing harmful requests, matching human preferences — tends to degrade performance on capability benchmarks that were not in the preference dataset. A model fine-tuned with RLHF to be more helpful in conversation may score lower on coding tasks, mathematical benchmarks, or factual question answering that it handled well before fine-tuning.

The alignment tax arises from the same mechanism as all forgetting in neural networks: gradient updates that improve performance on the RLHF objective shift parameters away from their pre-fine-tuning values, and if those pre-fine-tuning values encoded capabilities that the RLHF objective did not reward, those capabilities degrade. The KL penalty against $\pi_{\text{ref}}$ mitigates but does not eliminate this: it prevents the policy from drifting too far from the SFT model in distribution space, but it does not preserve all dimensions of capability equally. A model can remain close to $\pi_{\text{ref}}$ in KL while shifting probability mass in ways that affect specific capabilities.

The severity of the alignment tax depends on $\beta$, the KL coefficient, and on the diversity of the preference dataset. Large $\beta$ reduces the tax but also limits how much the policy can improve on the alignment objective. Diverse preference datasets that include examples requiring the capabilities being taxed can reduce the tax by ensuring the reward model rewards their preservation. In practice, InstructGPT reported modest alignment taxes on most benchmarks, with the largest degradations on tasks requiring precise factual recall — suggesting that RLHF fine-tuning shifted probability mass away from the specific token sequences encoding correct facts toward more fluent but occasionally less accurate alternatives.

## Practical Diagnostics and Stopping Criteria

Several practical monitoring signals have become standard in language model RL training. The **mean KL divergence** $\mathbb{E}[\sum_t \text{KL}_t]$ should increase gradually as training progresses — evidence that the policy is moving away from the reference — but should not grow to catastrophic values (above $10$–$20$ nats is typically a sign of reward hacking). The **reward model score** should increase, ideally tracking the improvement observed in human evaluations. The **policy entropy** should decrease slightly — the policy becoming more confident — but not collapse: collapse indicates reward hacking into a few stereotyped response patterns.

Stopping criteria are heuristic. Training is typically halted when the mean KL divergence reaches a predetermined budget, when a held-out human evaluation shows that further training no longer improves preference ratings, or when the reward model score increases but human evaluations plateau — the signal that the policy has begun to exploit the reward model rather than genuinely improve. No automatic criterion reliably identifies this transition; monitoring human evaluations at intervals throughout training remains the most reliable diagnostic.

## What Token-Level PPO Gets Right

The adaptation of PPO to the token-level MDP is, in retrospect, well-matched to the structure of language generation. The clipped objective prevents the large policy updates that would arise from single high-reward responses dominating the gradient. The per-token KL penalty distributes the proximity constraint across the sequence, providing dense gradient signal and preventing the specific form of reward hacking where the policy generates implausible token sequences that happen to score highly on $r_\phi$. The GAE advantage estimates, computed token by token, provide a credit assignment signal that identifies which parts of a response contributed to high or low reward — though imperfectly, since the reward model evaluates the complete response and per-token attribution involves substantial approximation.

What it does not address is the scale of the RL problem. A response of 500 tokens generates 500 state transitions per episode, and a training batch of 64 responses generates 32,000 token-level transitions — all of which must be processed through multi-billion-parameter networks in both the rollout and learning phases. The compute cost of on-policy PPO at language model scale is substantial, and the constraint that rollouts must come from the current policy limits data reuse. The training runs for InstructGPT, Anthropic's RLHF models, and subsequent large-scale RLHF systems each required significant compute infrastructure to execute what is, algorithmically, a straightforward application of Chapter 9's method.

The efficiency question — can we get the alignment benefit of PPO without its on-policy compute overhead — motivates both GRPO, which eliminates the value function entirely and computes advantages analytically from groups of rollouts, and DPO, which bypasses the RL loop altogether. Both are products of the same observation: the PPO-based RLHF pipeline works, but it is expensive, and the expense does not all buy proportional benefit.

---

*The token-level MDP formulation for language model RL and the per-token KL penalty are introduced in Ziegler, Stiennon, Wu, Brown, Radford, Amodei, Christiano, and Irving (2019), Fine-Tuning Language Models from Human Preferences, and refined in Stiennon et al. (2020), Learning to Summarize with Human Feedback. The full InstructGPT system, including the engineering of PPO at large model scale and the alignment tax observations, is described in Ouyang et al. (2022), Training Language Models to Follow Instructions with Human Feedback. Advantage whitening as a stabilization technique for language model RL is standard practice in open implementations including TRL (von Werra et al., 2020) and trlX (Havrilla et al., 2023). The reward-KL overoptimization and stopping criteria are analyzed in Gao, Hilton, Mićo, Askell, Ritter, Leike, Schulman, and Christiano (2022), Scaling Laws for Reward Model Overoptimization. The LoRA parameter-efficient fine-tuning approach used to make large-model RLHF tractable is from Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, and Chen (2022), LoRA: Low-Rank Adaptation of Large Language Models.*
