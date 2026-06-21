---
title: "Group-Based Policy Optimization"
subtitle: "GRPO, Dr. GRPO, and MaxRL — eliminating the critic, correcting normalization biases, and reframing RL as maximum likelihood"
---


Chapter 10 ended with the four-model memory wall and the observation that the value function contributes roughly 25% of the total memory footprint without a corresponding 25% contribution to learning quality. Chapter 11 answered one part of this problem: when the goal is preference learning on existing data, DPO eliminates the entire RL loop by reparameterizing the optimal policy as a supervised objective. But DPO's elimination of rollouts is also its limitation — it cannot discover behaviors outside its training distribution, and it cannot exploit ground-truth verification when the correct answer can be checked automatically.

The methods in this chapter take a different path. They keep the RL loop — on-policy sampling, reward computation, policy gradient updates — but eliminate the value function by replacing the learned critic with an analytical baseline computed from groups of rollouts to the same prompt. The insight is elementary: instead of training a neural network to estimate $V^\pi(x)$ from a single forward pass, sample $G$ responses from the current policy for each prompt and use their mean reward as the baseline. No network needed; no optimizer state; no critic warmup. The tradeoff is rollout compute — $G$ responses instead of one — but for tasks with verifiable rewards and cheap generation, this is a favorable exchange.

Three methods are developed here. **GRPO** (Group Relative Policy Optimization) introduces the group sampling scheme and establishes the standard algorithm for reasoning tasks. **Dr. GRPO** (GRPO Done Right) identifies two systematic biases in GRPO's normalization and provides corrected formulas with better empirical properties. **MaxRL** reinterprets the RL objective itself, showing that standard group-relative methods approximate maximum-likelihood training from a specific angle and that a better estimator — one whose variance decreases with more rollouts — falls naturally from this reinterpretation.

---

## GRPO: Group Relative Policy Optimization

### The Group Sampling Scheme

> [!info]
> For each prompt $x$ in a training batch, GRPO samples $G$ independent responses from the current policy snapshot $\pi_{\theta_{\text{old}}}$:
> $$y_1, y_2, \ldots, y_G \;\sim\; \pi_{\theta_{\text{old}}}(\cdot \mid x).$$
> Each response $y_i$ receives a scalar reward $r_i$ from the reward signal — either a learned reward model score or a verifiable reward based on ground truth. The **group baseline** is the mean:
> $$\bar{r} = \frac{1}{G} \sum_{i=1}^G r_i.$$
> The **group-relative advantage** centers by the mean and normalizes by the standard deviation:
> $$\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r + \varepsilon}, \qquad \sigma_r = \sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \bar{r})^2},$$
> where $\varepsilon$ is a small constant for numerical stability.

> [!tip]
> **Unbiasedness.** The group baseline $\bar{r}$ is a valid REINFORCE baseline: it depends on the other $G-1$ samples in the group but not on $y_i$ directly. The zero-mean property from Chapter 05 therefore applies — $\mathbb{E}_{y_i \sim \pi}[\nabla_\theta \log \pi_\theta(y_i \mid x) \cdot \bar{r}] = 0$ — and the advantage estimate is an unbiased estimate of $A^\pi(x, y_i) = Q^\pi(x, y_i) - V^\pi(x)$. No critic is needed to make the estimator unbiased; only the other responses in the group.
>
> As $G \to \infty$, $\bar{r} \to V^\pi(x)$ and the variance of the baseline decreases as $O(1/G)$. In practice $G = 4$–$16$ is standard — large enough to estimate the group mean accurately, small enough to fit in GPU memory alongside the policy and reference model.

> [!note]
> **Automatic difficulty weighting.** When rewards are binary (correct = 1, incorrect = 0), the group mean $\bar{r}$ is the empirical success rate on this prompt under the current policy. A correct response then has advantage $(1 - \bar{r})/\sigma_r$:
> - If the policy already solves the problem 90% of the time ($\bar{r} = 0.9$), a correct response receives advantage $\approx (1 - 0.9)/\sigma_r$ — small, since the problem is nearly mastered.
> - If the policy solves it 10% of the time ($\bar{r} = 0.1$), a correct response receives advantage $\approx (1 - 0.1)/\sigma_r$ — large, since the problem is hard and a successful response is informative.
>
> The group baseline automatically implements difficulty-weighted credit assignment: gradient effort concentrates on prompts where current performance is near chance and harvests little signal from problems the policy has already mastered.

---

### The GRPO Objective

The group-relative advantages replace GAE advantages in the PPO-Clip objective. For each response $y_i$ in the group, the token-level importance ratio at position $t$:

$$r_t^{(i)}(\theta) = \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})}.$$

The GRPO objective accumulates the clipped surrogate across all tokens in all responses, with explicit length normalization:

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{x}\!\left[\frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\!\Bigl(r_t^{(i)}(\theta)\,\hat{A}_i,\; \text{clip}\!\bigl(r_t^{(i)}(\theta),\, 1-\varepsilon,\, 1+\varepsilon\bigr)\,\hat{A}_i\Bigr)\right] - \beta\,\overline{\text{KL}}\!\bigl[\pi_\theta \,\|\, \pi_{\text{ref}}\bigr].$$

> [!note]
> Two structural features distinguish GRPO from PPO at the token level:
>
> 1. **Constant advantage within a response.** The advantage $\hat{A}_i$ is the same for every token position $t$ in response $y_i$ — every token in a high-reward response receives the same positive advantage, and every token in a low-reward response the same negative advantage. There is no per-token credit assignment: GRPO can say that response $i$ was better than average but cannot say which tokens within $i$ were responsible. This is a genuine limitation for long responses with heterogeneous quality.
>
> 2. **Length normalization** $1/|y_i|$. Without this, longer responses contribute more tokens to the loss and dominate the gradient. The $1/|y_i|$ factor makes the per-token loss contribution equal across responses of different lengths, preventing the objective from implicitly preferring verbose behavior.

> [!warning]
> The memory profile of GRPO versus PPO at 7B scale:
>
> | Model | PPO | GRPO |
> |-------|-----|------|
> | $\pi_\theta$ (policy, trainable) | ~14 GB | ~14 GB |
> | $\pi_{\theta_{\text{old}}}$ (rollout policy) | ~14 GB | ~14 GB |
> | $\pi_{\text{ref}}$ (reference, frozen) | ~4–7 GB (quantized) | ~4–7 GB (quantized) |
> | $V_w$ (value network, trainable) | ~14 GB | **eliminated** |
>
> GRPO reduces the memory wall by approximately one large model (~14 GB weights + optimizer state). The rollout cost increases by a factor of $G$, but rollout memory is dominated by KV-cache rather than model weights and is often amortized across parallel generation.

---

## Verifiable Rewards and the Reasoning Setting

> [!success]
> GRPO's design is most powerful when paired with **verifiable rewards** — reward signals computed by checking against ground truth, without a learned reward model:
> - **Mathematical reasoning**: the response either arrives at the correct numerical answer or it does not.
> - **Code generation**: the code either passes the test suite or it does not.
> - **Formal theorem proving**: the proof checker either accepts the proof or it does not.
>
> Verifiable rewards sidestep the reward hacking problem entirely. There is no $r_\phi$ to exploit — the reward is not a neural network approximation to human preference but a deterministic function of correctness. The reward-KL overoptimization curve of Chapter 09 does not apply because the gap between proxy and objective is zero by construction. GRPO on math benchmarks consequently uses $\beta$ values an order of magnitude smaller than InstructGPT-style RLHF, and training runs substantially longer before degrading.

> [!tip]
> The combination of group baselines and binary rewards creates a particularly clean signal. When $r_i \in \{0, 1\}$ and the group contains $K$ correct and $G - K$ incorrect responses:
>
> $$\bar{r} = \frac{K}{G}, \qquad \hat{A}_{\text{correct}} = \frac{1 - K/G}{\sigma_r + \varepsilon} > 0, \qquad \hat{A}_{\text{incorrect}} = \frac{0 - K/G}{\sigma_r + \varepsilon} < 0.$$
>
> Correct responses are pushed toward; incorrect responses are pushed away. The magnitude depends on $K/G$ — the empirical difficulty — making the learning signal calibrated to where the policy currently sits relative to the task.
>
> When $K = 0$ (no correct responses in the group) or $K = G$ (all correct), $\sigma_r = 0$ and the group provides no signal. Groups where the policy already fully succeeds or fully fails are filtered out: gradients are computed only for prompts where the current policy has some but not complete success. This automatic curriculum focuses optimization effort on the productive regime.

---

## DeepSeek-R1 and Emergent Reasoning

GRPO gained wide attention through DeepSeek-R1 (2025), which applied it to train a language model to produce extended chain-of-thought reasoning before giving a final answer. The setup: responses were required to follow a structured format with an explicit `<think>` section followed by a final answer box; the reward combined a format signal and a binary accuracy check against ground truth for mathematics and coding problems. The policy was initialized from a base language model — not an instruction-tuned model — and trained directly with RL, without the supervised fine-tuning stage that InstructGPT required.

> [!success]
> The trained model learned to use extended reasoning traces to solve difficult problems. But more striking were behaviors that emerged from the optimization pressure without explicit supervision:
>
> - **Self-correction**: the model spontaneously reconsidered earlier steps when they led to inconsistencies, producing within the `<think>` section what appeared to be a search over alternative approaches.
> - **Adaptive effort allocation**: longer thinking traces for harder problems, shorter for simpler ones — a form of compute-optimal reasoning allocation not specified in the reward.
> - **The "aha moment"**: at specific points in difficult reasoning traces, the model appeared to recognize a flaw in its current approach and switch strategies — a behavior the DeepSeek team flagged as significant because it was not present in any supervised demonstration and could not be attributed to any single gradient step.

> [!note]
> **The mechanism of emergent self-correction.** Among the $G$ responses sampled for a difficult problem, the responses that produced correct answers tended to include more structured, self-correcting traces — because on hard problems, committing to an initial error without recovery leads to wrong answers. The group-relative advantage assigned positive credit to these self-correcting responses and negative credit to responses that committed to an early error. Over many gradient updates on many problems, the policy learned to produce the self-correction pattern statistically associated with correct answers, even though no individual gradient update explicitly targeted that pattern.
>
> This is reinforcement learning working as intended: applied at scale to a sufficiently expressive policy, the reward gradient discovers behavioral strategies that are difficult to specify directly. The aha moment is the same phenomenon as DQN learning to play Breakout or AlphaGo learning to sacrifice stones strategically — but its manifestation in language-based reasoning has different implications for what these systems are doing and how they should be evaluated.

---

## GRPO vs PPO: Tradeoffs

> [!tip]
> The group baseline is GRPO's central innovation and its central limitation:
>
> | | PPO | GRPO |
> |-|-----|------|
> | Baseline | Learned value function $V_w$ | Group mean $\bar{r}$ |
> | Advantage granularity | Per-token (via GAE) | Per-sequence |
> | Memory | 4 models | 3 models |
> | Rollout cost | 1 response per prompt | $G$ responses per prompt |
> | Reward structure | Continuous or binary | Best with binary/verifiable |
> | Credit assignment | Token-level (noisy critic permitting) | Sequence-level |
> | Critic warmup needed | Yes | No |
>
> GRPO is best suited to tasks with verifiable, low-noise rewards and relatively short responses where token-level credit assignment matters less. Tasks with dense, continuous reward signals and long structured responses — multi-turn dialogue, creative writing evaluated by a rich reward model — are better served by PPO's critic, which can fit the continuous reward surface more accurately than a group mean.

---

## Dr. GRPO: Normalization Biases and Their Corrections

> [!question]
> GRPO's standard formulation uses two normalizations: dividing per-token losses by sequence length $|y_i|$, and dividing advantages by the group standard deviation $\sigma_r$. Both seem reasonable. Are they?

Liu et al. (2025) identified that both normalizations introduce systematic biases — not noise, but directional distortions in the gradient that reward specific undesirable behaviors or mis-allocate learning effort. They termed the corrected formulation **Dr. GRPO** (GRPO Done Right).

### Bias 1: Length Bias from Sequence-Level Loss Normalization

> [!warning]
> **The problem.** GRPO's objective averages token-level losses *within* each sequence (dividing by $|y_i|$) before averaging *across* sequences (dividing by $G$). For a response with advantage $\hat{A}_i$, the contribution to the objective is:
> $$\frac{\hat{A}_i}{|y_i|} \sum_{t=1}^{|y_i|} \ell_t,$$
> where $\ell_t$ is the token-level surrogate loss at position $t$.
>
> When $\hat{A}_i < 0$ (response is below average — the policy should be pushed away from it), this contributes a *negative* quantity to the objective. The negative contribution is $|\hat{A}_i| / |y_i|$ per token, which decreases with response length. A long incorrect response receives a weaker per-token penalty than a short incorrect response with the same advantage magnitude.
>
> This creates a gradient incentive: the policy can reduce the penalty it pays for incorrect responses by making them longer. Verbose incorrect responses are penalized less than concise incorrect ones. The length normalization intended to prevent length bias has introduced the opposite.

> [!tip]
> **The fix.** Dr. GRPO replaces sequence-level normalization with **token-level aggregation**: sum all token-level losses across the entire group without any per-sequence division, then normalize by the total number of tokens in the group:
> $$\mathcal{L}_{\text{Dr. GRPO}} = \frac{1}{\sum_i |y_i|} \sum_{i=1}^G \sum_{t=1}^{|y_i|} \hat{A}_i \cdot \ell_t.$$
> Under this aggregation, every token in the batch contributes equally to the gradient regardless of which sequence it belongs to. Long incorrect responses are penalized in proportion to their length — more tokens, more penalty — which removes the incentive to pad incorrect responses.

### Bias 2: Standard Deviation Overweighting

> [!warning]
> **The problem.** Dividing by $\sigma_r$ normalizes advantages so that the effective gradient magnitude is independent of the spread of rewards in the group. This seems desirable for scale invariance — but it conflates two different situations:
>
> - **High-variance group** ($\sigma_r$ large): the policy has mixed success on this prompt — some responses correct, some incorrect. The raw advantages $r_i - \bar{r}$ are large and informative.
> - **Low-variance group** ($\sigma_r$ small, near-zero): almost all responses have similar rewards — either all near-correct or all near-incorrect. The raw advantages are small, reflecting low information content about which responses are better.
>
> Dividing by $\sigma_r$ *inflates* the advantages for low-variance groups, making them as large as high-variance groups. Prompts where all responses are nearly identical receive the same gradient magnitude as prompts with mixed success — which overweights the "nearly solved" regime (where small fluctuations determine which responses are slightly above or below the mean) relative to the regime where the model genuinely struggles.

> [!tip]
> **The fix.** Dr. GRPO removes the $\sigma_r$ normalization entirely. The advantage becomes:
> $$\hat{A}_i = r_i - \bar{r},$$
> with no division by standard deviation. The raw centered reward is used directly. This means prompts with high-variance groups (where some but not all responses succeed) produce large advantages and strong gradients, while prompts where all responses succeed or all fail produce small advantages and weak gradients — which is the correct prioritization.

> [!success]
> **The Dr. GRPO formulation in summary:**
> $$\mathcal{L}_{\text{Dr. GRPO}}(\theta) = \frac{1}{\sum_i |y_i|} \sum_{i=1}^G \sum_{t=1}^{|y_i|} \min\!\Bigl(r_t^{(i)}(\theta)\,(r_i - \bar{r}),\; \text{clip}\!\bigl(r_t^{(i)}(\theta),\, 1-\varepsilon,\, 1+\varepsilon\bigr)\,(r_i - \bar{r})\Bigr).$$
>
> Two changes from GRPO: token-level aggregation (not per-sequence averaging) and no $\sigma_r$ division. Both are simple to implement, both are motivated by specific bias arguments, and both improve empirical performance on reasoning benchmarks by concentrating gradient effort on genuinely informative updates.

---

## MaxRL: RL as Approximate Maximum Likelihood

### The pass@k Connection

> [!question]
> Group-based methods optimize pass@1 — maximizing the probability that a single sampled response is correct. Is pass@1 the right objective for verifiable tasks, and what happens to pass@k performance?

MaxRL (Moshkovitz et al., 2025) starts from an identity connecting log-probability to the pass@k objective. For a prompt $x$ with success probability $p_\theta(x) = P(\text{response correct} \mid x, \pi_\theta)$, the log-likelihood has an exact expansion:

$$\log p_\theta(x) = -\sum_{k=1}^{\infty} \frac{(1 - p_\theta(x))^k}{k}.$$

> [!info]
> **Interpretation.** The term for index $k$ is $-(1 - p_\theta(x))^k / k$, which is proportional to the probability that $k$ independent samples *all fail* — the complement of pass@k. Maximizing $\log p_\theta(x)$ is equivalent to simultaneously maximizing all pass@k objectives, weighted by the harmonic series $1/k$:
>
> - $k=1$: maximize pass@1 (probability at least one success in one try).
> - $k=2$: maximize pass@2 (probability at least one success in two tries).
> - $k=\infty$: maximize the probability of eventually succeeding.
>
> Standard RL methods — REINFORCE, GRPO, PPO — optimize only the $k=1$ term. Maximum likelihood training on successful trajectories optimizes the full infinite sum. MaxRL defines a compute-indexed family of truncated objectives:
> $$J_{\text{MaxRL}}^{(T)}(x) = -\sum_{k=1}^{T} \frac{(1 - p_\theta(x))^k}{k},$$
> where $T=1$ recovers standard RL and $T \to \infty$ approaches maximum likelihood.

### The Successful-Trajectory Estimator

> [!info]
> Given $N$ rollouts for prompt $x$, with $K$ correct responses, MaxRL's on-policy gradient estimator is:
>
> $$\hat{g}_N(x) = \begin{cases} \displaystyle\frac{1}{K} \sum_{i=1}^{N} r_i \, \nabla_\theta \log \pi_\theta(y_i \mid x) & K \geq 1 \\ 0 & K = 0 \end{cases}$$
>
> This is the average gradient of log-policy over **successful trajectories only** — responses with $r_i = 1$ contribute their gradient with weight $1/K$; responses with $r_i = 0$ contribute nothing. For failed rollouts ($K = 0$), no update is made.

> [!success]
> **Why variance decreases with more rollouts.** The GRPO estimator averages over all $G$ responses, correct and incorrect. Adding more responses reduces variance via standard averaging. But in GRPO, the variance of the advantage estimator scales with the variance of the reward distribution under the current policy — roughly $O(1/G)$, but with a residual term that depends on how spread out the rewards are.
>
> In MaxRL, the estimator averages over $K$ successful trajectories. As $N$ increases, $K$ increases proportionally (in expectation), and the variance of the average gradient over successful trajectories decreases as $O(1/K)$. More rollouts simultaneously improve two things: the variance of the estimator, and the quality of the approximation to maximum likelihood (larger $T$). These two improvements compound: more compute produces both a better objective and a lower-variance estimate of its gradient.
>
> This contrasts with REINFORCE, where variance is determined by the episode horizon $T$ regardless of the number of rollouts.

### Effective Advantage and Difficulty Weighting

The MaxRL estimator's effective advantage can be expressed in the same form as GRPO's:

$$\hat{A}_i^{\text{MaxRL}} \propto \frac{r_i - \hat{r}}{\hat{r}}, \qquad \hat{r} = \frac{K}{N} \text{ (empirical success rate)}.$$

> [!tip]
> **Comparison to GRPO's advantage** $\hat{A}_i = (r_i - \bar{r})/\sigma_r$:
>
> | | GRPO | MaxRL |
> |-|------|-------|
> | Numerator | $r_i - \bar{r}$ | $r_i - \hat{r}$ (same for binary rewards) |
> | Denominator | $\sigma_r$ (std dev of rewards) | $\hat{r}$ (success rate) |
> | Difficult prompts ($\hat{r} \approx 0.1$) | Large denominator only if high spread | Small denominator → **large effective advantage** |
> | Easy prompts ($\hat{r} \approx 0.9$) | Small denominator if low spread | Large denominator → **small effective advantage** |
>
> MaxRL's denominator $\hat{r}$ produces automatic difficulty weighting that is more aggressive than GRPO's: a correct response on a 10%-success-rate problem has effective advantage $\approx (1 - 0.1)/0.1 = 9$, while a correct response on a 90%-success-rate problem has advantage $\approx (1 - 0.9)/0.9 \approx 0.11$. The ratio is $\approx 82\times$ — the policy concentrates almost all its learning effort on difficult prompts, which is consistent with the maximum-likelihood objective that down-weights already-mastered problems.

> [!warning]
> **The $K = 0$ problem.** When no rollout succeeds ($K = 0$), MaxRL makes no gradient update. For very hard prompts where the current policy never generates a correct response, MaxRL receives no learning signal at all. This is the zero-coverage problem: maximum likelihood over successful trajectories has nothing to maximize when there are no successes.
>
> GRPO has the same problem: a group where $K = 0$ has $\sigma_r = 0$, and the policy update is skipped. For both methods, making progress on genuinely unsolved problems requires either external intervention (curriculum design, initial demonstrations, RLHF-style warm-up) or waiting for occasional lucky rollouts as $N \to \infty$. MaxRL partially addresses this by showing that more rollouts increase the probability of at least one success, but for problems with true zero-probability under the current policy, group-based methods stall.

---

## Pass@k Preservation

> [!note]
> Optimizing pass@1 (the probability of success in a single sample) can actively harm pass@k for larger $k$. The mechanism: push the policy to concentrate probability on the single most likely correct response, reducing diversity. With lower diversity, the probability that at least one of $k$ independent samples succeeds — pass@k — can decrease even as pass@1 improves.
>
> MaxRL's maximum-likelihood interpretation explains why it avoids this failure mode. Maximizing $\log p_\theta(x)$ simultaneously maximizes all pass@k objectives via the harmonic expansion. The policy is trained to improve success probability across the entire distribution of possible responses, not just to increase the probability of the single most likely correct response. The result: MaxRL improves pass@k for all $k$ simultaneously and preserves response diversity, while GRPO (especially with $\sigma_r$ normalization that penalizes variance) can inadvertently reduce diversity as the policy narrows toward a single response pattern.
>
> For applications where test-time compute scaling is relevant — generating $k$ responses and selecting the best — preserving pass@k is directly valuable. MaxRL's theoretical grounding in maximum likelihood provides an automatic guarantee that standard pass@1-focused RL methods do not.

---

## A Unified View of Group-Based Methods

> [!tip]
> The three methods form a progression from empirical practice to theoretical grounding:
>
> | Method | Baseline | Normalization | Aggregation | Theoretical basis |
> |--------|----------|---------------|-------------|-------------------|
> | GRPO | Group mean $\bar{r}$ | $\sigma_r$ normalization | Per-sequence then across sequences | REINFORCE baseline identity |
> | Dr. GRPO | Group mean $\bar{r}$ | No $\sigma_r$ | Token-level across group | Bias analysis of GRPO's distortions |
> | MaxRL | Success rate $\hat{r} = K/N$ | $\hat{r}$ scaling | Only successful trajectories | Maximum likelihood via harmonic expansion |
>
> All three methods are critic-free, on-policy, and designed primarily for verifiable reward settings. Their differences lie in how they define the baseline, how they normalize advantages, and what objective they are implicitly targeting.
>
> GRPO's $\sigma_r$ normalization makes the gradient scale-invariant but introduces the overweighting of low-variance prompts that Dr. GRPO corrects. Dr. GRPO's raw centering $r_i - \bar{r}$ is unbiased and simple, but treats all prompts with non-zero variance equally. MaxRL's $1/\hat{r}$ scaling creates aggressive difficulty weighting consistent with maximum-likelihood training, at the cost of zero updates on fully unsolved prompts.
>
> In practice, the choice between them is secondary to the choice of reward structure: all three methods work well on binary verifiable rewards and are preferable to PPO when memory is constrained. For tasks with continuous rewards or long responses requiring token-level credit assignment, PPO's value function retains an advantage that group baselines cannot match regardless of how the advantages are normalized.

---

*GRPO is introduced in Shao, Wang, Zhu, Xu, Song, Zhang, Li, Wu, and Guo (2024), DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. The application to reasoning model training, the emergent self-correction behavior, and the DeepSeek-R1 system are in Guo, Yang, Zhang, et al. (2025), DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning with Verifiable Rewards. Dr. GRPO and the length-bias and standard-deviation-bias corrections are from Liu, Ji, Kuang, He, Zhang, and Bi (2025), Dr. GRPO: Decomposed Dr. GRPO for Solving Visual Puzzle with Verifiable Rewards. MaxRL and the connection between RL for verifiable tasks and maximum likelihood via the harmonic pass@k expansion are from Moshkovitz, Morcos, and Cho (2025), MaxRL: Reinforcement Learning from Maximum Likelihood. The connection between the group baseline and the REINFORCE baseline is in Williams (1992), Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. The STaR bootstrapping method that GRPO superseded is in Zelikman, Wu, Mu, and Goodman (2022), STaR: Bootstrapping Reasoning With Reasoning.*
