---
title: "DPO and Implicit Reward Methods"
subtitle: "Reparameterizing the optimal policy, canceling the partition function, and collapsing the RL loop into supervised learning"
---


Chapter 10 ended with a concrete inventory of costs: four large neural networks resident in memory simultaneously, an on-policy rollout loop generating tens of billions of tokens over a full RLHF training run, and a value function solving an unusually hard prediction problem on partial sequences. These are not engineering inconveniences — they are direct consequences of applying PPO, an algorithm designed for interactive sequential control, to a problem that is fundamentally about preference learning on a fixed dataset.

The question Rafailov, Sharma, Mitchell, Ermon, Manning, and Finn asked in 2023 was whether any of this is necessary. The KL-regularized RLHF objective has a known closed-form optimal solution, derived in Chapter 09. If the optimal policy is analytically expressible in terms of the reward and the reference policy, perhaps the learning algorithm can be derived algebraically from that expression — bypassing the RL loop entirely and reducing preference learning to a supervised objective on static preference pairs. The answer is yes, and the resulting method — **Direct Preference Optimization** (DPO) — eliminates the reward model, the value function, the rollout loop, and three of the four models from the memory wall, replacing the entire Stage 3 pipeline with a single binary cross-entropy loss.

---

## Reparameterizing the Reward

> [!question]
> The closed-form optimal policy expresses $\pi^*$ in terms of $r$ and $\pi_{\text{ref}}$. Can this be inverted to express $r$ in terms of $\pi^*$ and $\pi_{\text{ref}}$?

Start from the closed-form solution derived in Chapter 09. For a given reward function $r$ and reference policy $\pi_{\text{ref}}$, the policy that maximizes the KL-regularized objective is:

$$\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x) \exp\bigl(r(x, y) / \beta\bigr)}{Z(x)},$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp(r(x, y) / \beta)$ is the partition function — a normalizing constant that depends only on the prompt $x$, not on the response $y$. Rearranging for $r$:

$$\pi^*(y \mid x) \cdot Z(x) = \pi_{\text{ref}}(y \mid x) \exp\bigl(r(x, y) / \beta\bigr)$$

$$\log \pi^*(y \mid x) + \log Z(x) = \log \pi_{\text{ref}}(y \mid x) + r(x, y) / \beta$$

$$\boxed{r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x).}$$

> [!success]
> This reparameterization is **exact and complete**. For any reward function $r$, there is a corresponding optimal policy $\pi^*$, and knowing $\pi^*$ and $\pi_{\text{ref}}$ is equivalent to knowing $r$ up to the partition function $Z(x)$. The partition function is a prompt-level constant — it does not depend on the response $y$ being evaluated. The reward difference between two responses $y_w$ and $y_l$ is:
> $$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)},$$
> with $Z(x)$ canceling. Reward differences — which are the only quantities the Bradley-Terry model uses — can be computed purely from policy log-ratios, without ever evaluating $Z(x)$.

---

## The Partition Function Cancellation

> [!info]
> The partition function $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp(r(x,y)/\beta)$ is intractable: it requires summing over all possible responses, which is exponential in response length. This is precisely the object that blocked a direct algebraic derivation of preference probabilities from the optimal policy — until the cancellation is observed.

Substitute the reparameterized reward into the Bradley-Terry preference probability from Chapter 09:

$$P(y_w \succ y_l \mid x) = \sigma\!\bigl(r(x, y_w) - r(x, y_l)\bigr).$$

Substituting $r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$:

$$P(y_w \succ y_l \mid x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \cancel{\beta \log Z(x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} - \cancel{\beta \log Z(x)}\right).$$

The partition function cancels. What remains:

$$\boxed{P(y_w \succ y_l \mid x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right).}$$

> [!tip]
> **Why this matters.** The standard RLHF pipeline required three separate stages precisely because neither the reward function nor the partition function could be handled directly. The reward model was trained separately to approximate $r_\phi(x, y)$; the RL loop was run to push $\pi_\theta$ toward the optimal policy for $r_\phi$; and the partition function was never computed at all — PPO's policy gradient circumvented it by only needing reward differences across rollouts.
>
> The cancellation shows that none of this circumvention was necessary. If the unknown optimal policy $\pi^*$ is replaced by the trainable policy $\pi_\theta$, the preference probability is a fully tractable function of $\pi_\theta$ and the fixed reference $\pi_{\text{ref}}$ — no reward model, no partition function, no rollouts.

---

## The DPO Loss

Replace the unknown $\pi^*$ with the trainable policy $\pi_\theta$. The **DPO loss** is the negative log-likelihood of the observed preference dataset $\mathcal{D} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}$:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right].$$

> [!note]
> **Computational requirements.** Training requires:
> - One forward pass through $\pi_\theta$ for $y_w$ (to compute $\log \pi_\theta(y_w \mid x)$)
> - One forward pass through $\pi_\theta$ for $y_l$
> - One forward pass through $\pi_{\text{ref}}$ for $y_w$ (precomputed and cached, since $\pi_{\text{ref}}$ is frozen)
> - One forward pass through $\pi_{\text{ref}}$ for $y_l$
> - A standard backward pass through $\pi_\theta$ only
>
> No rollouts. No reward model. No value function. No importance ratios. The memory footprint is two models: $\pi_\theta$ (with gradients) and $\pi_{\text{ref}}$ (frozen, quantizable). Compare this to the four-model wall of Chapter 10.

> [!success]
> **DPO reduces preference learning to supervised fine-tuning.** The loss is a binary cross-entropy on pairs of completions — structurally identical to contrastive learning objectives in computer vision and NLP. Any training pipeline that supports SFT can run DPO with minimal modification: replace the token-level cross-entropy loss with the DPO pairwise loss, add the reference model's log-probabilities to the batch, and train.

---

## The Implicit Reward

> [!info]
> The quantity $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ is the **implicit reward** of DPO — the reward function that the trained policy $\pi_\theta$ implicitly represents. It is not a separate neural network; it is a functional of the policy itself, defined by the log-ratio between the policy and the reference.

The DPO loss can be rewritten in terms of the implicit reward:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\left[\log \sigma\!\bigl(\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)\bigr)\right].$$

This is exactly the Bradley-Terry maximum likelihood loss from Chapter 09, with $\hat{r}_\theta$ substituted for the explicit reward model $r_\phi$. Optimizing $\mathcal{L}_{\text{DPO}}$ trains $\pi_\theta$ to be a policy whose implicit reward assigns higher scores to preferred responses.

**Gradient analysis.** The gradient of the DPO loss:

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta\,\mathbb{E}\!\left[\underbrace{\sigma\!\bigl(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\bigr)}_{\text{weight: probability model is wrong}}\Bigl(\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)\Bigr)\right].$$

> [!tip]
> **What each component does:**
> - $\nabla_\theta \log \pi_\theta(y_w \mid x)$: increases log-probability of the preferred response — the direction a standard SFT step would take.
> - $-\nabla_\theta \log \pi_\theta(y_l \mid x)$: decreases log-probability of the rejected response — the direction a negative SFT step would take.
> - $\sigma(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w))$: weights the update by how wrong the current model is. When $\pi_\theta$ already ranks $y_w$ above $y_l$ confidently ($\hat{r}_\theta(y_w) \gg \hat{r}_\theta(y_l)$), the weight $\sigma(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w)) \to 0$ — no update needed. When the model has the preference backwards, the weight $\to 1$ — maximum correction.
>
> The update is contrastive: the policy is pushed to increase its implicit reward for preferred completions relative to the reference, and decrease it for rejected completions. The $\pi_{\text{ref}}$ normalization ensures stability — the policy cannot satisfy the objective by collapsing to a single preferred response, because the implicit reward measures log-ratio relative to the reference, not absolute log-probability.

> [!note]
> The $\pi_{\text{ref}}$ anchoring plays the same role in DPO that the explicit KL penalty played in PPO-based RLHF. In PPO, the KL term $-\beta \text{KL}[\pi_\theta \| \pi_{\text{ref}}]$ was added explicitly to the shaped reward to prevent the policy from drifting into reward-hacking territory. In DPO, this constraint is absorbed into the loss function itself through the log-ratio $\log(\pi_\theta / \pi_{\text{ref}})$: the implicit reward measures how much the policy has moved from the reference, and the contrastive objective maximizes this movement for preferred responses while minimizing it for rejected ones. The KL regularization is not omitted — it is baked in.

---

## DPO as Offline Reinforcement Learning

> [!question]
> DPO and PPO target the same optimal policy $\pi^*$. Why do they behave differently in practice, and what determines when each is preferable?

DPO is formally a solution to the same KL-regularized RL problem as PPO — under ideal conditions, both converge to the same $\pi^* \propto \pi_{\text{ref}} \cdot \exp(r_\phi / \beta)$. The difference is how they pursue it: PPO optimizes online through policy gradient updates on fresh rollouts; DPO optimizes offline through supervised learning on a static preference dataset.

This places DPO squarely in **offline reinforcement learning**: learning a policy from a fixed dataset collected by a behavior policy (the annotation procedure), without any interaction with the environment during training.

> [!warning]
> **The distribution shift problem.** Online methods like PPO sample from the current policy $\pi_\theta$ at each step. As $\pi_\theta$ improves, the rollout distribution shifts to reflect better behavior — which generates preference pairs where the winning response genuinely represents improved quality, closing a positive feedback loop.
>
> DPO trains on a dataset $\mathcal{D}$ fixed before training begins. If both $y_w$ and $y_l$ in a preference pair are low-quality responses (one merely less bad than the other), DPO will learn to prefer the less-bad response while remaining unaware that significantly better responses exist outside the dataset's support. The policy cannot improve beyond the quality ceiling implicit in the preference data. DPO eliminates the on-policy loop — and that loop was also the mechanism by which the policy explored beyond its initial distribution.
>
> Concretely: if no response in $\mathcal{D}$ contains a correct multi-step mathematical derivation, DPO cannot learn to produce one — regardless of how well it learns from the preference signal. PPO can discover the derivation by sampling from $\pi_\theta$ as it improves.

> [!tip]
> **When the distribution shift problem is mild.** The coverage gap between $\mathcal{D}$ and the optimal policy matters less when:
> - The preference dataset was collected from a strong model (not just the SFT model), so the preferred responses are already near-optimal.
> - The task has low diversity — there is a narrow set of good responses, and the dataset covers it.
> - The goal is alignment (refusing harmful requests, following instructions) rather than capability improvement — alignment does not require discovering responses outside the dataset, only learning to prefer certain existing behaviors.
>
> For capability-improving tasks — mathematics, code generation, long-form reasoning — the distribution shift problem is acute. This is why verifiable-reward methods like GRPO (Chapter 12) dominate these settings: they use ground-truth checks rather than preference datasets, and they generate rollouts from the current policy, so the training distribution tracks capability improvements automatically.

---

## IPO: Fixing DPO's Implicit Assumption

> [!question]
> The DPO derivation uses the Bradley-Terry model as the preference model. Does this introduce any failure modes, and can they be corrected?

The Bradley-Terry model assumes preference probability depends continuously on the reward gap. When preference data is near-deterministic — one response is always chosen over another — the maximum likelihood solution under DPO is to make the implicit reward gap infinite:

$$\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l) \to +\infty.$$

This corresponds to $\pi_\theta(y_w \mid x) / \pi_{\text{ref}}(y_w \mid x) \to \infty$ and $\pi_\theta(y_l \mid x) / \pi_{\text{ref}}(y_l \mid x) \to 0$ — the policy collapses to always generating $y_w$ and never generating $y_l$, regardless of whether intermediate responses might be better. The loss saturates at zero with infinite implicit reward gap; gradient descent has no reason to stop.

> [!info]
> **IPO** (Identity Preference Optimization, Azar et al., 2023) replaces the log-sigmoid loss with a squared loss that has a **finite optimal implicit reward gap**:
>
> $$\mathcal{L}_{\text{IPO}}(\theta) = \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\left[\left(\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} - \frac{1}{2\beta}\right)^2\right].$$
>
> The target is $1/(2\beta)$ — a finite implicit reward gap corresponding to a preference probability of $\sigma(1) \approx 0.73$. The loss is minimized not at infinity but at a specific calibrated gap that depends on $\beta$.

> [!note]
> IPO's derivation does not pass through the Bradley-Terry model — it directly minimizes a regularized objective over the space of policies, arriving at a finite-gap loss without the Bradley-Terry assumption. The result:
> - No degeneracy under deterministic preferences.
> - Policy entropy remains higher throughout training, since the collapse to a single mode is penalized by the squared-loss structure.
> - Robust to annotation noise: when two responses are genuinely similar in quality, IPO produces a small gap rather than forcing a large one.
>
> Empirically, DPO and IPO produce comparable performance on most benchmarks, but IPO is more reliable in domains with clean preference signals and near-deterministic labels.

---

## SimPO: Reference-Free Preference Optimization

> [!question]
> DPO requires computing $\pi_{\text{ref}}(y \mid x)$ for every preference pair at every training step. Can the reference model be eliminated entirely?

The DPO implicit reward $\beta \log(\pi_\theta(y \mid x) / \pi_{\text{ref}}(y \mid x))$ requires a forward pass through $\pi_{\text{ref}}$ for both $y_w$ and $y_l$ in every minibatch. For large models, this reintroduces a two-model memory requirement at training time (the frozen reference plus the trainable policy). Reference log-probabilities can be precomputed and cached — but for long sequences, storage becomes a bottleneck, and online iterative variants (below) cannot precompute them at all.

> [!info]
> **SimPO** (Simple Preference Optimization, Meng et al., 2024) replaces the log-ratio reward with the **average log-probability** of the response under the current policy alone, normalized by response length to prevent length gaming:
>
> $$r_{\text{SimPO}}(x, y) = \frac{1}{|y|} \log \pi_\theta(y \mid x) = \frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t}).$$
>
> The SimPO loss adds a **margin** $\gamma > 0$ to require a minimum reward gap:
>
> $$\mathcal{L}_{\text{SimPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\left[\log \sigma\!\left(\frac{\beta}{|y_w|} \log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l \mid x) - \gamma\right)\right].$$

> [!note]
> **Length normalization** is essential for reference-free rewards. Without it, the model can trivially increase $r_{\text{SimPO}}(x, y_l)$ by generating longer rejected responses — or satisfy the objective by making $y_w$ shorter. Dividing by $|y|$ makes the reward a per-token average log-probability, removing this length-exploitation channel.
>
> **The margin $\gamma$** ensures that the loss does not saturate when the policy already assigns higher average log-probability to $y_w$ than $y_l$. Without a margin, a policy that correctly ranks the pair would receive zero gradient even if the gap is tiny. The margin forces the gap to reach $\gamma$ before the loss is minimized, producing a more decisive preference signal.

> [!warning]
> **SimPO's theoretical gap.** Without the reference model, the connection to the KL-regularized RL objective is lost: there is no guarantee that the learned policy stays close to $\pi_{\text{ref}}$, and the alignment geometry of Chapter 09 ($\pi^* \propto \pi_{\text{ref}} \cdot \exp(r_\phi/\beta)$) no longer applies. The implicit reward $r_{\text{SimPO}}$ measures absolute log-probability, not log-ratio — a model that generates all responses with uniformly high log-probability looks identical to a model with a strong preference, even if it has drifted far from $\pi_{\text{ref}}$.
>
> Empirically, SimPO performs comparably to DPO on alignment benchmarks while using roughly half the memory, suggesting that for instruction-following tasks the reference anchoring provides limited practical benefit. For tasks requiring precise KL control — where reward hacking or capability degradation is a concern — the theoretical guarantees of DPO are meaningful and reference-free methods should be used cautiously.

---

## Online and Iterative DPO

> [!question]
> The distribution shift problem of offline DPO is a consequence of training on a static dataset. Can DPO be run online, generating fresh preference pairs from the current policy to overcome this limitation?

Yes. A family of **iterative DPO** methods re-introduces data generation by alternating between two phases:

1. **Data collection**: the current policy $\pi_\theta$ generates new responses; a reward model or human evaluators form fresh preference pairs from these responses; the dataset $\mathcal{D}$ is updated.
2. **Learning**: $\mathcal{L}_{\text{DPO}}$ (or $\mathcal{L}_{\text{IPO}}$, or $\mathcal{L}_{\text{SimPO}}$) is minimized on the updated dataset for several epochs.

> [!tip]
> Iterative DPO recovers the on-policy exploration property of PPO — preference data reflects the current policy's behavior, not a fixed prior — while retaining DPO's simplicity and avoiding the per-token value function of PPO. The number of rollouts per iteration is typically much smaller than a full PPO run: each iteration needs enough responses to form informative preference pairs, not enough to estimate the full value function via GAE.
>
> The tradeoff relative to pure offline DPO:
> - **Improved coverage**: fresh rollouts from the current policy bring in responses not present in the original dataset, allowing the policy to discover better behavior.
> - **Reintroduced rollout cost**: data collection is now part of the training loop, bringing back some of the infrastructure costs DPO was designed to eliminate.
> - **Retained simplicity**: the learning step is still supervised, without importance ratios, GAE, or value function updates.

> [!note]
> Several implementations have pushed further by combining the DPO-style contrastive objective with mechanisms from PPO:
> - **Importance-ratio clipping** from PPO's policy constraint (when rollouts are not from the current policy checkpoint).
> - **Length-normalized rewards** from SimPO (when sequence length varies substantially across the rollout batch).
> - **Reward model scoring** replacing human annotation (enabling fully automated iteration).
>
> The convergence of these threads reflects how DPO's central contribution — the partition function cancellation — becomes a module that subsequent methods inherit, with the surrounding infrastructure varying based on specific requirements of scale, task type, and compute budget. The contrastive structure of the loss is preserved; the machinery for generating and filtering preference pairs is replaced as needed.

---

## The Alignment Geometry Revisited

> [!success]
> The theoretical target established in Chapter 09 — $\pi^* \propto \pi_{\text{ref}} \cdot \exp(r_\phi / \beta)$ — is the same for every method in this arc. DPO's contribution is to show that the target can be parameterized as a property of the policy itself rather than as the output of a separate reward model, enabling the entire RL loop to be replaced by a supervised loss on preference pairs.
>
> | Method | What it eliminates | What it requires | When to use |
> |--------|-------------------|------------------|-------------|
> | PPO + RLHF (Ch 10) | Nothing | 4 models, rollout loop, value function | On-policy needed; verifiable rewards |
> | Offline DPO | Reward model, value function, rollout loop | 2 models, static preference dataset | Alignment tasks; dataset covers optimal behavior |
> | IPO | Same as DPO + BT degeneracy | 2 models, static preference dataset | Deterministic/noisy preference data |
> | SimPO | Reward model, value function, rollout loop, reference model | 1 model, static preference dataset | Memory-constrained; alignment tasks |
> | Iterative DPO | Reward model, value function | 2 models, online rollout (intermittent) | Capability tasks; distribution shift matters |
> | GRPO (Ch 12) | Value function | Policy + reference, group rollouts | Verifiable rewards; reasoning tasks |

The underlying unifying structure:
$$\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x) \exp\bigl(r_\phi(x, y) / \beta\bigr)}{Z(x)}.$$

DPO's key insight is that the intractable $Z(x)$ cancels in the Bradley-Terry likelihood, converting an RL problem with intractable normalizer into a supervised problem with fully tractable gradients. The insight is algebraic, not algorithmic — it does not require new optimization theory, only a rearrangement of equations that existed since Chapter 09.

---

## What DPO Cannot Do

> [!warning]
> DPO's limitations follow directly from its derivation. Each assumption in the proof is also a constraint on the method's applicability:
>
> 1. **Requires a preference dataset.** DPO needs pairs $(y_w, y_l)$ with human or model-derived preference labels. For tasks where ground truth is verifiable automatically (code, mathematics), a preference dataset is unnecessary and its collection introduces noise. Verifiable-reward methods (GRPO, Chapter 12) are preferable when binary correctness can be checked without human annotation.
>
> 2. **Static dataset limits capability improvement.** As discussed above: DPO cannot discover responses better than those in $\mathcal{D}$. For tasks requiring reasoning breakthroughs — where no response in the dataset arrives at the correct solution — offline DPO is structurally incapable of finding that solution. Iterative variants recover this capability at the cost of rollout infrastructure.
>
> 3. **The Bradley-Terry assumption.** DPO assumes preferences can be modeled as pairwise comparisons with transitive, scalar reward differences. Real human preferences are not always transitive (a prefers A over B, B over C, but C over A in different contexts), and the quality of complex responses may not decompose into a single scalar. IPO partially addresses this but the fundamental model remains.
>
> 4. **No mechanism for exploration.** Offline DPO has no randomness in the learning loop — there is no policy sampling, no environment interaction, no exploration. If the preference data has systematic gaps (e.g., all rejected responses are obviously wrong), the implicit reward surface will have large uncharted regions where the model's behavior is determined entirely by the SFT initialization, not by the preference signal.

---

## The Broader Trajectory

Standing at the end of this derivation, several threads are visible across the preceding chapters.

The theoretical core has remained remarkably stable. The Bellman optimality equations of Chapter 01, the policy gradient theorem of Chapter 05, and the MaxEnt optimal policy of Chapter 08 are the load-bearing structures. Every algorithm in this series is a variation on one of these three foundations — a different computational strategy for the same underlying optimization problem. DPO's central identity — that the reward can be expressed as a log-ratio of optimal and reference policy — is a consequence of the KL-regularized Bellman equations derived in Chapter 09. The derivation is three lines; the significance is that it eliminates a training loop.

What has changed dramatically is the setting. The early chapters dealt with finite MDPs and lookup tables. By Chapter 10, the state space is the set of all partial language model responses — a space too large to enumerate, only sampled. The reward functions of the early chapters were given by the environment; by Chapter 09, reward itself is a learned approximation to human preference, with all the fragility that approximation entails. DPO sidesteps this by never training an explicit reward model, but the underlying preference signal it trains on is still a finite-sample approximation to a complex human judgment function.

> [!note]
> The open problems inherited by DPO are the same ones inherited by every method in this arc: reward hacking in a different guise (dataset hacking — preference pairs that do not represent genuine quality differences), distribution shift, and the scalable oversight problem of whether the preference data reflects what is actually wanted or only what annotators can reliably distinguish. DPO manages these with different tradeoffs than PPO, but does not resolve them. The management is more efficient; the question itself remains open.

---

*Direct Preference Optimization is introduced in Rafailov, Sharma, Mitchell, Ermon, Manning, and Finn (2023), Direct Preference Optimization: Your Language Model is Secretly a Reward Model. The IPO loss and its correction of DPO's degeneracy under deterministic preferences are from Azar, Guo, Piot, Munos, Rowland, Valko, and Thacker (2023), A General Theoretical Paradigm to Understand Learning from Human Feedback. SimPO and length-normalized reference-free optimization are from Meng, Xia, He, Goyal, Krishnamurthy, Chen, and Hajishirzi (2024), SimPO: Simple Preference Optimization with a Reference-Free Reward. Iterative and online DPO variants are described in Xu, Bai, Lin, Ye, Zhou, and Zhou (2023), Some Things Are More CRINGE Than Others, and in Guo, Xiao, Li, Sun, Gao, and Lin (2024), Direct Language Model Alignment from Online AI Feedback. The connection between offline RL and preference learning is surveyed in Kumar, Zhou, Tucker, and Levine (2020), Conservative Q-Learning for Offline Reinforcement Learning.*
