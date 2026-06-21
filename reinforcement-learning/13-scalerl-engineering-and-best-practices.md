---
title: "ScaleRL: Engineering and Best Practices"
subtitle: "Async pipelines, FP32 logits, loss aggregation, curriculum design, and the convergent recipe from 400,000 GPU-hours of ablations"
---


The preceding chapters developed the algorithmic landscape of RL for language model post-training: the theoretical foundations in Chapters 01–08, the RLHF pipeline and the token-level MDP in Chapters 09–10, and the menu of modern methods — DPO, GRPO, Dr. GRPO, MaxRL, DAPO, CISPO, DPPO — in Chapters 11–12. Each algorithm was presented as a solution to a specific problem: DPO eliminates the rollout loop; GRPO eliminates the value function; Dr. GRPO corrects normalization biases; CISPO recovers zeroed gradients; DAPO handles asymmetric exploration.

But correct algorithms frequently fail to reach their theoretical performance in practice. Implementation details — numerical precision, loss aggregation, data pipeline architecture, training curriculum — interact with algorithm design in ways that are not visible in the pseudocode. A method that works correctly at small scale can produce meaningfully worse results at large scale because a seemingly minor implementation choice distorts the gradient signal in the high-compute regime.

**ScaleRL** (2025) is a systematic empirical study of exactly these interactions, conducted at a scale unusual in academic research: over 400,000 GPU-hours of ablations, fitting sigmoidal performance-versus-compute curves rather than comparing single training checkpoints. The distinction is important. Comparing single checkpoints conflates two different properties: early learning speed (how quickly the method makes progress in the first few hours of training) and asymptotic performance (how high the performance ceiling is, given sufficient compute). An intervention can improve one while harming the other, and a paper that evaluates only at a single checkpoint — common in the field — cannot distinguish the two. ScaleRL's sigmoidal fitting separates them, yielding a more reliable characterization of which implementation choices matter and in what direction.

This chapter synthesizes ScaleRL's findings alongside converging evidence from the algorithm-development literature. The result is a practical engineering reference for large-scale RL training on language models.

---

## The Generate-Then-Update Loop and Its Cost

> [!question]
> Standard RL training for language models alternates between a rollout phase (generate $G$ responses per prompt, compute rewards) and an update phase (run PPO or GRPO epochs on the rollout buffer). What is the compute cost of this architecture, and can it be improved?

The **generate-then-update loop** is the straightforward implementation of on-policy RL: generate a full batch of rollouts, compute advantages, run $M$ gradient update epochs, then generate a new batch. The policy and all other models are kept on GPU throughout.

> [!warning]
> **Idle time in the generate-then-update loop.** The two phases use different compute resources:
> - **Rollout phase**: GPU is bottlenecked by the autoregressive generation of long sequences. The backward pass machinery (gradient buffers, optimizer states) is idle.
> - **Update phase**: GPU executes forward and backward passes through the policy. The rollout infrastructure (KV-cache management, sampling kernels) is idle.
>
> On a large-scale training run, these alternations mean that substantial fractions of GPU-hours are spent waiting. For a 7B model generating 512-token responses, the rollout phase can consume more wall-clock time than the update phase, depending on batch size and the ratio of generation to update compute.

> [!tip]
> **The pipelined asynchronous architecture.** ScaleRL's central infrastructure finding: replace the generate-then-update loop with a **pipelined async setup** where rollout generation and weight updates proceed concurrently rather than sequentially.
>
> In the async architecture:
> - A **rollout worker** generates responses continuously against the current policy snapshot, immediately depositing completed trajectories into a shared replay buffer.
> - A **training worker** draws minibatches from the replay buffer and applies gradient updates, immediately updating the policy weights that the rollout worker will use for the next generation batch.
>
> The two workers overlap in time: while training is updating on one batch of rollouts, generation is producing the next. GPU utilization increases; wall-clock time per policy improvement step decreases.
>
> The policy received by the rollout worker is slightly stale — by the time a trajectory is used for a gradient update, the policy weights have been updated by some number of prior batches. This introduces a small off-policy gap, requiring importance-ratio correction (which the PPO-style clipping already provides). ScaleRL's finding: the off-policy staleness in a well-tuned async pipeline is small enough that final performance is competitive with or better than synchronous training, while compute efficiency improves substantially.

---

## Numerical Precision: FP32 Logits

> [!warning]
> **The importance-sampling ratio distortion problem.** The PPO and GRPO objectives both depend on the importance ratio:
> $$r_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\theta_{\text{old}}}(y_t \mid x, y_{<t})}.$$
> At training time, this ratio is computed from log-probabilities produced by the current policy $\pi_\theta$ (computed in the update phase) and the rollout policy $\pi_{\theta_{\text{old}}}$ (computed during generation). If the generation kernel and the training kernel use different numerical precision — a common situation when generation uses BF16/FP16 for speed while training uses FP32 — small mismatches accumulate across the softmax computation, producing log-probability values that differ between the two phases even for identical weights.
>
> The effect: the importance ratio $r_t(\theta)$ is nonzero not just from the true policy update $\theta \to \theta_{\text{new}}$ but also from numerical noise introduced by the precision mismatch. For tokens with very low probability (rare vocabulary entries), the log-probability difference from numerical noise can be comparable to or larger than the difference from the actual gradient step. The resulting importance ratios are distorted in ways that are not corrected by the clipping mechanism — clipping operates on the ratio, but the ratio itself is wrong.

> [!success]
> **Fix: compute the language model head in FP32.** Following the MiniMax report and confirmed by ScaleRL, computing the final linear projection (vocabulary logits → softmax → log-probabilities) in FP32 — while keeping the rest of the model in BF16 — sharply reduces the numerical mismatch between generation and training. The computation cost is modest (the LM head is small relative to the attention blocks); the impact on asymptotic performance is substantial.
>
> This is one of ScaleRL's most reproducible and impactful single-variable findings: FP32 logits improve asymptotic performance across all methods tested, without any change to algorithm or hyperparameters. It is a low-cost, high-reward implementation detail that should be treated as a default rather than an optional optimization.

---

## Loss Aggregation: Token, Sample, and Prompt Level

> [!question]
> The policy gradient loss can be aggregated in multiple ways: averaging token-level losses within sequences (sample-level averaging), then across sequences; or aggregating directly across all tokens in the batch (token-level); or averaging across prompts first (prompt-level). Do these produce the same gradient?

They do not. The aggregation level determines the implicit weighting scheme applied to the gradient, and different schemes can introduce or correct systematic biases.

> [!info]
> **Three aggregation schemes:**
>
> Let $\ell_{i,t}$ denote the token-level surrogate loss for token $t$ of response $i$, with response length $|y_i|$.
>
> **Sample-level (standard)**: average within each sequence, then across sequences:
> $$\mathcal{L}_{\text{sample}} = \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \ell_{i,t}.$$
> Each sequence contributes equally regardless of length. A 1000-token response and a 50-token response have the same weight in the gradient.
>
> **Token-level**: aggregate across all tokens in the batch uniformly:
> $$\mathcal{L}_{\text{token}} = \frac{1}{\sum_i |y_i|} \sum_{i=1}^G \sum_{t=1}^{|y_i|} \ell_{i,t}.$$
> Each *token* contributes equally. A 1000-token response contributes 20× more than a 50-token response.
>
> **Prompt-level**: for each prompt $x$, first aggregate over all $G$ responses and all their tokens, then average across prompts:
> $$\mathcal{L}_{\text{prompt}} = \frac{1}{|\mathcal{B}|} \sum_{x \in \mathcal{B}} \frac{1}{\sum_i |y_i^x|} \sum_{i=1}^G \sum_{t=1}^{|y_i^x|} \ell_{i,t}^x.$$
> Each *prompt* contributes equally, regardless of response length or group size.

> [!tip]
> **ScaleRL's finding: prompt-level averaging shows the best asymptotic performance.** The mechanism: prompt-level averaging ensures that difficult prompts (those where the policy currently struggles and the reward signal is informative) are not down-weighted relative to prompts with longer but less informative responses. If a difficult prompt generates short responses (because the policy gives up quickly) while an easy prompt generates long ones (because the policy confidently produces extended correct reasoning), sample-level averaging would give more weight to the easy prompt. Prompt-level averaging equalizes this.
>
> Token-level averaging, as the Dr. GRPO analysis in Chapter 12 established, removes the length-bias that sample-level averaging introduces for negative-advantage responses. It is an improvement over sample-level, but still gives more weight to verbose prompts (through longer aggregate sequence lengths) than prompt-level averaging does. ScaleRL's empirical finding ranks them: prompt-level > token-level > sample-level, in terms of asymptotic performance on reasoning benchmarks.

---

## Variance Filtering and Curriculum Design

> [!info]
> **Zero-variance filtering.** When all $G$ responses for a prompt have identical rewards — either all correct ($K = G$) or all incorrect ($K = 0$) — the group provides no learning signal:
> - All-correct: the group mean equals the maximum reward; all advantages are zero; the gradient update does zero work.
> - All-incorrect: the group mean equals the minimum reward; all advantages are zero; same.
>
> Computing rollouts, rewards, and log-probabilities for these prompts consumes GPU compute and storage without contributing to learning. Zero-variance filtering excludes such prompts from the optimization step (while still generating their rollouts if needed for reward estimation). ScaleRL confirms that filtering zero-variance prompts improves training efficiency without harming final performance — which is expected, since the excluded updates were null updates.

> [!tip]
> **Positive resampling: the 90% correctness threshold.** A more aggressive curriculum intervention: when a prompt achieves an empirical success rate above 90% ($K/G > 0.9$) across recent batches, exclude it from future training epochs — even though it still provides non-zero advantages on 10% of samples.
>
> The rationale: a prompt where the policy succeeds 90% of the time is nearly mastered. The remaining 10% failures provide some signal, but the gradient from this prompt is dominated by reinforcing already-correct behavior (the 90%). The marginal learning value is low relative to prompts where the policy succeeds 50% or 30% of the time. Resampling difficult prompts into the freed training slots concentrates gradient effort where it is most informative.
>
> ScaleRL's finding: positive resampling slightly *slows* early training (because some informative-but-easy prompts are excluded) but *improves asymptotic performance* (because the freed compute is spent on harder, more informative prompts). The early-vs-asymptotic distinction that ScaleRL's sigmoidal fitting makes possible is essential here — a single-checkpoint evaluation would classify positive resampling as harmful.

---

## Algorithm Comparison at Scale

> [!note]
> ScaleRL's algorithm comparison, conducted at production-relevant scale (not single-checkpoint comparisons), produces a result that differs from the impression given by individual papers optimized for benchmark performance at their point of publication:
>
> **Among off-policy loss functions, CISPO and GSPO outperform DAPO in asymptotic performance, with CISPO selected as the default for combining strong results with relative robustness.**
>
> This is a practically significant finding. DAPO introduced four innovations (token aggregation, asymmetric clipping, overlong shaping, dynamic sampling) and reported strong results at the time of publication. ScaleRL's sustained-compute comparison finds that CISPO — which focuses on a single change (stopping gradients through clipped importance weights rather than masking them entirely) — achieves better asymptotic performance than DAPO's more elaborate construction.

> [!tip]
> The comprehensive algorithm comparison across all methods discussed in this series:
>
> | Method | Baseline/Advantage | Clipping | Loss Aggregation | Key property |
> |--------|-------------------|----------|-----------------|-------------|
> | REINFORCE | EMA or batch mean | None | Sample avg | Baseline; high variance |
> | PPO | GAE with critic | Symmetric IS | Sample avg | Per-token credit; high memory |
> | RLOO | Leave-one-out $r_i - \frac{1}{G-1}\sum_{j\neq i}r_j$ | None | Sample avg | Critic-free; pure REINFORCE style |
> | GRPO | $(r_i-\mu_G)/\sigma_G$ | Symmetric IS | Length norm | Group baseline; reasoning tasks |
> | Dr. GRPO | $r_i - \mu_G$ | Symmetric IS | Token avg | Bias-corrected GRPO |
> | DAPO | $(r_i-\mu_G)/\sigma_G$ | Asymmetric ($\varepsilon_\text{low}=0.2$, $\varepsilon_\text{high}=0.28$) | Token avg | Exploration-preserving clip |
> | CISPO | Group-normalized | Upper bound only + stop-gradient | Token avg | Recovers zeroed gradients |
> | DPPO | Within-group norm | Symmetric DV (TV/KL threshold) | Sample avg | Divergence-based trust region |
> | MaxRL | $(r_i-\hat{r})/(N\hat{r})$, successful only | None | Sample avg | Approx. max-likelihood |
> | ScaleRL | $(r_i-\mu_\mathcal{B})/\sigma_\mathcal{B}$ | Upper bound only | Prompt avg | Scale-validated defaults |
>
> **Reading the table:** the columns encode the four design decisions that ScaleRL, Dr. GRPO, DAPO, and CISPO all identify as consequential: how to define the baseline, how to clip, how to aggregate, and which prompts to include. Each method makes a specific choice on each axis; the variance across choices is the engineering design space for RL post-training.

---

## A Provisional Recipe

> [!success]
> Synthesizing ScaleRL's findings alongside the convergent evidence from Dr. GRPO, DAPO, CISPO, and the algorithm comparisons across Chapters 07–12, a provisional best-practices recipe for large-scale RL post-training of language models on verifiable tasks:
>
> **Architecture**
> - Use a pipelined async setup: overlap rollout generation and policy updates.
> - Maintain $\pi_{\text{ref}}$ in 4-bit or 8-bit quantization; no gradients needed.
> - Eliminate the value network; use group-relative baselines.
>
> **Numerical precision**
> - Compute the LM head (vocabulary projection) in FP32; keep attention blocks in BF16.
> - Verify that log-probabilities computed during generation and training match numerically.
>
> **Loss and advantage**
> - Normalize at prompt level (not sample level or token level).
> - Use $\hat{A}_i = r_i - \mu_G$ (no $\sigma_r$ normalization) or the MaxRL advantage $(r_i - \hat{r})/\hat{r}$.
> - Use upper-bound-only IS clipping with stop-gradient (CISPO style), or symmetric clipping with $\varepsilon \approx 0.2$.
>
> **Curriculum**
> - Filter zero-variance prompts (all-correct or all-incorrect groups) from gradient updates.
> - Apply positive resampling: exclude prompts with $>90\%$ recent success rate from future epochs.
> - Maintain a dataset with diversity in difficulty; single-difficulty datasets fail to produce the group-baseline dynamics that drive learning.
>
> **KL control**
> - Use a small $\beta$ (RLHF-style values such as 0.1 are often too conservative for verifiable rewards; values in $[0.01, 0.05]$ are more common).
> - Monitor mean KL divergence from $\pi_{\text{ref}}$; training is typically stable up to KL $\approx 10$–$30$ nats for reasoning tasks, substantially more than the $\approx 1$–$3$ nat budgets used in instruction-following RLHF.

> [!warning]
> **Treat this recipe as provisional.** ScaleRL's own framing is explicit: the convergent findings represent "a provisional recipe emerging from the current data" that "can quickly change with new methods or details." The recipe above is validated for:
> - Verifiable binary rewards (mathematics, code)
> - Single-turn generation
> - 7B–70B parameter scale
> - Standard instruction-following tasks as a downstream target
>
> Generalization beyond these conditions — multi-turn tasks, dense continuous rewards, very small or very large model scales, multilingual settings, multimodal inputs — is not established. The recipe is a starting point, not a universal prescription.

---

## Empirical Reliability and the Reproducibility Gap

> [!warning]
> **The most underappreciated open problem in RL for language models, as identified by ScaleRL:** the evidence base is narrow, empirical, and expensive to reproduce.
>
> The typical evaluation structure in the field: a single model family (often a specific DeepSeek or Qwen checkpoint), a single verifier setup (MATH500, AIME, or a code benchmark), a single dataset mix, and a single compute budget. Papers that show strong improvements under these conditions may be finding algorithmic improvements — or they may be finding that their method works particularly well for one model/reward/data configuration, with no guarantee of transfer.
>
> Specific unknowns that remain unresolved:
> - **Scale robustness**: does a method that outperforms alternatives at 7B also outperform at 70B? At 1B? The relationship between model scale and RL method efficacy is not characterized.
> - **Reward generalization**: most methods are validated on mathematical reasoning. Performance on code, formal proof, factual question answering, and tasks requiring multi-step interaction is less established.
> - **Learning trajectory vs final performance**: as ScaleRL emphasizes, an intervention that improves early training may plateau early, while a slower-starting approach may reach higher asymptotic performance. Single-checkpoint comparisons cannot distinguish these.
> - **Base model sensitivity**: the degree to which RL improvements are genuine policy improvements versus the policy recovering behavior latent in the base model (but not expressed in the SFT initialization) is actively debated.

---

## What ScaleRL Establishes and What It Does Not

> [!success]
> The validated conclusions from ScaleRL's 400,000 GPU-hour study:
>
> 1. **Async pipelines improve compute efficiency** without sacrificing final performance. This is a hardware engineering result, not an algorithmic one, and generalizes broadly.
> 2. **FP32 logits improve asymptotic performance** across all methods tested. This is a numerical precision result that applies whenever importance-sampling ratios are computed between generation and training phases.
> 3. **Prompt-level loss aggregation outperforms sample-level** on asymptotic performance. This generalizes the Dr. GRPO and DAPO findings to a larger-scale empirical setting.
> 4. **Zero-variance filtering is efficient** with no asymptotic cost. This is consistent with the theoretical observation that zero-variance groups contribute no gradient.
> 5. **Positive resampling at 90% threshold improves asymptotic performance** at the cost of early training speed — a tradeoff favorable for long training runs.
> 6. **CISPO achieves better asymptotic performance than DAPO** at production scale, suggesting that DAPO's asymmetric clipping and overlong shaping are less important than CISPO's stop-gradient mechanism.

> [!note]
> What ScaleRL does not establish:
> - Which algorithm is theoretically correct (all methods discussed are asymptotically sound variants of policy gradient).
> - Why FP32 logits help by exactly as much as observed (the mechanism is clear; the magnitude is empirical).
> - Whether the recipe transfers to non-mathematical verifiable tasks, multi-turn RL, or dense reward settings.
> - The relative importance of the six findings: whether, say, FP32 logits matter more than prompt-level aggregation, or whether the two interact.
>
> These remain active questions. The practical implication: implement all six validated findings as defaults, since each is individually low-cost and none conflict, and treat the remaining design space (algorithm choice, KL budget, group size, curriculum specifics) as requiring task-specific validation.

---

*ScaleRL is described in Li et al. (2025), ScaleRL: Scaling RL Training for Language Models with Principled Empirical Evaluation. The MiniMax report documenting the FP32 logits finding is from the MiniMax team (2025). GSPO (Group Sequence Policy Optimization) is introduced in Zeng et al. (2025), GSPO: Group Sequence Policy Optimization. The comprehensive algorithm comparison across REINFORCE, PPO, GRPO, RLOO, Dr. GRPO, DAPO, CISPO, DPPO, and MaxRL is synthesized in Aweers (2026), Reinforcement Learning for Reasoning Language Models: A Survey and Empirical Comparison. The sigmoidal fitting methodology for separating early learning speed from asymptotic performance is discussed in the ScaleRL paper and is the primary methodological advance of the study over prior comparisons.*
