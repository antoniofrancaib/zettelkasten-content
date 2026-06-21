---
title: "Open Problems"
subtitle: "Credit assignment, sample efficiency, hard-problem coverage, domain generalization, and empirical reliability — the frontier where the theory ends"
---


The thirteen preceding chapters built a coherent arc. It began with Bellman's recursion in 1957 — the observation that optimal decisions can be decomposed into optimal sub-decisions, reducing a planning problem to a fixed-point equation. It passed through temporal difference learning, policy gradient methods, trust regions, maximum entropy RL, and reward modeling to arrive at the contemporary toolkit for fine-tuning language models: PPO, GRPO, DPO, CISPO, and their variants. Chapter 13 grounded this toolkit in the engineering reality of 400,000-GPU-hour ablations.

At each stage, the theoretical foundations are sound. The Bellman equations have convergence proofs. The policy gradient theorem is exact. The MaxEnt optimal policy is derivable from first principles. DPO's partition function cancellation is algebraically exact. The methods work — in the sense that training curves improve, benchmarks rise, and models fine-tuned with these techniques are measurably more capable or aligned than their untreated counterparts.

But the arc does not close. The gap between what the theory guarantees and what the practice requires is not a gap that engineering refinements will fill — it reflects genuine unresolved theoretical and empirical questions. This chapter maps that frontier.

---

## Open Problem 1: Token-Level Credit Assignment

> [!question]
> A language model response consists of hundreds of tokens. Some tokens are causally responsible for a correct or incorrect final answer; others are connective tissue that would appear in any response. All receive the same reward signal. Why is this a problem in principle, and what would a solution look like?

All methods in this series that use outcome-based rewards — PPO, GRPO, Dr. GRPO, MaxRL, RLOO — assign the reward for a complete response to every token in that response. The token that states the incorrect final answer receives the same negative signal as the tokens that correctly set up the problem context. The insight that caused the reasoning error receives the same positive signal as the closing punctuation.

> [!warning]
> **The credit assignment problem is not merely inefficient — it is a source of systematic misdirection.** Consider a response with the structure:
>
> 1. Correct problem setup (tokens 1–50): these tokens are well-written and would appear in any good response.
> 2. Incorrect intermediate step (tokens 51–80): a subtle error that propagates to a wrong final answer.
> 3. Fluent but wrong conclusion (tokens 81–150): confident language around the incorrect result.
>
> With a binary reward of $r = 0$ (wrong), every token receives the same negative gradient signal. Tokens 1–50 — which were correct and should be reinforced — are penalized equally with tokens 51–80, which caused the failure. The gradient update pushes the policy away from the correct setup as much as it pushes away from the incorrect step. For long responses with heterogeneous quality, this misdirection compounds: correct reasoning in one part of the response is penalized because a later part was wrong.
>
> GAE with a value function partially addresses this: in principle, a well-trained critic assigns higher value to the correct setup than to the state after the incorrect step, so the token-level advantages would correctly identify the error location. In practice, the value function faces the hard prediction problem discussed in Chapter 10 — estimating quality from partial responses is difficult, and the critic's approximation errors reintroduce noise that smears the intended per-token signal.

**Process reward models.** The natural solution is to provide a dense reward signal at each reasoning step, rather than only at the final answer. A **process reward model** (PRM) is a neural network trained to evaluate individual steps in a reasoning trace — assigning positive reward to correct steps and negative reward to incorrect or circular ones, before the final answer is known.

> [!note]
> The PRM approach faces three challenges that have prevented it from becoming the standard solution:
>
> 1. **Step boundary definition.** Natural language reasoning does not decompose into clean, discrete steps. A human reasoning trace may mix forward deduction with meta-commentary, hedges, and self-corrections, without clear boundaries separating "steps." Automatic step segmentation is an unsolved problem; human annotation of step boundaries is expensive and inconsistent.
>
> 2. **Annotation cost.** Training a PRM requires labeled preference data at the step level, not just the response level. Each step in each response requires a quality judgment. At the same annotation budget, a PRM can be trained on fewer responses than an outcome-based reward model, and the step-level signal is harder to provide reliably (annotators often disagree on whether an intermediate step is correct even when they agree on the final answer).
>
> 3. **PRM exploitation.** A PRM is a learned proxy for step quality. A sufficiently capable policy will find step patterns that score highly according to the PRM but that do not represent genuine correct reasoning — the same Goodhart's Law dynamics that affect outcome-based reward models, applied at finer granularity and potentially harder to detect because the exploitation occurs within individual steps rather than across the full response.
>
> The problem is real; the proposed solutions (PRMs, step-level verifiers, search-based training, branch-sensitive objectives) each address part of it. None has become standard, and none fully solves the credit assignment problem without introducing new difficulties.

---

## Open Problem 2: Sample Efficiency

> [!question]
> Supervised fine-tuning uses each labeled example as a direct gradient signal. RL post-training uses each rollout as a single bit of information (correct or incorrect). How severe is this gap, and can it be closed?

The information bottleneck is fundamental. In supervised learning on a labeled mathematical problem, the model receives the correct solution directly — thousands of tokens, each providing signal about which token sequences are acceptable. In RL with a binary verifiable reward, the model generates a complete response and learns only whether that specific response was correct or incorrect. The gradient must infer, from this single bit, which tokens in the response were responsible for success or failure.

> [!warning]
> **The quantitative gap.** A supervised training step on a 200-token solution provides approximately $200 \times \log_2(|\mathcal{V}|)$ bits of information — around 3,000 bits for a vocabulary of 32,000 tokens. A binary RL reward provides 1 bit. A single supervised step is information-equivalent to roughly 3,000 binary RL rewards for the same response length. This is not a gap that better algorithms can fully close — it is a consequence of not having labeled solutions.
>
> The practical consequence: RL methods require many rollouts per prompt ($G = 8$–$64$ is standard) just to construct a useful group baseline. The gradient from a group of 8 rollouts is roughly an 8-response averaged estimate of the policy gradient — itself a noisy estimator. The effective information extraction rate per GPU-hour is orders of magnitude lower than in supervised training.

> [!tip]
> **Directions toward better sample efficiency:**
>
> - **Better reuse of failed rollouts.** All group-based methods use only the reward signal from failed rollouts (to compute the group mean and pull policy away from incorrect responses). But failed responses contain information about *how* the policy currently fails — which reasoning patterns, which error types, which superficially plausible but incorrect approaches. Methods that extract richer information from unsuccessful trajectories (beyond the binary $r = 0$ label) could improve per-rollout efficiency.
>
> - **Offline-to-online mixing.** Combining a static dataset of labeled solutions (where each solution provides full supervised signal) with online RL rollouts (where the model explores) could improve early training efficiency. The challenge is off-policy correction: supervised examples are drawn from a fixed distribution, not the current policy.
>
> - **Prompt selection policies.** Rather than sampling prompts uniformly from a training set, an active learning component could select prompts based on the current policy's estimated information gain — preferring prompts where the policy is near the 50% success-rate threshold and the binary signal is maximally informative. Positive resampling (Chapter 13) is a heuristic in this direction; a principled information-theoretic curriculum could do better.
>
> None of these fully closes the supervised-to-RL efficiency gap. The gap is intrinsic to the information content of the reward signal; it can be managed but not eliminated without richer feedback.

---

## Open Problem 3: Hard-Problem Coverage

> [!question]
> When the current policy never produces a correct response for a prompt — $K = 0$ in all $G$ rollouts — every group-based method provides zero gradient. How should a training procedure behave when no correct response exists in the rollout data?

This is the **zero-coverage problem**, and it affects every method in this series. GRPO with $K = 0$ has $\sigma_r = 0$ and skips the update. MaxRL's estimator is explicitly defined as zero when $K = 0$. PPO's value function produces advantages from its own estimates, but if the reward is always zero, the value function converges to predict zero everywhere and the advantages carry no signal. The fundamental constraint is the same: outcome-based learning requires at least one successful trajectory to generate a positive gradient.

> [!warning]
> **The cold-start problem.** For genuinely hard problems — those beyond the current frontier of the model's capability — the probability of producing a correct response in $G$ rollouts may be effectively zero. If the training set contains such problems, the model makes no progress on them regardless of how many GPU-hours are spent. Curriculum learning (starting with easy problems and progressively including harder ones) is the standard mitigation: train on problems the model can occasionally solve, then introduce harder problems as capability improves.
>
> Curriculum learning works in practice but is a workaround rather than a solution. It requires a difficulty oracle (a way to estimate problem difficulty for the current policy, which is itself a nontrivial problem) and a progression schedule (how quickly to introduce harder problems as capability grows). Both are heuristic, and a poorly calibrated curriculum can either waste compute on already-mastered problems (too conservative) or stall by introducing unsolvable problems too early (too aggressive).

> [!note]
> **Partial solutions that have been explored:**
>
> - **STaR (Self-Taught Reasoner, Zelikman et al., 2022)**: For problems where the model cannot generate a correct response, provide the correct final answer as a hint and ask the model to rationalize it — generating a reasoning trace that leads to the known correct answer. The rationalization is used as supervised training data, bootstrapping capability on problems that RL alone cannot reach. This effectively uses a small amount of privileged information (the correct answer) to generate labeled data, rather than learning from pure exploration.
>
> - **Rejection sampling fine-tuning**: Sample many responses for each prompt, keep only the correct ones, and fine-tune on the kept responses. This is supervised learning on a curated self-generated dataset — not RL, but capable of bootstrapping capability on problems where the success rate is very low (even 1% success over 100 rollouts yields a usable example).
>
> - **Warmup on demonstrated solutions**: Before RL training, do one round of SFT on human or model demonstrations of solved hard problems, seeding the policy in a region where RL rollouts occasionally succeed. This is the SFT stage of the InstructGPT pipeline, now recognized as having a dual role: alignment (producing reasonable initial behavior) and bootstrapping (making RL rollouts informative on the hard end of the distribution).
>
> All three approaches require external information (correct answers, demonstrations, or curated data) that may not be available for genuinely frontier tasks. For problems where no solution exists anywhere in the training data or its augmentations, these methods reduce to hoping the model randomly discovers a solution — which, at sufficiently high difficulty, is not a reliable training strategy.

---

## Open Problem 4: Domain Generalization Beyond Mathematics and Code

> [!question]
> Nearly all validated progress in RL post-training of language models uses mathematical reasoning or code generation as the target task. What makes these settings special, and what would be required to extend the methods to other domains?

Mathematical reasoning and code generation are structurally unusual among tasks that language models perform:

- **Binary, unambiguous correctness**: $2 + 2 = 4$ regardless of annotation context. The test suite either passes or it does not. There is no spectrum of "partly correct" that requires human judgment.
- **Automatic verification**: a Python interpreter or a symbolic math checker can evaluate the response without human involvement, making the reward label cheap and scalable.
- **Dense training signal**: for a curriculum of mathematical problems, every problem has a ground-truth answer. There is no label scarcity.
- **Short-horizon evaluation**: each response is evaluated on its own, without reference to previous exchanges.

> [!warning]
> **Most tasks that users actually care about do not satisfy any of these properties:**
>
> | Property | Math/Code | Conversational helpfulness | Medical advice | Creative writing |
> |----------|-----------|--------------------------|---------------|----------------|
> | Binary correctness | Yes | No | Partially | No |
> | Automatic verification | Yes | No | No | No |
> | Label availability | Dense | Sparse, expensive | Expert-only | Subjective |
> | Short-horizon | Yes | Multi-turn | Multi-turn | Variable |
>
> Extending RL post-training to the right half of this table requires solving reward specification for tasks where ground truth is not computable — which brings back the full weight of the problems from Chapter 09 (Bradley-Terry approximations, reward hacking, scalable oversight) that verifiable rewards sidestep.

> [!note]
> **Multi-turn RL** introduces additional structure beyond single-turn generation. In a multi-turn dialogue:
>
> - The state at turn $t$ includes the full conversation history — prompt, prior model responses, and user follow-ups.
> - Actions at turn $t$ affect which future states are reachable: a response that misleads the user at turn 2 forecloses valuable follow-up questions at turns 3–5.
> - The reward may not be evaluable until the conversation concludes, reintroducing the terminal-reward credit assignment problem at a much longer horizon (a 10-turn conversation with a 200-token response per turn has a horizon of 2,000 tokens, most of which receive zero intermediate reward).
> - The reference policy $\pi_{\text{ref}}$ must now define appropriate behavior across multiple turns, and the KL penalty must apply across the full conversation trajectory, not just a single generation.
>
> None of the methods developed in this series addresses multi-turn RL directly. PPO's token-level MDP can in principle be extended to multi-turn settings — the state simply includes more context — but the value function faces an even harder prediction problem (predicting conversation-level outcome from early-turn partial responses), and the KL budget must be allocated across a much longer trajectory.

> [!warning]
> **The coverage problem compounds in novel domains.** For mathematical reasoning, a base language model trained on internet text has seen millions of worked examples of mathematical problems, so even before RL fine-tuning, the base model can solve some fraction of problems at every difficulty level — establishing the positive-coverage condition that RL requires. For novel domains — medical diagnosis from patient records, multi-party negotiation, scientific experiment design — the base model may have seen few or no examples of the specific decision-making format required, and the zero-coverage problem is the default rather than the exception.
>
> This is arguably the deepest structural limitation of the current paradigm: RL post-training amplifies capabilities that exist in the base model; it does not install new ones. What the base model cannot do at all, RL cannot teach it to do better.

---

## Open Problem 5: Empirical Reliability and Unpredictable Scaling

> [!question]
> The methods in this series were validated on specific model families, reward setups, and compute budgets. How confident should practitioners be that results transfer across these conditions?

> [!warning]
> The evidence base for RL post-training methods is systematically narrow. The typical evaluation structure: a single model family (DeepSeek-R1, Qwen 2.5, or Llama 3), a single benchmark suite (MATH500, AIME, HumanEval), a single compute budget ($10^{21}$–$10^{23}$ FLOPs), a single training dataset. Papers showing improvements in this specific intersection may be finding:
>
> - Genuine algorithmic improvements that transfer across conditions.
> - Improvements that hold for this model family but not others (sensitivity to pretraining data distribution, tokenization, or architecture details).
> - Improvements that hold at this benchmark but not others (overfitting to the specific task distribution of the benchmark).
> - Improvements that hold at this compute budget but not at larger or smaller budgets.
>
> These cases are observationally indistinguishable from single-condition evaluations. ScaleRL's contribution (Chapter 13) was to conduct multi-condition evaluations with fixed methodology — but even 400,000 GPU-hours covers only a fraction of the relevant design space.

> [!note]
> **Specific reliability gaps:**
>
> **Learning speed vs asymptotic performance.** As Chapter 13 emphasized, an intervention can improve early training while harming final performance, or vice versa. Methods that show strong early benchmark improvements — which dominate the published literature because evaluations happen at a single checkpoint — may plateau earlier than methods with slower initial learning curves. Without fitting performance-vs-compute curves across a wide range of training durations, the relative ordering of methods cannot be established reliably.
>
> **Base model recovery vs genuine capability improvement.** Language models trained on large text corpora contain substantial latent capability that SFT fine-tuning does not always elicit. RL post-training may be discovering and amplifying capability that exists in the base model weights but is suppressed by the SFT distribution — rather than genuinely teaching the model new reasoning strategies. If so, the improvements would be expected to plateau when the latent capability is exhausted, and the methods would not scale to capabilities beyond the base model's knowledge horizon. Distinguishing these two mechanisms requires probing with problems that are genuinely out-of-distribution for the base model — a methodologically difficult task.
>
> **The scale robustness question.** Most empirical results are established at 7B or 70B parameter scale. Whether CISPO outperforms DAPO at 7B implies nothing about the comparison at 7M or 700B. The relationship between model scale and RL method efficacy — whether larger models are more or less sensitive to implementation details, whether certain methods require minimum scale to work, whether advantages compound or diminish at larger scale — is not characterized. ScaleRL's results are at production scale but for specific model families; the generalization is unknown.

---

## What Genuine Progress Would Look Like

> [!tip]
> Against this landscape of open problems, it is useful to specify what would constitute genuine progress rather than incremental improvement:
>
> **On credit assignment**: a process reward model that (a) can be trained with annotation costs comparable to outcome-based reward models, (b) does not introduce new exploitation surfaces, and (c) consistently improves policy gradient quality on held-out tasks that were not in its training distribution. The bar is not "sometimes useful" but "reliably better than outcome reward in expectation."
>
> **On sample efficiency**: a method that closes the information gap between RL (1 bit per rollout) and supervised learning (thousands of bits per labeled example) by at least an order of magnitude — either through richer reward signals, better reuse of failed trajectories, or principled offline-online mixing — without requiring labeled solutions as input.
>
> **On hard-problem coverage**: an approach that can make genuine progress on problems where the base model's current success rate is zero, without requiring external demonstrations or correct-answer hints. This probably requires fundamentally different exploration strategies than the current rollout-and-reward paradigm.
>
> **On domain generalization**: a reward specification method that works for at least one non-verifiable task type (multi-turn dialogue, factual accuracy, scientific reasoning) with empirical robustness comparable to what verifiable reward methods achieve for mathematics — meaning it does not reward-hack, does not degrade on capability benchmarks not in the preference dataset, and generalizes across base models and scales.
>
> **On empirical reliability**: evaluation methodology that distinguishes learning speed from asymptotic performance, tests across at least three distinct model families and two compute scales, and reports confidence intervals over random seeds and dataset splits. This is not an algorithmic contribution but an epistemic one — it is the standard of evidence that would allow practitioners to trust that a method's advantages are real and transferable.

---

## What the Arc Establishes

Stepping back from the open frontier, it is worth stating clearly what the arc from Chapter 01 to Chapter 13 does establish.

> [!success]
> **The theoretical foundations are complete.** The Bellman optimality equations, the policy gradient theorem, the MaxEnt optimal policy, and the DPO partition function cancellation form a coherent mathematical structure. Any method in the space of KL-regularized policy optimization can be located within this structure, and the structure clarifies what each method is doing and what assumptions it makes. The framework is not fragile: it accommodates continuous and discrete action spaces, on-policy and off-policy data, learned and verifiable rewards, sequence-level and token-level MDPs, within a common information-theoretic language.

> [!success]
> **The engineering toolkit is mature for verifiable-reward settings.** For tasks where binary ground-truth verification is available, the combination of group-relative baselines, token-aware or prompt-aware loss aggregation, FP32 logits, async pipelines, and the zero-variance curriculum produces reliable policy improvement at scale. This is not trivial — five years ago, RL post-training of a 7B language model to produce structured chain-of-thought reasoning was not achievable by any open-source method. The toolkit makes it routine.

> [!note]
> **The open problems are not failures of the framework.** Credit assignment, sample efficiency, and domain generalization are not bugs in the theory — they are consequences of the framework being applied to problems whose structure (subjective reward, multi-turn, no ground truth) lies outside the regime where the theory's assumptions hold. The MaxEnt framework is correct; the Bradley-Terry model is a useful approximation; GRPO's group baseline is unbiased. The problems arise when the inputs to these correct frameworks are imprecise (a learned reward model instead of a true reward, a binary signal instead of a dense one), when the environment is not Markovian in the intended sense (multi-turn conversations), or when the policy lacks the latent capability that RL amplifies.

> [!warning]
> **The remaining gap is between the scope of the theory and the scope of the ambition.** The theory is well-suited to optimizing a policy against a well-specified reward in a well-defined environment. The ambition — building systems that behave helpfully, honestly, and appropriately in the full range of situations that users present — goes far beyond this. The tools developed across these fourteen chapters are genuine progress toward that ambition, not a solution to it. They establish the conceptual vocabulary, the mathematical language, and the engineering infrastructure for approaching it systematically. The approach is incomplete, actively being extended, and occasionally wrong about important things. That is the normal condition of a field in productive development.

---

*The credit assignment problem in the context of language model RL is discussed in Lightman, Kosaraju, Burda, Edwards, Baker, Lee, Leike, Schulman, Sutskever, and Cobbe (2023), Let's Verify Step by Step. The STaR bootstrapping approach for hard-problem coverage is in Zelikman, Wu, Mu, and Goodman (2022), STaR: Bootstrapping Reasoning With Reasoning. The sample efficiency perspective and information-theoretic framing are discussed in Aweers (2026), Reinforcement Learning for Reasoning Language Models: A Survey and Empirical Comparison. The scalable oversight problem and its relationship to reward specification are analyzed in Bowman, Hyun, Perez, Chen, Pettit, Heiner, Lukošiūtė, Askell, Jones, Chen, Goldie, Mirhoseini, McKinnon, Chen, Olsson, Olah, Hernandez, Drain, Ganguli, Li, Tran-Johnson, Perez, Kerr, Mueller, Ladish, Landau, Ndousse, Sakaguchi, Showk, Clark, Kambadur, Anthropic, and Irving (2022), Measuring Progress on Scalable Oversight for Large Language Models. The RL-as-base-model-recovery question is examined empirically in Guo, Yang, Zhang, et al. (2025), DeepSeek-R1, and theoretically questioned in Yue, Chen, Sun, Wei, Zhang, and Liu (2025), Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs?*
