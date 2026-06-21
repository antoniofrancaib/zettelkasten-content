---
title: "Reward Modeling and RLHF"
subtitle: "Preference learning, the Bradley-Terry model, the KL-regularized objective, and the InstructGPT pipeline"
---


Every algorithm in the preceding eight chapters assumed the reward function was given. The Bellman equations take $r(s, a)$ as input. Q-learning bootstraps toward it. Policy gradients weight log-probability updates by it. The theory is elegant and the algorithms are powerful, but they all defer the hardest question: where does the reward come from?

For video games, the reward is the game score — a number computed by the environment, unambiguous and dense. For robotic locomotion, the reward is forward velocity plus a penalty for falling — an engineering choice that happens to produce the right behavior. For problems that matter most — training a system to be helpful, to be honest, to produce outputs that humans actually prefer — there is no natural reward function. A language model generating a response to a question does not receive a numeric signal from the environment. The world does not score its answers.

This is the **reward specification problem**, and it is not peripheral to the project of building capable AI systems. It is central. A system that optimizes the wrong reward — one easy to specify but imperfectly correlated with what is actually wanted — will find ways to maximize it that are correct by the metric and wrong by the intent. The history of RL applied to the real world is partly a history of reward functions that seemed right and produced unexpected or undesirable behavior. Specification is hard; misspecification is common; the consequences scale with the capability of the optimizer.

---

## Learning Reward from Preferences

> [!question]
> If we cannot specify reward directly, can we learn it from human behavior — specifically from pairwise comparisons between outputs?

The core insight: ask a person not to assign a numeric score to an output but to say which of two outputs they prefer. Preferences are easier to elicit than scores — they require no calibration of what a "7 out of 10" means, no cardinal encoding of qualitative judgment. Collecting many pairwise preferences from human annotators produces a dataset from which a reward model can be learned: a neural network that, given a prompt and response, outputs a scalar representing how much a human would prefer this output.

This is the foundation of **reinforcement learning from human feedback** (RLHF). The key empirical finding (Christiano et al., 2017): with on the order of hundreds to low thousands of preferences, a learned reward model is accurate enough to drive policy optimization to competitive performance — even without any engineered reward signal.

> [!tip]
> The pipeline has three stages: collect preferences, train a reward model, optimize a policy against the reward model. The designer's role shifts from writing reward functions to designing the process by which preferences are collected. The objective is no longer to maximize a reward the designer believes is correct; it is to maximize a reward that tracks human preferences as revealed through comparisons.

---

## The Bradley-Terry Model

> [!info]
> A preference dataset consists of triples $(x, y_w, y_l)$: a prompt $x$, a preferred response $y_w$ (the winner), and a less-preferred response $y_l$ (the loser), labeled by a human annotator. The **Bradley-Terry model** converts a scalar reward function $r_\phi$ into a preference probability:
>
> $$P(y_w \succ y_l \mid x) = \sigma\!\bigl(r_\phi(x, y_w) - r_\phi(x, y_l)\bigr) = \frac{\exp\bigl(r_\phi(x, y_w)\bigr)}{\exp\bigl(r_\phi(x, y_w)\bigr) + \exp\bigl(r_\phi(x, y_l)\bigr)},$$
>
> where $\sigma$ is the sigmoid function. The probability of preferring $y_w$ depends only on the difference of scalar rewards — larger gap means more certain preference. The model has no notion of absolute scale: only differences are identified from preferences.

The reward model is trained by maximum likelihood on the preference dataset $\mathcal{D} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}$:

$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\Bigl[\log \sigma\!\bigl(r_\phi(x, y_w) - r_\phi(x, y_l)\bigr)\Bigr].$$

This is a binary cross-entropy loss on the log-odds of the pairwise preference. The gradient pushes $r_\phi$ to increase the reward gap between preferred and less-preferred responses.

> [!note]
> In the language model setting, $r_\phi$ is typically initialized from the same pretrained language model as the policy, with the final token-embedding head replaced by a scalar output head. The initialization matters: a reward model initialized from a strong language model already understands text semantics and can focus its learning on the preference signal rather than on basic language understanding. The full sequence is passed through the model; the scalar output at the final token position is the reward.

---

## The KL-Regularized Objective

> [!question]
> With a learned reward model $r_\phi$, the naive approach is to maximize $\mathbb{E}[r_\phi(x, y)]$ directly. Why does this fail in a precise and predictable way?

Maximizing $r_\phi$ without constraint produces **reward hacking**: the policy learns to exploit weaknesses of $r_\phi$, generating outputs that score highly according to the reward model but that no human would actually prefer. The reward model is a statistical approximation trained on finite data; it has regions where its predictions are unreliable, and a capable optimizer will find and exploit them.

> [!tip]
> **The KL-regularized objective** adds a penalty keeping the optimized policy close to a **reference policy** $\pi_{\text{ref}}$ — typically the SFT model before RL training:
>
> $$\max_{\pi_\theta} \; \mathbb{E}_{x \sim \mathcal{X},\, y \sim \pi_\theta(\cdot \mid x)}\!\bigl[r_\phi(x, y)\bigr] - \beta \, \text{KL}\!\bigl[\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\bigr].$$
>
> The KL term serves two purposes simultaneously:
> 1. **Limits reward hacking**: outputs that exploit $r_\phi$'s weaknesses tend to be improbable under $\pi_{\text{ref}}$; the KL penalty discourages moving probability mass to such regions.
> 2. **Preserves pretraining capabilities**: a policy that deviates far from $\pi_{\text{ref}}$ may improve the reward metric while losing fluency, factuality, or coherence acquired during pretraining.

---

## The Optimal Policy in Closed Form

> [!success]
> The KL-regularized objective has an exact optimal solution derivable in closed form. Taking the functional derivative of the objective with respect to $\pi_\theta$ and setting it to zero gives:
>
> $$\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x) \exp\bigl(r_\phi(x, y) / \beta\bigr)}{Z(x)},$$
>
> where $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp(r_\phi(x, y) / \beta)$ is a normalizing partition function. The optimal policy is the reference policy re-weighted by the exponential of the reward, tempered by $\beta$. High-reward responses receive higher probability than under $\pi_{\text{ref}}$; low-reward responses receive lower probability; $\beta$ controls how sharply the distribution concentrates on high-reward responses.

> [!tip]
> **This is the Boltzmann distribution of Chapter 08**, with the soft Q-function replaced by $r_\phi$ and the temperature $\alpha$ replaced by $\beta$. The RLHF objective is a **maximum entropy RL problem** in the space of text generations, with the reference policy playing the role of the prior and the reward model playing the role of the energy function. The theoretical unity is complete: language model alignment is not a new invention — it is the MaxEnt framework applied to a new domain.
>
> This closed-form solution is not merely theoretical: DPO (Chapter 11) exploits it directly to bypass the RL training loop entirely, reparameterizing the reward model in terms of the optimal policy and solving the alignment problem with a simple supervised objective.

---

## The Three-Stage Pipeline

> [!note]
> **Stage 1 — Supervised Fine-Tuning (SFT).** Starting from a pretrained language model, fine-tune on a dataset of high-quality demonstrations: prompts paired with responses written by human contractors. This produces $\pi_{\text{ref}}$. The SFT stage is not RL — it is standard maximum likelihood training — but it initializes the policy in a region of behavior space where responses are already reasonable, giving the subsequent RL stage a useful starting point and a meaningful reference distribution.
>
> **Stage 2 — Reward Model Training.** Present human annotators with pairs of responses to the same prompt — sampled from $\pi_{\text{ref}}$ or intermediate policy checkpoints — and collect binary preferences. Train $r_\phi$ by minimizing the Bradley-Terry loss. In InstructGPT, annotators labeled roughly 50,000 prompt-response pairs; the reward model was initialized from the 6B SFT model with a scalar head.
>
> **Stage 3 — RL Fine-Tuning.** Run PPO (Chapter 10) with $r_\phi$ as the reward function and a KL penalty against $\pi_{\text{ref}}$. At each step: sample a prompt $x$, generate a response $y \sim \pi_\theta(\cdot \mid x)$, compute the shaped reward $r_\phi(x, y) - \beta \text{KL}[\pi_\theta \| \pi_{\text{ref}}]$, and update $\pi_\theta$ using PPO.

> [!success]
> **InstructGPT (Ouyang et al., 2022)** demonstrated that RLHF fine-tuning with a relatively small model could outperform a much larger model without RLHF on dimensions that annotators cared about. The 1.3B InstructGPT model — fine-tuned on roughly 13,000 preference comparisons — was preferred over the 175B GPT-3 model by human evaluators on the vast majority of prompts. Size does not substitute for alignment. The specific behaviors RLHF improved — following multi-step instructions, declining harmful requests, giving more truthful answers — responded directly to the preference signal and were difficult to improve by scaling alone.

---

## Reward Hacking and the Limits of Proxy Reward

> [!warning]
> **The reward-KL frontier** (Gao et al., 2022). As KL divergence from $\pi_{\text{ref}}$ increases with training, the reward model score increases — but the true human preference score eventually turns over. There is a point past which optimizing $r_\phi$ harder makes outputs worse by human judgment, even though the reward model predicts they are better. The gap between proxy $r_\phi$ and the true preference grows with the degree of optimization.
>
> This is Goodhart's Law precisely: *when a measure becomes a target, it ceases to be a good measure.* The overoptimization phenomenon is predictable from the reward model's training distribution. $r_\phi$ was trained on pairs where quality differences were clear enough for annotators to express a preference. In the tails of the distribution — where the policy generates outputs very different from anything in the preference dataset — $r_\phi$'s predictions are extrapolations with no training support. A capable optimizer finds these regions and produces outputs that score well by extrapolation while being incoherent, repetitive, or subtly wrong by any other standard.

> [!note]
> **Mitigations** include larger and more diverse preference datasets, iterative reward model updates as the policy evolves, ensemble-based uncertainty estimates, and conservative KL coefficients. None fully eliminates overoptimization — they push the overoptimization threshold further out along the KL axis. The fundamental issue is that $r_\phi$ is a finite-sample approximation to an infinite-complexity preference function, and sufficiently capable optimization will always find the gaps.

---

## Scalable Oversight

> [!question]
> Human annotators can evaluate short, simple responses readily but struggle with long, complex, or technical ones. What happens when the reward model is trained on comparisons where annotators cannot reliably judge quality?

If annotators cannot evaluate quality reliably, the reward model learns an approximation to what annotators *think* is good rather than what is actually good — which diverges substantially for tasks requiring technical expertise or long-form reasoning. The reward model inherits the annotators' blindspots.

> [!warning]
> **Scalable oversight** is the research agenda addressing this gap: how can human supervisors maintain meaningful oversight of AI systems whose outputs they cannot fully evaluate? Approaches include:
> - **Debate**: AI systems argue against each other; humans judge the argument rather than the object-level claim.
> - **Recursive reward modeling**: use the AI to assist humans in evaluating AI outputs.
> - **Constitutional AI**: the AI applies explicit principles to evaluate and revise its own outputs.
>
> None fully resolves the fundamental problem. The scalable oversight problem is ultimately a consequence of capability: the better the language model, the harder it becomes for a human to evaluate its outputs exhaustively. RLHF works well when the gap between what the model can produce and what humans can evaluate is small; it becomes unreliable as that gap grows. Every alignment method that follows inherits this limitation.

---

## Verifiable Rewards: An Exit from the Proxy Problem

> [!tip]
> A different path around reward hacking is to sidestep the learned reward model entirely. **Verifiable rewards** are reward signals computed by checking against ground truth automatically, without a learned model:
> - Mathematical reasoning: the response either arrives at the correct numerical answer or it does not.
> - Code generation: the code either passes the test suite or it does not.
> - Formal theorem proving: the proof checker either accepts the proof or it does not.
>
> Verifiable rewards eliminate the proxy-objective gap entirely because the proxy and the true objective coincide. A response that gives the correct answer is rewarded; one that gives the wrong answer is not — regardless of how fluent or confident it appears. The reward-KL overoptimization curve does not apply: there is no $r_\phi$ to overfit.
>
> The limitation is coverage: verifiable rewards exist only for tasks where ground truth can be checked automatically. The vast majority of tasks that humans care about — helpfulness in conversation, quality of creative writing, accuracy of medical advice — do not have computable ground-truth checks. For those tasks, a learned reward model is unavoidable, and all the limitations above apply.

---

## The Alignment Geometry

The theoretical content of this chapter can be summarized as a single structure. The space of language model outputs has a natural probability geometry — the reference policy $\pi_{\text{ref}}$ defines a prior; human preferences define an energy function via the reward model $r_\phi$. The optimal aligned policy is the Boltzmann distribution that balances these two:

$$\pi^* \propto \pi_{\text{ref}} \cdot \exp(r_\phi / \beta).$$

> [!tip]
> Everything in Chapters 09–12 is a different method for reaching or approximating this optimal distribution:
>
> | Method | Approach |
> |--------|----------|
> | PPO + RLHF (Chapter 10) | Iteratively optimizes toward $\pi^*$ via policy gradient on the shaped reward |
> | DPO (Chapter 11) | Directly estimates $r_\phi$ implicitly from $\pi^*$ and $\pi_{\text{ref}}$; collapses the RL loop |
> | GRPO (Chapter 12) | Eliminates the value function; uses group rollouts to estimate advantages toward $\pi^*$ |
> | Dr. GRPO, DAPO, CISPO (Chapters 07, 12) | Correct biases in the gradient estimator used to reach $\pi^*$ |
>
> The target is the same in every case. The methods differ in how they estimate the gradient toward it, what computational infrastructure they require, and what biases they introduce or correct.

---

*Reinforcement learning from human preferences is introduced in Christiano, Leike, Brown, Martic, Legg, and Amodei (2017), Deep Reinforcement Learning from Human Preferences. The Bradley-Terry model is from Bradley and Terry (1952), Rank Analysis of Incomplete Block Designs. RLHF applied to language model fine-tuning is developed in Ziegler, Stiennon, Wu, Brown, Radford, Amodei, Christiano, and Irving (2019), Fine-Tuning Language Models from Human Preferences, and extended to summarization in Stiennon et al. (2020), Learning to Summarize with Human Feedback. The InstructGPT system is described in Ouyang et al. (2022), Training Language Models to Follow Instructions with Human Feedback. The reward-KL overoptimization phenomenon is characterized in Gao, Hilton, Mićo, Askell, Ritter, Leike, Schulman, and Christiano (2022), Scaling Laws for Reward Model Overoptimization. The closed-form optimal policy under KL-regularized reward is derived in Rafailov, Sharma, Mitchell, Ermon, Manning, and Finn (2023), Direct Preference Optimization. Constitutional AI is described in Bai, Jones, Ndousse, Askell et al. (2022), Constitutional AI: Harmlessness from AI Feedback.*
