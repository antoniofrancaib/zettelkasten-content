---

title: "Reward Modeling and RLHF"
subtitle: "Preference learning, the Bradley-Terry model, KL-regularized objectives, and the InstructGPT recipe"

---


Every algorithm in the preceding ten chapters assumed the reward function was given. The Bellman equations take $r(s, a)$ as input. Q-learning bootstraps toward it. Policy gradients weight log-probability updates by it. The theory is elegant and the algorithms are powerful, but they all defer the hardest question: where does the reward come from?

For video games, the reward is the game score — a number already computed by the environment, unambiguous and dense. For robotic locomotion, the reward is forward velocity plus a penalty for falling — an engineering choice that happens to produce the right behavior for the specific task. For problems that matter most — training a system to be helpful, to be honest, to produce outputs that humans actually prefer — there is no natural reward function. A language model generating a response to a question does not receive a numeric signal from the environment. The world does not score its answers.

This is the **reward specification problem**, and it is not peripheral to the project of building capable AI systems. It is central. A system that optimizes the wrong reward — one that is easy to specify but imperfectly correlated with what is actually wanted — will find ways to maximize it that are correct by the metric and wrong by the intent. The history of RL applied to the real world is partly a history of reward functions that seemed right and produced unexpected or undesirable behavior. Specification is hard; misspecification is common; the consequences scale with the capability of the optimizer.

## Learning Reward from Preferences

The core insight that broke the deadlock was simple: if you cannot specify reward directly, learn it from human judgments. Ask a person not to assign a numeric score to an output but to say which of two outputs they prefer. Preferences are easier to elicit than scores — they require no calibration of what a "7 out of 10" means, no cardinal encoding of qualitative judgment. Collecting many pairwise preferences from human annotators produces a dataset from which a reward model can be learned: a neural network that, given a state and action (or in the language setting, a prompt and response), outputs a scalar representing how much a human would prefer this output.

This is the foundation of **reinforcement learning from human feedback** (RLHF). The idea appears in several places in the RL literature before its application to language models — most directly in Christiano, Leike, Brown, Martic, Legg, and Amodei (2017), who demonstrated that a reward model learned from pairwise preferences between short video clips of agent behavior could guide PPO to solve tasks in MuJoCo and Atari without any engineered reward signal. The key empirical finding: with enough preferences — on the order of hundreds to low thousands — the learned reward model is accurate enough to drive policy optimization to competitive performance.

The pipeline has three stages: collect preferences, train a reward model, optimize a policy against the reward model. Each stage builds on the infrastructure of the preceding chapters, but the combination is qualitatively different in purpose. The goal is no longer to maximize a reward that the designer believes is correct; it is to maximize a reward that tracks human preferences as revealed through comparisons. The designer's role shifts from writing reward functions to designing the process by which preferences are collected.

## The Bradley-Terry Model

A preference dataset consists of triples $(x, y_w, y_l)$: a prompt $x$, a preferred response $y_w$ (the winner), and a less-preferred response $y_l$ (the loser), labeled by a human annotator. The model for converting a scalar reward function $r_\phi$ into a preference probability is the **Bradley-Terry model**, introduced in 1952 for ranking sports teams from pairwise match outcomes:

$$P(y_w \succ y_l \mid x) = \sigma\!\bigl(r_\phi(x, y_w) - r_\phi(x, y_l)\bigr) = \frac{\exp\bigl(r_\phi(x, y_w)\bigr)}{\exp\bigl(r_\phi(x, y_w)\bigr) + \exp\bigl(r_\phi(x, y_l)\bigr)},$$

where $\sigma$ is the sigmoid function. The model asserts that the probability of preferring $y_w$ over $y_l$ depends only on the difference of their scalar rewards under $r_\phi$. If $r_\phi(x, y_w) > r_\phi(x, y_l)$, the model assigns more than 50% probability to the human preferring $y_w$; the larger the gap, the more certain the preference. The model has no notion of the absolute scale of reward — only differences are identified from preferences — which is why the KL regularization discussed below is essential: without it, the reward model is defined only up to an additive constant per context.

The reward model is trained by maximum likelihood on the preference dataset $\mathcal{D} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}$:

$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\Bigl[\log \sigma\!\bigl(r_\phi(x, y_w) - r_\phi(x, y_l)\bigr)\Bigr].$$

This is a binary cross-entropy loss on the log-odds of the pairwise preference. The gradient pushes $r_\phi$ to increase the reward gap between preferred and less-preferred responses: winning responses get higher reward, losing responses get lower reward, with the magnitude of the update proportional to how wrong the current model is about the preference.

In the language model setting, the reward model is typically initialized from the same pretrained language model as the policy, with the final token-embedding head replaced by a scalar output head. The initialization matters: a reward model initialized from a strong language model already understands text semantics and can focus its learning on the preference signal rather than on basic language understanding.

## The KL-Regularized Objective

With a learned reward model $r_\phi$ in hand, the naive approach is to optimize a policy $\pi_\theta$ to maximize $\mathbb{E}[r_\phi(x, y)]$. This fails in a precise and predictable way: the policy learns to exploit weaknesses of $r_\phi$, producing outputs that score highly according to the reward model but that no human would actually prefer. The reward model is a statistical approximation trained on a finite dataset; it has regions where its predictions are unreliable, and a capable policy optimizer will find and exploit them. This is **reward hacking**: maximizing the proxy rather than the intended objective.

The standard countermeasure is to add a KL penalty that keeps the optimized policy close to a **reference policy** $\pi_{\text{ref}}$ — typically the supervised fine-tuned (SFT) language model before RL training:

$$\max_{\pi_\theta} \; \mathbb{E}_{x \sim \mathcal{X},\, y \sim \pi_\theta(\cdot \mid x)}\!\bigl[r_\phi(x, y)\bigr] - \beta \, \text{KL}\!\bigl[\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\bigr],$$

where $\beta > 0$ is the KL coefficient and the expectation over $x$ is over the distribution of prompts. The KL term penalizes the optimized policy for deviating from the reference: high-reward responses that are very improbable under $\pi_{\text{ref}}$ are discouraged, because the policy must pay a KL cost proportional to $\log(\pi_\theta / \pi_{\text{ref}})$ to assign them high probability.

The KL penalty serves two purposes simultaneously. First, it limits reward hacking: outputs that exploit $r_\phi$'s weaknesses tend to be improbable under $\pi_{\text{ref}}$, and the KL penalty discourages moving probability mass to such regions. Second, it preserves the capabilities learned during pretraining: a policy that deviates far from $\pi_{\text{ref}}$ may improve on the reward metric while losing fluency, factuality, or coherence that $\pi_{\text{ref}}$ had acquired from the pretraining corpus.

## The Optimal Policy in Closed Form

The KL-regularized objective has an exact optimal solution that can be derived in closed form — a result that is both theoretically illuminating and practically consequential.

Define the **pointwise KL-penalized reward** for a response $y$ given prompt $x$:

$$\hat{r}(x, y) = r_\phi(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}.$$

The objective is $\mathbb{E}[\hat{r}(x, y)]$ maximized over $\pi_\theta$. Taking the functional derivative with respect to $\pi_\theta$ and setting it to zero gives the optimal policy:

$$\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x) \exp\bigl(r_\phi(x, y) / \beta\bigr)}{Z(x)},$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp(r_\phi(x, y) / \beta)$ is a normalizing partition function. The optimal policy is the reference policy re-weighted by the exponential of the reward, tempered by $\beta$. Responses with high reward under $r_\phi$ receive higher probability than under $\pi_{\text{ref}}$; responses with low reward receive lower probability; the temperature $\beta$ controls how sharply the distribution concentrates on high-reward responses.

This is exactly the Boltzmann distribution of the MaxEnt framework from Chapter 10, with the soft Q-function replaced by $r_\phi$ and the temperature $\alpha$ replaced by $\beta$. The RLHF objective is a MaxEnt problem in the space of text generations, with the reference policy playing the role of the prior and the reward model playing the role of the energy function. The theoretical unity is complete: the algorithms for language model alignment are not a new invention but an application of the maximum entropy RL framework to a new domain.

## The Three-Stage Pipeline

The RLHF pipeline as applied to language models — formalized in Ziegler et al. (2019) for stylistic fine-tuning, developed for summarization in Stiennon, Ouyang, Wu, Ziegler, Lowe, Voss, Radford, Amodei, and Christiano (2020), and brought to large scale by Ouyang, Wu, Jiang, Almeida, Wainwright, Mishkin, Zhang, Agarwal, Slama, Ray, Schulman, Hilton, Kelton, Miller, Simens, Askell, Welinder, Christiano, Leike, and Lowe (2022) in InstructGPT — has three stages.

**Stage 1: Supervised fine-tuning (SFT).** Starting from a pretrained language model, fine-tune on a dataset of high-quality demonstrations: prompts paired with responses written by human contractors to show what good behavior looks like. This produces $\pi_{\text{ref}}$, the SFT model. The SFT stage is not RL — it is standard maximum likelihood training on demonstration data — but it initializes the policy in a region of behavior space where responses are already reasonable, giving the subsequent RL stage a useful starting point and a meaningful reference distribution.

**Stage 2: Reward model training.** Present human annotators with pairs of responses to the same prompt — both sampled from $\pi_{\text{ref}}$ or from intermediate policy checkpoints — and collect binary preferences. Train $r_\phi$ by minimizing the Bradley-Terry loss on the resulting dataset. In InstructGPT, annotators labeled roughly 50,000 prompt-response pairs; the reward model was initialized from the 6B parameter SFT model with a scalar head.

**Stage 3: RL fine-tuning.** Run PPO with $r_\phi$ as the reward function and a KL penalty against $\pi_{\text{ref}}$. At each step: sample a prompt $x$ from the prompt distribution, generate a response $y \sim \pi_\theta(\cdot \mid x)$, compute the reward $r_\phi(x, y) - \beta \log(\pi_\theta(y \mid x) / \pi_{\text{ref}}(y \mid x))$, and update $\pi_\theta$ using PPO. The details of applying PPO to token generation — computing per-token KL penalties, defining the episode structure for a language model — are the subject of the next chapter.

## InstructGPT: The Empirical Result

InstructGPT demonstrated that RLHF fine-tuning with a relatively small model could outperform a much larger model without RLHF on the dimensions that annotators actually cared about. The 1.3B InstructGPT model — fine-tuned on roughly 13,000 preference comparisons — was preferred over the 175B GPT-3 model by human evaluators on the vast majority of prompts. Size did not substitute for alignment. The specific behaviors that RLHF improved — following multi-step instructions, declining harmful requests, maintaining consistent personas, giving more truthful answers — were difficult to improve by scaling alone and responded directly to the preference signal.

The result had an important subtext. GPT-3 was already a highly capable model by the standards of 2022; the gap between it and InstructGPT was not in language modeling ability but in the alignment of its outputs with human preferences. RLHF, in other words, was not making models smarter — it was steering the distribution of their outputs toward the region that humans found useful and appropriate. The reward model learned a projection from the space of all plausible language model outputs onto the subset that annotators preferred, and PPO moved the policy toward that projection.

## Reward Hacking and the Limits of Proxy Reward

The KL penalty is a partial solution to reward hacking, not a complete one. As $\beta \to 0$, the KL penalty vanishes and the policy is free to exploit $r_\phi$ arbitrarily; as $\beta \to \infty$, the policy cannot move from $\pi_{\text{ref}}$ at all. The value of $\beta$ that produces useful behavior occupies a range that is task-specific and must be tuned.

Even within that range, Gao, Hilton, Mićo, Askell, Ritter, Leike, Schulman, and Christiano (2022) demonstrated a systematic phenomenon: the **reward-KL frontier**. As the KL divergence from $\pi_{\text{ref}}$ increases, the reward model score increases — but the true human preference score eventually turns over. There is a point past which optimizing $r_\phi$ harder makes outputs worse by human judgment, even though the reward model predicts they are better. The gap between the proxy $r_\phi$ and the true preference grows with the degree of optimization. This is Goodhart's law precisely: when a measure becomes a target, it ceases to be a good measure.

The overoptimization phenomenon is predictable from the reward model's training distribution. $r_\phi$ was trained on pairs where the difference in quality was large enough for annotators to express a clear preference. In the tails of the distribution — where the policy generates outputs that are very different from anything in the preference dataset — $r_\phi$'s predictions are extrapolations with no training support. A capable policy optimizer will find these regions and produce outputs that score well by extrapolation while being incoherent, repetitive, or subtly wrong by any other standard.

Mitigations include larger and more diverse preference datasets, iterative reward model updates as the policy evolves, ensemble-based uncertainty estimates that penalize reward predictions with high variance, and conservative KL coefficients. None of these fully eliminates the overoptimization problem; they only push the overoptimization threshold further out along the KL axis. The fundamental issue is that $r_\phi$ is a finite-sample approximation to an infinite-complexity preference function, and sufficiently capable optimization will always find the gaps.

## Scalable Oversight

The reward hacking problem is an instance of a deeper challenge: human annotators can evaluate short, simple responses readily but struggle with long, complex, or technical ones. A human can judge whether a brief customer service reply is helpful, but evaluating a multi-step mathematical proof, a long software architecture document, or a nuanced policy analysis requires expertise and time that may not scale to the volume of comparisons needed for RLHF. If the reward model is trained on comparisons where annotators cannot actually judge quality reliably, it will learn an approximation to what annotators think is good rather than what is actually good — which may diverge substantially for tasks where human judgment is unreliable.

**Scalable oversight** is the research agenda addressing this gap: how can human supervisors maintain meaningful oversight of AI systems whose outputs they cannot fully evaluate? Approaches include debate (having AI systems argue against each other and asking humans to judge the argument), recursive reward modeling (using the AI to assist humans in evaluating AI outputs), and constitutional AI (having the AI apply explicit principles to evaluate its own outputs). None of these fully resolves the fundamental problem, but each shifts the point at which human judgment is applied to a location where it is more reliable.

The scalable oversight problem is ultimately a consequence of capability: the better the language model, the harder it is for a human to evaluate its outputs exhaustively. RLHF works well when the gap between what the model can produce and what humans can evaluate is small; it becomes unreliable as that gap grows. The alignment methods of the next three chapters — PPO for token generation, GRPO, and DPO — all inherit this limitation while addressing various aspects of the computational and statistical infrastructure around it.

---

*Reinforcement learning from human preferences is introduced in Christiano, Leike, Brown, Martic, Legg, and Amodei (2017), Deep Reinforcement Learning from Human Preferences. The Bradley-Terry model for pairwise comparison is from Bradley and Terry (1952), Rank Analysis of Incomplete Block Designs by the Method of Paired Comparisons. RLHF applied to language model fine-tuning is developed in Ziegler, Stiennon, Wu, Brown, Radford, Amodei, Christiano, and Irving (2019), Fine-Tuning Language Models from Human Preferences, and extended to summarization in Stiennon, Ouyang, Wu, Ziegler, Lowe, Voss, Radford, Amodei, and Christiano (2020), Learning to Summarize with Human Feedback. The InstructGPT system and the three-stage RLHF pipeline at scale are described in Ouyang et al. (2022), Training Language Models to Follow Instructions with Human Feedback. The reward-KL overoptimization phenomenon is characterized in Gao, Hilton, Mićo, Askell, Ritter, Leike, Schulman, and Christiano (2022), Scaling Laws for Reward Model Overoptimization. The closed-form optimal policy under KL-regularized reward is derived in the DPO literature; see Rafailov, Sharma, Mitchell, Ermon, Manning, and Finn (2023), Direct Preference Optimization. Constitutional AI and scalable oversight are described in Bai, Jones, Ndousse, Askell, Chen, DasSarma, Drain, Fort, Ganguli, Henighan, et al. (2022), Constitutional AI: Harmlessness from AI Feedback.*
