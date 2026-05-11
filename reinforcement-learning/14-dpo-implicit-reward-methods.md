---

title: "DPO and Implicit Reward Methods"
subtitle: "Reparameterizing the reward, bypassing the RL loop, and the convergence of offline RL with preference learning"

---


Every algorithm in Part IV — PPO for token generation, GRPO — requires the policy to generate text at training time. Rollouts are the irreducible cost: samples must be drawn from the current policy, rewards must be computed for each, and the policy must be updated before the next round. This on-policy loop, inherited from the robotics RL literature, consumes compute proportionally to the number of tokens generated. For large models, generating a single training batch of responses costs as much as several forward passes through the model; a full RLHF training run generates tens of billions of tokens. The loop is not merely expensive in isolation — it is the bottleneck that determines how quickly the policy can be updated and how large a training signal can be extracted from a fixed compute budget.

The question Rafailov, Sharma, Mitchell, Ermon, Manning, and Finn asked in 2023 was whether the rollout loop is necessary at all. The RLHF objective has a known closed-form optimal solution — Chapter 11 derived it explicitly. If the optimal policy can be expressed analytically, perhaps the learning algorithm can be derived algebraically from that expression, bypassing the RL loop entirely and reducing preference learning to a supervised loss on static preference data. The answer was yes, and the resulting algorithm — **Direct Preference Optimization**, or DPO — became the most widely adopted preference learning method of 2023–2025.

## The Reparameterization

Start from the closed-form optimal policy derived in Chapter 11. For a given reward function $r$ and reference policy $\pi_\text{ref}$, the policy that maximizes the KL-regularized objective is:

$$\pi^*(y \mid x) = \frac{\pi_\text{ref}(y \mid x) \exp\bigl(r(x, y) / \beta\bigr)}{Z(x)},$$

where $Z(x) = \sum_y \pi_\text{ref}(y \mid x) \exp(r(x, y) / \beta)$ is the partition function. This equation expresses the optimal policy in terms of the reward. Rearranging, it expresses the reward in terms of the optimal policy:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_\text{ref}(y \mid x)} + \beta \log Z(x).$$

This reparameterization is exact: for any reward function $r$, there is a corresponding optimal policy $\pi^*$, and knowing $\pi^*$ and $\pi_\text{ref}$ is equivalent to knowing $r$ up to the partition function $Z(x)$, which depends only on the prompt $x$ and not on the response $y$. The reward is encoded in the ratio of the optimal policy to the reference policy, scaled by $\beta$.

## The DPO Derivation

Now substitute this reparameterization into the Bradley-Terry preference model of Chapter 11. The probability that a human prefers response $y_w$ over $y_l$ given prompt $x$ is:

$$P(y_w \succ y_l \mid x) = \sigma\!\bigl(r(x, y_w) - r(x, y_l)\bigr).$$

Substituting the reparameterized reward:

$$P(y_w \succ y_l \mid x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} + \beta \log Z(x) - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)} - \beta \log Z(x)\right).$$

The partition function $Z(x)$ cancels — it appears with opposite signs in the two terms. What remains is:

$$P(y_w \succ y_l \mid x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right).$$

The partition function was the intractable object in RLHF: computing it requires summing over all possible responses, which is exponential in response length. It cancels because the Bradley-Terry model depends only on reward differences, and $Z(x)$ contributes equally to both sides of the difference.

Replacing the unknown $\pi^*$ with the trainable policy $\pi_\theta$, the **DPO loss** is the negative log-likelihood of the observed preferences:

$$\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right)\right].$$

This is a supervised binary cross-entropy loss. The inputs are pairs of completions from a static preference dataset $\mathcal{D}$; there are no rollouts, no reward model, and no RL training loop. The gradient can be computed with two forward passes — one through $\pi_\theta$ and one through $\pi_\text{ref}$ — and a standard backward pass. Training is as straightforward as fine-tuning a language model on labeled sequences.

## The Implicit Reward and What DPO Is Doing

The quantity $\beta \log(\pi_\theta(y \mid x) / \pi_\text{ref}(y \mid x))$ is the **implicit reward** of DPO — the reward function that the trained policy $\pi_\theta$ implicitly represents. It is not a separate network; it is a functional of the policy itself. Optimizing $\mathcal{L}_\text{DPO}$ adjusts $\pi_\theta$ so that the implicit reward of the preferred response exceeds the implicit reward of the rejected response by a margin determined by the observed preference probability.

The gradient of the DPO loss with respect to $\theta$ illuminates the update direction:

$$\nabla_\theta \mathcal{L}_\text{DPO} = -\beta\,\mathbb{E}\!\left[\sigma\!\bigl(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w)\bigr)\Bigl(\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)\Bigr)\right],$$

where $\hat{r}_\theta(y) = \beta \log(\pi_\theta(y \mid x) / \pi_\text{ref}(y \mid x))$ is the implicit reward. The gradient increases the log-probability of preferred responses and decreases the log-probability of rejected responses, weighted by $\sigma(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w))$ — the probability that the current model gets the preference wrong. When the model already ranks $y_w$ above $y_l$ confidently, the weight is small; when the ranking is wrong or uncertain, the weight is large. This is contrastive learning: the policy is pushed to concentrate probability on preferred completions relative to the reference, and away from rejected completions, with the strength of the push proportional to current model error.

The normalization by $\pi_\text{ref}$ is essential. Without it, the gradient would simply increase $\log \pi_\theta(y_w)$ and decrease $\log \pi_\theta(y_l)$ — a contrastive loss on absolute probabilities. This would be unstable: the model could increase $\pi_\theta(y_w)$ by making $y_w$ more likely in absolute terms, but it could also satisfy the constraint by making $y_l$ less likely while keeping $y_w$ unchanged, potentially collapsing probability mass onto a small set of preferred responses. The $\pi_\text{ref}$ normalization anchors the comparison: what matters is not whether $y_w$ is likely in absolute terms but whether $y_w$ is more likely than $\pi_\text{ref}$ predicts and $y_l$ is less likely than $\pi_\text{ref}$ predicts. This is the KL regularization of Chapter 11 expressed implicitly through the loss function rather than explicitly as a penalty term.

## DPO as Offline RL

DPO is formally equivalent to solving the KL-regularized RL problem — it converges to the same optimal policy $\pi^*$ that PPO targets — but it does so via supervised learning on a fixed offline dataset rather than via policy gradient updates on online rollouts. This places DPO squarely in the literature on **offline reinforcement learning**: learning a policy from a static dataset of transitions collected by a possibly different behavior policy, without any interaction with the environment.

The connection to offline RL illuminates both the strength and the limitation of DPO. The strength: offline methods are vastly more data-efficient in the sense that each data point (a preference pair) is used many times across training epochs, without the cost of generating new rollouts. The limitation: offline methods can only learn about the policy from the support of the dataset — the distribution of prompts and responses that appear in $\mathcal{D}$. If the preference dataset contains pairs where both $y_w$ and $y_l$ are poor responses (both marked relative to each other, neither representing the truly best behavior), DPO will learn to prefer the less-bad response while remaining unaware of better responses not in the dataset. The policy cannot improve beyond the quality ceiling implicit in the preference data.

This is the **distribution shift problem** of offline RL, manifesting in the preference learning context. PPO-based RLHF samples from the current policy at each training step — if the current policy starts generating better responses, the rollout data reflects that improvement and the reward signal reinforces it. DPO trains on a fixed dataset collected before training begins; if the optimal policy lies outside the support of that dataset, DPO cannot reach it. The on-policy loop that DPO eliminates was also the mechanism by which the policy could explore and improve beyond the initial data distribution.

## IPO: Fixing DPO's Implicit Assumption

The DPO derivation implicitly assumes that the Bradley-Terry model is the correct model for human preferences — that the probability of preferring $y_w$ over $y_l$ depends only on the difference of their scalar rewards. Azar, Guo, Piot, Munos, Rowland, Valko, and Thacker (2023) identified a more subtle issue: DPO's loss can be driven to zero by making the implicit reward difference $\hat{r}_\theta(y_w) - \hat{r}_\theta(y_l)$ arbitrarily large, regardless of whether this is consistent with the observed preference probability. When the dataset contains deterministic preferences (one response always preferred over another), the optimal DPO solution is to make the reward gap infinite — which corresponds to $\pi_\theta(y_w | x) / \pi_\text{ref}(y_w | x) \to \infty$ while $\pi_\theta(y_l | x) / \pi_\text{ref}(y_l | x) \to 0$. The policy collapses to always generating $y_w$ and never generating $y_l$, regardless of whether intermediate responses might be better than $y_w$.

**IPO** (Identity Preference Optimization) addresses this by replacing the log-sigmoid loss with a squared loss that has a finite optimum:

$$\mathcal{L}_\text{IPO}(\theta) = \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\left[\left(\log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)} - \frac{1}{2\beta}\right)^2\right].$$

The target reward gap $1/(2\beta)$ is finite — the optimal implicit reward difference is not infinity but the value consistent with a preference probability of $\sigma(1/2) \approx 0.62$. IPO avoids the degeneracy of DPO while retaining the same offline, rollout-free training structure. Empirically, IPO is more robust on datasets with near-deterministic preferences and tends to maintain higher policy entropy throughout training.

## SimPO and Reference-Free Methods

A further simplification removes the reference model entirely. The DPO implicit reward $\beta \log(\pi_\theta(y \mid x) / \pi_\text{ref}(y \mid x))$ requires computing $\pi_\text{ref}(y \mid x)$ at each training step — a full forward pass through the reference model for every preference pair in every minibatch. For large models, keeping $\pi_\text{ref}$ resident in memory while training $\pi_\theta$ reintroduces the dual-model memory pressure that GRPO eliminated.

**SimPO** (Simple Preference Optimization, Meng, Xia, He, Goyal, Krishnamurthy, Chen, and Hajishirzi, 2024) replaces the log-ratio reward with the average log-probability of the response under the current policy alone:

$$r_\text{SimPO}(x, y) = \frac{1}{|y|} \log \pi_\theta(y \mid x),$$

normalized by response length to prevent the model from assigning high reward simply by generating shorter responses. The SimPO loss is:

$$\mathcal{L}_\text{SimPO}(\theta) = -\mathbb{E}\!\left[\log \sigma\!\left(\frac{\beta}{|y_w|} \log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l \mid x) - \gamma\right)\right],$$

where $\gamma > 0$ is a margin that encourages a minimum reward gap between preferred and rejected responses. Without the reference model, SimPO loses the theoretical connection to the KL-regularized RL objective — there is no guarantee that the learned policy stays close to any reference — but gains memory efficiency and avoids the need to store or load the reference weights. Empirically, SimPO performs comparably to DPO on alignment benchmarks while requiring roughly half the memory.

## Online DPO and Iterative Methods

The distribution shift limitation of offline DPO motivated a family of **online** and **iterative** variants that re-introduce some form of fresh data generation. In iterative DPO (also called online DPO), the pipeline alternates between two phases: a **data collection phase** in which the current policy $\pi_\theta$ generates new responses to a prompt set, a reward model or human evaluators label the new responses to form fresh preference pairs, and the dataset $\mathcal{D}$ is updated; and a **learning phase** in which $\mathcal{L}_\text{DPO}$ is minimized on the updated dataset. The policy trained in the learning phase becomes the generation policy for the next collection phase.

Iterative DPO recovers the on-policy exploration property of PPO — the preference data reflects the current policy's behavior rather than a fixed prior distribution — while retaining DPO's simplicity and avoiding the per-token value function of PPO. The cost is reintroducing the data collection step that offline DPO eliminated. The number of rollouts per iteration is typically much smaller than a full PPO training run, since each iteration needs only enough responses to build informative preference pairs rather than enough to estimate the full value function. The result is a method that sits between offline DPO and full PPO on the spectrum of on-policy exposure, trading the theoretical cleanliness of offline DPO for better coverage of the current policy's distribution.

**DAPO** (Direct Alignment with Policy Optimization, 2025) and related methods from ByteDance, Meta, and others pushed this further by combining the DPO-style contrastive objective with online sampling, clipping mechanisms borrowed from PPO, and length-normalized rewards — effectively synthesizing the main ideas of the previous chapters into a single training loop that is simultaneously more stable than PPO and more data-efficient than offline DPO. The convergence of these threads illustrates how quickly the frontier moves: DPO's central contribution — the cancellation of the partition function — becomes the module that every subsequent method inherits, with the surrounding infrastructure varying based on the specific requirements of scale, task type, and compute budget.

## The Landscape in 2025

Standing at the end of this arc, from Bellman's recursion in 1957 to the proliferation of DPO variants in 2025, several threads are visible.

The theoretical core has remained remarkably stable. The Bellman optimality equations of Chapter 1, the policy gradient theorem of Chapter 6, and the MaxEnt optimal policy of Chapter 10 are the load-bearing structures. Every algorithm in this notebook is a variation on one of these three foundations — a different computational strategy for the same underlying optimization problem. DPO's central identity, that the reward can be expressed as a log-ratio of optimal and reference policy, is a consequence of the KL-regularized Bellman equations that Chapter 10 derived. The derivation is three lines; the significance is that it eliminates a training loop.

What has changed dramatically is the setting. The early chapters dealt with finite MDPs and lookup tables; by Chapter 13, the state space is the set of all partial language model responses — a space so large that it cannot be enumerated, only sampled. The reward functions of the early chapters were given by the environment; by Chapter 11, reward itself is a learned approximation to human preference, with all the fragility that approximation entails. The policies of the early chapters were small parametric functions; by Chapter 14, they are billion-parameter neural networks whose internal structure is not interpretable even to their designers.

The open problems are correspondingly larger. Reward specification remains unsolved for tasks where ground truth is not computable and human evaluation is unreliable at scale. The distribution shift problem of offline DPO, the reward hacking problem of PPO-based RLHF, and the scalable oversight problem of Chapter 11 are variations on a single deeper question: how can a learning system optimize a proxy for what is wanted when the gap between proxy and intent is not directly observable? No method in this arc answers that question; they manage it with different tradeoffs. The management is sophisticated — the engineering is remarkable — but the question itself remains open.

What the arc does establish is the conceptual vocabulary for approaching it. The tools are in place: the policy gradient theorem gives gradient directions in policy space; the trust region and clipping machinery controls step sizes; the MaxEnt framework provides the information-theoretic language for quantifying how much the policy has changed from a reference; DPO shows how preference data can be used without explicit reward modeling when the optimality conditions are tractable. The progress from 1957 to 2025 is not merely a sequence of algorithms but the construction of a coherent theory — incomplete, actively extended, occasionally wrong about important things — for understanding what it means to learn to act well in a world that responds to your actions.

---

*Direct Preference Optimization is introduced in Rafailov, Sharma, Mitchell, Ermon, Manning, and Finn (2023), Direct Preference Optimization: Your Language Model is Secretly a Reward Model. The degeneracy of DPO under deterministic preferences and the IPO correction are analyzed in Azar, Guo, Piot, Munos, Rowland, Valko, and Thacker (2023), A General Theoretical Paradigm to Understand Learning from Human Feedback. SimPO and length-normalized reference-free optimization are from Meng, Xia, He, Goyal, Krishnamurthy, Chen, and Hajishirzi (2024), SimPO: Simple Preference Optimization with a Reference-Free Reward. Iterative and online DPO variants are described in Xu, Bai, Lin, Ye, Zhou, and Zhou (2023), Some Things Are More CRINGE Than Others: Preference Optimization with the Pairwise Cringe Loss, and in Guo, Xiao, Li, Sun, Gao, and Lin (2024), Direct Language Model Alignment from Online AI Feedback. DAPO and the synthesis of online RL with DPO-style objectives are described in Yu, Liu, Liu, Wu, Zhu, Zheng, Zhang, Zhu, and Guo (2025), DAPO: An Open-Source LLM Reinforcement Learning System at Scale from ByteDance. The connection between offline RL and preference learning is surveyed in the context of conservative Q-learning in Kumar, Zhou, Tucker, and Levine (2020), Conservative Q-Learning for Offline Reinforcement Learning.*
