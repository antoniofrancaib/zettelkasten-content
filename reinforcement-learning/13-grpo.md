---

title: "GRPO — Group Relative Policy Optimization"
subtitle: "Eliminating the critic, group baselines, verifiable rewards, and the emergence of reasoning"

---


Token-level PPO requires two large neural networks to be resident in memory simultaneously: the policy $\pi_\theta$ and the value function $V_w$. In robotics applications, the value network has the same architecture as the policy but a scalar output; at the scale of a 7B-parameter language model, the value network costs 7B additional parameters in memory plus the gradient buffers and optimizer states needed to train it. For a 70B model trained with full precision, the value function alone consumes tens of gigabytes of GPU memory that might otherwise accommodate a larger batch size, a longer context, or a second rollout stream. The memory pressure is not a peripheral engineering concern — it is the binding constraint that determines what can be trained on what hardware.

Beyond memory, the value function introduces a subtle epistemological problem. In robotics, the value function estimates future discounted reward from a given state, and the state changes at every time step. In the token-level MDP, the value function estimates future reward from a partial response $(x, y_{1:t})$, but since the only reward is the terminal reward model score, the value at position $t$ is essentially the expected final reward given the tokens generated so far — a function that is hard to estimate accurately from prefixes, because the quality of a response is not in general predictable from its first half. A value network trained on terminal rewards learns a slow, noisy approximation to this expectation, and the advantage estimates it produces carry the noise of that approximation through every token in every response.

GRPO — Group Relative Policy Optimization, introduced by Shao, Wang, Zhu, Xu, Song, Zhang, Li, Wu, and Guo (2024) in the DeepSeekMath paper — eliminates the value function entirely. It replaces the learned critic with an analytical baseline computed from a group of responses to the same prompt. The insight is old — the Monte Carlo baseline of Chapter 6 — applied in a new context that makes it unusually effective: when the reward is binary or otherwise easily computable, and when sampling multiple responses to the same prompt is cheap relative to training a critic, the group mean is a better baseline than a learned value function.

## The Group Sampling Scheme

For each prompt $x$ in a training batch, GRPO samples $G$ independent responses from the current policy:

$$y_1, y_2, \ldots, y_G \;\sim\; \pi_{\theta_{\text{old}}}(\cdot \mid x).$$

Each response $y_i$ receives a scalar reward $r_i$ from the reward signal — either a learned reward model score $r_\phi(x, y_i)$ or a verifiable reward based on ground truth. The group of $G$ rewards $\{r_1, \ldots, r_G\}$ provides a local sample of the reward distribution for this prompt under the current policy.

The **group baseline** is the mean reward within the group:

$$\bar{r} = \frac{1}{G} \sum_{i=1}^G r_i.$$

The **group-relative advantage** for response $i$ is its reward centered by the group mean and normalized by the group standard deviation:

$$\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r + \varepsilon}, \quad \sigma_r = \sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \bar{r})^2},$$

where $\varepsilon$ is a small constant for numerical stability. Responses that scored above the group average have positive advantage; responses that scored below have negative advantage. The normalization ensures that advantage magnitudes are comparable across prompts whose reward distributions have different scales.

The group-relative advantage is an unbiased estimate of $A^\pi(x, y_i) = Q^\pi(x, y_i) - V^\pi(x)$ under the policy $\pi_{\theta_\text{old}}$, for exactly the same reason that the REINFORCE baseline is unbiased: $\bar{r}$ depends on the other samples in the group but not on $y_i$ directly, so it satisfies the zero-mean property required of a valid baseline. The bias-variance tradeoff is explicit: larger $G$ reduces the variance of the baseline estimate (the group mean is a better estimate of $V^\pi(x)$) at the cost of $G$ forward passes to generate the group. In practice, $G$ between 4 and 16 is standard.

## The GRPO Objective

The group-relative advantages replace the GAE advantages in the PPO-Clip objective. For each response $y_i$ in the group, the token-level importance ratio at position $t$ is:

$$r_t^{(i)}(\theta) = \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})},$$

and the GRPO objective accumulates the clipped surrogate across all tokens in all responses in the group:

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{x}\!\left[\frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\!\Bigl(r_t^{(i)}(\theta)\,\hat{A}_i,\; \text{clip}\!\bigl(r_t^{(i)}(\theta),\, 1 - \varepsilon,\, 1 + \varepsilon\bigr)\,\hat{A}_i\Bigr)\right] - \beta\,\overline{\text{KL}}\!\bigl[\pi_\theta \,\|\, \pi_{\text{ref}}\bigr],$$

where $|y_i|$ is the length of response $i$ and the $1/|y_i|$ normalization prevents longer responses from dominating the loss by contributing more tokens. The KL term is computed as in Chapter 12 — either per-token against $\pi_\text{ref}$ embedded in the shaped reward, or as a sequence-level penalty.

Two features distinguish this from PPO. First, the advantage $\hat{A}_i$ is constant across all token positions $t$ within response $i$: every token in a high-reward response receives the same positive advantage, and every token in a low-reward response receives the same negative advantage. There is no per-token credit assignment within a response — the group baseline can say that response $i$ was better than average but cannot say which tokens within $i$ were responsible. This is a genuine limitation relative to a well-trained value function, and it matters for long responses where some parts are good and others are poor. Second, the baseline is computed analytically from the group rather than from a trained network, which means the advantage estimates are available immediately from the rollout batch without any additional forward pass.

## Verifiable Rewards and the Math Setting

GRPO's design is most powerful when paired with **verifiable rewards** — reward signals that can be computed automatically by checking against ground truth, without a learned reward model. For mathematical reasoning tasks, the reward is binary: the response either arrives at the correct numerical answer or it does not. For code generation, the reward is whether the generated code passes a suite of test cases. For formal theorem proving, the reward is whether the proof checker accepts the proof.

Verifiable rewards sidestep the reward hacking problem entirely. There is no $r_\phi$ to exploit: the reward is not a neural network approximation to human preference but a deterministic function of correctness. A response that produces the right answer to a math problem is rewarded; one that produces the wrong answer is not, regardless of how fluent, confident, or plausible it appears. The reward surface has no exploitable proxy structure because the proxy and the objective coincide.

The consequence for training stability is substantial. Without a learned reward model that can be overfit, the overoptimization curve of Chapter 11 does not apply — there is no reward-KL frontier where the proxy score diverges from true quality. The policy can be trained more aggressively, with lower KL penalties, because reward hacking requires a gap between proxy and objective that verifiable rewards eliminate. In practice, GRPO on math benchmarks uses $\beta$ values an order of magnitude smaller than InstructGPT-style RLHF, and training runs substantially longer before degrading.

The binary reward structure also fits naturally with the group baseline. When $G = 8$ responses are generated for a math problem and some subset solves it correctly, the group mean $\bar{r}$ is the empirical success rate on that problem under the current policy. A response that solves the problem has advantage $(1 - \bar{r}) / \sigma_r$ — normalized by how difficult the problem currently is. If the policy already solves the problem 90% of the time, a correct response has low positive advantage (the problem is easy); if it solves it 10% of the time, a correct response has high positive advantage (the problem is hard). The group baseline automatically implements a form of difficulty-weighted credit assignment: the policy concentrates learning effort on problems where its current performance is near chance and harvests little gradient from problems it has already mastered.

## DeepSeek-R1 and Reasoning Incentives

GRPO gained wide attention through DeepSeek-R1 (Guo, Yang, Zhang, and colleagues, 2025), which used it to train a language model to produce extended chain-of-thought reasoning before giving a final answer. The architecture was specific: responses were required to follow a format with an explicit thinking section delimited by special tokens, followed by a final answer. The reward combined a format reward (whether the response used the required structure) and an accuracy reward (whether the answer was correct, verified against ground truth for mathematics and coding problems).

The training setup applied GRPO with $G = 8$ samples per problem, binary accuracy rewards, and a small format penalty for responses that violated the think-then-answer structure. The policy was initialized from a base language model — not an instruction-tuned or RLHF-fine-tuned model — and trained directly with RL, without the supervised fine-tuning stage that InstructGPT required.

The result was a model that had learned to use extended reasoning traces to solve difficult problems. But the more striking observation was about the nature of what was learned. The trained model exhibited behaviors not present in the base model and not in any supervised demonstration: it spontaneously produced **self-correction** within the thinking section, reconsidering earlier steps when they led to inconsistencies; it **allocated reasoning effort adaptively**, producing longer thinking traces for harder problems and shorter ones for simpler ones; and it occasionally exhibited what the paper called the "aha moment" — a point within a thinking trace where the model appeared to recognize a flaw in its approach and switch strategies.

## The Aha Moment as an Emergent Behavior

The aha moment is worth examining precisely because of what it is and what it is not. It is not a behavior that was demonstrated in supervised training data — the base model was trained without think-then-answer examples before RL. It is not an explicit reward signal — the reward function gave credit only for correct final answers, not for the quality or structure of the reasoning process. It emerged from the optimization pressure of a reward that required correct answers to hard problems, given a policy parameterization (the large language model) that could in principle produce extended, structured reasoning.

The mechanism is speculative but coherent. Among the $G$ responses sampled for a difficult problem, the ones that produced correct answers tended to include more structured, self-correcting reasoning traces — because on hard problems, first attempts at the answer are often wrong and a response that catches and corrects early errors is more likely to arrive at the right answer. The group-relative advantage assigned positive credit to these responses and negative credit to responses that committed to an early error without recovering. Over many gradient updates on many problems, the policy learned to produce the self-correction pattern that was statistically associated with correct answers, even though no individual gradient update explicitly targeted that pattern.

This is reinforcement learning working as intended: the reward gradient, applied at scale to a sufficiently expressive policy, discovers behavioral strategies that are difficult to specify directly. The aha moment is not magical — it is the same phenomenon as DQN learning to play Breakout, or AlphaGo learning to sacrifice stones strategically — but its manifestation in language-based reasoning has different implications for what these systems are doing and how they should be evaluated.

## GRPO Versus PPO: The Tradeoffs

The elimination of the value function is GRPO's central innovation and its central limitation. On the benefit side: no value network means no additional memory, no additional training objective, no critic warmup phase, and no risk that a poorly trained critic generates noisy advantages that destabilize the actor. The implementation is simpler, the memory footprint is smaller, and the training loop is shorter.

On the limitation side: the group baseline is a high-variance estimator of $V^\pi(x)$ when $G$ is small, and it is a sequence-level rather than token-level baseline. PPO's value function — when it is well trained — provides a per-token estimate of expected future return that can identify which specific parts of a response were above or below average. GRPO assigns the same advantage to every token in a response, which is correct in expectation but noisy in practice for long responses with heterogeneous quality. The variance reduction from a well-trained critic exceeds the variance reduction from a group of 8 or 16 samples unless the response length is short or the reward is binary with high variance in the group.

The practical implication: GRPO is best suited to tasks with verifiable, low-noise rewards and relatively short responses where token-level credit assignment matters less. Tasks with dense, continuous reward signals and long, structured responses — multi-turn dialogue, creative writing evaluated by a rich reward model — are better served by PPO's critic, which can fit the continuous reward surface more accurately than a group mean. The two methods occupy different niches in the same space, and the choice between them should be driven by the reward structure of the target task.

## Group Size and Practical Considerations

The group size $G$ governs the bias-variance tradeoff of the baseline. At $G = 1$, GRPO degenerates to REINFORCE with no baseline — maximum variance, no centering. At $G \to \infty$, the group mean converges to the true value $V^\pi(x)$ — minimum variance, but requiring infinitely many rollouts per prompt. In practice, $G$ is limited by the memory available to store $G$ concurrent response contexts during the rollout phase and by the latency cost of generating $G$ responses before any gradient update can be computed.

A related design choice is whether to normalize within the group (dividing by $\sigma_r$ as above) or only center (subtracting $\bar{r}$ without normalizing). Normalization makes the gradient magnitude independent of the reward scale — useful when different prompts have different reward ranges — but discards information about how much variance there is in the group. When all $G$ responses have the same reward (e.g., all correct or all incorrect), $\sigma_r = 0$ and the normalized advantages are undefined; the standard fix is to skip the gradient update for such groups entirely, since a group with no variance provides no useful learning signal.

The DeepSeekMath and DeepSeek-R1 implementations used $G = 8$ throughout, running 8 parallel generations per prompt and computing group statistics before each gradient step. The per-prompt batch structure maps naturally to multi-GPU rollout: each GPU generates a subset of the $G$ responses for each prompt in parallel, the rewards are computed, and the group statistics are synchronized across devices before the backward pass. The implementation overhead relative to PPO is primarily the $G$-fold increase in rollout compute, which is offset by the elimination of the value network's forward and backward passes.

## What GRPO Does Not Address

The absence of a value function eliminates the critic bottleneck but does not address the fundamental constraint shared by all on-policy methods: rollouts must come from the current policy. GRPO samples $G$ responses from $\pi_{\theta_\text{old}}$ at each step and discards them after the update, just as PPO discards its rollout batch. The per-prompt cost is $G$ times higher than PPO's single rollout, trading memory for compute. There is no replay buffer, no reuse of past responses, no off-policy correction.

For reasoning tasks with verifiable rewards, this is acceptable — generating mathematical reasoning traces is cheap relative to real-world interaction, and the binary reward signal provides a strong learning signal per response. For tasks where generating a rollout is expensive — robot manipulation in the physical world, tasks requiring expensive API calls, any setting where the environment is slow — the on-policy requirement remains the binding constraint, and GRPO does not improve on PPO in this regard.

The other invariant is the reward specification problem. GRPO with verifiable rewards sidesteps the learned reward model and its failure modes, but only for tasks where ground truth can be checked automatically. The vast majority of tasks that humans care about — helpfulness in conversation, quality of creative writing, accuracy of medical advice — do not have computable ground-truth checks. For those tasks, a learned reward model is unavoidable, and all the limitations of Chapter 11 apply equally to GRPO and PPO.

---

*GRPO is introduced in Shao, Wang, Zhu, Xu, Song, Zhang, Li, Wu, and Guo (2024), DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. The application of GRPO to reasoning model training, the aha moment observations, and the DeepSeek-R1 system are described in Guo, Yang, Zhang, Yang, Dong, Zhu, Wu, Zhu, Liu, Luo, Bi, Chen, Xu, Yu, Wu, Li, et al. (2025), DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning with Verifiable Rewards. The connection between the group baseline and the REINFORCE baseline is analyzed in Williams (1992), Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. The broader context of reinforcement learning for verifiable reasoning, and the STaR self-taught reasoning approach that GRPO superseded, is in Zelikman, Wu, Mu, and Goodman (2022), STaR: Bootstrapping Reasoning With Reasoning.*
