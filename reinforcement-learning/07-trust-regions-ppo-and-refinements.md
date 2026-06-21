---
title: "Trust Regions: TRPO, PPO, and Their Modern Refinements"
subtitle: "From Fisher geometry to clipped surrogates, and from PPO to DAPO, CISPO, and DPPO"
---


The actor-critic methods of Chapter 06 resolved the variance problem that made REINFORCE impractical. They did not resolve — and in some ways sharpened — a different problem: the sensitivity of policy gradient updates to step size.

The difficulty is structural. In supervised learning, the parameters $\theta$ live in a Euclidean space: the distance $\|\theta' - \theta\|$ is a meaningful measure of how much the model has changed, and a gradient step of size $\alpha$ produces a change in behavior that scales predictably with $\alpha$. In reinforcement learning with parameterized policies, this correspondence breaks down. A small change in $\theta$ in one region of parameter space can radically alter the policy — collapsing a diffuse distribution to a near-deterministic one, or reversing the ordering of action probabilities — while an equally large change elsewhere may be nearly invisible in behavior space. The Euclidean geometry of parameter space is the wrong geometry for measuring policy change.

The consequence in practice: an aggressive step size that works well for ten iterations suddenly destroys the policy on the eleventh, because the gradient happened to point into a region of parameter space where the policy landscape is steep. Practitioners resorted to heuristics — small fixed step sizes, gradient clipping, periodic restarts — that traded speed for stability without theoretical grounding. The problem was not implementation but geometry.

---

## The Natural Gradient

> [!question]
> What is the right geometry for measuring how much a policy has changed, and what gradient direction does it imply?

Policies $\pi_\theta$ are probability distributions, and probability distributions form a statistical manifold equipped with a canonical metric: the **Fisher information matrix**:

$$F(\theta) = \mathbb{E}_{\pi_\theta}\!\bigl[\nabla_\theta \log \pi_\theta(A \mid S)\, \nabla_\theta \log \pi_\theta(A \mid S)^\top\bigr].$$

$F(\theta)$ is positive semi-definite and encodes local curvature of the distribution manifold: directions in parameter space that cause large changes in the policy have large Fisher information; directions that are nearly redundant — where moving the parameters has little effect on the distribution — have small Fisher information.

> [!info]
> The **natural gradient** (Amari, 1998; Kakade, 2001) pre-multiplies the ordinary gradient by the inverse Fisher:
> $$\tilde{\nabla}_\theta J(\theta) = F(\theta)^{-1} \nabla_\theta J(\theta).$$
> It solves $\arg\max_{\Delta\theta} \nabla_\theta J(\theta)^\top \Delta\theta$ subject to $\Delta\theta^\top F(\theta) \Delta\theta \leq \varepsilon$ — the steepest ascent direction under the constraint that the KL divergence between old and new policies stays within a ball of radius $\varepsilon/2$. In other words, the natural gradient is the direction of steepest ascent when progress is measured in **units of policy change**, not parameter change.

> [!tip]
> The natural gradient is **parameterization-invariant**: if two parameterizations of the same policy produce the same distribution, the natural gradient is the same in both. A reparameterization that stretches parameter space in one direction does not change the natural gradient direction, because the Fisher metric stretches correspondingly. This is the property the Euclidean gradient lacks — and the property that trust region methods need.

The practical obstruction: $F(\theta)$ is a $d \times d$ matrix where $d$ is the number of parameters. Forming and inverting it costs $O(d^2)$ memory and $O(d^3)$ time — infeasible for modern networks with millions of parameters.

---

## The Surrogate Objective and Monotone Improvement

> [!question]
> Is there a single objective function that, when maximized, guarantees the true performance $J(\theta)$ does not decrease?

Define the **surrogate objective** — the performance objective approximated by an importance-weighted average under the old policy:

$$L_{\theta_{\text{old}}}(\theta) = \mathbb{E}_{(S_t, A_t) \sim \pi_{\theta_{\text{old}}}}\!\left[\frac{\pi_\theta(A_t \mid S_t)}{\pi_{\theta_{\text{old}}}(A_t \mid S_t)} A^{\pi_{\theta_{\text{old}}}}(S_t, A_t)\right].$$

Two useful properties: (1) $L_{\theta_{\text{old}}}(\theta_{\text{old}}) = J(\theta_{\text{old}})$ — the surrogate and true objective agree at the current policy. (2) Their gradients agree at $\theta_{\text{old}}$. Locally, optimizing the surrogate is equivalent to optimizing the true objective.

> [!success]
> **Policy improvement bound** (Schulman et al., 2015, building on Kakade & Langford, 2002):
> $$J(\theta) \geq L_{\theta_{\text{old}}}(\theta) - C \cdot \max_s \,\text{KL}\!\bigl[\pi_{\theta_{\text{old}}}(\cdot \mid s) \,\|\, \pi_\theta(\cdot \mid s)\bigr],$$
> where $C = 4\gamma \epsilon / (1 - \gamma)^2$ and $\epsilon = \max_{s,a} |A^{\pi_{\theta_{\text{old}}}}(s,a)|$.
>
> Any $\theta$ that increases $L_{\theta_{\text{old}}}(\theta)$ by more than $C$ times the KL divergence is guaranteed to increase the true objective. **This is the monotone improvement guarantee**: a policy update is safe whenever the KL divergence is small enough to keep the surrogate correction smaller than the surrogate gain.

---

## TRPO: The Principled Trust Region

> [!note]
> **Trust Region Policy Optimization (TRPO)** (Schulman et al., 2015) translates the monotone improvement bound into an optimization problem. Rather than using the conservative theoretical bound (which depends on the worst-case advantage and is highly pessimistic), TRPO uses the mean KL as the constraint:
> $$\max_\theta \; L_{\theta_{\text{old}}}(\theta) \quad \text{subject to} \quad \mathbb{E}_{S}\!\bigl[\text{KL}\bigl(\pi_{\theta_{\text{old}}}(\cdot \mid S) \,\|\, \pi_\theta(\cdot \mid S)\bigr)\bigr] \leq \delta.$$
> The KL constraint is equivalent, to second order, to $(\theta - \theta_{\text{old}})^\top F(\theta_{\text{old}}) (\theta - \theta_{\text{old}}) \leq \delta$ — exactly a natural gradient step with step size chosen to saturate the constraint. The TRPO update direction and the natural gradient are the same thing.
>
> The constraint is enforced via **conjugate gradients** (to compute $F^{-1}g$ without forming $F$) plus a backtracking line search. Each update requires $\approx 10$ Fisher-vector products and 10–20 line search steps — roughly $10\text{–}20\times$ the compute of a standard gradient step.

> [!success]
> On the MuJoCo continuous control benchmark — humanoid locomotion requiring coordination of 20+ joints — TRPO solved tasks that had been essentially unachievable by prior methods. REINFORCE and A3C produced catastrophic policy collapses on tasks where large parameter-space steps dramatically changed behavior. TRPO's trust region prevented this entirely.

> [!warning]
> **TRPO's cost**: the conjugate gradient solver, Fisher-vector products (double backward passes), and line search make TRPO roughly 10–20× more expensive per update than a standard gradient step. The full implementation is several hundred lines of nontrivial numerical code. Extending it to recurrent policies or shared actor-critic architectures requires non-trivial modifications. TRPO is theoretically complete and practically difficult.

---

## PPO: The Practical Approximation

> [!question]
> Can the essential content of the trust region constraint — keep the policy from changing too much — be enforced by a mechanism that lives entirely within a standard first-order optimizer?

Schulman, Wolski, Dhariwal, Radford, and Klimov (2017) answered yes. The key object is the **importance ratio**:

$$r_t(\theta) = \frac{\pi_\theta(A_t \mid S_t)}{\pi_{\theta_{\text{old}}}(A_t \mid S_t)}, \qquad r_t(\theta_{\text{old}}) = 1.$$

> [!tip]
> **PPO-Clip objective**: clip the importance ratio to $[1-\varepsilon,\, 1+\varepsilon]$ before multiplying by the advantage, and take the minimum of clipped and unclipped terms:
> $$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\Bigl[\min\bigl(r_t(\theta)\,\hat{A}_t,\; \text{clip}(r_t(\theta),\, 1 - \varepsilon,\, 1 + \varepsilon)\,\hat{A}_t\bigr)\Bigr].$$
>
> **Why the minimum, not just the clip?** The construction is asymmetrically conservative:
> - When $\hat{A}_t > 0$: gradient is zeroed once $r_t > 1 + \varepsilon$ — no incentive to push the probability further above old probability.
> - When $\hat{A}_t < 0$: gradient is zeroed once $r_t < 1 - \varepsilon$ — no incentive to push probability further below old probability.
> - But: if $r_t$ has already drifted in the *wrong* direction (e.g., $r_t < 1-\varepsilon$ with $\hat{A}_t > 0$), the gradient is nonzero and corrects it back.
>
> The minimum creates a **pessimistic lower bound** on the unclipped surrogate: it removes the incentive for large changes in either direction, while preserving gradients that correct the ratio when it has already moved too far.

> [!note]
> **Multiple epochs of minibatch updates** — the practical advantage that separates PPO from both REINFORCE and TRPO. A batch of $T \times K$ transitions (e.g., $2048 \times 8 = 16{,}384$) is collected, then $M = 10$ epochs of minibatch gradient ascent are run on the same data. The clipped objective keeps importance ratios bounded throughout all epochs, making this safe. The result: $\approx 10\times$ more gradient signal per environment interaction than any single-step method.
>
> **Combined loss**:
> $$\mathcal{L}(\theta) = \mathbb{E}_t\!\Bigl[L^{\text{CLIP}}_t - c_1(\hat{V}_\theta(S_t) - V_t^{\text{target}})^2 + c_2\,H[\pi_\theta(\cdot \mid S_t)]\Bigr].$$

> [!success]
> PPO became the most widely deployed policy optimization algorithm in deep RL: OpenAI Five (Dota 2), AlphaStar, InstructGPT, and a large fraction of the robotics learning literature all use PPO as their core optimizer. The reason is not theoretical elegance but **engineering fit** — the full algorithm fits in a few hundred lines of standard deep learning code, uses no specialized solvers, and the defaults $(\varepsilon = 0.2,\, \lambda = 0.95,\, \gamma = 0.99,\, M = 10)$ work well across a broad range of tasks with minimal tuning.

> [!warning]
> **What PPO gives up.** The clipped objective is a heuristic — it creates a flat region in the landscape that *discourages* large updates, but does not *prevent* them. A large learning rate can push the policy across the clip boundary, producing an update that the objective assigns zero gradient but the true objective may penalize severely. TRPO's monotone improvement guarantee does not carry over. Occasional catastrophic updates occur in practice, particularly late in training when entropy is low and small parameter changes produce large KL divergences.

---

## From Classical RL to Language Models: New Failure Modes

PPO was designed for continuous control and Atari — environments with dense rewards, relatively short horizons, and a single response per state. Applying it to autoregressive language model post-training exposed failure modes that the original design did not anticipate. Three empirical observations, accumulated across RLOO (Chapter 06), Dr. GRPO (Chapter 12), and the work below, motivated systematic refinements to the trust region machinery:

> [!note]
> **Observation 1 — Clipping is rarely active.** Across multiple LLM post-training experiments, PPO's ratio clipping was active in fewer than 5% of update steps. The trust region mechanism that cost so much to design and motivated so much theoretical work turns out to be largely idle in this regime. This suggests the importance ratios are naturally staying near 1 — not because of the clip, but because on-policy rollouts and small learning rates keep policies close together.
>
> **Observation 2 — Symmetric clipping is wrong for rare tokens.** PPO clips equally in both directions: $r_t \in [1-\varepsilon, 1+\varepsilon]$. For rare tokens — tokens with low probability under the reference policy — even a small absolute probability increase corresponds to a large ratio. The symmetric upper clip $1+\varepsilon$ suppresses the gradient for these tokens, preventing the model from learning to produce them even when they are strongly rewarded. The upward clip is too tight for the left tail of the token distribution.
>
> **Observation 3 — Loss aggregation level matters.** Whether the loss is averaged over tokens, sequences, or prompts produces meaningfully different per-token learning signals and creates systematic biases (e.g., toward longer or shorter responses). This is not a minor implementation detail.

Each of the three methods below responds to one or more of these observations.

---

## DAPO: Decoupled Advantage Policy Optimization

> [!question]
> If PPO's clipping is rarely the binding constraint, and symmetric clipping penalizes rare tokens incorrectly, can we redesign the trust region from first principles for the LLM setting?

**DAPO** (Decoupled Advantage Policy Optimization; Yu et al., 2025) proposes four modifications to the standard PPO objective, each motivated by a specific failure mode:

> [!note]
> **Modification 1 — Token-level loss aggregation.** Standard PPO averages the surrogate loss at the sequence level — each sequence contributes equally regardless of length. This causes longer sequences to contribute more total gradient signal (more tokens summed before averaging), creating an implicit incentive for verbosity. DAPO switches to token-level averaging: loss is summed over all tokens across all sequences in a batch, then divided by the total token count. Each token contributes equally to the gradient, regardless of which sequence it came from.
>
> **Modification 2 — Asymmetric clipping bounds.** Rather than symmetric $[1-\varepsilon, 1+\varepsilon]$, DAPO uses:
> $$r_t(\theta) \in \bigl[1 - \varepsilon_{\text{low}},\; 1 + \varepsilon_{\text{high}}\bigr], \quad \varepsilon_{\text{low}} = 0.2,\; \varepsilon_{\text{high}} = 0.28.$$
> The wider upper bound allows rare tokens — those with small reference probability — more room to grow their probability when rewarded, without triggering the clip. The lower bound is unchanged: suppressing probabilities for penalized tokens requires tighter control.
>
> **Modification 3 — Overlong response shaping.** Long responses that exceed a target length are penalized with a gradually increasing reward penalty proportional to the excess length, rather than hard truncation. This prevents the model from gaming sequence-level averaging by generating arbitrarily long responses.
>
> **Modification 4 — Dynamic sampling.** At each update step, DAPO filters out prompts where all $G$ rollouts received the same reward (all correct or all incorrect). Such prompts produce zero-variance advantage estimates and contribute no learning signal. Dynamic sampling ensures every prompt in the active batch has mixed outcomes, concentrating compute on the informative margin.

> [!tip]
> The four modifications are orthogonal: token-level aggregation corrects a length bias in the loss; asymmetric clipping corrects a probability-scale bias in the trust region; overlong shaping corrects a gaming incentive; dynamic sampling corrects a sample efficiency waste. Each addresses a specific failure mode independently of the others, making ablations straightforward.

> [!warning]
> The asymmetric clipping bound ($\varepsilon_{\text{high}} > \varepsilon_{\text{low}}$) introduces an asymmetry in the trust region that is not theoretically grounded in the monotone improvement bound — that bound is symmetric in the KL divergence. DAPO's justification is empirical: the symmetric bound provably suppresses rare-token probability growth, and asymmetric bounds fix this without measurably destabilizing training. Whether this asymmetry holds across domains beyond mathematical reasoning and code generation is an open question.

---

## CISPO: Clipped Importance Sampling Policy Optimization

> [!question]
> PPO's clipping works by zeroing the gradient for importance ratios outside $[1-\varepsilon, 1+\varepsilon]$. Is zeroing the right operation — or does it throw away too much information?

**CISPO** (Clipped Importance Sampling Policy Optimization; Hu et al., 2025) targets PPO's hard gradient masking. When $r_t(\theta)$ falls outside the clip range, PPO's objective is flat and the gradient is exactly zero for that token. The update for that token contributes nothing to the parameter step — neither reinforcing nor correcting. CISPO argues this is too aggressive.

> [!tip]
> **CISPO's modification**: clip only the importance-sampling *weight* and apply a stop-gradient to it, but keep the gradient of the log-policy flowing:
> $$L^{\text{CISPO}}_t(\theta) = \text{clip}(r_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon)\big|_{\text{stop-grad}} \cdot \hat{A}_t \cdot \log \pi_\theta(A_t \mid S_t).$$
>
> - The IS weight $r_t(\theta)$ is clipped for variance control — extreme weights do not dominate the gradient.
> - The weight is treated as a constant (stop-gradient) — no gradient flows through $r_t$.
> - But the gradient of $\log \pi_\theta$ is always active — the policy always receives a gradient signal proportional to the clipped weight, regardless of whether $r_t$ is inside or outside the clip range.
>
> In PPO: tokens with $r_t > 1+\varepsilon$ receive *zero* gradient. In CISPO: the same tokens receive gradient weighted by $1+\varepsilon$ (the clipped constant). The learning signal is attenuated but not zeroed.

> [!success]
> Empirically, CISPO achieves $\approx 2\times$ step efficiency improvement over DAPO in controlled experiments — reaching the same performance level in half the gradient steps. The intuition: PPO wastes many tokens by zeroing their gradients when they fall outside the clip; CISPO recovers those gradients at a reduced weight, extracting more signal from each update step. At scale (ScaleRL findings, Chapter 13), CISPO and similar soft-clipping methods outperform DAPO asymptotically.

> [!note]
> The stop-gradient on the IS weight is essential: without it, differentiating through $r_t(\theta)$ would produce a second-order term (gradient of the ratio times the advantage) that destabilizes training. The stop-gradient reduces the update to a reweighted policy gradient where the weights happen to be clipped — a clean interpolation between PPO's hard masking and the unclipped surrogate.

---

## DPPO: Divergence PPO

> [!question]
> PPO's trust region uses importance ratios $\pi_\theta / \pi_{\text{old}}$ as a proxy for distributional change. How faithful is this proxy — and can we replace it with an actual measure of distributional divergence?

**DPPO** (Divergence PPO; Zuo et al., 2025) questions the fundamental premise of ratio-based clipping. The probability ratio $r_t(\theta) = \pi_\theta(a \mid s) / \pi_{\text{old}}(a \mid s)$ measures the relative change in the probability of a *single token*. But what we actually care about is the change in the *distribution over sequences* — a quantity that depends on all tokens jointly, not on individual token ratios.

> [!warning]
> **The rare-token problem, stated precisely.** For a token with reference probability $p = 0.001$, a new probability of $p' = 0.002$ gives ratio $r = 2.0$ — outside PPO's clip at $\varepsilon = 0.2$, triggering zeroed gradient. But the absolute probability change is $0.001$ — a change so small it barely shifts the output distribution. Conversely, for a common token with $p = 0.5$, a ratio of $r = 1.3$ is within the clip, yet the absolute probability change is $0.15$ — a substantial shift. The ratio is a poor proxy for actual distributional change when token probabilities span several orders of magnitude, as they do in language model distributions.

> [!tip]
> **DPPO's alternative**: replace the ratio-based clip with a constraint on the estimated total variation (TV) or KL divergence between the current and old policy, computed at the *sequence level* rather than token-by-token:
> $$\text{Update if: } \widehat{\text{TV}}(\pi_\theta,\, \pi_{\text{old}}) \leq \delta_{\text{TV}} \quad \text{or} \quad \widehat{\text{KL}}(\pi_{\text{old}} \,\|\, \pi_\theta) \leq \delta_{\text{KL}}.$$
> These divergences are estimated from the rollout batch using Monte Carlo samples. Tokens whose inclusion would push the estimated divergence over threshold receive zero gradient weight; all others receive full gradient. The threshold is a hard distributional constraint rather than a per-token ratio bound.

> [!success]
> DPPO finds that fewer than 0.5% of gradient updates cause instability when using divergence thresholds calibrated to typical training dynamics — consistent with the earlier finding that PPO's ratio clip was active in only 5% of steps. The divergence-based criterion is more principled: it measures what matters (distributional distance) rather than a local proxy (per-token ratio), and it is less sensitive to the scale of individual token probabilities.

> [!warning]
> The divergence estimation from a finite batch introduces its own noise: estimated TV or KL divergence is a random variable, and thresholding on a noisy estimate can incorrectly mask or pass tokens. DPPO requires careful calibration of the estimator and threshold to avoid either excessive masking (degrading learning signal) or insufficient masking (allowing large distributional shifts). This calibration is more complex than tuning a single $\varepsilon$ parameter, and the method is less mature than PPO or DAPO in terms of practical deployment experience.

---

## The Trust Region Evolutionary Line: A Summary

> [!tip]
> Reading these methods as an evolutionary sequence reveals a consistent theme: each generation tightens the correspondence between what the trust region mechanism *claims* to constrain and what it *actually* constrains.
>
> | Method | Trust region mechanism | What it measures |
> |--------|----------------------|-----------------|
> | TRPO | Hard KL constraint via conjugate gradient | True average KL divergence |
> | PPO | Per-token ratio clipping, symmetric | Per-token probability ratio |
> | DAPO | Asymmetric ratio clipping + token aggregation | Per-token ratio, corrected for scale bias |
> | CISPO | Clipped IS weight, stop-gradient | Per-token ratio (attenuated, not zeroed) |
> | DPPO | Sequence-level divergence threshold | Actual distributional distance |
>
> TRPO had the most principled mechanism but was computationally infeasible at LLM scale. PPO replaced the hard constraint with a heuristic that works well in practice. DAPO, CISPO, and DPPO each address a specific failure of the heuristic in the LLM regime: DAPO corrects for aggregation-level and probability-scale biases; CISPO recovers gradient signal that PPO wastes by hard masking; DPPO replaces the per-token proxy with an actual distributional measure.
>
> No consensus has emerged on which is definitively best. ScaleRL (Chapter 13) provides the most controlled large-scale comparison and favors CISPO-style soft clipping at large compute budgets — but the ranking is compute- and domain-dependent.

---

*Natural gradient for statistical models is developed in Amari (1998), Natural Gradient Works Efficiently in Learning. The natural policy gradient is introduced in Kakade (2001), A Natural Policy Gradient. The policy improvement bound is from Kakade and Langford (2002), Approximately Optimal Approximate Reinforcement Learning, sharpened in Schulman, Levine, Jordan, Abbeel, and Moritz (2015), Trust Region Policy Optimization. Proximal Policy Optimization is introduced in Schulman, Wolski, Dhariwal, Radford, and Klimov (2017), Proximal Policy Optimization Algorithms. DAPO is from Yu, Liu, Liu, Luo, Qin, Zhang, and others (2025), DAPO: An Open-Source LLM Reinforcement Learning System at Scale. CISPO is from Hu and colleagues (2025), CISPO: Clipped Importance Sampling Policy Optimization. DPPO is from Zuo and colleagues (2025), DPPO: Divergence-Based Proximal Policy Optimization for LLM Post-Training. ScaleRL comparative findings are described in Chapter 13.*
