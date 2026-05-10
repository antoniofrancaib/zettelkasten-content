---

title: "Steering the Flow"
subtitle: "Classifier guidance, classifier-free guidance, reward steering, conditional flow matching, inverse problems, and compositional generation — 2021 to 2025"
---



An unconditional generative model learns to sample from a distribution $p_{\text{data}}$. But in practice, we almost never want a random sample from the full data distribution — we want an image of a specific subject, a molecule with specific properties, a protein that binds a specific target. The question of how to steer a trained generative model toward a desired subset or property of the data distribution is the subject of this chapter. The answer turns out to involve a single mathematical operation — tilting the distribution by a likelihood or reward function — realized through different mechanisms depending on whether the guidance signal is available at training time, inference time, or both.

The central tension is between **fidelity** (how well the guided sample satisfies the condition) and **diversity** (how much variation remains among guided samples). Perfect fidelity means every sample exactly matches the condition — but if the condition is "a photo of a cat," there should still be many different cats. The guidance strength parameter $w$, which appears in every method in this chapter, controls this tradeoff: low $w$ gives diverse but weakly conditioned samples; high $w$ gives precise but repetitive samples. Understanding this tradeoff — and its mathematical origin as a temperature parameter on a tilted distribution — is the unifying theme.

## 15.1 The Conditional Generation Problem

The formal setting: given a condition $y$ (a class label, a text prompt, a partial observation, an energy function), generate samples from the conditional distribution $p(x | y)$. By Bayes' rule:

> [!equation] 15.1
> $$p(x | y) = \frac{p(y | x)\, p(x)}{p(y)} \propto p(y | x)\, p(x),$$


where $p(x)$ is the unconditional data distribution (the prior), $p(y|x)$ is the likelihood of the condition given the data, and $p(y)$ is the evidence (a normalizing constant). The conditional distribution is the unconditional distribution *tilted* by the likelihood.

In the language of generative flows: the unconditional model provides access to $p(x)$ through its score function $\nabla_x \log p_t(x)$ or velocity field $v_t(x)$. The guidance signal provides access to $p(y|x)$ or its gradient. The conditional score is:

> [!equation] 15.2
> $$\nabla_x \log p_t(x | y) = \nabla_x \log p_t(x) + \nabla_x \log p_t(y | x),$$


where $p_t(y|x)$ is the likelihood at noise level $t$ — the probability of the condition given a noisy version of the data. The conditional velocity field is derived from this conditional score via the probability flow ODE relationship.

The challenge is computing $\nabla_x \log p_t(y|x)$ — the gradient of the likelihood with respect to the noisy data $x$ at intermediate noise levels. Different methods handle this differently, and the differences define the landscape of guided generation.

## 15.2 Classifier Guidance

The most direct approach: train a separate classifier $p_\phi(y|x, t)$ on noisy data at all noise levels, and use its gradient to steer the generative process.

> [!definition] 15.1 — Classifier Guidance
> Given an unconditional diffusion model with score $s_\theta(x, t) \approx \nabla_x \log p_t(x)$ and a noise-conditional classifier $p_\phi(y|x, t)$, the **classifier-guided score** is:
>
> $$\tilde{s}(x, t) = s_\theta(x, t) + w \cdot \nabla_x \log p_\phi(y | x, t),$$
>
> where $w \geq 0$ is the **guidance scale**. At $w = 0$, sampling is unconditional. At $w = 1$, sampling targets $p(x|y)$ exactly (in the limit of perfect models). At $w > 1$, sampling targets a *sharpened* conditional $p(x|y)^w / Z_w$ that concentrates on the most likely samples given $y$.


The guided SDE or ODE replaces the unconditional score with the guided score $\tilde{s}$ — everything else (the noise schedule, the solver, the architecture) remains unchanged. The classifier gradient $\nabla_x \log p_\phi(y|x, t)$ acts as an additional force that pushes samples toward regions where the condition is satisfied.

> [!remark] 15.1
> The guidance scale $w > 1$ has no Bayesian justification — it corresponds to sampling from a tempered posterior $p(x|y)^w$ that is *more concentrated* than the true conditional. In practice, $w \in [2, 10]$ produces the best samples by human evaluation, despite being statistically "wrong." The explanation is that the unconditional model $p(x)$ already assigns some mass to low-quality samples (blurry images, implausible molecules), and the extra guidance removes this mass. The price is reduced diversity — high $w$ produces near-identical samples for a given condition.


## 15.3 Classifier-Free Guidance

Classifier guidance requires training a separate classifier on noisy data — an additional model with its own training pipeline. **Classifier-free guidance** (CFG) eliminates this requirement by using the generative model itself as an implicit classifier.

The key identity: the classifier gradient can be written as the difference between conditional and unconditional scores:

$$\nabla_x \log p_t(y|x) = \nabla_x \log p_t(x|y) - \nabla_x \log p_t(x).$$

Substituting into the guided score:

> [!equation] 15.3
> $$\tilde{s}(x, t) = s_\theta(x, t) + w \cdot [s_\theta(x, t | y) - s_\theta(x, t)] = (1 + w)\, s_\theta(x, t | y) - w \cdot s_\theta(x, t).$$


> [!definition] 15.2 — Classifier-Free Guidance
> A **classifier-free guided** model is trained as a *conditional* model $s_\theta(x, t, y)$ that can also run unconditionally by replacing the condition with a null token $\varnothing$: $s_\theta(x, t) \equiv s_\theta(x, t, \varnothing)$. During training, the condition is dropped (replaced by $\varnothing$) with probability $p_{\text{uncond}}$ (typically 10–20%). At inference, the guided prediction is:
>
> $$\tilde{s}(x, t) = (1 + w)\, s_\theta(x, t, y) - w \cdot s_\theta(x, t, \varnothing),$$
>
> which requires two forward passes per step: one conditional, one unconditional.


CFG has become the dominant guidance method for large-scale generative models (Stable Diffusion, DALL-E, Imagen) for several reasons: (1) no separate classifier is needed; (2) the unconditional model is obtained for free by label dropout during training; (3) the method applies to any type of condition (text, class, image) without modification.

> [!proposition] 15.1 — CFG as Tempered Bayes
> Classifier-free guidance with scale $w$ samples from the tilted distribution:
>
> $$\tilde{p}_w(x | y) \propto p(x | y) \cdot \left(\frac{p(x|y)}{p(x)}\right)^w \propto \frac{p(y|x)^{1+w}\, p(x)}{p(y)^{1+w}}.$$
>
> This is Bayes' rule with the likelihood raised to the power $1+w$: the guidance sharpens the influence of the condition relative to the prior. At $w = 0$, this is the true posterior; at large $w$, the posterior concentrates on the maximum-likelihood estimate $\arg\max_x p(y|x)$.


## 15.4 Conditional Flow Matching

For flow-based models, conditional generation can be built directly into the flow matching framework without the detour through scores and guidance.

> [!definition] 15.3 — Conditional Flow Matching
> Given paired data $(x_1, y)$ where $x_1$ is the data and $y$ is the condition, **conditional flow matching** trains a velocity field $v_\theta(x, t, y)$ using the conditional flow matching objective:
>
> $$\mathcal{L}_{\mathrm{CFM}}(\theta) = \mathbb{E}_{t,\, (x_1, y) \sim p_{\text{data}},\, \epsilon \sim \mathcal{N}(0,I)}\!\left[\|v_\theta(x_t, t, y) - (x_1 - \epsilon)\|^2\right],$$
>
> where $x_t = tx_1 + (1-t)\epsilon$ is the linear interpolant. At inference, the ODE $\dot{x} = v_\theta(x, t, y)$ is solved with the desired condition $y$.


Conditional flow matching is simply flow matching with the condition as an additional input to the network. No guidance scale is needed at inference — the model directly generates from $p(x|y)$ rather than requiring post-hoc steering. CFG can still be applied on top of conditional flow matching for additional sharpening, using the same formula (15.3) with $v_\theta$ replacing $s_\theta$.

The advantage of direct conditional training is efficiency: one forward pass per step (vs. two for CFG). The disadvantage is that the condition must be available at training time — you cannot guide toward conditions that were not seen during training.

## 15.5 Reward-Based and Energy-Based Steering

A more general setting: instead of a discrete condition $y$, the guidance signal is a **reward function** $R(x)$ or **energy function** $U(x)$ that assigns a scalar score to each generated sample. The goal is to generate samples that maximize the reward (or minimize the energy) while maintaining diversity.

The tilted distribution is:

> [!equation] 15.4
> $$p_\beta(x) \propto p(x) \cdot e^{\beta R(x)},$$


where $\beta > 0$ is the inverse temperature. At $\beta = 0$, sampling is unconditional. As $\beta \to \infty$, sampling concentrates on the mode $\arg\max_x R(x)$ within the support of $p(x)$.

For a flow-based or diffusion-based model, the score of the tilted distribution at noise level $t$ is:

$$\nabla_x \log p_{t,\beta}(x) = \nabla_x \log p_t(x) + \beta\, \nabla_x R_t(x),$$

where $R_t(x) = \mathbb{E}[R(X_1) | X_t = x]$ is the expected reward conditioned on the current noisy state. This is analogous to classifier guidance, with the reward gradient replacing the classifier gradient.

> [!remark] 15.2
> Computing $R_t(x) = \mathbb{E}[R(X_1) | X_t = x]$ — the expected reward at the endpoint given a noisy intermediate — is the central challenge of reward-based guidance. Two approaches: (1) **one-step denoising**: approximate $R_t(x) \approx R(\hat{x}_1(x, t))$ where $\hat{x}_1$ is the one-step denoising estimate (Tweedie formula); (2) **learned value function**: train a neural network $V_\phi(x, t) \approx R_t(x)$ by regressing against the reward of denoised samples. Approach (1) is cheap but noisy at high noise levels; approach (2) is accurate but requires an additional training phase.


This framework connects to the reinforcement learning literature: the reward function $R(x)$ plays the role of a terminal reward, the diffusion process is the dynamics, and the guidance is the optimal policy. The value function $R_t(x)$ is the state-value function, and the guidance gradient $\nabla_x R_t$ is the policy gradient. Fine-tuning generative models with reinforcement learning (DDPO, DRaFT, ReFL) makes this connection explicit.

## 15.6 Posterior Sampling and Inverse Problems

A particularly important class of conditional generation problems is **inverse problems**: given a measurement $y = A(x) + \eta$ (where $A$ is a known forward operator and $\eta$ is noise), recover $x$ from $y$. This arises in medical imaging (MRI reconstruction from undersampled k-space), computational photography (super-resolution, inpainting, deblurring), and scientific measurement (cryo-EM structure determination).

The posterior is $p(x|y) \propto p(y|x)\, p(x)$, where $p(y|x) = \mathcal{N}(y; A(x), \sigma_\eta^2 I)$ for Gaussian noise. The posterior score is:

> [!equation] 15.5
> $$\nabla_x \log p(x|y) = \nabla_x \log p(x) - \frac{1}{\sigma_\eta^2} A^\top(A(x) - y).$$


For *linear* inverse problems ($A$ is a matrix), the likelihood gradient is known analytically: $\nabla_x \log p(y|x) = -\sigma_\eta^{-2} A^\top(Ax - y)$. The guided score at noise level $t$ uses the one-step denoising estimate $\hat{x}_0(x_t, t)$ to approximate the likelihood:

$$\nabla_{x_t} \log p(y | x_t) \approx -\frac{1}{\sigma_\eta^2} \nabla_{x_t}\!\left[\|A\hat{x}_0(x_t, t) - y\|^2\right],$$

which requires backpropagating through the denoiser — one additional backward pass per solver step.

> [!definition] 15.4 — Diffusion Posterior Sampling (DPS)
> **Diffusion Posterior Sampling** replaces the intractable noise-level likelihood $p_t(y|x_t)$ with an approximation based on the one-step denoising estimate:
>
> $$p_t(y | x_t) \approx \mathcal{N}\!\left(y;\; A(\hat{x}_0(x_t, t)),\; r_t^2 I\right),$$
>
> where $\hat{x}_0(x_t, t) = \mathbb{E}[X_0 | X_t = x_t]$ is the Tweedie denoiser and $r_t$ is a noise scale that accounts for the uncertainty in the denoising estimate. The guided score is:
>
> $$\tilde{s}(x_t, t) = s_\theta(x_t, t) - \frac{1}{r_t^2}\nabla_{x_t}\|A(\hat{x}_0(x_t, t)) - y\|^2.$$


DPS and its variants (PSLD, $\Pi$GDM, RED-diff) have achieved state-of-the-art results on inverse problems across imaging modalities, often outperforming task-specific methods that were trained end-to-end for each inverse problem.

## 15.7 Compositional Generation

A powerful consequence of the score-based formulation is **compositionality**: multiple conditions can be combined by adding their score contributions.

Given conditions $y_1, \ldots, y_K$ with independent likelihoods $p(y_k|x)$, the joint conditional score decomposes as:

> [!equation] 15.6
> $$\nabla_x \log p(x | y_1, \ldots, y_K) = \nabla_x \log p(x) + \sum_{k=1}^K \nabla_x \log p(y_k | x).$$


Each condition contributes an additive guidance term, and the terms can be combined freely at inference time without retraining. This enables:

- **Multi-attribute generation**: "a red car on a rainy street" = color guidance + object guidance + scene guidance
- **Negation**: "a landscape without people" = landscape guidance $-$ person guidance
- **Logical operations**: conjunction (add scores), disjunction (logsumexp of scores), negation (subtract scores)

> [!remark] 15.3
> Compositionality is exact when the conditions are truly independent given $x$ — meaning $p(y_1, y_2 | x) = p(y_1|x) p(y_2|x)$. In practice, conditions are rarely independent (the color and shape of an object are correlated), so the composed guidance is approximate. The approximation is usually good enough for practical use, but it can produce artifacts when conditions conflict (e.g., "a round square"). The severity of the approximation depends on the degree of conditional dependence and on the guidance scale — high $w$ amplifies both the guidance and its artifacts.


## 15.8 The Guidance-Diversity Tradeoff

All guidance methods in this chapter share a common parameter — the guidance strength $w$ (or inverse temperature $\beta$) — that controls the tradeoff between fidelity to the condition and diversity of the generated samples. This tradeoff has a precise information-theoretic characterization.

The guided distribution $\tilde{p}_w(x|y) \propto p(x|y)^{1+w} / p(x)^w$ has entropy:

$$H(\tilde{p}_w) = H(p(\cdot|y)) - w \cdot \mathrm{KL}(p(\cdot|y) \| p) + O(w^2),$$

which decreases linearly with $w$ for small $w$. Each unit of guidance strength costs approximately $\mathrm{KL}(p(\cdot|y) \| p)$ nats of diversity — the mutual information between $x$ and $y$.

In practice, the optimal $w$ depends on the application:
- **Image generation**: $w \in [3, 8]$ (high guidance, favoring quality over diversity)
- **Scientific sampling**: $w = 1$ (exact posterior, diversity is essential)
- **Creative applications**: $w \in [1, 3]$ (moderate guidance, balancing novelty and relevance)
- **Inverse problems**: $w$ is determined by the measurement noise level $\sigma_\eta$ (higher SNR in the measurement allows stronger guidance)

The guidance-diversity tradeoff is the generative analog of the exploration-exploitation tradeoff in reinforcement learning: guidance exploits the condition, while unconditional sampling explores the data distribution.

## 15.9 Historical Notes

**Classifier guidance** was introduced by **Dhariwal and Nichol (2021)** ("Diffusion Models Beat GANs on Image Synthesis"), who trained noise-conditional classifiers and demonstrated that guided diffusion models produce higher-quality images than GANs on ImageNet. The guidance scale $w > 1$ and its effect on the FID-IS tradeoff were characterized in this paper.

**Classifier-free guidance** was introduced by **Ho and Salimans (2022)** ("Classifier-Free Diffusion Guidance"), who showed that label dropout during training provides a free unconditional model and that the two-evaluation guidance formula (15.3) matches or exceeds classifier guidance without a separate classifier. CFG has since become the standard guidance method for large-scale text-to-image models: **Ramesh et al. (2022)** (DALL-E 2), **Saharia et al. (2022)** (Imagen), and **Rombach et al. (2022)** (Stable Diffusion) all use CFG.

The tilted-distribution interpretation of guidance was formalized by **Karras et al. (2022)** and **Bradley (2024)** ("Classifier-Free Guidance is a Predictor-Corrector"). The connection to reward maximization was developed by **Black et al. (2024)** (DDPO), **Clark et al. (2024)** (DRaFT), and **Xu et al. (2024)** (Imagereward), who fine-tuned diffusion models with reinforcement learning objectives.

**Diffusion Posterior Sampling (DPS)** was introduced by **Chung, Kim, Mccann, Klasky, and Ye (2023)** ("Diffusion Posterior Sampling for General Noisy Inverse Problems"). Related approaches include **Song, Shen, Xing, and Ermon (2022)** for linear inverse problems, **Kawar, Elad, Ermon, and Song (2022)** (DDRM) for spectral methods, and **Chung et al. (2023)** ($\Pi$GDM). The application to cryo-EM reconstruction was demonstrated by **Levy, Poitevin, Marber, et al. (2024)** (CryoFM).

Compositional generation was explored by **Liu, Karalias, Rempel, and Fidler (2022)** ("Compositional Visual Generation with Composable Diffusion Models") and **Du, Li, and Mordatch (2023)**. The information-theoretic analysis of the guidance-diversity tradeoff appears in **Ho and Salimans (2022)** and was further developed by **Kynkäänniemi, Karras, Laine, Lehtinen, and Aila (2024)**.

Conditional flow matching was developed as a natural extension of the flow matching framework by **Lipman et al. (2023)** and **Tong et al. (2024)**. The application to molecular design — conditioning on molecular properties, binding targets, or experimental constraints — is an active area developed by **Jing et al. (2024)** (AlphaFold 3's diffusion module), **Yim et al. (2023)** (FrameFlow), and **Song et al. (2024)** (EquiFM).
