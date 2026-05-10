---

title: "The Learning Problem"
subtitle: "Maximum likelihood, regression losses, score matching, energy-based training, noise schedules, and the art of choosing what the network predicts — 2014 to 2025"
---



The preceding chapters have described *what* to learn — a velocity field, a score function, a drift, a generator — and *where* to learn it — Euclidean space, manifolds, discrete spaces, product spaces. This chapter addresses *how* to learn it: what loss function to minimize, what the neural network should output, how to weight the loss across noise levels and time steps, and what to do when you have an energy function instead of data. These are not minor implementation details. The choice of training objective, parameterization, and noise schedule can change FID by a factor of two without changing any other component of the system. Understanding the full design space — and why the choices interact the way they do — is essential for practice.

The chapter also resolves a puzzle that has been implicit since Chapter 3: the denoising objective of DDPM, the score matching objective of NCSN, the velocity regression of flow matching, and the conditional flow matching loss are all different-looking objectives that produce the same (or equivalent) models. Why? The answer is that they are all instances of a single principle — regression against conditional targets — applied with different parameterizations and different weighting functions. The framework that unifies them is the subject of this chapter.

## 12.1 Maximum Likelihood and Change of Variables

The oldest training objective for generative models is **maximum likelihood**: choose parameters $\theta$ to maximize $\mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)]$, where $p_\theta$ is the model density. For normalizing flows (Chapter 1), the model density is computed via the change-of-variables formula:

> [!equation] 12.1
> $$\log p_\theta(x) = \log p_0(T_\theta^{-1}(x)) + \log \left|\det \frac{\partial T_\theta^{-1}}{\partial x}\right|,$$


where $T_\theta$ is the flow map and $p_0$ is the base distribution. For continuous normalizing flows (Chapter 2), the log-determinant is replaced by a trace integral:

$$\log p_\theta(x) = \log p_0(x_0) - \int_0^1 \nabla \cdot v_\theta(x_t, t)\, dt,$$

where $x_t$ is the ODE trajectory starting from $x$ and running backward to $x_0$.

Maximum likelihood has a compelling theoretical property: minimizing $-\mathbb{E}[\log p_\theta(x)]$ is equivalent to minimizing $\mathrm{KL}(p_{\text{data}} \| p_\theta)$, the forward KL divergence. The forward KL is **mode-covering** — it penalizes the model for assigning low density to any region where the data has support, encouraging the model to spread its mass across all data modes.

> [!remark] 12.1
> The fundamental problem with maximum likelihood for flow models is computational cost. Evaluating $\log p_\theta(x)$ requires either computing the Jacobian determinant (which costs $O(d^3)$ for a general $d$-dimensional map, reduced to $O(d)$ for coupling layers with triangular Jacobians) or integrating the trace $\nabla \cdot v_\theta$ along an ODE trajectory (which requires $O(100)$ neural network evaluations per training step for CNFs). The entire trajectory from flow matching (Chapter 4) to the present was driven by the desire to avoid this cost.


## 12.2 Regression Losses: The Simulation-Free Revolution

The central insight of flow matching is that the intractable maximum-likelihood objective can be replaced by a **regression loss** — a simple squared-error between the network output and a known target — without changing the optimal model. This is the simulation-free revolution, and it applies far beyond flow matching.

> [!definition] 12.1 — Conditional Regression Loss
> Given a probability path $\{p_t\}$ and a target function $\tau_t(x, x_1)$ that depends on the current state $x$ and the conditioning data point $x_1$, the **conditional regression loss** is:
>
> $$\mathcal{L}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\; x_1 \sim p_{\text{data}},\; x \sim p_t(\cdot|x_1)}\!\left[\|f_\theta(x, t) - \tau_t(x, x_1)\|^2\right],$$
>
> where $f_\theta$ is the neural network output.


The regression loss has three decisive advantages over maximum likelihood: (1) no Jacobian determinant or trace integral is needed; (2) no ODE simulation is needed — $x$ is sampled directly from the known conditional distribution $p_t(\cdot|x_1)$; (3) the target $\tau_t$ is known in closed form for standard interpolants. The cost per training step is a single forward-backward pass through the network — the same as for any supervised learning problem.

Different choices of target $\tau_t$ and interpolant produce all the standard training objectives:

| **Method** | **Target $\tau_t$** | **Interpolant** | **Network predicts** |
|---|---|---|---|
| Flow matching | $u_t(x \mid x_1) = x_1 - \epsilon$ | Linear: $x_t = tx_1 + (1-t)\epsilon$ | Velocity field $v_\theta$ |
| DDPM ($\epsilon$-prediction) | $\epsilon$ | VP: $x_t = \sqrt{\bar\alpha_t}x_1 + \sqrt{1-\bar\alpha_t}\epsilon$ | Noise $\epsilon_\theta$ |
| DDPM ($x$-prediction) | $x_1$ | VP: $x_t = \sqrt{\bar\alpha_t}x_1 + \sqrt{1-\bar\alpha_t}\epsilon$ | Clean data $\hat{x}_\theta$ |
| Score matching | $-\epsilon / \sqrt{1-\bar\alpha_t}$ | VP: same | Score $s_\theta \approx \nabla\log p_t$ |

> [!proposition] 12.1 — Equivalence of Parameterizations
> For a fixed interpolant (e.g., the VP-SDE schedule), the $\epsilon$-prediction, $x$-prediction, velocity-prediction, and score-prediction parameterizations are related by invertible affine transformations. At convergence, they produce the same model — the same probability flow ODE, the same generated distribution. They differ only in the weighting of the loss across noise levels and in the variance of the training gradient.


The proof is direct: for the VP-SDE interpolant $x_t = \sqrt{\bar\alpha_t}x_1 + \sqrt{1-\bar\alpha_t}\epsilon$, the relationships are:

$$v = \frac{\dot{\bar\alpha}_t}{2\sqrt{\bar\alpha_t}}x_1 - \frac{\dot{\bar\alpha}_t}{2\sqrt{1-\bar\alpha_t}}\epsilon, \qquad s = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}, \qquad x_1 = \frac{x_t - \sqrt{1-\bar\alpha_t}\epsilon}{\sqrt{\bar\alpha_t}}.$$

Given any one of $(v, s, \epsilon, x_1)$ and the current state $x_t$, the other three can be computed. The choice of parameterization does not change what is learned — only how the loss weights different noise levels.

## 12.3 Score Matching and Its Variants

**Score matching** trains a model $s_\theta(x) \approx \nabla_x \log p(x)$ to approximate the score function of the data distribution. The original objective (Hyvärinen, 2005) is the **Fisher divergence**:

> [!equation] 12.2
> $$\mathcal{L}_{\mathrm{SM}} = \frac{1}{2}\mathbb{E}_{p_{\text{data}}}\!\left[\|s_\theta(x) - \nabla_x \log p_{\text{data}}(x)\|^2\right].$$


This is intractable as written — $\nabla_x \log p_{\text{data}}$ is unknown. Three techniques make it tractable:

**Implicit score matching** (Hyvärinen, 2005): expand the squared norm and integrate by parts to eliminate the unknown score:

$$\mathcal{L}_{\mathrm{ISM}} = \mathbb{E}_{p_{\text{data}}}\!\left[\mathrm{tr}(\nabla_x s_\theta(x)) + \tfrac{1}{2}\|s_\theta(x)\|^2\right].$$

The trace term $\mathrm{tr}(\nabla_x s_\theta)$ involves the diagonal of the Jacobian of $s_\theta$ — computable but expensive ($O(d)$ backward passes or Hutchinson trace estimation).

**Denoising score matching** (Vincent, 2011): instead of matching the score of $p_{\text{data}}$, match the score of the noised distribution $p_\sigma(x) = \int p_{\text{data}}(x_1)\mathcal{N}(x; x_1, \sigma^2 I)\, dx_1$, whose score at a noisy sample $x = x_1 + \sigma\epsilon$ is simply:

$$\nabla_x \log p_\sigma(x) \approx \frac{x_1 - x}{\sigma^2} = -\frac{\epsilon}{\sigma}.$$

This is the Tweedie formula: the score of the noised distribution points from the noisy observation toward the clean data point. Training reduces to regression against $(x_1 - x)/\sigma^2$ — no Jacobian, no integration by parts.

**Sliced score matching** (Song, Garg, Shi, and Ermon, 2020): project the score onto random directions $v \sim \mathcal{N}(0, I)$, reducing the $d$-dimensional matching problem to a one-dimensional one:

$$\mathcal{L}_{\mathrm{SSM}} = \mathbb{E}_{v, p_{\text{data}}}\!\left[v^\top \nabla_x s_\theta(x)\, v + \tfrac{1}{2}(v^\top s_\theta(x))^2\right].$$

Each term requires only one directional derivative $v^\top \nabla_x s_\theta v$ — a single backward pass regardless of $d$.

## 12.4 Energy-Based Training: When You Have the Density

All the objectives above assume access to samples $x \sim p_{\text{data}}$. In some applications — notably molecular simulation and statistical physics — the target distribution is known up to a normalizing constant: $p_{\text{target}}(x) = e^{-\beta U(x)} / Z$, where $U(x)$ is the energy function and $Z = \int e^{-\beta U(x)}\, dx$ is the (intractable) partition function. Samples are not available, but the energy can be evaluated at any point.

> [!definition] 12.2 — Energy-Based Training (Reverse KL)
> Given a target density $p_{\text{target}}(x) \propto e^{-\beta U(x)}$ and a generative model $q_\theta$ (defined by a flow map $T_\theta$), the **reverse KL** training objective is:
>
> $$\mathcal{L}_{\mathrm{RKL}}(\theta) = \mathrm{KL}(q_\theta \| p_{\text{target}}) = \mathbb{E}_{x \sim q_\theta}\!\left[\log q_\theta(x) - \log p_{\text{target}}(x)\right] = \mathbb{E}_{x \sim q_\theta}\!\left[\log q_\theta(x) + \beta U(x)\right] + \log Z.$$


The reverse KL $\mathrm{KL}(q_\theta \| p_{\text{target}})$ is **mode-seeking**: it penalizes the model for placing mass where the target has low density, producing sharp but potentially incomplete approximations. This contrasts with the forward KL $\mathrm{KL}(p_{\text{data}} \| q_\theta)$ used in maximum likelihood, which is mode-covering.

The gradient of the reverse KL is:

> [!equation] 12.3
> $$\nabla_\theta \mathcal{L}_{\mathrm{RKL}} = \mathbb{E}_{z \sim p_0}\!\left[\nabla_\theta \log q_\theta(T_\theta(z)) + \beta \nabla_\theta U(T_\theta(z))\right],$$


which requires: (1) sampling from the model $q_\theta$ (by running the flow forward on noise $z$), (2) evaluating $\log q_\theta$ (by computing the log-determinant or trace integral), and (3) evaluating $U$ and its gradient at the generated sample. The partition function $Z$ drops out of the gradient — it is a constant with respect to $\theta$.

> [!remark] 12.2
> Energy-based training inverts the usual data pipeline: instead of learning from examples, the model learns from the energy function directly. This is natural for physics — the Boltzmann distribution is defined by $U$, not by samples — and it avoids the sampling bottleneck that makes molecular simulation expensive in the first place. The model is both the approximation to the distribution and the sampler: generate from $q_\theta$, then reweight by importance sampling if needed. The effective sample size $\mathrm{ESS} = (\sum_i w_i)^2 / \sum_i w_i^2$, where $w_i = p_{\text{target}}(x_i) / q_\theta(x_i)$, measures how close $q_\theta$ is to $p_{\text{target}}$.


## 12.5 The Equivalence of Objectives

A unifying perspective emerges: all the training objectives in this chapter are instances of **regression against conditional targets, weighted by a time- and noise-dependent function**. The general form is:

> [!equation] 12.4
> $$\mathcal{L}(\theta) = \mathbb{E}_{t,\, x_1,\, x_t}\!\left[\lambda(t)\, \|f_\theta(x_t, t) - \tau(x_t, x_1, t)\|^2\right],$$


where $\lambda(t) \geq 0$ is the **weighting function**, $\tau$ is the target, and $x_t$ is sampled from the conditional path. Different choices of $(\lambda, \tau)$ recover:

- **Unweighted flow matching**: $\lambda(t) = 1$, $\tau = x_1 - \epsilon$
- **DDPM simple loss**: $\lambda(t) = 1$, $\tau = \epsilon$ (noise prediction)
- **DDPM variational loss**: $\lambda(t) = \beta_t / (2\sigma_t^2(1-\bar\alpha_t))$, $\tau = \epsilon$ (derived from the ELBO)
- **Score matching at noise level $\sigma$**: $\lambda(\sigma) = \sigma^2$, $\tau = -\epsilon/\sigma$ (the SNR-weighted Fisher divergence)

The weighting function $\lambda(t)$ determines how much the loss emphasizes different noise levels. This has a dramatic effect on sample quality:

- **Uniform weighting** ($\lambda = 1$): treats all noise levels equally. The loss is dominated by high-noise regions where the target is large (the velocity or score is large when the signal-to-noise ratio is low), which biases the model toward coarse structure at the expense of fine details.
- **SNR weighting** ($\lambda(t) \propto \text{SNR}(t)$): upweights low-noise regions where fine details are resolved. Produces sharper samples but can underfit the coarse structure.
- **Min-SNR weighting** (Hang et al., 2024): $\lambda(t) = \min(\text{SNR}(t), \gamma)$ for a clipping threshold $\gamma$. Balances coarse and fine structure by capping the weight at high SNR. This is the current best practice for diffusion training.

## 12.6 Noise Schedules and Their Geometry

The **noise schedule** determines how the signal-to-noise ratio evolves over time. For the general Gaussian interpolant $x_t = \alpha_t x_1 + \sigma_t \epsilon$, the SNR at time $t$ is:

> [!equation] 12.5
> $$\text{SNR}(t) = \frac{\alpha_t^2}{\sigma_t^2}.$$


Common choices:

- **Linear** (flow matching): $\alpha_t = t$, $\sigma_t = 1-t$. SNR $= t^2/(1-t)^2$, monotonically increasing from 0 to $\infty$. The schedule spends equal time at all SNR levels — a uniform distribution on the "information content" of the noisy sample.
- **VP (variance-preserving)**: $\alpha_t = \sqrt{\bar\alpha_t}$ where $\bar\alpha_t = \prod_{s=1}^t (1-\beta_s)$. SNR $= \bar\alpha_t/(1-\bar\alpha_t)$. The schedule is determined by $\beta_t$, typically linear or cosine.
- **Cosine** (Nichol and Dhariwal, 2021): $\bar\alpha_t = \cos^2(\pi t/2 \cdot (1+s)/(1+s))$ for a small offset $s$. Produces a nearly uniform distribution of $\log\text{SNR}(t)$ over time, which empirically gives the best image quality.
- **Trigonometric** (Stable Diffusion 3): $\alpha_t = \cos(\pi t/2)$, $\sigma_t = \sin(\pi t/2)$. SNR $= \cot^2(\pi t/2)$. The schedule traverses the full SNR range smoothly, with $\log\text{SNR}$ distributed as a shifted logistic.

> [!proposition] 12.2 — Noise Schedule Equivalence
> For a fixed model architecture and fixed training budget, the optimal noise schedule is the one that distributes $\log\text{SNR}(t)$ uniformly over the time interval $[0,1]$. Schedules that spend too much time at extreme SNR values (very high or very low) waste training capacity on noise levels that are either trivial (pure signal) or uninformative (pure noise).


The geometric interpretation: in the space of $\log\text{SNR}$, the optimal schedule moves at constant speed. This is the "geodesic" in the space of noise levels — the schedule that extracts maximum information per training step.

## 12.7 Parameterizations: What Should the Network Predict?

Given the same interpolant and noise schedule, the network can predict different quantities. The three standard parameterizations are:

**$\epsilon$-prediction.** The network outputs $\epsilon_\theta(x_t, t) \approx \epsilon$, the noise that was added to the data. Given $x_t = \alpha_t x_1 + \sigma_t \epsilon$, the predicted data is $\hat{x}_1 = (x_t - \sigma_t \epsilon_\theta) / \alpha_t$.

**$x$-prediction.** The network outputs $\hat{x}_\theta(x_t, t) \approx x_1$, the clean data directly. The predicted noise is $\hat\epsilon = (x_t - \alpha_t \hat{x}_\theta) / \sigma_t$.

**$v$-prediction** (Salimans and Ho, 2022). The network outputs $v_\theta(x_t, t) \approx \alpha_t \epsilon - \sigma_t x_1$, a linear combination of noise and data that is numerically well-conditioned across all noise levels:

> [!equation] 12.6
> $$v = \alpha_t \epsilon - \sigma_t x_1 = \dot{\alpha}_t x_1 + \dot{\sigma}_t \epsilon,$$


where the second equality holds for the trigonometric schedule $\alpha_t = \cos(\pi t/2)$, $\sigma_t = \sin(\pi t/2)$.

> [!remark] 12.3
> The practical differences between parameterizations are entirely about numerical conditioning. At high noise ($\sigma_t \gg \alpha_t$), the $\epsilon$-prediction target is $O(1)$ while the $x$-prediction target is $O(\sigma_t/\alpha_t)$ — huge and poorly conditioned. At low noise ($\sigma_t \ll \alpha_t$), the situation reverses: $x$-prediction is $O(1)$ while $\epsilon$-prediction is $O(\alpha_t/\sigma_t)$. The $v$-prediction is $O(1)$ at both extremes — it is the "natural" parameterization for schedules where $\alpha_t$ and $\sigma_t$ have comparable magnitude throughout.


This is why $v$-prediction has become the default for flow matching and modern diffusion models: the linear interpolant $x_t = tx_1 + (1-t)\epsilon$ gives $v = x_1 - \epsilon$, which is $O(1)$ everywhere. The DDPM $\epsilon$-prediction works well for the VP-SDE schedule (where high-noise regions dominate training) but is poorly conditioned for the linear schedule. Matching the parameterization to the noise schedule is essential for stable training.

## 12.8 Practical Recipes

Combining the design choices of this chapter, the current best practices for training transport-based generative models are:

**For image generation with flow matching:**
- Interpolant: linear $x_t = tx_1 + (1-t)\epsilon$ or trigonometric $x_t = \cos(\pi t/2)\epsilon + \sin(\pi t/2)x_1$
- Parameterization: velocity ($v$-prediction)
- Weighting: uniform or lognormal time sampling (sample $t$ from a lognormal centered at the midpoint of the SNR range)
- Architecture: Diffusion Transformer (DiT) with adaptive layer norm
- OT conditioning: mini-batch OT with batch size $\leq 256$

**For molecular generation with equivariant flows:**
- Interpolant: linear on coordinates, often with center-of-mass removal
- Parameterization: velocity ($v$-prediction) with $SE(3)$-equivariant architecture
- Weighting: uniform time sampling
- Conservation: projected zero-COM velocity (Equation 11.2)
- Architecture: EGNN, PaiNN, or MACE backbone

**For energy-based training (Boltzmann generators):**
- Objective: reverse KL (Equation 12.3) or flow matching with energy-based reweighting
- Architecture: equivariant flow with exact log-likelihood (for importance sampling)
- Evaluation: effective sample size (ESS) as primary metric
- Refinement: sequential Monte Carlo or annealed importance sampling post-training

## 12.9 Historical Notes

Maximum likelihood training for normalizing flows was introduced by **Rezende and Mohamed (2015)** and developed through the coupling-flow era (NICE, RealNVP, Glow). The simulation-free insight arose simultaneously in **Lipman et al. (2022)**, **Albergo and Vanden-Eijnden (2022)**, and **Liu et al. (2022)**, as described in Chapter 4.

Score matching was introduced by **Hyvärinen (2005)** ("Estimation of Non-Normalized Statistical Models by Score Matching"). Denoising score matching was established by **Vincent (2011)** ("A Connection Between Score Matching and Denoising Autoencoders"), who proved the equivalence with Tweedie's formula. Sliced score matching was introduced by **Song, Garg, Shi, and Ermon (2020)**. The multi-scale score matching approach (training at multiple noise levels simultaneously) was developed by **Song and Ermon (2019, 2020)** (NCSN, NCSNv2).

The DDPM training objective and the equivalence between noise prediction and score matching was established by **Ho, Jain, and Abbeel (2020)** and analyzed systematically by **Kingma, Salimans, Poole, and Ho (2021)** ("Variational Diffusion Models"), who derived the optimal time-weighting from the variational bound. The $v$-prediction parameterization was introduced by **Salimans and Ho (2022)** ("Progressive Distillation for Fast Sampling of Diffusion Models"). The min-SNR weighting strategy was proposed by **Hang, Gu, Li, Bao, Li, Zhang, and Guo (2024)** ("Efficient Diffusion Training via Min-SNR Weighting Strategy").

The cosine noise schedule was introduced by **Nichol and Dhariwal (2021)** ("Improved Denoising Diffusion Probabilistic Models"). The systematic exploration of the design space — noise schedule, parameterization, weighting, sampler — was carried out by **Karras, Aittala, Aila, and Laine (2022)** ("Elucidating the Design Space of Diffusion-Based Generative Models"), which remains the most comprehensive analysis of these choices and their interactions.

Energy-based training for generative flows was pioneered by **Noé, Olsson, Köhler, and Wu (2019)** ("Boltzmann Generators: Sampling Equilibrium States of Many-Body Systems with Deep Learning"), who demonstrated reverse KL training for molecular systems. The importance sampling framework and the ESS diagnostic were developed in this paper and extended by **Midgley, Stimper, Simm, Schölkopf, and Hernández-Lobato (2023)** ("Flow Annealed Importance Sampling Bootstrap") and **Klein, Krämer, and Noé (2024)** ("Transferable Boltzmann Generators").
