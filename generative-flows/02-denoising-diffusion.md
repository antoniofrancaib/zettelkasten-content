---

title: "Diffusion"
subtitle: "Score matching, denoising diffusion, and the probability flow ODE — 2015 to 2021"
---



Every physical system evolves toward equilibrium. Drop a drop of ink into water and it spreads, slowly and irreversibly, until the mixture is uniform. The direction of time is the direction of increasing entropy — of increasing disorder. This is not a rule that can be escaped; it is encoded in the statistical mechanics of the world.

But the direction of time can be reversed in the imagination, if not in fact. Given the final, disordered state of the ink and the water, can you recover the initial configuration — the single drop, before it spread? In general, no. But if the process is diffusion — which has a precise stochastic structure — then the reverse process also has a precise stochastic structure, one that can be characterized exactly in terms of how the probability density of configurations evolves.

This is the insight from non-equilibrium thermodynamics that Sohl-Dickstein, Weiss, Maheswaranathan, and Ganguli brought to machine learning in 2015. Data is the signal. Noise is the equilibrium. The generative process is the reversal of diffusion. Their paper — "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" — was largely ignored for five years, during which the transport track was busy building CNFs and Glow. Then, in 2019 and 2020, the idea came back with a force that would permanently reshape the field.

Understanding why requires following a parallel mathematical thread that begins not with transport maps or Jacobians, but with a much older question in statistical estimation: can you estimate a probability density without ever computing its normalization constant?

## 2.1 The Score Function: Density Without Normalization

Many probability models in machine learning arise as $p(x) \propto \tilde{p}(x)$, where $\tilde{p}(x)$ is an unnormalized energy function that can be evaluated pointwise, but the normalization constant $Z = \int \tilde{p}(x)\, dx$ is intractable. Energy-based models, Boltzmann machines, Markov random fields — all share this structure. Maximum likelihood training requires $\log p(x) = \log \tilde{p}(x) - \log Z$, and estimating $Z$ by Monte Carlo is itself a hard problem.

The escape route is the **score function**. Define:

> [!equation] 2.1
> $$s(x) = \nabla_x \log p(x) = \nabla_x \log \tilde{p}(x),$$


where the last equality holds because $\nabla_x \log Z = 0$ — the normalization constant does not depend on $x$, so its gradient vanishes. The score is the gradient of the log-density, and it is computable from the unnormalized model without knowing $Z$.

The score has a direct geometric interpretation: at any point $x$, $s(x)$ points in the direction of steepest ascent of the log-density — toward the nearest mode, along the path of highest probability. In thermodynamics, this gradient is the force that drives a particle toward equilibrium. In statistics, it encodes the local shape of the distribution more finely than any scalar summary.

> [!definition] 2.1 — Score Function and Score Model
> The **score function** of a distribution $p(x)$ is $s(x) = \nabla_x \log p(x) \in \mathbb{R}^d$. A **score model** $s_\theta(x)$ is a neural network trained to approximate the true score $\nabla_x \log p_{\text{data}}(x)$. A generative model built around a score model is called a **score-based generative model**.


The immediate challenge: $\nabla_x \log p_{\text{data}}(x)$ is unknown. We have samples from $p_{\text{data}}$, not an analytic expression for its density. Training $s_\theta$ by direct regression against the true score creates a circular dependency — to fit the score, you need the score. Breaking this circle is the central problem of score estimation, and its resolution became the mathematical foundation for the entire denoising track.

## 2.2 Score Matching: Learning the Gradient (Hyvärinen, 2005)

Aapo Hyvärinen resolved the circular dependency in 2005 with a trick that is simple to state and not obvious to discover. The natural training objective for a score model is the **Fisher divergence** — the expected squared distance between model and truth:

$$J(\theta) = \frac{1}{2}\,\mathbb{E}_{p_{\text{data}}}\!\left[\|s_\theta(x) - \nabla_x \log p_{\text{data}}(x)\|^2\right].$$

This cannot be minimized directly, since $\nabla_x \log p_{\text{data}}(x)$ is unknown. But Hyvärinen showed that, after expanding the square and applying integration by parts, $J(\theta)$ is equivalent — up to a constant independent of $\theta$ — to:

> [!equation] 2.2
> $$J_{\text{SM}}(\theta) = \mathbb{E}_{p_{\text{data}}}\!\left[\mathrm{tr}\!\left(\nabla_x s_\theta(x)\right) + \frac{1}{2}\|s_\theta(x)\|^2\right].$$


This **implicit score matching** objective involves only $s_\theta$ and samples from $p_{\text{data}}$ — no knowledge of the true score is required. The derivation proceeds by writing the cross term $\mathbb{E}_p[s_\theta \cdot \nabla_x \log p] = \mathbb{E}_p[s_\theta \cdot (\nabla_x p)/p]$ and applying integration by parts, yielding $-\mathbb{E}_p[\mathrm{tr}(\nabla_x s_\theta)]$, which cancels the unknown score and leaves the computable form (2.2).

The first term, $\mathrm{tr}(\nabla_x s_\theta(x))$, is the trace of the Jacobian of $s_\theta$ at a data point. Computing it exactly requires $d$ backward passes through the network — one per output dimension — at a cost of $O(d)$ times that of a single forward pass. For moderate dimensions this is manageable. For images, where $d \sim 10^5$, it is not. Score matching found immediate applications in density estimation for tabular data, but the Jacobian trace was a hard wall against image-scale problems.

> [!remark] 2.1
> The integration by parts in Hyvärinen's derivation requires $p_{\text{data}}$ to vanish at the boundaries of $\mathbb{R}^d$ — or equivalently, that $s(x) \to 0$ sufficiently fast as $\|x\| \to \infty$. This is satisfied for all practical continuous data distributions with finite variance. An alternative proof via Stein's identity makes the same assumption explicit: $\mathbb{E}_p[g(x) \nabla_x \log p(x) + \nabla_x g(x)] = 0$ for suitable test functions $g$. The identity (2.2) is a special case with $g = s_\theta$.


## 2.3 Denoising as Score Matching (Vincent, 2011)

Pascal Vincent's 2011 insight converted the intractable Jacobian-trace objective into a simple regression, and in doing so revealed the deep connection between density estimation and denoising.

Define the **noise-perturbed distribution** at noise level $\sigma > 0$ as the convolution of $p_{\text{data}}$ with a Gaussian kernel:

$$p_\sigma(\tilde{x}) = \int p_{\text{data}}(x)\, \mathcal{N}(\tilde{x};\, x,\, \sigma^2 I)\, dx.$$

This is the marginal distribution of $\tilde{x} = x + \sigma\epsilon$ when $x \sim p_{\text{data}}$ and $\epsilon \sim \mathcal{N}(0, I)$. The score of $p_\sigma$ at a noisy point $\tilde{x}$ has a closed-form expression in terms of conditional expectations, known as the **Tweedie formula**:

$$\nabla_{\tilde{x}} \log p_\sigma(\tilde{x}) = \frac{\mathbb{E}_{p_{\text{data}}(x|\tilde{x})}\![x] - \tilde{x}}{\sigma^2}.$$

The score at a noisy point $\tilde{x}$ points toward the conditional mean of the underlying clean data, scaled by $1/\sigma^2$. Substituting this into the Fisher divergence for $p_\sigma$ and expanding gives the **denoising score matching** objective:

> [!equation] 2.3
> $$J_{\text{DSM}}(\theta;\, \sigma) = \mathbb{E}_{x \sim p_{\text{data}},\, \epsilon \sim \mathcal{N}(0,I)}\!\left[\left\|s_\theta(x + \sigma\epsilon;\, \sigma) + \frac{\epsilon}{\sigma}\right\|^2\right].$$


No trace, no Jacobians, no extra backward passes. Each training step requires one forward pass of $s_\theta$ to evaluate the squared norm and one backward pass for the gradient with respect to $\theta$. The cost is exactly that of supervised regression — as cheap as training a denoising autoencoder.

> [!theorem] 2.1 — Equivalence of Score Matching and Denoising
> For any $\sigma > 0$, the denoising score matching objective $J_{\text{DSM}}(\theta;\, \sigma)$ and the implicit score matching objective (2.2) applied to $p_\sigma$ have the same minimizer $\theta^*$. In other words, minimizing the denoising loss (2.3) is equivalent to fitting the score of the noise-perturbed distribution, $\nabla_{\tilde{x}} \log p_\sigma(\tilde{x})$.


The theorem converts a hard density-estimation problem into an easy regression problem. Train a network to predict the noise $\epsilon$ given the noisy observation $x + \sigma\epsilon$ — and you are, exactly and without approximation, learning the score function of the perturbed distribution. This equivalence is not heuristic; it is algebraic. And it removes the only computational barrier to training score models at scale.

## 2.4 Non-Equilibrium Thermodynamics: The 2015 Foundation

In 2015 — the same year NICE appeared — Sohl-Dickstein, Weiss, Maheswaranathan, and Ganguli published their thermodynamics paper. Their framing was different from score matching: no Fisher divergence, no Tweedie formula, no explicit score functions. Instead, they described the generative process using the language of Markov chains and the theory of non-equilibrium statistical mechanics.

**The forward process.** Define a Markov chain $x_0, x_1, \ldots, x_T$ that gradually corrupts a data point $x_0$ into pure noise:

> [!equation] 2.4
> $$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\; \sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t I\right),$$


where $\{\beta_t\}_{t=1}^T \subset (0,1)$ is a **noise schedule** chosen so that $q(x_T) \approx \mathcal{N}(0,I)$. The factor $\sqrt{1-\beta_t}$ contracts the signal toward zero, while $\beta_t$ injects Gaussian noise. The two together preserve the variance: if $x_{t-1}$ has unit variance, so does $x_t$.

The key algebraic property of this specific construction is that the marginal $q(x_t | x_0)$ is Gaussian for all $t$ simultaneously:

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\; \sqrt{\bar{\alpha}_t}\, x_0,\; (1-\bar{\alpha}_t)\, I\right), \qquad \bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s).$$

Equivalently, any noisy sample at time $t$ can be written as $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon$ for $\epsilon \sim \mathcal{N}(0,I)$. This closed-form marginal — a direct consequence of the Gaussian structure — is the algebraic miracle that makes diffusion models tractable. Without it, sampling from $q(x_t|x_0)$ would require running the Markov chain step by step.

**The reverse process.** The time-reversed chain $q(x_{t-1}|x_t, x_0)$ has a known closed form when $x_0$ is given:

$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\!\left(\tilde{\mu}_t(x_t, x_0),\; \tilde{\beta}_t I\right),$$

where $\tilde{\mu}_t$ and $\tilde{\beta}_t$ are functions of $\beta_t$ and $\bar{\alpha}_t$ given by explicit formulas. The problem: $x_0$ is not available at generation time. A learned model $p_\theta(x_{t-1} | x_t)$ must approximate this reverse without access to $x_0$.

Training maximizes the evidence lower bound on $\log p_\theta(x_0)$, which decomposes as a sum of KL divergences between true and learned reverse conditionals at each step:

$$\log p_\theta(x_0) \geq -\sum_{t=1}^T D_{\mathrm{KL}}\!\left(q(x_{t-1}\mid x_t, x_0)\;\|\; p_\theta(x_{t-1}\mid x_t)\right) + C.$$

Sohl-Dickstein et al. showed this works in principle. But the results — blurry samples from low-resolution datasets — could not compete with GANs or VAEs of the era. The paper went largely uncited for five years.

> [!remark] 2.2
> The 2015 paper lay dormant for two reasons. First, the training procedure required evaluating KL divergences at all $T$ noise levels, and $T$ had to be large (hundreds to thousands) for the Gaussian approximation of the reverse to be accurate at each step — expensive for networks of the time. Second, the network architecture — fully connected layers on small images — was insufficiently expressive to model the complex reverse transitions. Both problems would be resolved by 2020 through a combination of deeper networks, U-Net architectures, and an objective simplification that removed the explicit KL computation entirely.


## 2.5 Noise-Conditional Score Networks (Song & Ermon, 2019)

The dormancy ended with Yang Song and Stefano Ermon's "Generative Modeling by Estimating Gradients of the Data Distribution" (NeurIPS 2019). Their approach dispensed with the Markov chain framing and returned to score matching — but with a key innovation that made it competitive for the first time.

The problem with training a score model at a single noise level $\sigma$ is that the score is useful only near that noise level. For generation, you need to start from pure noise (high $\sigma$) and gradually refine toward data (low $\sigma$). A score model trained at high $\sigma$ learns nothing about the fine-grained structure of $p_{\text{data}}$; a model trained at low $\sigma$ gives vanishing gradients in the high-noise regime where sampling must begin. Neither alone is sufficient.

Song and Ermon's solution was to train a **single** neural network $s_\theta(x, \sigma)$ to estimate $\nabla_x \log p_\sigma(x)$ simultaneously across a geometric sequence of noise levels $\sigma_1 > \sigma_2 > \cdots > \sigma_L$, with $\sigma_1$ large enough to cover the full data space and $\sigma_L$ small enough to be close to $p_{\text{data}}$. The combined training objective is:

$$\mathcal{L}(\theta) = \sum_{i=1}^L \lambda(\sigma_i)\, J_{\text{DSM}}(\theta;\, \sigma_i), \qquad \lambda(\sigma_i) \propto \sigma_i^2.$$

The weight $\lambda(\sigma_i) \propto \sigma_i^2$ normalizes the loss across scales: without it, small-$\sigma$ terms would dominate and the network would learn a poor score at high noise levels, where the coarse structure of the distribution lives.

**Annealed Langevin dynamics.** Sampling with a trained $s_\theta$ uses **Langevin MCMC** — a stochastic gradient ascent on the log-density:

$$x_{k+1} = x_k + \frac{\epsilon_i}{2}\, s_\theta(x_k, \sigma_i) + \sqrt{\epsilon_i}\, z_k, \qquad z_k \sim \mathcal{N}(0, I).$$

Run at noise level $\sigma_i$ with step size $\epsilon_i$, this Markov chain converges (in the limit of small $\epsilon_i$ and many steps) to $p_{\sigma_i}$. The full **annealed Langevin** procedure runs a short Langevin chain at $\sigma_1$, then uses the terminal samples as initial conditions for $\sigma_2$, and so on down to $\sigma_L \approx 0$. The progressive coarsening from global structure to fine detail mirrors the multi-scale architecture of RealNVP, but here it emerges from the noise schedule rather than from architectural design.

The **Noise-Conditional Score Network (NCSN)**, trained with $L = 10$ noise levels and a U-Net architecture, produced CIFAR-10 samples competitive with contemporary GANs — the first score-based model to do so. NCSNv2 (Song & Ermon, NeurIPS 2020) refined the noise schedule and architecture further, establishing score-based models as a serious alternative to the adversarial paradigm.

## 2.6 DDPM: The Simplification That Changed Everything (Ho et al., 2020)

Jonathan Ho, Ajay Jain, and Pieter Abbeel's "Denoising Diffusion Probabilistic Models" (NeurIPS 2020) returned to the Sohl-Dickstein Markov chain framework — but with a simplification so effective that it transformed diffusion models from a research curiosity into the dominant paradigm for generative modeling.

The key observation: the optimal reverse posterior mean $\tilde{\mu}_t(x_t, x_0)$ can be rewritten entirely in terms of the noise $\epsilon$ that was added at step $t$. Because $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon$, we have $x_0 = (x_t - \sqrt{1-\bar{\alpha}_t}\, \epsilon)/\sqrt{\bar{\alpha}_t}$, and substituting:

$$\tilde{\mu}_t(x_t, x_0) = \frac{1}{\sqrt{1-\beta_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\, \epsilon\right).$$

Learning the reverse process is therefore equivalent to learning to predict $\epsilon$ from $x_t$. Parametrizing the learned reverse as $p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t, t), \beta_t I)$ with $\mu_\theta$ in the noise-prediction form, and substituting into the ELBO, Ho et al. showed that the dominant term simplifies to:

> [!equation] 2.5
> $$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t,\, x_0 \sim p_{\text{data}},\, \epsilon \sim \mathcal{N}(0,I)}\!\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right], \qquad x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon.$$


This training objective is extraordinarily simple: sample a data point $x_0$, sample a timestep $t$ uniformly from $\{1, \ldots, T\}$, corrupt $x_0$ with the appropriate Gaussian noise to obtain $x_t$, and train the network to predict which noise was added. No explicit KL divergences, no Jacobians, no ODE solvers, no Langevin chains during training. Just noise prediction by regression.

The connection to score matching is exact: since $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$, the optimal noise predictor satisfies

$$\epsilon_\theta(x_t, t) \approx -\sqrt{1 - \bar{\alpha}_t}\;\nabla_{x_t} \log p_t(x_t),$$

making the DDPM training objective a disguised form of denoising score matching — the same objective Vincent derived in 2011 from the Fisher divergence, applied now at discrete timesteps with the specific Gaussian forward process (2.4). NCSN and DDPM, developed from entirely different motivations, are training the same function.

The results were immediately recognized as a step change. FID scores on CIFAR-10 and LSUN surpassed contemporaneous GANs. Samples from $256 \times 256$ face generation were visually indistinguishable from photographs. And — unlike GANs — training was stable, reproducible, and required no adversarial game between competing networks. Within months, DALL-E 2, Stable Diffusion, and Imagen had all been built on DDPM foundations.

> [!remark] 2.3
> The DDPM sampling procedure generates samples by running the reverse chain $p_\theta(x_{t-1}|x_t)$ backward from $x_T \sim \mathcal{N}(0,I)$ to $x_0$. This requires $T = 1000$ sequential neural network evaluations — one per denoising step — making generation slow compared to a single normalizing flow forward pass. The problem of reducing the number of function evaluations (NFE) without degrading sample quality became a central research direction. Song et al. (DDIM, 2020) showed that $T$ could be reduced to 50 with minimal quality loss by using a deterministic non-Markovian sampler; the systematic theory of fast sampling is the subject of Chapter 8.


## 2.7 The Score SDE: A Continuous-Time Unification (Song et al., 2021)

By early 2021, the field had two distinct score-based approaches — NCSN (continuous noise levels, annealed Langevin sampling) and DDPM (discrete Markov chain, reverse diffusion sampling) — that were clearly related but whose precise connection had never been articulated. Yang Song, Jascha Sohl-Dickstein, Diederik Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole's "Score-Based Generative Modeling through Stochastic Differential Equations" (ICLR 2021, outstanding paper) provided the unification by passing to the continuous-time limit.

As the number of discrete noise steps $T \to \infty$ and the step size simultaneously goes to zero, the forward Markov chain (2.4) converges to a **stochastic differential equation** (SDE). In general, the forward process is:

> [!equation] 2.6
> $$dx = f(x, t)\, dt + g(t)\, dW,$$


where $W$ is a standard Brownian motion, $f : \mathbb{R}^d \times [0,T] \to \mathbb{R}^d$ is the **drift**, and $g : [0,T] \to \mathbb{R}$ is the **diffusion coefficient**. The SDE (2.6) defines a continuous-time stochastic process $\{x_t\}_{0 \leq t \leq T}$ whose marginal distribution at time $T$ approaches $\mathcal{N}(0,I)$ for appropriate choices of $f$ and $g$.

Song et al. identified two canonical cases corresponding to the two established approaches:

**VP-SDE** (Variance Preserving, corresponding to DDPM): $f(x,t) = -\tfrac{1}{2}\beta(t)\, x$ and $g(t) = \sqrt{\beta(t)}$, where $\beta(t)$ is the continuous-time noise schedule interpolating the discrete $\{\beta_t\}$. The signal is contracted linearly while noise is injected proportionally — a *variance-preserving* process because the variance of $x_t$ remains bounded.

**VE-SDE** (Variance Exploding, corresponding to NCSN): $f(x,t) = 0$ and $g(t) = \sqrt{d[\sigma^2(t)]/dt}$, where $\sigma(t)$ is the noise level schedule. The drift is zero — the signal is not contracted — and noise is injected until the variance *explodes*. The endpoint distribution is not Gaussian in the strict sense but has variance $\sigma_T^2 \gg 1$, which dominates any fixed-variance data distribution.

Both SDEs transform $x_0 \sim p_{\text{data}}$ into $x_T \approx \mathcal{N}(0, I)$ as $T \to \infty$. The key quantity connecting forward and reverse is the score of the marginal distribution at each time: $\nabla_x \log p_t(x)$.

**The reverse-time SDE.** Brian Anderson (1982) showed that every forward Itô SDE has a corresponding reverse-time SDE that runs from $t = T$ back to $t = 0$ with the same marginal distributions. Applied to (2.6), the reverse process satisfies:

> [!equation] 2.7
> $$dx = \left[f(x, t) - g(t)^2\, \nabla_x \log p_t(x)\right] dt + g(t)\, d\bar{W},$$


where $\bar{W}$ is a Brownian motion running backward in time. The score function $\nabla_x \log p_t(x)$ appears explicitly as the correction term that reverses the forward diffusion. Approximating this score with a learned network $s_\theta(x, t) \approx \nabla_x \log p_t(x)$ — trained by denoising score matching across all $t$ — gives a complete generative model.

> [!definition] 2.2 — Score-Based SDE Model
> A **score-based SDE generative model** consists of a forward SDE (2.6), a learned score network $s_\theta(x, t) \approx \nabla_x \log p_t(x)$ trained by denoising score matching, and the reverse SDE (2.7) with $\nabla_x \log p_t$ replaced by $s_\theta$. Sampling proceeds by numerically integrating the reverse SDE from $x_T \sim \mathcal{N}(0, I)$ to $x_0$.


The continuous-time training objective is a natural generalization of (2.3): for a weight function $\lambda(t) > 0$,

$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,T],\; x_0 \sim p_{\text{data}},\; x_t \sim q(x_t|x_0)}\!\left[\lambda(t)\,\|s_\theta(x_t, t) - \nabla_{x_t} \log q(x_t \mid x_0)\|^2\right].$$

Because $q(x_t | x_0)$ is Gaussian for both the VP- and VE-SDE, $\nabla_{x_t} \log q(x_t | x_0)$ is a simple linear function of $x_t$ and $x_0$ — and the objective reduces to denoising score matching. No Jacobians. No ODE simulation during training. The DDPM objective (2.5) is the VP-SDE version of this, discretized to $T = 1000$ steps.

## 2.8 The Probability Flow ODE

The most striking result of Song et al. (2021) was not the unification of NCSN and DDPM — that was expected — but the discovery that every diffusion model has an equivalent **deterministic** counterpart.

For any forward SDE (2.6) with marginals $\{p_t\}$, there exists an ordinary differential equation — no Brownian motion, no randomness — whose solution trajectories $\{x_t\}$ share the same marginal distribution $p_t$ at every time $t$:

> [!equation] 2.8
> $$\frac{dx}{dt} = f(x, t) - \frac{1}{2}\,g(t)^2\,\nabla_x \log p_t(x).$$


This is the **probability flow ODE**. It is a deterministic ordinary differential equation whose right-hand side involves the score function. Compared to the reverse SDE (2.7), the noise term $g(t)\, d\bar{W}$ has been removed and the score drift has been halved — the factor of $\tfrac{1}{2}$ is not an approximation but an exact result from the Fokker-Planck equation governing the evolution of $p_t$.

> [!theorem] 2.2 — Probability Flow ODE
> Let $\{p_t\}_{0 \leq t \leq T}$ be the marginal densities of the forward SDE (2.6). The probability flow ODE (2.8) defines a deterministic flow $\phi_t : \mathbb{R}^d \to \mathbb{R}^d$ such that $\phi_t(x_0) \sim p_t$ whenever $x_0 \sim p_0$. Equivalently, integrating the ODE backward from $x_T \sim \mathcal{N}(0,I)$ produces a sample distributed as $p_0 = p_{\text{data}}$.


The implications of this theorem are profound, and they become clearer when stated as a connection between the two tracks. The probability flow ODE (2.8) is exactly a continuous normalizing flow in the sense of Chapter 1. Its vector field is

$$u_t(x) = f(x,t) - \tfrac{1}{2}\,g(t)^2\, \nabla_x \log p_t(x).$$

If we know the exact score $\nabla_x \log p_t$ at every time and position, we can integrate this ODE forward or backward to generate samples or compute exact likelihoods — exactly as FFJORD does, with the same NFE cost. The simulation bottleneck of Chapter 1 reappears: generating one sample requires numerically integrating the probability flow ODE, which costs as many network evaluations as a CNF.

But the training is free of that bottleneck. DDPM trains by denoising regression. The probability flow ODE is a consequence of that training — the deterministic dynamics that emerge from a trained score model, without any ODE solve during the training loop.

> [!remark] 2.4
> The probability flow ODE enables **deterministic sampling**: fixing the initial condition $x_T$ fixes the generated output completely. This gives diffusion models a latent space — the space of initial Gaussian noise vectors — with meaningful geometry. Two data points $x^{(1)}_0$ and $x^{(2)}_0$ correspond to initial conditions $x^{(1)}_T$ and $x^{(2)}_T$; interpolating linearly between these in the latent space and integrating the ODE produces a smooth perceptual morph between the two data points. This latent interpolation, unavailable from the stochastic reverse SDE sampler, made the probability flow ODE an immediate tool for image editing applications (DALL-E 2, InstructPix2Pix, and their successors all use related ideas).


The probability flow ODE also enables **exact likelihood computation**: running the ODE forward from $x_0$ to $x_T$ and applying the instantaneous change-of-variables formula (from Chapter 1) gives $\log p_0(x_0)$ exactly, as the sum of $\log \mathcal{N}(x_T; 0, I)$ and the integrated divergence of $u_t$. Diffusion models — which were thought to offer no tractable likelihood — turn out to have exactly computable likelihoods via their probability flow ODE, using the same Hutchinson trace estimator as FFJORD.

## 2.9 What the Denoising Track Got Right

The denoising track arrived at the same destination as the transport track but via different mathematics, and its tools had properties that the transport track's lacked. Understanding the contrast clarifies why diffusion models displaced CNFs in practice — and why the question of Chapter 3 is so natural.

**Scalability by construction.** The DDPM training objective (2.5) requires exactly one forward pass of $\epsilon_\theta$ per training step, regardless of data dimension, distribution complexity, or model depth. There is no ODE to simulate, no Jacobian to estimate, no adjoint to integrate backward. The cost of a training step for DDPM equals the cost of supervised regression — and this cost does not grow as the target distribution becomes more complex. The simulation bottleneck of Chapter 1 is entirely absent from training.

**The score as the universal object.** The score function $\nabla_x \log p_t(x)$ is not merely a computational device — it is the sufficient statistic for the reverse dynamics. Everything needed to generate from any distribution reachable by diffusion is encoded in the score, at every noise level. This is both mathematically clean and practically powerful: it means the architecture and training procedure are the same regardless of the target, and theoretical guarantees can be stated in terms of score estimation accuracy alone.

**The SDE framework and its connections.** By casting diffusion models as SDEs, Song et al. connected the denoising track to a rich body of mathematical theory: Fokker-Planck equations, Langevin dynamics, and — via the probability flow ODE — continuous normalizing flows. The framework is extensible: new noise processes, new reverse samplers (predictor-corrector, DDIM, DPM-Solver), and new training objectives all fit within the SDE language.

**The implicit multi-scale decomposition.** Training a score model across all noise levels $t \in [0,T]$ implicitly teaches the model to decompose the data distribution at all scales simultaneously. At high $t$, the score model learns global structure — which region of the space a sample belongs to. At low $t$, it learns fine-grained local detail. This coarse-to-fine decomposition happens automatically from the training objective, without architectural engineering — and gives diffusion models their characteristic ability to generate globally coherent samples with faithful local texture.

## 2.10 Two Tracks, One Step From Convergence

Chapter 1 introduced normalizing flows and continuous normalizing flows: exact likelihoods, tractable training for flows, but either restricted architectures (coupling layers) or the simulation bottleneck (CNFs). This chapter introduced the denoising track, which removes the simulation bottleneck from training entirely — at the cost of slow, multi-step sampling.

The situation in late 2021 was this:

- CNFs: free architectures, exact likelihoods, **simulation bottleneck at training**.
- Diffusion models: cheap training (one forward pass per step), **many function evaluations at sampling**.

Both bottlenecks trace to the same root: both methods define a complex, curved path from noise to data — the Riemannian path induced by the SDE, or the constrained path imposed by the Jacobian structure — and that path requires many steps to traverse accurately. Is there a *simpler path*?

The answer is yes, and it is almost embarrassingly direct: define the transportation path yourself, as a straight line between noise and data, and derive the vector field from that path. No diffusion, no score functions, no ODE simulation during training. Just regression.

This is the **flow matching** insight of Lipman et al. (2022) and Albergo & Vanden-Eijnden (2022). The probability flow ODE (2.8) is a particular choice of continuous normalizing flow — the one induced by a specific diffusion SDE. Flow matching says: why not choose a *different* flow, one that is simpler to train and cheaper to sample? The score function is not fundamental to the vector field; it is one way to derive a vector field that transports $p_0$ to $p_1$. There are others.

Understanding precisely what those others are, and how they unify the entire landscape of generative transport, is the subject of Chapter 3.

## 2.11 Historical Notes

The mathematical foundation for score matching is **Hyvärinen (2005)** ("Estimation of Non-Normalized Statistical Models by Score Matching", *JMLR*), who proved the implicit score matching identity (2.2) and demonstrated its use for estimating energy-based models. **Vincent (2011)** ("A Connection Between Score Matching and Denoising Autoencoders", *Neural Computation*) established Theorem 2.1 — that denoising and score matching are equivalent — converting the intractable Jacobian-trace objective into a simple regression.

The Tweedie formula connecting the posterior mean to the score appears in **Robbins (1956)** and was connected to diffusion models explicitly by **Efron (2011)** ("Tweedie's Formula and Selection Bias", *JASA*) and formalized in the diffusion context by **Luo (2022)** ("Understanding Diffusion Models: A Unified Perspective").

The non-equilibrium thermodynamics framing is due to **Sohl-Dickstein, Weiss, Maheswaranathan, and Ganguli (ICML 2015)** ("Deep Unsupervised Learning using Nonequilibrium Thermodynamics"). The physics background — fluctuation theorems, Jarzynski's equality, the thermodynamic arrow of time — is surveyed in **Seifert (2012)** ("Stochastic thermodynamics, fluctuation theorems and molecular machines", *Reports on Progress in Physics*).

The revival of diffusion models began with **Song and Ermon (NeurIPS 2019)** ("Generative Modeling by Estimating Gradients of the Data Distribution") and **NCSNv2 (NeurIPS 2020)** ("Improved Techniques for Training Score-Based Generative Models"). The DDPM simplification is from **Ho, Jain, and Abbeel (NeurIPS 2020)**. The v-prediction reparametrization — predicting the velocity $v = \sqrt{\bar\alpha_t}\epsilon - \sqrt{1-\bar\alpha_t}x_0$ — appears in **Salimans and Ho (2022)** ("Progressive Distillation for Fast Sampling of Diffusion Models").

The continuous-time SDE unification is **Song, Sohl-Dickstein, Kingma, Kumar, Ermon, and Poole (ICLR 2021, outstanding paper)** ("Score-Based Generative Modeling through Stochastic Differential Equations"). The reverse-time SDE identity (2.7) is Anderson's result (**Anderson, 1982**, "Reverse-time diffusion equation models", *Stochastic Processes and their Applications*). The mathematical background for SDEs and Fokker-Planck equations is in **Øksendal (2003)** (*Stochastic Differential Equations*) and **Risken (1989)** (*The Fokker-Planck Equation*).

Theoretical convergence rates for score-based models — relating total variation distance between model and $p_{\text{data}}$ to score estimation error and discretization step size — are developed in **De Bortoli (2022)**, **Chen, Chewi, Li, Li, Salim, and Zhang (2022)**, and **Lee, Lu, and Tan (2022)**. The sharpest current results give polynomial dependence on dimension under log-Sobolev or Poincaré inequalities on $p_{\text{data}}$.

The large-scale applications that made diffusion models dominant — **DALL-E 2** (Ramesh et al., 2022), **Stable Diffusion** (Rombach et al., CVPR 2022), **Imagen** (Saharia et al., NeurIPS 2022) — all build on the DDPM backbone with classifier-free guidance (Ho & Salimans, 2021) and latent-space diffusion (Rombach et al.). The diffusion transformer architecture (DiT, Peebles & Xie, 2023), which replaced U-Nets with transformers as the score network backbone, is now the standard architecture for large-scale image and video generation.

## References

- [Sohl-Dickstein, Weiss, Maheswaranathan, and Ganguli (ICML 2015)](https://arxiv.org/abs/1503.03585) — Deep Unsupervised Learning using Nonequilibrium Thermodynamics
- [Ho et al., 2020](https://arxiv.org/abs/2006.11239) — Denoising Diffusion Probabilistic Models
- [Song & Ermon, NeurIPS 2019](https://arxiv.org/abs/1907.05600) — Generative Modeling by Estimating Gradients of the Data Distribution
- [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, and Poole (ICLR 2021, outstanding paper)](https://arxiv.org/abs/2011.13456) — Score-Based Generative Modeling through Stochastic Differential Equations
- [Luo (2022)](https://arxiv.org/abs/2208.11970) — Understanding Diffusion Models: A Unified Perspective
