---

title: "Guidance"
subtitle: "Feynman-Kac correctors, SMC samplers, classifier guidance, CFG, reward steering, and posterior sampling — 2001 to 2026"
---



Every generative model is an approximation. The neural network has finite capacity; the training data is finite; the training procedure converges to a local optimum; the ODE solver introduces discretization error. The result is a model distribution $q_\theta$ that is close to, but not equal to, the target $p_{\text{target}}$. For many applications — generating images, synthesizing speech — "close" is good enough. But for scientific applications where quantitative accuracy matters — computing free energies, sampling from Boltzmann distributions, estimating posterior probabilities — "close" is not enough. The gap between $q_\theta$ and $p_{\text{target}}$ must be measured and, ideally, closed.

The first half of this chapter develops the tools for closing that gap through **correction**: given an imperfect generative model $q_\theta$, use importance sampling, sequential Monte Carlo, or Feynman-Kac reweighting to transform samples from $q_\theta$ into (asymptotically) exact samples from $p_{\text{target}}$. The correction is always possible in principle, but its efficiency depends on how close $q_\theta$ is to $p_{\text{target}}$: a good generative model makes the correction cheap; a poor one makes it exponentially expensive.

The second half addresses a complementary problem: **guidance** — steering a trained model toward a desired subset or property of the data distribution. An unconditional model learns to sample from $p_{\text{data}}$; guidance tilts that distribution by a likelihood or reward function, concentrating samples on a desired subregion. The central tension is between fidelity (how well guided samples satisfy the condition) and diversity (how much variation remains among them). A single parameter — the guidance strength $w$ — controls this tradeoff in every method of the second half.

## 9.1 The Correction Problem

Let $q_\theta$ be the distribution induced by a generative model (a flow, a diffusion model, a bridge) and $p_{\text{target}}$ be the true target distribution. We assume that $p_{\text{target}}$ can be evaluated up to a normalizing constant: $p_{\text{target}}(x) = \tilde{p}(x) / Z$, where $\tilde{p}(x)$ is computable but $Z = \int \tilde{p}(x)\, dx$ is not. This is the standard setting for Boltzmann distributions ($\tilde{p}(x) = e^{-\beta U(x)}$), Bayesian posteriors ($\tilde{p}(x) = p(\text{data}|x)p(x)$), and energy-based models.

The correction problem is: given $n$ samples $\{x_i\}_{i=1}^n$ from $q_\theta$, produce estimates of expectations $\mathbb{E}_{p_{\text{target}}}[h(x)]$ for arbitrary test functions $h$.

> [!definition] 9.1 — Self-Normalized Importance Sampling
> Given samples $x_1, \ldots, x_n \sim q_\theta$ and unnormalized importance weights:
>
> $$\tilde{w}_i = \frac{\tilde{p}(x_i)}{q_\theta(x_i)},$$
>
> the **self-normalized importance sampling (SNIS)** estimator of $\mathbb{E}_{p_{\text{target}}}[h(x)]$ is:
>
> $$\hat{\mu}_{\mathrm{SNIS}} = \frac{\sum_{i=1}^n \tilde{w}_i\, h(x_i)}{\sum_{i=1}^n \tilde{w}_i}.$$
>
> This is a consistent estimator: $\hat{\mu}_{\mathrm{SNIS}} \to \mathbb{E}_{p_{\text{target}}}[h(x)]$ as $n \to \infty$.


The quality of the estimator depends on the **effective sample size (ESS)**:

> [!equation] 9.1
> $$\mathrm{ESS} = \frac{\left(\sum_i \tilde{w}_i\right)^2}{\sum_i \tilde{w}_i^2} \in [1, n].$$


ESS $= n$ when all weights are equal ($q_\theta = p_{\text{target}}$); ESS $= 1$ when one weight dominates (catastrophic mismatch). An ESS of $n/2$ means the importance sampling is half as efficient as direct sampling — a reasonable correction. An ESS of $1$ means the correction has failed entirely.

> [!remark] 9.1
> The ESS depends exponentially on the KL divergence: $\mathrm{ESS} \approx n \cdot e^{-\mathrm{KL}(p \| q)}$ for large $n$. This means importance sampling is only practical when $q_\theta$ is already close to $p_{\text{target}}$ — a KL divergence of 10 nats reduces the ESS by a factor of $e^{10} \approx 22{,}000$. The generative model is doing most of the work; importance sampling handles the residual error. This is why correction methods are *complements* to generative models, not substitutes.


## 9.2 The Feynman-Kac Formula

The Feynman-Kac formula connects the solution of a PDE to an expectation over stochastic paths, providing a natural framework for correcting generative models that produce paths (flows, diffusions) rather than just endpoints.

> [!theorem] 9.1 — Feynman-Kac Formula
> Let $\{X_t\}_{t \in [0,T]}$ be the solution of the SDE $dX_t = f_t(X_t)\, dt + g_t\, dW_t$, and let $V : \mathbb{R}^d \times [0,T] \to \mathbb{R}$ be a potential function. Then the function:
>
> $$u(x, t) = \mathbb{E}\!\left[h(X_T)\, \exp\!\left(-\int_t^T V(X_s, s)\, ds\right) \;\Big|\; X_t = x\right]$$
>
> solves the PDE $\partial_t u + f_t \cdot \nabla u + \frac{g_t^2}{2}\Delta u - V\, u = 0$ with terminal condition $u(x, T) = h(x)$.


For generative modeling, the Feynman-Kac formula provides a way to *reweight paths* rather than just endpoints. Consider a generative diffusion model with learned drift $f_\theta$ that approximately samples from $p_{\text{target}}$. The true target satisfies a Fokker-Planck equation; the approximate model satisfies a different one. The ratio between the two — the Radon-Nikodym derivative of the true path measure with respect to the approximate one — can be expressed as a Feynman-Kac exponential:

> [!equation] 9.2
> $$\frac{dP_{\text{target}}}{dP_\theta}(\{X_t\}) = \exp\!\left(-\int_0^T V_t(X_t)\, dt\right),$$


where $V_t$ is a potential that depends on the discrepancy between the true and approximate drifts. This path-level reweighting is more informative than endpoint reweighting: it uses the entire trajectory $\{X_t\}$ to assess the quality of the sample, not just the final point $X_T$.

## 9.3 Feynman-Kac Correctors for Diffusion Models

The Feynman-Kac framework specializes cleanly to diffusion-based generative models. Suppose the generative model runs the reverse SDE:

$$dX_t = \hat{f}_t(X_t)\, dt + g_t\, dW_t, \qquad X_0 \sim p_0,$$

with approximate drift $\hat{f}_t \approx f_t^*$ (the true reverse drift). The Feynman-Kac potential for the path-level correction is:

> [!equation] 9.3
> $$V_t(x) = \frac{1}{2g_t^2}\|\hat{f}_t(x) - f_t^*(x)\|^2 + \frac{1}{2}\nabla \cdot (\hat{f}_t(x) - f_t^*(x)).$$


The first term is the squared drift discrepancy, normalized by the noise level. The second is a divergence correction that accounts for the density change induced by the drift error. Together, they define the Girsanov change of measure between the approximate and exact path measures.

> [!definition] 9.2 — Feynman-Kac Corrector
> Given a generative SDE with approximate drift $\hat{f}_t$ and a target defined by exact drift $f_t^*$, the **Feynman-Kac corrector** assigns to each generated path $\{X_t\}_{t=0}^T$ the weight:
>
> $$w(\{X_t\}) = \exp\!\left(-\int_0^T V_t(X_t)\, dt\right),$$
>
> where $V_t$ is the potential (9.3). The corrected estimator for $\mathbb{E}_{p_{\text{target}}}[h(x)]$ is:
>
> $$\hat{\mu}_{\mathrm{FK}} = \frac{\sum_{i=1}^n w(\{X_t^{(i)}\})\, h(X_T^{(i)})}{\sum_{i=1}^n w(\{X_t^{(i)}\})}.$$


The FK corrector has a crucial advantage over endpoint importance sampling: it uses information from the *entire trajectory*, not just the final sample. This means the effective sample size degrades more gracefully — the path-level weights have lower variance than endpoint weights when the model is approximately correct along most of the trajectory but makes localized errors.

> [!remark] 9.2
> Computing the FK corrector requires knowing the exact drift $f_t^*$, which involves the true score $\nabla \log p_t$. In practice, this is not available — if we had the exact score, we wouldn't need the correction. The practical implementations use approximate scores from an independently trained model, or exploit the fact that for certain targets (e.g., Boltzmann distributions), the score at the endpoint ($t = T$) is known: $\nabla \log p_T(x) = -\beta \nabla U(x)$ if $p_T$ is the Boltzmann distribution. The FK correction then only needs the approximate score at intermediate times, with the exact endpoint information anchoring the estimate.


## 9.4 Sequential Monte Carlo Samplers

**Sequential Monte Carlo (SMC)** combines importance sampling with resampling to maintain a population of particles that collectively approximate the target distribution. Applied to generative models, SMC runs multiple particles through the generative SDE simultaneously, periodically resampling according to importance weights to eliminate poor particles and duplicate good ones.

> [!definition] 9.3 — SMC for Generative Diffusion
> Given a generative process $\{X_t\}_{t=0}^T$ with approximate drift $\hat{f}_t$:
>
> 1. **Initialize**: Sample $N$ particles $\{X_0^{(i)}\}_{i=1}^N$ from $p_0$.
> 2. **For each time step** $t_k \to t_{k+1}$:
>    - **Propagate**: Run each particle forward by one SDE step: $X_{t_{k+1}}^{(i)} = X_{t_k}^{(i)} + \hat{f}_{t_k}(X_{t_k}^{(i)})\Delta t + g_{t_k}\sqrt{\Delta t}\, z^{(i)}$.
>    - **Weight**: Compute incremental weights $w_k^{(i)} = \exp(-V_{t_k}(X_{t_k}^{(i)})\Delta t)$.
>    - **Resample**: If the ESS drops below a threshold $N/2$, resample the particles with replacement according to normalized weights, then reset all weights to 1.
> 3. **Output**: The final particles $\{X_T^{(i)}\}$ with their accumulated weights approximate $p_{\text{target}}$.


The resampling step is what distinguishes SMC from simple importance sampling: by eliminating low-weight particles early, SMC prevents the weight distribution from degenerating. The cost is that resampling introduces correlations between particles, reducing the effective independence of the final samples. The tradeoff is controlled by the resampling threshold — more frequent resampling reduces weight variance but increases particle correlation.

SMC with $N$ particles provides an **unbiased estimate** of the normalizing constant $Z$ as a byproduct:

> [!equation] 9.4
> $$\hat{Z}_{\mathrm{SMC}} = \prod_{k=0}^{K-1} \left(\frac{1}{N}\sum_{i=1}^N w_k^{(i)}\right),$$


a property that is invaluable for model comparison and free energy estimation. The estimate is unbiased regardless of $N$, though its variance decreases with $N$.

## 9.5 Annealed Importance Sampling

**Annealed importance sampling (AIS)** is a special case of SMC where the intermediate distributions are defined by *tempering* — interpolating between an easy-to-sample distribution and the target:

> [!equation] 9.5
> $$p_k(x) \propto p_0(x)^{1-\beta_k}\, p_{\text{target}}(x)^{\beta_k}, \qquad 0 = \beta_0 < \beta_1 < \cdots < \beta_K = 1,$$


where $\beta_k$ is the inverse temperature schedule. At $\beta_0 = 0$, $p_0 = p_0$ (the easy base distribution); at $\beta_K = 1$, $p_K = p_{\text{target}}$.

Each step of AIS runs a Markov transition (e.g., a few steps of MCMC) targeting $p_k$, followed by an importance weight update accounting for the density change from $p_{k-1}$ to $p_k$. The result is a particle with a cumulative weight that, when properly normalized, provides an unbiased estimate of the ratio $Z_{\text{target}} / Z_0$.

> [!remark] 9.3
> AIS and generative flows address the same problem — transporting $p_0$ to $p_{\text{target}}$ — from opposite directions. The generative flow learns a continuous transport map; AIS constructs a discrete sequence of intermediate distributions connected by MCMC transitions. The natural synthesis is to use the generative flow *as the transport* within AIS: propagate particles along the learned flow (which is cheap but approximate), and correct with MCMC transitions at each intermediate distribution (which is exact but expensive). This hybrid — flow-guided AIS — achieves the efficiency of the flow with the exactness of the Monte Carlo correction.


## 9.6 Twisted Diffusion Samplers

A recent development combines the Feynman-Kac framework with a *learned twist function* to improve the efficiency of the correction. The idea is to modify the generative SDE's drift by a twist — a function that anticipates the future reweighting — so that the particles are already biased toward high-weight regions before the correction is applied.

> [!definition] 9.4 — Twisted Diffusion Sampler
> Given a target $p_{\text{target}}$ and a reference diffusion with drift $f_t$, a **twisted diffusion sampler** uses a modified drift:
>
> $$\hat{f}_t^{\text{twist}}(x) = f_t(x) + g_t^2\, \nabla_x \log \psi_t(x),$$
>
> where $\psi_t(x) \approx \mathbb{E}[\tilde{p}(X_T) \mid X_t = x]$ is the **twist function** — an approximation to the expected (unnormalized) target density evaluated at the endpoint, conditioned on the current position.


The twist function acts as a "lookahead": at time $t$, it estimates how likely the current particle is to end up in a high-density region of the target, and adjusts the drift to steer toward such regions. The optimal twist makes the FK weights exactly equal to 1 — all particles contribute equally — but learning the optimal twist is as hard as solving the original problem. In practice, a parameterized twist $\psi_\phi$ is trained to minimize the variance of the importance weights, and even a rough approximation substantially improves the ESS.

The twisted sampler has a natural interpretation as a **controlled diffusion**: the twist $\nabla \log \psi_t$ is the optimal control that steers the reference diffusion toward the target, in the sense of minimizing the KL divergence between the controlled and target path measures. This connects the correction problem back to the Schrödinger bridge of Chapter 4 — the optimal twist produces the Schrödinger bridge drift.

## 9.7 Particle Filtering Along the Generative Path

The SMC framework can be applied not just at the endpoint but *continuously along the generative path*, using the structure of the flow or diffusion to define intermediate targets. This is **particle filtering** for generative models.

For a generative flow with velocity field $v_\theta$ and probability path $\{q_t^\theta\}$, suppose the true target path is $\{p_t\}$. At each intermediate time $t_k$, the ratio $p_{t_k}(x) / q_{t_k}^\theta(x)$ provides an incremental weight. Resampling at intermediate times prevents weight degeneracy and keeps the particle population concentrated in high-density regions of the *true* intermediate distributions.

The practical challenge is computing $p_{t_k}(x)$ — the true intermediate density, which is generally unknown. Two approaches exist:

1. **Score-based reweighting**: If an approximate score $s_\theta(x, t_k) \approx \nabla \log p_{t_k}(x)$ is available, the weight ratio can be estimated from the score discrepancy.
2. **Energy-based reweighting**: If the target is a Boltzmann distribution and the intermediate distributions are defined by a temperature schedule $p_{t_k} \propto e^{-\beta_k U(x)}$, the weights involve only the energy $U(x)$, which is computable.

In both cases, the combination of a trained generative model (providing the proposal dynamics) with SMC correction (providing exactness guarantees) produces an estimator that is asymptotically exact and practically efficient — significantly better than either component alone.

## 9.8 When Corrections Close the Gap

The correction methods of this chapter have the following asymptotic guarantees:

- **Importance sampling** (endpoint): converges as $O(1/\sqrt{n})$ but with a constant that depends exponentially on $\mathrm{KL}(p_{\text{target}} \| q_\theta)$.
- **SMC** (path-level): converges as $O(1/\sqrt{N})$ with a constant that depends on the maximum incremental weight variance — typically much smaller than the endpoint KL.
- **Twisted SMC**: the twist reduces the incremental weight variance, improving the constant further.

The practical consequence: for a generative model with moderate accuracy ($\mathrm{KL} \leq 5$ nats), path-level SMC with $N = 100$–$1000$ particles provides corrected estimates with ESS $> 0.5N$. For a highly accurate model ($\mathrm{KL} \leq 1$ nat), endpoint importance sampling with $N = 100$ particles suffices. For a poor model ($\mathrm{KL} > 10$ nats), no amount of correction can help — the model must be retrained.

The correction methods also provide a *diagnostic*: the ESS tells you how good the model is, without requiring access to the true distribution's normalizing constant. An ESS that is a substantial fraction of $N$ means the model is accurate; an ESS near 1 means it is not. This diagnostic is particularly valuable for energy-based targets, where no other quantitative evaluation is available.

## 9.9 Historical Notes (Correction)

Importance sampling is one of the oldest Monte Carlo techniques, dating to **Kahn and Marshall (1953)**. Self-normalized importance sampling and its variance properties were analyzed by **Hesterberg (1995)** and **Owen (2013)**. The ESS diagnostic was introduced by **Kong, Liu, and Wong (1994)**.

Sequential Monte Carlo methods were developed by **Gordon, Salmond, and Smith (1993)** for state estimation (the "bootstrap particle filter") and generalized to arbitrary target sequences by **Del Moral, Doucet, and Jasra (2006)**. The application of SMC to sample from complex distributions (rather than filtering) was formalized as "SMC samplers" by **Del Moral, Doucet, and Jasra (2006)**.

Annealed importance sampling was introduced by **Neal (2001)** ("Annealed Importance Sampling"), combining tempering with importance sampling. The connection to the Jarzynski equality from non-equilibrium statistical mechanics was noted by **Neal (2001)** and formalized by **Crooks (2000)**.

The Feynman-Kac formula is a classical result of stochastic analysis; see **Karatzas and Shreve (1991)** for the standard treatment. Its application to correcting diffusion generative models was developed by **Richter, Berner, and Liu (2024)** ("Improved Sampling via Learned Diffusions") and **Wu, Trippe, Naesseth, Blei, and Cunningham (2020)** ("Practical and Asymptotically Exact Conditional Sampling in Diffusion Models").

Twisted diffusion samplers were introduced by **Wu, Trippe, Naesseth, Blei, and Cunningham (2023)** and **Cardoso, Idrissi, Rector-Brooks, Bengio, and Malkin (2024)**. The controlled diffusion interpretation connects to the stochastic control literature of **Fleming and Soner (2006)** and to the path integral control methods of **Kappen (2005)**.

The synthesis of generative models with SMC — using the flow as a proposal within a particle filter — was developed by **Dou and Song (2024)** ("Diffusion Posterior Sampling with Learned Twisting"), **Phillips, Seymour, Sarkar, and Doucet (2024)**, and **Berner, Richter, Seidl, and Ullrich (2024)**. The application to Boltzmann generation with quantitative accuracy guarantees was demonstrated by **Midgley, Stimper, Simm, Schölkopf, and Hernández-Lobato (2023)** ("Flow Annealed Importance Sampling Bootstrap, FAB").

---

An unconditional generative model learns to sample from a distribution $p_{\text{data}}$. But in practice, we almost never want a random sample from the full data distribution — we want an image of a specific subject, a molecule with specific properties, a protein that binds a specific target. The question of how to steer a trained generative model toward a desired subset or property of the data distribution is the subject of the remaining sections. The answer turns out to involve a single mathematical operation — tilting the distribution by a likelihood or reward function — realized through different mechanisms depending on whether the guidance signal is available at training time, inference time, or both.

## 9.10 The Conditional Generation Problem

The formal setting: given a condition $y$ (a class label, a text prompt, a partial observation, an energy function), generate samples from the conditional distribution $p(x | y)$. By Bayes' rule:

> [!equation] 9.6
> $$p(x | y) = \frac{p(y | x)\, p(x)}{p(y)} \propto p(y | x)\, p(x),$$


where $p(x)$ is the unconditional data distribution (the prior), $p(y|x)$ is the likelihood of the condition given the data, and $p(y)$ is the evidence (a normalizing constant). The conditional distribution is the unconditional distribution *tilted* by the likelihood.

In the language of generative flows: the unconditional model provides access to $p(x)$ through its score function $\nabla_x \log p_t(x)$ or velocity field $v_t(x)$. The guidance signal provides access to $p(y|x)$ or its gradient. The conditional score is:

> [!equation] 9.7
> $$\nabla_x \log p_t(x | y) = \nabla_x \log p_t(x) + \nabla_x \log p_t(y | x),$$


where $p_t(y|x)$ is the likelihood at noise level $t$ — the probability of the condition given a noisy version of the data. The conditional velocity field is derived from this conditional score via the probability flow ODE relationship.

The challenge is computing $\nabla_x \log p_t(y|x)$ — the gradient of the likelihood with respect to the noisy data $x$ at intermediate noise levels. Different methods handle this differently, and the differences define the landscape of guided generation.

## 9.11 Classifier Guidance

The most direct approach: train a separate classifier $p_\phi(y|x, t)$ on noisy data at all noise levels, and use its gradient to steer the generative process.

> [!definition] 9.5 — Classifier Guidance
> Given an unconditional diffusion model with score $s_\theta(x, t) \approx \nabla_x \log p_t(x)$ and a noise-conditional classifier $p_\phi(y|x, t)$, the **classifier-guided score** is:
>
> $$\tilde{s}(x, t) = s_\theta(x, t) + w \cdot \nabla_x \log p_\phi(y | x, t),$$
>
> where $w \geq 0$ is the **guidance scale**. At $w = 0$, sampling is unconditional. At $w = 1$, sampling targets $p(x|y)$ exactly (in the limit of perfect models). At $w > 1$, sampling targets a *sharpened* conditional $p(x|y)^w / Z_w$ that concentrates on the most likely samples given $y$.


The guided SDE or ODE replaces the unconditional score with the guided score $\tilde{s}$ — everything else (the noise schedule, the solver, the architecture) remains unchanged. The classifier gradient $\nabla_x \log p_\phi(y|x, t)$ acts as an additional force that pushes samples toward regions where the condition is satisfied.

> [!remark] 9.4
> The guidance scale $w > 1$ has no Bayesian justification — it corresponds to sampling from a tempered posterior $p(x|y)^w$ that is *more concentrated* than the true conditional. In practice, $w \in [2, 10]$ produces the best samples by human evaluation, despite being statistically "wrong." The explanation is that the unconditional model $p(x)$ already assigns some mass to low-quality samples (blurry images, implausible molecules), and the extra guidance removes this mass. The price is reduced diversity — high $w$ produces near-identical samples for a given condition.


## 9.12 Classifier-Free Guidance

Classifier guidance requires training a separate classifier on noisy data — an additional model with its own training pipeline. **Classifier-free guidance** (CFG) eliminates this requirement by using the generative model itself as an implicit classifier.

The key identity: the classifier gradient can be written as the difference between conditional and unconditional scores:

$$\nabla_x \log p_t(y|x) = \nabla_x \log p_t(x|y) - \nabla_x \log p_t(x).$$

Substituting into the guided score:

> [!equation] 9.8
> $$\tilde{s}(x, t) = s_\theta(x, t) + w \cdot [s_\theta(x, t | y) - s_\theta(x, t)] = (1 + w)\, s_\theta(x, t | y) - w \cdot s_\theta(x, t).$$


> [!definition] 9.6 — Classifier-Free Guidance
> A **classifier-free guided** model is trained as a *conditional* model $s_\theta(x, t, y)$ that can also run unconditionally by replacing the condition with a null token $\varnothing$: $s_\theta(x, t) \equiv s_\theta(x, t, \varnothing)$. During training, the condition is dropped (replaced by $\varnothing$) with probability $p_{\text{uncond}}$ (typically 10–20%). At inference, the guided prediction is:
>
> $$\tilde{s}(x, t) = (1 + w)\, s_\theta(x, t, y) - w \cdot s_\theta(x, t, \varnothing),$$
>
> which requires two forward passes per step: one conditional, one unconditional.


CFG has become the dominant guidance method for large-scale generative models (Stable Diffusion, DALL-E, Imagen) for several reasons: (1) no separate classifier is needed; (2) the unconditional model is obtained for free by label dropout during training; (3) the method applies to any type of condition (text, class, image) without modification.

> [!proposition] 9.1 — CFG as Tempered Bayes
> Classifier-free guidance with scale $w$ samples from the tilted distribution:
>
> $$\tilde{p}_w(x | y) \propto p(x | y) \cdot \left(\frac{p(x|y)}{p(x)}\right)^w \propto \frac{p(y|x)^{1+w}\, p(x)}{p(y)^{1+w}}.$$
>
> This is Bayes' rule with the likelihood raised to the power $1+w$: the guidance sharpens the influence of the condition relative to the prior. At $w = 0$, this is the true posterior; at large $w$, the posterior concentrates on the maximum-likelihood estimate $\arg\max_x p(y|x)$.


## 9.13 Conditional Flow Matching

For flow-based models, conditional generation can be built directly into the flow matching framework without the detour through scores and guidance.

> [!definition] 9.7 — Conditional Flow Matching
> Given paired data $(x_1, y)$ where $x_1$ is the data and $y$ is the condition, **conditional flow matching** trains a velocity field $v_\theta(x, t, y)$ using the conditional flow matching objective:
>
> $$\mathcal{L}_{\mathrm{CFM}}(\theta) = \mathbb{E}_{t,\, (x_1, y) \sim p_{\text{data}},\, \epsilon \sim \mathcal{N}(0,I)}\!\left[\|v_\theta(x_t, t, y) - (x_1 - \epsilon)\|^2\right],$$
>
> where $x_t = tx_1 + (1-t)\epsilon$ is the linear interpolant. At inference, the ODE $\dot{x} = v_\theta(x, t, y)$ is solved with the desired condition $y$.


Conditional flow matching is simply flow matching with the condition as an additional input to the network. No guidance scale is needed at inference — the model directly generates from $p(x|y)$ rather than requiring post-hoc steering. CFG can still be applied on top of conditional flow matching for additional sharpening, using the same formula (9.8) with $v_\theta$ replacing $s_\theta$.

The advantage of direct conditional training is efficiency: one forward pass per step (vs. two for CFG). The disadvantage is that the condition must be available at training time — you cannot guide toward conditions that were not seen during training.

## 9.14 Reward-Based and Energy-Based Steering

A more general setting: instead of a discrete condition $y$, the guidance signal is a **reward function** $R(x)$ or **energy function** $U(x)$ that assigns a scalar score to each generated sample. The goal is to generate samples that maximize the reward (or minimize the energy) while maintaining diversity.

The tilted distribution is:

> [!equation] 9.9
> $$p_\beta(x) \propto p(x) \cdot e^{\beta R(x)},$$


where $\beta > 0$ is the inverse temperature. At $\beta = 0$, sampling is unconditional. As $\beta \to \infty$, sampling concentrates on the mode $\arg\max_x R(x)$ within the support of $p(x)$.

For a flow-based or diffusion-based model, the score of the tilted distribution at noise level $t$ is:

$$\nabla_x \log p_{t,\beta}(x) = \nabla_x \log p_t(x) + \beta\, \nabla_x R_t(x),$$

where $R_t(x) = \mathbb{E}[R(X_1) | X_t = x]$ is the expected reward conditioned on the current noisy state. This is analogous to classifier guidance, with the reward gradient replacing the classifier gradient.

> [!remark] 9.5
> Computing $R_t(x) = \mathbb{E}[R(X_1) | X_t = x]$ — the expected reward at the endpoint given a noisy intermediate — is the central challenge of reward-based guidance. Two approaches: (1) **one-step denoising**: approximate $R_t(x) \approx R(\hat{x}_1(x, t))$ where $\hat{x}_1$ is the one-step denoising estimate (Tweedie formula); (2) **learned value function**: train a neural network $V_\phi(x, t) \approx R_t(x)$ by regressing against the reward of denoised samples. Approach (1) is cheap but noisy at high noise levels; approach (2) is accurate but requires an additional training phase.


This framework connects to the reinforcement learning literature: the reward function $R(x)$ plays the role of a terminal reward, the diffusion process is the dynamics, and the guidance is the optimal policy. The value function $R_t(x)$ is the state-value function, and the guidance gradient $\nabla_x R_t$ is the policy gradient. Fine-tuning generative models with reinforcement learning (DDPO, DRaFT, ReFL) makes this connection explicit.

## 9.15 Posterior Sampling and Inverse Problems

A particularly important class of conditional generation problems is **inverse problems**: given a measurement $y = A(x) + \eta$ (where $A$ is a known forward operator and $\eta$ is noise), recover $x$ from $y$. This arises in medical imaging (MRI reconstruction from undersampled k-space), computational photography (super-resolution, inpainting, deblurring), and scientific measurement (cryo-EM structure determination).

The posterior is $p(x|y) \propto p(y|x)\, p(x)$, where $p(y|x) = \mathcal{N}(y; A(x), \sigma_\eta^2 I)$ for Gaussian noise. The posterior score is:

> [!equation] 9.10
> $$\nabla_x \log p(x|y) = \nabla_x \log p(x) - \frac{1}{\sigma_\eta^2} A^\top(A(x) - y).$$


For *linear* inverse problems ($A$ is a matrix), the likelihood gradient is known analytically: $\nabla_x \log p(y|x) = -\sigma_\eta^{-2} A^\top(Ax - y)$. The guided score at noise level $t$ uses the one-step denoising estimate $\hat{x}_0(x_t, t)$ to approximate the likelihood:

$$\nabla_{x_t} \log p(y | x_t) \approx -\frac{1}{\sigma_\eta^2} \nabla_{x_t}\!\left[\|A\hat{x}_0(x_t, t) - y\|^2\right],$$

which requires backpropagating through the denoiser — one additional backward pass per solver step.

> [!definition] 9.8 — Diffusion Posterior Sampling (DPS)
> **Diffusion Posterior Sampling** replaces the intractable noise-level likelihood $p_t(y|x_t)$ with an approximation based on the one-step denoising estimate:
>
> $$p_t(y | x_t) \approx \mathcal{N}\!\left(y;\; A(\hat{x}_0(x_t, t)),\; r_t^2 I\right),$$
>
> where $\hat{x}_0(x_t, t) = \mathbb{E}[X_0 | X_t = x_t]$ is the Tweedie denoiser and $r_t$ is a noise scale that accounts for the uncertainty in the denoising estimate. The guided score is:
>
> $$\tilde{s}(x_t, t) = s_\theta(x_t, t) - \frac{1}{r_t^2}\nabla_{x_t}\|A(\hat{x}_0(x_t, t)) - y\|^2.$$


DPS and its variants (PSLD, $\Pi$GDM, RED-diff) have achieved state-of-the-art results on inverse problems across imaging modalities, often outperforming task-specific methods that were trained end-to-end for each inverse problem.

## 9.16 Compositional Generation

A powerful consequence of the score-based formulation is **compositionality**: multiple conditions can be combined by adding their score contributions.

Given conditions $y_1, \ldots, y_K$ with independent likelihoods $p(y_k|x)$, the joint conditional score decomposes as:

> [!equation] 9.11
> $$\nabla_x \log p(x | y_1, \ldots, y_K) = \nabla_x \log p(x) + \sum_{k=1}^K \nabla_x \log p(y_k | x).$$


Each condition contributes an additive guidance term, and the terms can be combined freely at inference time without retraining. This enables:

- **Multi-attribute generation**: "a red car on a rainy street" = color guidance + object guidance + scene guidance
- **Negation**: "a landscape without people" = landscape guidance $-$ person guidance
- **Logical operations**: conjunction (add scores), disjunction (logsumexp of scores), negation (subtract scores)

> [!remark] 9.6
> Compositionality is exact when the conditions are truly independent given $x$ — meaning $p(y_1, y_2 | x) = p(y_1|x) p(y_2|x)$. In practice, conditions are rarely independent (the color and shape of an object are correlated), so the composed guidance is approximate. The approximation is usually good enough for practical use, but it can produce artifacts when conditions conflict (e.g., "a round square"). The severity of the approximation depends on the degree of conditional dependence and on the guidance scale — high $w$ amplifies both the guidance and its artifacts.


## 9.17 The Guidance-Diversity Tradeoff

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

## 9.18 Historical Notes (Guidance)

**Classifier guidance** was introduced by **Dhariwal and Nichol (2021)** ("Diffusion Models Beat GANs on Image Synthesis"), who trained noise-conditional classifiers and demonstrated that guided diffusion models produce higher-quality images than GANs on ImageNet. The guidance scale $w > 1$ and its effect on the FID-IS tradeoff were characterized in this paper.

**Classifier-free guidance** was introduced by **Ho and Salimans (2022)** ("Classifier-Free Diffusion Guidance"), who showed that label dropout during training provides a free unconditional model and that the two-evaluation guidance formula (9.8) matches or exceeds classifier guidance without a separate classifier. CFG has since become the standard guidance method for large-scale text-to-image models: **Ramesh et al. (2022)** (DALL-E 2), **Saharia et al. (2022)** (Imagen), and **Rombach et al. (2022)** (Stable Diffusion) all use CFG.

The tilted-distribution interpretation of guidance was formalized by **Karras et al. (2022)** and **Bradley (2024)** ("Classifier-Free Guidance is a Predictor-Corrector"). The connection to reward maximization was developed by **Black et al. (2024)** (DDPO), **Clark et al. (2024)** (DRaFT), and **Xu et al. (2024)** (Imagereward), who fine-tuned diffusion models with reinforcement learning objectives.

**Diffusion Posterior Sampling (DPS)** was introduced by **Chung, Kim, Mccann, Klasky, and Ye (2023)** ("Diffusion Posterior Sampling for General Noisy Inverse Problems"). Related approaches include **Song, Shen, Xing, and Ermon (2022)** for linear inverse problems, **Kawar, Elad, Ermon, and Song (2022)** (DDRM) for spectral methods, and **Chung et al. (2023)** ($\Pi$GDM). The application to cryo-EM reconstruction was demonstrated by **Levy, Poitevin, Marber, et al. (2024)** (CryoFM).

Compositional generation was explored by **Liu, Karalias, Rempel, and Fidler (2022)** ("Compositional Visual Generation with Composable Diffusion Models") and **Du, Li, and Mordatch (2023)**. The information-theoretic analysis of the guidance-diversity tradeoff appears in **Ho and Salimans (2022)** and was further developed by **Kynkäänniemi, Karras, Laine, Lehtinen, and Aila (2024)**.

Conditional flow matching was developed as a natural extension of the flow matching framework by **Lipman et al. (2023)** and **Tong et al. (2024)**. The application to molecular design — conditioning on molecular properties, binding targets, or experimental constraints — is an active area developed by **Jing et al. (2024)** (AlphaFold 3's diffusion module), **Yim et al. (2023)** (FrameFlow), and **Song et al. (2024)** (EquiFM).

## References

- [Dhariwal and Nichol (2021)](https://arxiv.org/abs/2105.05233) — Diffusion Models Beat GANs on Image Synthesis
- [Ho and Salimans (2022)](https://arxiv.org/abs/2207.12598) — Classifier-Free Diffusion Guidance
- [Chung, Kim, Mccann, Klasky, and Ye (2023)](https://arxiv.org/abs/2209.14687) — Diffusion Posterior Sampling for General Noisy Inverse Problems
- [Ramesh et al. (2022)](https://arxiv.org/abs/2204.06125) — Hierarchical Text-Conditional Image Generation with CLIP Latents (DALL-E 2)
- [Lipman et al. (ICLR 2023)](https://arxiv.org/abs/2210.02747) — Flow Matching for Generative Modeling
