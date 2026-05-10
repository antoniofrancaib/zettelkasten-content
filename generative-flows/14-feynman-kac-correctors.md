---

title: "Correcting the Approximation"
subtitle: "Importance sampling, the Feynman-Kac formula, SMC samplers, annealed importance sampling, and how to close the gap between the learned and true distribution — 2001 to 2026"
---



Every generative model is an approximation. The neural network has finite capacity; the training data is finite; the training procedure converges to a local optimum; the ODE solver introduces discretization error. The result is a model distribution $q_\theta$ that is close to, but not equal to, the target $p_{\text{target}}$. For many applications — generating images, synthesizing speech — "close" is good enough. But for scientific applications where quantitative accuracy matters — computing free energies, sampling from Boltzmann distributions, estimating posterior probabilities — "close" is not enough. The gap between $q_\theta$ and $p_{\text{target}}$ must be measured and, ideally, closed.

This chapter develops the tools for closing that gap. The key idea is **correction**: given an imperfect generative model $q_\theta$, use importance sampling, sequential Monte Carlo, or Feynman-Kac reweighting to transform samples from $q_\theta$ into (asymptotically) exact samples from $p_{\text{target}}$. The correction is always possible in principle — importance sampling is exact given enough samples — but the efficiency of the correction depends on how close $q_\theta$ is to $p_{\text{target}}$. A good generative model makes the correction cheap; a poor one makes it exponentially expensive. The generative model and the correction method are partners, not alternatives.

## 14.1 The Correction Problem

Let $q_\theta$ be the distribution induced by a generative model (a flow, a diffusion model, a bridge) and $p_{\text{target}}$ be the true target distribution. We assume that $p_{\text{target}}$ can be evaluated up to a normalizing constant: $p_{\text{target}}(x) = \tilde{p}(x) / Z$, where $\tilde{p}(x)$ is computable but $Z = \int \tilde{p}(x)\, dx$ is not. This is the standard setting for Boltzmann distributions ($\tilde{p}(x) = e^{-\beta U(x)}$), Bayesian posteriors ($\tilde{p}(x) = p(\text{data}|x)p(x)$), and energy-based models.

The correction problem is: given $n$ samples $\{x_i\}_{i=1}^n$ from $q_\theta$, produce estimates of expectations $\mathbb{E}_{p_{\text{target}}}[h(x)]$ for arbitrary test functions $h$.

> [!definition] 14.1 — Self-Normalized Importance Sampling
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

> [!equation] 14.1
> $$\mathrm{ESS} = \frac{\left(\sum_i \tilde{w}_i\right)^2}{\sum_i \tilde{w}_i^2} \in [1, n].$$


ESS $= n$ when all weights are equal ($q_\theta = p_{\text{target}}$); ESS $= 1$ when one weight dominates (catastrophic mismatch). An ESS of $n/2$ means the importance sampling is half as efficient as direct sampling — a reasonable correction. An ESS of $1$ means the correction has failed entirely.

> [!remark] 14.1
> The ESS depends exponentially on the KL divergence: $\mathrm{ESS} \approx n \cdot e^{-\mathrm{KL}(p \| q)}$ for large $n$. This means importance sampling is only practical when $q_\theta$ is already close to $p_{\text{target}}$ — a KL divergence of 10 nats reduces the ESS by a factor of $e^{10} \approx 22{,}000$. The generative model is doing most of the work; importance sampling handles the residual error. This is why correction methods are *complements* to generative models, not substitutes.


## 14.2 The Feynman-Kac Formula

The Feynman-Kac formula connects the solution of a PDE to an expectation over stochastic paths, providing a natural framework for correcting generative models that produce paths (flows, diffusions) rather than just endpoints.

> [!theorem] 14.1 — Feynman-Kac Formula
> Let $\{X_t\}_{t \in [0,T]}$ be the solution of the SDE $dX_t = f_t(X_t)\, dt + g_t\, dW_t$, and let $V : \mathbb{R}^d \times [0,T] \to \mathbb{R}$ be a potential function. Then the function:
>
> $$u(x, t) = \mathbb{E}\!\left[h(X_T)\, \exp\!\left(-\int_t^T V(X_s, s)\, ds\right) \;\Big|\; X_t = x\right]$$
>
> solves the PDE $\partial_t u + f_t \cdot \nabla u + \frac{g_t^2}{2}\Delta u - V\, u = 0$ with terminal condition $u(x, T) = h(x)$.


For generative modeling, the Feynman-Kac formula provides a way to *reweight paths* rather than just endpoints. Consider a generative diffusion model with learned drift $f_\theta$ that approximately samples from $p_{\text{target}}$. The true target satisfies a Fokker-Planck equation; the approximate model satisfies a different one. The ratio between the two — the Radon-Nikodym derivative of the true path measure with respect to the approximate one — can be expressed as a Feynman-Kac exponential:

> [!equation] 14.2
> $$\frac{dP_{\text{target}}}{dP_\theta}(\{X_t\}) = \exp\!\left(-\int_0^T V_t(X_t)\, dt\right),$$


where $V_t$ is a potential that depends on the discrepancy between the true and approximate drifts. This path-level reweighting is more informative than endpoint reweighting: it uses the entire trajectory $\{X_t\}$ to assess the quality of the sample, not just the final point $X_T$.

## 14.3 Feynman-Kac Correctors for Diffusion Models

The Feynman-Kac framework specializes cleanly to diffusion-based generative models. Suppose the generative model runs the reverse SDE:

$$dX_t = \hat{f}_t(X_t)\, dt + g_t\, dW_t, \qquad X_0 \sim p_0,$$

with approximate drift $\hat{f}_t \approx f_t^*$ (the true reverse drift). The Feynman-Kac potential for the path-level correction is:

> [!equation] 14.3
> $$V_t(x) = \frac{1}{2g_t^2}\|\hat{f}_t(x) - f_t^*(x)\|^2 + \frac{1}{2}\nabla \cdot (\hat{f}_t(x) - f_t^*(x)).$$


The first term is the squared drift discrepancy, normalized by the noise level. The second is a divergence correction that accounts for the density change induced by the drift error. Together, they define the Girsanov change of measure between the approximate and exact path measures.

> [!definition] 14.2 — Feynman-Kac Corrector
> Given a generative SDE with approximate drift $\hat{f}_t$ and a target defined by exact drift $f_t^*$, the **Feynman-Kac corrector** assigns to each generated path $\{X_t\}_{t=0}^T$ the weight:
>
> $$w(\{X_t\}) = \exp\!\left(-\int_0^T V_t(X_t)\, dt\right),$$
>
> where $V_t$ is the potential (14.3). The corrected estimator for $\mathbb{E}_{p_{\text{target}}}[h(x)]$ is:
>
> $$\hat{\mu}_{\mathrm{FK}} = \frac{\sum_{i=1}^n w(\{X_t^{(i)}\})\, h(X_T^{(i)})}{\sum_{i=1}^n w(\{X_t^{(i)}\})}.$$


The FK corrector has a crucial advantage over endpoint importance sampling: it uses information from the *entire trajectory*, not just the final sample. This means the effective sample size degrades more gracefully — the path-level weights have lower variance than endpoint weights when the model is approximately correct along most of the trajectory but makes localized errors.

> [!remark] 14.2
> Computing the FK corrector requires knowing the exact drift $f_t^*$, which involves the true score $\nabla \log p_t$. In practice, this is not available — if we had the exact score, we wouldn't need the correction. The practical implementations use approximate scores from an independently trained model, or exploit the fact that for certain targets (e.g., Boltzmann distributions), the score at the endpoint ($t = T$) is known: $\nabla \log p_T(x) = -\beta \nabla U(x)$ if $p_T$ is the Boltzmann distribution. The FK correction then only needs the approximate score at intermediate times, with the exact endpoint information anchoring the estimate.


## 14.4 Sequential Monte Carlo Samplers

**Sequential Monte Carlo (SMC)** combines importance sampling with resampling to maintain a population of particles that collectively approximate the target distribution. Applied to generative models, SMC runs multiple particles through the generative SDE simultaneously, periodically resampling according to importance weights to eliminate poor particles and duplicate good ones.

> [!definition] 14.3 — SMC for Generative Diffusion
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

> [!equation] 14.4
> $$\hat{Z}_{\mathrm{SMC}} = \prod_{k=0}^{K-1} \left(\frac{1}{N}\sum_{i=1}^N w_k^{(i)}\right),$$


a property that is invaluable for model comparison and free energy estimation. The estimate is unbiased regardless of $N$, though its variance decreases with $N$.

## 14.5 Annealed Importance Sampling

**Annealed importance sampling (AIS)** is a special case of SMC where the intermediate distributions are defined by *tempering* — interpolating between an easy-to-sample distribution and the target:

> [!equation] 14.5
> $$p_k(x) \propto p_0(x)^{1-\beta_k}\, p_{\text{target}}(x)^{\beta_k}, \qquad 0 = \beta_0 < \beta_1 < \cdots < \beta_K = 1,$$


where $\beta_k$ is the inverse temperature schedule. At $\beta_0 = 0$, $p_0 = p_0$ (the easy base distribution); at $\beta_K = 1$, $p_K = p_{\text{target}}$.

Each step of AIS runs a Markov transition (e.g., a few steps of MCMC) targeting $p_k$, followed by an importance weight update accounting for the density change from $p_{k-1}$ to $p_k$. The result is a particle with a cumulative weight that, when properly normalized, provides an unbiased estimate of the ratio $Z_{\text{target}} / Z_0$.

> [!remark] 14.3
> AIS and generative flows address the same problem — transporting $p_0$ to $p_{\text{target}}$ — from opposite directions. The generative flow learns a continuous transport map; AIS constructs a discrete sequence of intermediate distributions connected by MCMC transitions. The natural synthesis is to use the generative flow *as the transport* within AIS: propagate particles along the learned flow (which is cheap but approximate), and correct with MCMC transitions at each intermediate distribution (which is exact but expensive). This hybrid — flow-guided AIS — achieves the efficiency of the flow with the exactness of the Monte Carlo correction.


## 14.6 Twisted Diffusion Samplers

A recent development combines the Feynman-Kac framework with a *learned twist function* to improve the efficiency of the correction. The idea is to modify the generative SDE's drift by a twist — a function that anticipates the future reweighting — so that the particles are already biased toward high-weight regions before the correction is applied.

> [!definition] 14.4 — Twisted Diffusion Sampler
> Given a target $p_{\text{target}}$ and a reference diffusion with drift $f_t$, a **twisted diffusion sampler** uses a modified drift:
>
> $$\hat{f}_t^{\text{twist}}(x) = f_t(x) + g_t^2\, \nabla_x \log \psi_t(x),$$
>
> where $\psi_t(x) \approx \mathbb{E}[\tilde{p}(X_T) \mid X_t = x]$ is the **twist function** — an approximation to the expected (unnormalized) target density evaluated at the endpoint, conditioned on the current position.


The twist function acts as a "lookahead": at time $t$, it estimates how likely the current particle is to end up in a high-density region of the target, and adjusts the drift to steer toward such regions. The optimal twist makes the FK weights exactly equal to 1 — all particles contribute equally — but learning the optimal twist is as hard as solving the original problem. In practice, a parameterized twist $\psi_\phi$ is trained to minimize the variance of the importance weights, and even a rough approximation substantially improves the ESS.

The twisted sampler has a natural interpretation as a **controlled diffusion**: the twist $\nabla \log \psi_t$ is the optimal control that steers the reference diffusion toward the target, in the sense of minimizing the KL divergence between the controlled and target path measures. This connects the correction problem back to the Schrödinger bridge of Chapter 6 — the optimal twist produces the Schrödinger bridge drift.

## 14.7 Particle Filtering Along the Generative Path

The SMC framework can be applied not just at the endpoint but *continuously along the generative path*, using the structure of the flow or diffusion to define intermediate targets. This is **particle filtering** for generative models.

For a generative flow with velocity field $v_\theta$ and probability path $\{q_t^\theta\}$, suppose the true target path is $\{p_t\}$. At each intermediate time $t_k$, the ratio $p_{t_k}(x) / q_{t_k}^\theta(x)$ provides an incremental weight. Resampling at intermediate times prevents weight degeneracy and keeps the particle population concentrated in high-density regions of the *true* intermediate distributions.

The practical challenge is computing $p_{t_k}(x)$ — the true intermediate density, which is generally unknown. Two approaches exist:

1. **Score-based reweighting**: If an approximate score $s_\theta(x, t_k) \approx \nabla \log p_{t_k}(x)$ is available, the weight ratio can be estimated from the score discrepancy.
2. **Energy-based reweighting**: If the target is a Boltzmann distribution and the intermediate distributions are defined by a temperature schedule $p_{t_k} \propto e^{-\beta_k U(x)}$, the weights involve only the energy $U(x)$, which is computable.

In both cases, the combination of a trained generative model (providing the proposal dynamics) with SMC correction (providing exactness guarantees) produces an estimator that is asymptotically exact and practically efficient — significantly better than either component alone.

## 14.8 When Corrections Close the Gap

The correction methods of this chapter have the following asymptotic guarantees:

- **Importance sampling** (endpoint): converges as $O(1/\sqrt{n})$ but with a constant that depends exponentially on $\mathrm{KL}(p_{\text{target}} \| q_\theta)$.
- **SMC** (path-level): converges as $O(1/\sqrt{N})$ with a constant that depends on the maximum incremental weight variance — typically much smaller than the endpoint KL.
- **Twisted SMC**: the twist reduces the incremental weight variance, improving the constant further.

The practical consequence: for a generative model with moderate accuracy ($\mathrm{KL} \leq 5$ nats), path-level SMC with $N = 100$–$1000$ particles provides corrected estimates with ESS $> 0.5N$. For a highly accurate model ($\mathrm{KL} \leq 1$ nat), endpoint importance sampling with $N = 100$ particles suffices. For a poor model ($\mathrm{KL} > 10$ nats), no amount of correction can help — the model must be retrained.

The correction methods also provide a *diagnostic*: the ESS tells you how good the model is, without requiring access to the true distribution's normalizing constant. An ESS that is a substantial fraction of $N$ means the model is accurate; an ESS near 1 means it is not. This diagnostic is particularly valuable for energy-based targets, where no other quantitative evaluation is available.

## 14.9 Historical Notes

Importance sampling is one of the oldest Monte Carlo techniques, dating to **Kahn and Marshall (1953)**. Self-normalized importance sampling and its variance properties were analyzed by **Hesterberg (1995)** and **Owen (2013)**. The ESS diagnostic was introduced by **Kong, Liu, and Wong (1994)**.

Sequential Monte Carlo methods were developed by **Gordon, Salmond, and Smith (1993)** for state estimation (the "bootstrap particle filter") and generalized to arbitrary target sequences by **Del Moral, Doucet, and Jasra (2006)**. The application of SMC to sample from complex distributions (rather than filtering) was formalized as "SMC samplers" by **Del Moral, Doucet, and Jasra (2006)**.

Annealed importance sampling was introduced by **Neal (2001)** ("Annealed Importance Sampling"), combining tempering with importance sampling. The connection to the Jarzynski equality from non-equilibrium statistical mechanics was noted by **Neal (2001)** and formalized by **Crooks (2000)**.

The Feynman-Kac formula is a classical result of stochastic analysis; see **Karatzas and Shreve (1991)** for the standard treatment. Its application to correcting diffusion generative models was developed by **Richter, Berner, and Liu (2024)** ("Improved Sampling via Learned Diffusions") and **Wu, Trippe, Naesseth, Blei, and Cunningham (2020)** ("Practical and Asymptotically Exact Conditional Sampling in Diffusion Models").

Twisted diffusion samplers were introduced by **Wu, Trippe, Naesseth, Blei, and Cunningham (2023)** and **Cardoso, Idrissi, Rector-Brooks, Bengio, and Malkin (2024)**. The controlled diffusion interpretation connects to the stochastic control literature of **Fleming and Soner (2006)** and to the path integral control methods of **Kappen (2005)**.

The synthesis of generative models with SMC — using the flow as a proposal within a particle filter — was developed by **Dou and Song (2024)** ("Diffusion Posterior Sampling with Learned Twisting"), **Phillips, Seymour, Sarkar, and Doucet (2024)**, and **Berner, Richter, Seidl, and Ullrich (2024)**. The application to Boltzmann generation with quantitative accuracy guarantees was demonstrated by **Midgley, Stimper, Simm, Schölkopf, and Hernández-Lobato (2023)** ("Flow Annealed Importance Sampling Bootstrap, FAB").
