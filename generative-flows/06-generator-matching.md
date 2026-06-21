---

title: "Generator Matching"
subtitle: "Markov generators, Bregman divergences, and one theorem to rule them all — 2025"
---



By the close of Chapter 5, the theory of generative flows had split into two formally distinct families for continuous data. Euclidean flow matching works through vector fields and the continuity equation. Diffusion models work through score functions and the Fokker-Planck equation. Each has its own training objective, its own proof that the conditional version equals the marginal version, its own collection of design choices. The methods look and feel like the same idea. But the same idea as what?

The answer came in 2025, in a paper titled "Generator Matching: Generative Modeling with Arbitrary Markov Processes" by Holderrieth, Yim, and Jaakkola. It does not introduce a new generative model. It identifies the right level of abstraction at which all existing models become special cases of a single construction: **matching the generator of a Markov process** using a **Bregman divergence**. When you see the framework, the proliferation of objectives does not look like multiple different ideas. It looks like one idea written in different fonts — and as Chapter 7 will show, the same principle extends seamlessly to discrete flow matching (rate matrices, Kolmogorov forward equation) and Riemannian flow matching (geodesic interpolants on curved manifolds).

## 6.1 A Zoo of Objectives

Before presenting the unification, it is worth confronting the fragmentation it resolves. Across the preceding chapters, several training objectives have accumulated — each derived from first principles, each enjoying a conditional-equals-marginal equivalence, each requiring nothing more than sampling from conditional distributions at training time.

The **conditional flow matching** objective for Euclidean data ($x_1 \in \mathbb{R}^d$, linear interpolant):

$$\mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t,\; x_1,\; x_t | x_1}\!\left[\left\|u_t^\theta(x_t) - u_t(x_t \mid x_1)\right\|^2\right].$$

The **denoising score matching** objective for Gaussian-noised diffusion:

$$\mathcal{L}_\text{DSM}(\theta) = \mathbb{E}_{t,\; x_1,\; x_t | x_1}\!\left[\left\|s_t^\theta(x_t) + \frac{x_t - \mu_t(x_1)}{\sigma_t^2}\right\|^2\right].$$

The **discrete conditional flow matching** objective for token sequences:

$$\mathcal{L}_\text{dCFM}(\theta) = \mathbb{E}_{t,\; x_1,\; x_t | x_1}\!\left[\sum_{y \neq x_t} \left| R_t^\theta(x_t, y) - R_t(x_t, y \mid x_1)\right|^2\right].$$

The **Riemannian flow matching** objective for manifold-valued data:

$$\mathcal{L}_\text{RFM}(\theta) = \mathbb{E}_{t,\; x_1,\; x_t | x_1}\!\left[\left\|v_t^\theta(x_t) - \frac{\log_{x_t}(x_1)}{1-t}\right\|_{g_{x_t}}^2\right].$$

The structural repetition is unmistakable. In every case: sample time $t$, draw a data point $x_1$, sample a noisy state $x_t$ from the conditional path, minimize a distance between a model output and a known conditional target. The differences lie in the object being regressed ($u$, $s$, $R$, $v$) and the geometry of the distance ($\ell^2$, weighted $\ell^2$, discrete $\ell^2$, Riemannian $\|\cdot\|_g^2$). These differences are, as the generator matching framework shows, consequences of a single underlying choice — made once, at the level of the Markov process.

## 6.2 The Generator of a Markov Process

The central concept is the **infinitesimal generator** of a Markov process — a linear operator that encodes, at each instant, how the process is about to evolve. It generalizes the velocity field of an ODE to the richer settings of stochastic and discrete dynamics.

> [!definition] 6.1 — Infinitesimal Generator
> Let $\{X_t\}_{t \in [0,1]}$ be a time-inhomogeneous Markov process on a state space $\mathcal{X}$. The **infinitesimal generator** $\mathcal{L}_t$ at time $t$ is the linear operator on functions $f : \mathcal{X} \to \mathbb{R}$ defined by:
>
> $$(\mathcal{L}_t f)(x) = \lim_{\varepsilon \to 0} \frac{\mathbb{E}\!\left[f(X_{t+\varepsilon}) \mid X_t = x\right] - f(x)}{\varepsilon}.$$
>
> It encodes the instantaneous rate of change of the expected value of any observable $f$ under the process, starting from state $x$ at time $t$.


The generator takes a different concrete form depending on the type of process. For a **deterministic flow** $\dot{X}_t = u_t(X_t)$ on $\mathcal{X} = \mathbb{R}^d$:

> [!equation] 6.1
> $$(\mathcal{L}_t f)(x) = u_t(x) \cdot \nabla f(x).$$


The generator is the directional derivative along $u_t$ — it tells you how fast $f$ changes if you move with the flow.

For a **stochastic diffusion** $dX_t = b_t(X_t)\,dt + g_t\,dW_t$ on $\mathbb{R}^d$:

$$(\mathcal{L}_t f)(x) = b_t(x) \cdot \nabla f(x) + \tfrac{1}{2}g_t^2\,\Delta f(x).$$

The Laplacian term $\tfrac{1}{2}g_t^2\,\Delta f$ reflects the diffusive spread of probability mass; the drift $b_t$ contributes a transport term identical to the ODE case.

For a **continuous-time Markov chain** with rate matrix $R_t$ on finite $\mathcal{X}$:

$$(\mathcal{L}_t f)(x) = \sum_{y \in \mathcal{X}} R_t(x, y)\,\bigl[f(y) - f(x)\bigr].$$

The generator evaluates $f$ at all reachable states $y$, weighted by the jump rate $R_t(x,y)$, and computes the expected instantaneous change.

For a **Riemannian flow** with velocity $v_t$ on a manifold $(\mathcal{M}, g)$:

$$(\mathcal{L}_t f)(x) = \bigl\langle v_t(x),\; \mathrm{grad}_g\, f(x) \bigr\rangle_{g_x},$$

where $\mathrm{grad}_g\, f$ is the Riemannian gradient and $\langle \cdot, \cdot \rangle_{g_x}$ is the metric inner product at $x$.

> [!remark] 6.1
> Every Markov process — deterministic or stochastic, continuous or discrete, flat or curved — is fully characterized by its generator family $\{\mathcal{L}_t\}$. This is not a simplification: by the Hille-Yosida theorem, a sufficiently regular generator uniquely determines the associated process and its semigroup. The generator is not a summary of the dynamics; it *is* the dynamics, stated at the infinitesimal level.


## 6.3 The Generalized Continuity Equation

The generator acts on functions; its $L^2$-adjoint $\mathcal{L}_t^*$ acts on probability distributions. The evolution of any marginal density $p_t$ under the corresponding Markov process is governed by a single equation:

> [!equation] 6.2
> $$\frac{\partial}{\partial t} p_t = \mathcal{L}_t^* p_t.$$


This is the **Kolmogorov forward equation** in its most general form, and (6.2) is the common root of every evolution equation encountered in the preceding chapters.

For the **ODE generator** (6.1), the adjoint acts by $\mathcal{L}_t^* p = -\nabla \cdot (p\,u_t)$, and (6.2) becomes:
$$\partial_t p_t + \nabla \cdot (p_t u_t) = 0 \quad \text{— the continuity equation (3.1).}$$

For the **diffusion generator**, $\mathcal{L}_t^* p = -\nabla \cdot (p\,b_t) + \tfrac{1}{2}g_t^2\,\Delta p$, and (6.2) becomes:
$$\partial_t p_t = -\nabla\cdot(p_t b_t) + \tfrac{g_t^2}{2}\,\Delta p_t \quad \text{— the Fokker-Planck equation (2.7).}$$

For the **CTMC generator**, $(\mathcal{L}_t^* p)(x) = \sum_y p(y)\,R_t(y,x) - p(x)\sum_y R_t(x,y)$, and (6.2) becomes:
$$\partial_t p_t(x) = \sum_y p_t(y)\,R_t(y, x) \quad \text{— the Kolmogorov forward equation (7.1).}$$

The message is plain: the continuity equation, the Fokker-Planck equation, and the Kolmogorov forward equation are all the same equation, stated in different coordinate systems for different types of generators. This is not a formal coincidence — it reflects the fact that every Markov process conserves total probability, and equation (6.2) is the expression of that conservation law at the infinitesimal level.

## 6.4 Bregman Divergences

The second ingredient in the framework is the **Bregman divergence** — the family of loss functions that measure how well a model's generator matches the true generator of the interpolant.

> [!definition] 6.2 — Bregman Divergence
> Let $\phi : \mathcal{C} \to \mathbb{R}$ be a strictly convex, differentiable function on a convex set $\mathcal{C}$. The **Bregman divergence** generated by $\phi$ is:
>
> $$D_\phi(a \,\|\, b) = \phi(a) - \phi(b) - \nabla\phi(b)^\top(a - b).$$
>
> It measures the gap between the value of $\phi$ at $a$ and the tangent-plane approximation to $\phi$ at $b$, evaluated at $a$.


Bregman divergences are non-negative ($D_\phi(a\|b) \geq 0$, equality iff $a = b$) but generally asymmetric. They satisfy a **generalized bias-variance decomposition**: for any random variable $A$ and any fixed $b$,

> [!equation] 6.3
> $$\mathbb{E}\!\left[D_\phi(A \,\|\, b)\right] = D_\phi\!\left(\mathbb{E}[A] \,\|\, b\right) + \mathbb{E}\!\left[D_\phi\!\left(A \,\|\, \mathbb{E}[A]\right)\right].$$


The second term on the right depends on $A$ but not on $b$. This identity is the Bregman generalization of the classical decomposition $\mathbb{E}[(A - b)^2] = (\mathbb{E}[A] - b)^2 + \mathrm{Var}(A)$, and it is the engine of the conditional generator matching theorem.

Two canonical choices cover the main cases encountered in generative modeling:

**Squared norm** ($\phi(v) = \tfrac{1}{2}\|v\|^2$ on $\mathbb{R}^d$): $D_\phi(a\|b) = \tfrac{1}{2}\|a-b\|^2$. Symmetric, geometrically natural, the basis of least-squares regression. The right choice when the generator output is a vector in a real space — a velocity field, a score, a tangent vector.

**KL divergence** ($\phi(p) = \sum_y p(y)\log p(y)$ on non-negative sequences): $D_\phi(a\|b) = \sum_y a(y)\log(a(y)/b(y)) - a(y) + b(y)$, which reduces to the standard KL divergence when $a$ and $b$ are probability distributions. The right choice when the generator output is a distribution over next states — as it is for CTMC rate matrices, where each row $R_t(x, \cdot)$ specifies a signed measure over $\mathcal{X}$.

## 6.5 The Generator Matching Objective

Given an interpolant process $\{p_t\}_{t \in [0,1]}$ with known generator $\{\mathcal{L}_t\}$ and a parametric family of generators $\{\mathcal{L}_t^\theta\}$ of the same structural type, the **generator matching objective** is:

> [!equation] 6.4
> $$\mathcal{L}_\mathrm{GM}(\theta) = \int_0^1 \mathbb{E}_{x_t \sim p_t}\!\left[D_\phi\!\left(\ell_t(x_t) \;\Big\|\; \ell_t^\theta(x_t)\right)\right] dt,$$


where $\ell_t(x)$ is the **local generator statistic** at state $x$ — the concrete object that the generator produces at each point:
- For flows: $\ell_t(x) = u_t(x) \in \mathbb{R}^d$ (the velocity vector).
- For CTMCs: $\ell_t(x) = R_t(x, \cdot) \in \mathbb{R}^{|\mathcal{X}|}$ (the row of outgoing rates).
- For Riemannian flows: $\ell_t(x) = v_t(x) \in T_x\mathcal{M}$ (the tangent-space velocity).

The minimizer of (6.4) is the parameter $\theta^*$ such that $\mathcal{L}_t^{\theta^*}$ matches the true generator $\mathcal{L}_t$ on the support of $p_t$, at every time $t$.

> [!remark] 6.2
> The objective (6.4) is intractable as stated, because computing $\ell_t(x_t)$ requires access to the marginal generator — which in turn requires integrating over all data that could produce the observed $x_t$. This is exactly the same intractability that made the original flow matching objective (3.2) intractable: the marginal velocity field $u_t(x)$ is the average over all data points $x_1$ weighted by $p_1(x_1|x_t)$. The resolution is the conditional generator matching theorem.


## 6.6 The Conditional Generator Matching Theorem

The key observation is that the marginal generator statistic is a conditional expectation of the conditional generator statistics. If $p_t(x) = \int p_t(x|x_1)\,p_1(x_1)\,dx_1$ is a mixture of conditional paths, then by the linearity of generators and the tower property:

$$\ell_t(x) = \mathbb{E}_{p_1(x_1 \mid x_t = x)}\!\left[\ell_t(x \mid x_1)\right],$$

where $\ell_t(x|x_1)$ is the conditional generator statistic of the path $p_t(\cdot|x_1)$. For the linear interpolant, this is $u_t(x_t|x_1) = (x_1-x_t)/(1-t)$; for the masking interpolant, it is the rate row from (7.3). In every case, the conditional target is computable from first principles, without marginalizing over data.

Define the **conditional generator matching objective**:

> [!equation] 6.5
> $$\mathcal{L}_\mathrm{CGM}(\theta) = \int_0^1 \mathbb{E}_{t,\; x_1 \sim p_1,\; x_t \sim p_t(\cdot | x_1)}\!\left[D_\phi\!\left(\ell_t(x_t \mid x_1) \;\Big\|\; \ell_t^\theta(x_t)\right)\right] dt.$$


> [!theorem] 6.1 — Conditional Generator Matching
> Let $D_\phi$ be any Bregman divergence, $\mathcal{X}$ any state space, $\{\mathcal{L}_t^\theta\}$ any parametric family of Markov generators, and $\{p_t(\cdot|x_1)\}$ any family of conditional paths whose mixture recovers the marginal $p_t$. Then:
>
> $$\nabla_\theta\,\mathcal{L}_\mathrm{GM}(\theta) = \nabla_\theta\,\mathcal{L}_\mathrm{CGM}(\theta).$$
>
> The marginal and conditional objectives have identical gradients with respect to $\theta$ at every parameter value.


The proof is a direct application of the generalized bias-variance decomposition (6.3). Fix $\theta$, $t$, and $x_t$. By (6.3) applied to the random variable $A = \ell_t(x_t|x_1)$ with expectation $\ell_t(x_t) = \mathbb{E}_{x_1|x_t}[\ell_t(x_t|x_1)]$:

$$\mathbb{E}_{x_1 | x_t}\!\left[D_\phi\!\left(\ell_t(x_t|x_1) \;\Big\|\; \ell_t^\theta(x_t)\right)\right] = D_\phi\!\left(\ell_t(x_t) \;\Big\|\; \ell_t^\theta(x_t)\right) + \underbrace{\mathbb{E}_{x_1|x_t}\!\left[D_\phi\!\left(\ell_t(x_t|x_1) \;\Big\|\; \ell_t(x_t)\right)\right]}_{\text{does not depend on } \theta}.$$

Integrating over $t$ and $x_t$ and differentiating in $\theta$ annihilates the second term, leaving $\nabla_\theta\,\mathcal{L}_\mathrm{GM} = \nabla_\theta\,\mathcal{L}_\mathrm{CGM}$.

This proof is four lines. It holds for any Bregman divergence, any state space, any generator class, any interpolant family. Every conditional-equals-marginal equivalence from every preceding chapter is an instance of it. The CFM = FM proof of Chapter 3 was the special case of squared-norm Bregman divergence applied to ODE generators on $\mathbb{R}^d$. The discrete case from Section 7.3 is the same proof applied to CTMC generators on a finite alphabet. What changes is the state space and the divergence. What does not change is the theorem.

## 6.7 Flow Matching as Generator Matching

Recovering conditional flow matching is the most immediate instantiation. Choose:
- **State space**: $\mathcal{X} = \mathbb{R}^d$.
- **Generator class**: deterministic flows, $(\mathcal{L}_t^\theta f)(x) = u_t^\theta(x) \cdot \nabla f(x)$.
- **Bregman divergence**: squared norm, $D_\phi(a\|b) = \tfrac{1}{2}\|a-b\|^2$.
- **Interpolant**: linear, with conditional velocity $u_t(x_t|x_1) = (x_1 - x_t)/(1-t)$.

The local generator statistic is $\ell_t(x) = u_t(x)$, the velocity vector. Substituting into the conditional objective (6.5):

$$\mathcal{L}_\mathrm{CGM}(\theta) = \int_0^1 \mathbb{E}_{t,\, x_1,\, x_t|x_1}\!\left[\tfrac{1}{2}\left\|u_t^\theta(x_t) - u_t(x_t|x_1)\right\|^2\right] dt = \mathcal{L}_\mathrm{CFM}(\theta).$$

The conditional flow matching objective of Chapter 3 is exactly generator matching with an ODE generator and squared-norm Bregman divergence. Similarly, **Riemannian flow matching** is generator matching on $(\mathcal{M}, g)$ with a Riemannian-flow generator and the Riemannian squared norm $\|\cdot\|_g^2$ as the Bregman divergence — the geodesic $\log_{x_t}(x_1)/(1-t)$ is the conditional generator statistic of the geodesic interpolant (7.5).

## 6.8 Diffusion as Generator Matching

The diffusion case is instructive because it highlights the role of the Bregman divergence. A diffusion model parametrizes the **score function** $s_t^\theta(x) \approx \nabla \log p_t(x)$ — the gradient of the log-density — rather than a drift directly. The connection arises from decomposing the SDE generator:

$$(\mathcal{L}_t f)(x) = \underbrace{\left(b_t(x) + \tfrac{g_t^2}{2}\,\nabla \log p_t(x)\right)}_{\text{probability flow ODE drift}} \!\cdot\nabla f(x) \;+\; \tfrac{g_t^2}{2}\underbrace{\bigl(\Delta f - \nabla\log p_t \cdot \nabla f\bigr)}_{\text{fluctuation term}}.$$

Restricting to models that share the fixed diffusion coefficient $g_t$ and vary only in their drift, and identifying the drift with the score via $b_t = f_t - g_t^2 s_t^\theta$, the generator matching objective reduces to:

$$\mathcal{L}_\mathrm{GM}(\theta) \;\propto\; \int_0^1 g_t^2\;\mathbb{E}\!\left[\left\|s_t^\theta(x_t) - \nabla\log p_t(x_t)\right\|^2\right] dt.$$

This is **likelihood-weighted denoising score matching** (Kingma et al., 2021). With Gaussian conditional paths and Tweedie's formula, the score $\nabla\log p_t(x_t|x_1)$ is the Gaussian score $-(x_t - \mu_t x_1)/\sigma_t^2$, and the objective becomes the DDPM noise-prediction loss of Chapter 2. Generator matching with a diffusive generator and squared-norm Bregman divergence recovers the full family of score-based diffusion objectives.

> [!remark] 6.3
> The diffusion case shows that the choice of Bregman divergence is not merely aesthetic. The score function $\nabla\log p_t$ is a logarithmic derivative — it measures *relative* changes in density, not absolute ones. Using the squared Euclidean norm treats it as a plain vector, ignoring its probabilistic origin. A Bregman divergence derived from an entropic potential $\phi$ would be sensitive to ratios rather than differences, yielding a different loss. The freedom to choose $D_\phi$ within the generator matching framework makes this choice explicit and principled, rather than buried in a reparametrization.


## 6.9 Discrete Flow Matching as Generator Matching

The discrete case makes the role of the Bregman divergence most transparent. Choose:
- **State space**: $\mathcal{X}$ finite (a vocabulary, alphabet, or product thereof).
- **Generator class**: CTMCs, $(\mathcal{L}_t^\theta f)(x) = \sum_y R_t^\theta(x,y)\,[f(y)-f(x)]$.
- **Local generator statistic**: $\ell_t(x) = R_t(x, \cdot)$, the row of outgoing rates.
- **Bregman divergence**: KL divergence, $D_{\text{KL}}(a\|b) = \sum_y a(y)\log(a(y)/b(y))$.

With the masking interpolant, the conditional generator statistic at state $\text{[M]}$ given target $x_1$ is the point-mass row $R_t(\text{[M]},\cdot|x_1) = \frac{1}{1-t}\delta_{\cdot,\, x_1}$ from equation (7.3). The KL divergence between this point mass and the model's predicted rate row is:

$$D_{\text{KL}}\!\left(R_t(\text{[M]}, \cdot \mid x_1) \;\Big\|\; R_t^\theta(\text{[M]}, \cdot)\right) = -\log R_t^\theta(\text{[M]}, x_1) + \text{const},$$

which is the **cross-entropy** of the model's next-token distribution against the true token $x_1$. Training with the KL Bregman divergence turns generator matching into maximum-likelihood estimation over categorical predictions — masked language model training, with time-dependent weighting by $1/(1-t)$.

The KL divergence is the natural choice for discrete generator outputs for a deeper reason: the rate matrix row $R_t(x, \cdot)$ is a measure over next states, and the appropriate divergence between measures is information-theoretic. Using the squared $\ell^2$ norm on rate rows — as the discrete CFM objective of Section 7.3 does — is technically valid (it corresponds to a different Bregman divergence) but ignores the probabilistic structure of the output. The KL version is the one that connects naturally to the standard cross-entropy training of language models.

## 6.10 The Design Space: A Unified Recipe

The generator matching framework reduces the problem of designing a new generative model to four choices. Every model encountered in this monograph is a point in this space; every future model is a new combination of the same four ingredients.

**Choice 1: State space $\mathcal{X}$.** Euclidean $\mathbb{R}^d$, a Riemannian manifold $(\mathcal{M},g)$, a finite alphabet $\mathcal{A}^L$, a Lie group, a product space — the state space determines what kind of data the model generates and constrains the generator class.

**Choice 2: Interpolant $\{p_t(\cdot|x_1)\}$.** The conditional path from reference to data. Linear for Euclidean, geodesic for Riemannian, masking or uniform noise for discrete, Gaussian forward process for diffusion. The interpolant determines the conditional target $\ell_t(x_t|x_1)$ and controls the geometry of the training signal.

**Choice 3: Generator class $\{\mathcal{L}_t^\theta\}$.** The parametric family of Markov processes the model can express. Deterministic flows, stochastic diffusions, CTMCs, or Riemannian flows — and mixtures thereof. This is the parametrization of the neural network: what the model outputs at each state.

**Choice 4: Bregman divergence $D_\phi$.** Squared norm for real-vector outputs (velocities, scores, tangent vectors); KL divergence for distributional outputs (rate rows, token predictors); other Bregman divergences for specialized output geometries.

Given these four choices, training proceeds identically across all settings: sample $t$, draw $x_1 \sim p_\mathrm{data}$, sample $x_t \sim p_t(\cdot|x_1)$, evaluate the conditional generator target $\ell_t(x_t|x_1)$, compute $D_\phi(\ell_t(x_t|x_1)\|\ell_t^\theta(x_t))$, backpropagate. No simulation. One forward pass per gradient step. Theorem 6.1 guarantees that this procedure is equivalent — in gradient — to optimizing the intractable marginal objective.

> [!remark] 6.4
> The four choices are not independent. The state space constrains the generator class (CTMCs are not defined on $\mathbb{R}^d$; ODE generators are not defined on finite alphabets). The interpolant should respect the geometry of the state space (masking for discrete, geodesic for Riemannian). The Bregman divergence should match the type of generator output (squared norm for vectors, KL for distributions). Within these structural constraints, there is genuine freedom — and exploring the design space has been the source of most practical advances since 2023. The generator matching framework makes these choices explicit rather than implicit, which is precisely its value as a research tool.


## 6.11 Historical Notes and the Road Ahead

The generator matching framework as presented here is due to **Holderrieth, Yim, and Jaakkola**, whose 2024/2025 paper unified the continuous, discrete, and Riemannian streams of the literature under a single theorem. The key technical insight — that the Bregman bias-variance decomposition (6.3) is the underlying algebraic fact, and that all the conditional-equals-marginal equivalences are instances of it — was anticipated in scattered form. The connection between flow matching and CTMC-based discrete generation had been noted independently by **Campbell, Yim, Barzilay, Jaakkola, and Gat (NeurIPS 2024)** and **Gat et al. (NeurIPS 2024)**. The tower-property proof in the continuous case goes back to **Lipman et al. (ICLR 2023)** and **Liu et al. (2022)**. The Riemannian extension had been established by **Chen and Lipman (ICLR 2024)**. What the generator matching paper contributes is the identification of the common algebraic structure — Bregman decomposition applied to Markov generators — and the proof of Theorem 6.1 in full generality.

The framework has been extended in several directions since. The **stochastic interpolant** perspective of Albergo, Boffi, and Vanden-Eijnden (2023) fits seamlessly: adding noise to the interpolant corresponds to switching from a deterministic ODE generator to a diffusive one, with the Bregman objective unchanged. **Riemannian diffusions** — SDEs on manifolds — arise when the CTMC case is replaced by a manifold-adapted diffusion generator; the score function becomes the Riemannian gradient of the log-density, and Riemannian denoising score matching follows automatically from Theorem 6.1. The unification is not just formal: knowing that all these methods are generator matching has guided the design of new ones, by making clear which ingredients can be varied independently.

Chapter 7 develops the two non-Euclidean instantiations in full: discrete flow matching on finite alphabets and Riemannian flow matching on curved manifolds. Both are special cases of Theorem 6.1, and both have the same four-choice structure — what changes is the state space, the interpolant, the generator class, and the Bregman divergence.

## 6.12 Historical Notes on the Bregman Connection

A remark on the mathematics is warranted, because the appearance of Bregman divergences in this context is not accidental. **Amari** (1985, 2016) established that the natural divergences on statistical manifolds are the $f$-divergences and Bregman divergences arising from the geometry of exponential families — and that denoising, regression, and maximum-likelihood estimation are all instances of Bregman projection. **Banerjee, Merugu, Dhillon, and Ghosh (JMLR 2005)** showed that every Bregman divergence corresponds to an exponential family and vice versa, and that the "generalized mean" defined by minimizing the expected Bregman divergence is simply the expectation. This is exactly the fact exploited in the proof of Theorem 6.1: $\argmin_b\,\mathbb{E}[D_\phi(A\|b)] = \mathbb{E}[A]$ for any Bregman divergence and any random variable $A$.

The generator matching framework, seen through this lens, is the answer to a natural question: what is the correct divergence between Markov generators? The answer is: any Bregman divergence, applied to the local generator statistics. The choice of divergence determines the implicit statistical model — squared norm corresponds to Gaussian regression, KL corresponds to categorical maximum-likelihood — and the choice of generator class determines the process being modeled. The two choices together determine the method. The theorem holds for all of them.

## References

- [Holderrieth, Yim, and Jaakkola](https://arxiv.org/abs/2409.11399) — Generator Matching: Generative Modeling with Arbitrary Markov Processes
- [Lipman et al. (ICLR 2023)](https://arxiv.org/abs/2210.02747) — Flow Matching for Generative Modeling
- [Liu et al. (2022)](https://arxiv.org/abs/2209.03003) — Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow
- [Chen and Lipman (ICLR 2024)](https://arxiv.org/abs/2302.03660) — Flow Matching on General Geometries
- [Campbell, Yim, Barzilay, Jaakkola, and Gat (NeurIPS 2024)](https://arxiv.org/abs/2406.04329) — Generative Flows on Discrete State-Spaces
