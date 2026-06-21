---

title: "Flows and ODEs"
subtitle: "Normalizing flows, Neural ODEs, continuous normalizing flows, and the simulation bottleneck — 2014 to 2019"
---

Every generative model, at its core, answers the same question: given access to samples from an unknown distribution $p_{\text{data}}$ over some space $\mathcal{X}$, can we construct a computational procedure that produces new samples indistinguishable from the real ones? By 2014, the field had two serious but unsatisfying answers. GANs — introduced by Goodfellow et al. that year — produced visually striking images but were notoriously unstable to train, and offered no way to evaluate the likelihood of a given sample. VAEs (Kingma & Welling, 2013) were stable and principled, but their samples passed through a bottleneck that blurred fine detail. Both approaches were fundamentally *implicit*: they described a procedure for generating samples without ever writing down an explicit formula for the probability of a given point.

The question that launched normalizing flows was different and more direct: can we write down an *explicit*, *tractable* probability density for our generative model, while simultaneously being expressive enough to model the complexity of real data? The answer, developed through a series of elegant papers from 2014 to 2018, was yes — with a caveat. The solution required a clever architectural constraint that made the math work, but at a price. Understanding both the elegance and the price is essential context for everything that follows.

## 1.1 The Transport Perspective

Let $p_0$ be a simple reference distribution — the standard Gaussian $\mathcal{N}(0, I_d)$ is the canonical choice. If we could find an invertible, differentiable map $T : \mathbb{R}^d \to \mathbb{R}^d$ such that $T(z) \sim p_{\text{data}}$ whenever $z \sim p_0$, we would have a perfect generative model: to sample, draw $z \sim p_0$ and compute $T(z)$. The image of $p_0$ under $T$ is the *pushforward distribution*, denoted $T_\# p_0$, and the condition we want is:

$$T_\# p_0 = p_{\text{data}}.$$

This is the **transport problem**. Its abstract form — finding a map that pushes one measure onto another — was studied extensively in 19th-century mathematics under the name of the Monge problem (1781), and in the 20th century as optimal transport (Kantorovich, 1942). But these classical formulations sought optimal maps subject to cost criteria. For machine learning, the goal is different: find a *learnable* map $T_\theta$ that can be fit to data via gradient descent.

The key computational requirement is the ability to evaluate the likelihood of any data point $x \sim p_{\text{data}}$ under the model. If $T_\theta$ is invertible with inverse $f_\theta = T_\theta^{-1}$, the change-of-variables formula gives:

> [!equation] 1.1
> $$\log p_\theta(x) = \log p_0(f_\theta(x)) + \log \left|\det \nabla f_\theta(x)\right|,$$


where $\nabla f_\theta(x) \in \mathbb{R}^{d \times d}$ is the Jacobian matrix of $f_\theta$ at $x$. This formula is exact and requires no approximation — it is a simple consequence of the substitution rule for integrals.

> [!definition] 1.1 — Normalizing Flow
> A **normalizing flow** is a generative model defined by an invertible, differentiable map $T_\theta : \mathbb{R}^d \to \mathbb{R}^d$ (or equivalently its inverse $f_\theta = T_\theta^{-1}$) with tractable Jacobian determinant. The model density is given by the change-of-variables formula (1.1). The name refers to the process of "normalizing" a complex distribution to a standard Gaussian through the inverse map.


The phrase *tractable Jacobian determinant* carries all the engineering weight. For a general $d$-dimensional vector-valued function, computing $\det \nabla f$ requires $O(d^3)$ operations via LU decomposition — completely infeasible for images where $d \sim 10^4$–$10^6$. The entire architecture design of normalizing flows from 2014 to 2018 is the story of finding neural network structures where this determinant is cheap to compute, without sacrificing too much representational power.

## 1.2 NICE: The First Tractable Architecture (2014)

Dinh, Krueger, and Bengio introduced **NICE** (Non-linear Independent Components Estimation) in 2014, published in 2015 as a workshop paper, with the key insight that a *triangular* Jacobian has a determinant equal to the product of its diagonal entries — computable in $O(d)$ rather than $O(d^3)$.

**Additive coupling layers.** Partition the $d$-dimensional input $x$ into two halves: $x = (x_{1:k}, x_{k+1:d})$. Define the transformation:

> [!equation] 1.2
> $$y_{1:k} = x_{1:k}, \qquad y_{k+1:d} = x_{k+1:d} + m_\theta(x_{1:k}),$$


where $m_\theta : \mathbb{R}^k \to \mathbb{R}^{d-k}$ is an arbitrary neural network (the **coupling function**). The Jacobian of this transformation is:

$$\nabla_x y = \begin{pmatrix} I_k & 0 \\ \frac{\partial m_\theta}{\partial x_{1:k}} & I_{d-k} \end{pmatrix},$$

a lower triangular matrix with ones on the diagonal. The determinant is exactly 1, regardless of the complexity of $m_\theta$. Crucially, $m_\theta$ can be a deep neural network with any architecture — the tractability of the Jacobian is guaranteed by the *structure* of the coupling, not the simplicity of $m_\theta$.

The inverse is trivially:

$$x_{1:k} = y_{1:k}, \qquad x_{k+1:d} = y_{k+1:d} - m_\theta(y_{1:k}).$$

Because the first half $y_{1:k} = x_{1:k}$ is copied unchanged, the inverse uses the same forward pass of $m_\theta$ — no additional computation is needed.

**Composing layers.** A single coupling layer is clearly insufficient: the first $k$ dimensions are unchanged. NICE addresses this by alternating which half is "active": layer 1 updates the second half given the first; layer 2 updates the first half given the second. After stacking $L$ such layers (with alternating masks or random permutations of dimensions), the composition $f_\theta = f_L \circ \cdots \circ f_1$ can in principle move every dimension. The log-determinant of the composition is the sum of the log-determinants, which is zero for additive coupling layers. This means the model is volume-preserving: it cannot expand or contract probability mass, only rearrange it.

> [!remark] 1.1
> Volume preservation is both a feature and a limitation. NICE trains purely as a latent-variable model: the reference $p_0$ is a product of independent logistic distributions, and the coupling layers rearrange this into the data distribution via composition. The model can in principle represent any continuous distribution — by the universal approximation theorem for flows, deep enough compositions of coupling layers are dense in the space of diffeomorphisms. But in practice, the volume constraint means the model cannot easily "flatten" a multimodal data distribution into a unimodal Gaussian without very deep stacks of layers.


## 1.3 RealNVP: Affine Coupling and the Scaling Stack (2017)

The limitation of additive coupling — unit Jacobian, no volume change — was removed by Dinh, Sohl-Dickstein, and Bengio in **RealNVP** (Real-valued Non-Volume Preserving, 2016/2017). The modification is minimal but consequential: replace the additive update with an *affine* one.

> [!definition] 1.2 — Affine Coupling Layer
> Given input $x = (x_{1:k}, x_{k+1:d})$, an **affine coupling layer** computes:
>
> $$y_{1:k} = x_{1:k}, \qquad y_{k+1:d} = x_{k+1:d} \odot \exp(s_\theta(x_{1:k})) + t_\theta(x_{1:k}),$$
>
> where $s_\theta, t_\theta : \mathbb{R}^k \to \mathbb{R}^{d-k}$ are neural networks (the *scale* and *translation* networks). The Jacobian determinant is $\exp(\sum_i s_\theta(x_{1:k})_i)$, computable in $O(d)$.


The inverse is:

$$x_{1:k} = y_{1:k}, \qquad x_{k+1:d} = (y_{k+1:d} - t_\theta(y_{1:k})) \odot \exp(-s_\theta(y_{1:k})).$$

The log-determinant of a composition of $L$ affine coupling layers is $\sum_{\ell=1}^L \sum_i s_\theta^{(\ell)}(\cdot)_i$ — a sum of $L \times (d-k)$ scalar outputs. Training by maximum likelihood requires computing this sum and backpropagating through it, which is perfectly tractable.

**Multi-scale architecture.** RealNVP introduced a crucial practical innovation: the **multi-scale architecture**. After several coupling layers, half the dimensions are "factored out" — decoded directly to the reference distribution $p_0$ — while the other half continue through additional coupling layers. This creates a coarse-to-fine structure: early layers capture global structure (color distributions, rough shapes), later layers capture fine details (textures, edges). The multi-scale architecture dramatically improves both expressivity and training stability, and became the template for subsequent flow architectures.

**Batch normalization and permutations.** RealNVP also introduced *batch normalization* layers between coupling steps (later replaced by the more principled *actnorm* in Glow) and showed that simple alternating masks — splitting dimensions as odd/even indices rather than top/bottom halves — are sufficient for mixing.

The results were striking for 2017: on CIFAR-10 and ImageNet $64\times 64$, RealNVP produced visually coherent samples and achieved competitive bits-per-dimension scores, while also supporting exact latent-space interpolation (a property unavailable to GANs) and exact log-likelihood evaluation (unavailable to VAEs).

## 1.4 Glow: Invertible 1×1 Convolutions (2018)

Kingma and Dhariwal's **Glow** (2018) resolved the remaining awkwardness in RealNVP: the alternating-mask permutation that interleaved which dimensions were active was hand-designed and potentially limiting. Glow replaced it with a **learned invertible 1×1 convolution** — a learnable permutation of channels at each layer.

For a $d$-channel feature map, an invertible 1×1 convolution is a matrix $W \in \mathbb{R}^{d \times d}$. The log-determinant is $\log |\det W|$. With an LU decomposition $W = PLU$ (permutation × lower triangular × upper triangular), the determinant is the product of the diagonal of $U$, computable in $O(d)$. This decomposition is computed once at the start of training and updated via gradient descent.

Glow also introduced **actnorm** — a data-dependent normalization layer whose scale and bias parameters are initialized to make the output of the first mini-batch have zero mean and unit variance. This stabilized training of very deep flow networks and allowed Glow to be scaled to high-resolution images ($256 \times 256$).

> [!proposition] 1.1 — Glow Likelihood
> A Glow model with $L$ steps per level and $K$ levels computes the log-likelihood:
>
> $$\log p_\theta(x) = \log p_0(z) + \sum_{\ell=1}^{L \cdot K} \left[\sum_i (s^{(\ell)})_i + \log |\det W^{(\ell)}| + \log |\det \mathrm{diag}(a^{(\ell)})|\right],$$
>
> where $z = f_\theta(x)$ is the latent code, $(s^{(\ell)})_i$ are the scale outputs of the affine coupling layers, $W^{(\ell)}$ are the 1×1 convolution matrices, and $a^{(\ell)}$ are the actnorm scale parameters. Each term is computed in a single forward pass — no ODE solving, no stochastic estimation.


Glow's key practical achievement was **high-quality face synthesis**: for the first time, a likelihood-based model without adversarial training produced photorealistic $256 \times 256$ faces. The latent space had a remarkable semantic structure — interpolating between two face latent codes produced a smooth perceptual morph, and individual latent directions corresponded to interpretable attributes like lighting and expression. This was a compelling demonstration that exact likelihood models could match GANs in output quality while offering properties GANs fundamentally lack.

## 1.5 Autoregressive Flows

Concurrently with the coupling-layer line of research, a parallel approach developed under the name **autoregressive flows**. The central observation is that an autoregressive model — one that factorizes $p_\theta(x) = \prod_{i=1}^d p_\theta(x_i | x_{1:i-1})$ — can be viewed as a normalizing flow.

Specifically, if each conditional $p_\theta(x_i | x_{1:i-1}) = \mathcal{N}(x_i;\, \mu_i(x_{1:i-1}),\, \sigma_i^2(x_{1:i-1}))$, then the transformation:

$$z_i = \frac{x_i - \mu_i(x_{1:i-1})}{\sigma_i(x_{1:i-1})}$$

maps $(x_1, \ldots, x_d) \mapsto (z_1, \ldots, z_d)$ with $z_i \sim \mathcal{N}(0,1)$ marginally. The Jacobian of this map is lower triangular (since $z_i$ depends only on $x_{1:i}$), with diagonal entries $1/\sigma_i$. The log-determinant is $-\sum_i \log \sigma_i(x_{1:i-1})$ — exactly the sum of log-standard-deviations in the autoregressive model.

This observation, formalized by Papamakarios, Murray, and Rainforth in **MAF** (Masked Autoregressive Flow, 2017) and Kingma et al. in **IAF** (Inverse Autoregressive Flow, 2016), revealed a deep connection: autoregressive models are normalizing flows in disguise, and normalizing flows are autoregressive models with a special parallel structure.

> [!remark] 1.2
> There is a fundamental asymmetry in autoregressive flows. In **MAF**, density evaluation ($\log p_\theta(x)$) is parallelizable — all $\mu_i, \sigma_i$ can be computed simultaneously using a masked neural network (MADE) — but *sampling* is sequential: $x_i$ depends on $x_{1:i-1}$, so each token must be generated in order ($O(d)$ sequential steps). In **IAF**, the roles are reversed: sampling is parallel but density evaluation is sequential. This trade-off — which one pays for the choice of masking direction — is a fundamental property of autoregressive flows that has no clean resolution in the coupling-layer framework.


## 1.6 The Structural Bottleneck

By 2018, normalizing flows had achieved something genuinely impressive: for the first time, likelihood-based generative models could produce high-quality images and high-fidelity density estimates. But a careful examination of the architecture reveals a fundamental limitation that would motivate the next era of research.

Every normalizing flow evaluated so far — NICE, RealNVP, Glow, MAF, IAF — achieves tractable Jacobian determinants through the *same mechanism*: the Jacobian is structured (triangular, block-diagonal, sparse) because the transformation updates only *some* coordinates as a function of *others*, leaving the rest unchanged. This is not a coincidence. It is a theorem:

> [!proposition] 1.2 — The Coupling Constraint
> Let $f : \mathbb{R}^d \to \mathbb{R}^d$ be a differentiable map. If $\log |\det \nabla f(x)|$ is computable in $O(d)$ for all $x$ via a single evaluation of $f$, then $f$ must have a triangular or block-triangular Jacobian — equivalently, a coordinate $y_i = f_i(x)$ can only depend on a subset of inputs.


The consequence is that every $O(d)$-determinant flow must leave an *ordering* on coordinates: some coordinates are updated before others can be used. This imposes an inductive bias on the model that is entirely extrinsic to the data — the architectural constraint, not the data structure, determines which coordinates influence which.

For images, this is tolerable: pixels in one corner of an image do genuinely influence pixels in other corners, and the ordering can be chosen to respect spatial locality. But for more structured data — molecules, protein backbones, graphs — the coordinate ordering has no natural correspondence to the data's underlying dependencies, and one pays a price in terms of the expressivity needed to compensate.

There is a second, subtler limitation. Because the log-determinant is computed as a sum over coordinates (e.g., $\sum_i s_i$ for affine coupling), it has a *linear* functional form. The only information about the transformation that enters the log-likelihood is this sum — all other geometric information about the Jacobian is discarded. This means the model cannot "see" whether the transformation is locally volume-expanding in one direction while volume-contracting in another; only the total volume change is tracked. For distributions with complex local geometry, this is a real constraint.

## 1.7 What Normalizing Flows Got Right

Before the limitations drove researchers toward continuous-time formulations, it is worth pausing to appreciate what the normalizing flow framework contributed that has endured.

**The transport framing as the organizing principle.** The statement $T_\# p_0 = p_{\text{data}}$ is not just one way to think about generative modeling — it is, as the rest of this monograph argues, *the* right way. It makes the generative process geometrically transparent, separates the question of model architecture from the question of training objective, and naturally accommodates the probability manipulations (likelihood evaluation, posterior sampling, Bayesian inference) that scientists actually need. Every method discussed in subsequent chapters can be understood as a different way to learn the transport map $T$.

**Exact likelihoods without simulation.** The change-of-variables formula (1.1) gives exact log-likelihoods in a single forward pass. This is something that VAEs (approximate ELBO), GANs (no likelihood at all), and — as we will see in Chapter 2 — diffusion models (intractable marginal likelihood) cannot offer. For scientific applications where likelihoods are needed for Bayesian inference or hypothesis testing, this matters.

**Latent space structure.** Because the latent code $z = f_\theta(x)$ is a deterministic function of $x$, the latent space of a normalizing flow is the *same space as the data space*. Interpolation in latent space has a clean geometric interpretation: the straight line $tz_1 + (1-t)z_2$ maps, under $T_\theta$, to a geodesic (under the pullback metric) in data space. This is qualitatively different from VAE latent spaces, where the posterior is stochastic and interpolation is less principled.

**The seeds of what comes next.** In retrospect, the coupling layer construction already contains the germ of flow matching. The conditional update $y_{k+1:d} = x_{k+1:d} + m_\theta(x_{1:k})$ is a discrete-time flow of the second half of the coordinates, driven by the first half. The move to continuous time — letting the step size go to zero — is exactly the step taken in the following sections.

## 1.8 The State of Affairs in 2018

By the end of 2018, the landscape was this. Normalizing flows were theoretically principled, practically working, and actively being improved. Glow had demonstrated that high-resolution image synthesis was possible with exact likelihoods. RealNVP had shown the multi-scale architecture pattern that would be adopted by subsequent work. Autoregressive flows had revealed the deep connection between density estimation and invertible transformations.

But the field was also beginning to feel the ceiling. Training deep flow stacks was computationally expensive. The best bits-per-dimension results on image datasets still lagged behind autoregressive models like PixelCNN, which could afford much larger architectures because they did not need to compute a Jacobian. And critically, the coupling constraint — the source of tractability — was also the source of a subtle rigidity that limited how well flows could model data with complex, entangled dependencies.

The question forming in several research groups simultaneously was: what if the architectural constraint is unnecessary? What if tractable Jacobians are not the price of exact likelihoods — what if there is another way?

That question had been brewing since 2015 in a completely different corner of the literature. But the most direct path to an answer came not from probability theory but from a deceptively simple observation about deep learning architectures. ResNet layers, written as $x_{t+1} = x_t + f_\theta(x_t)$, look exactly like Euler steps for an ODE. What happens if you take that analogy seriously and let the number of layers go to infinity?

The answer to that question is developed in the following sections.

## 1.9 Historical Notes (Normalizing Flows)

The history of normalizing flows as a machine learning tool properly begins with **Tabak & Turner (2013)** and **Tabak & Vanden-Eijnden (2010)**, who introduced the term "normalizing flow" and developed the mathematical framework for density estimation via composition of simple maps. The machine learning community's engagement began with **Rezende & Mohamed (2015)** (ICML), who introduced normalizing flows in the context of variational inference and named the variational normalizing flow objective. The architecture that made flows practical for generative modeling was **NICE** (Dinh, Krueger, Bengio, 2014/ICLR 2015), followed by **RealNVP** (Dinh, Sohl-Dickstein, Bengio, 2016/ICLR 2017), and culminating in **Glow** (Kingma & Dhariwal, NeurIPS 2018). Autoregressive flows were developed as **IAF** (Kingma et al., NeurIPS 2016) and **MAF** (Papamakarios, Murray, Rainforth, NeurIPS 2017). The connection between normalizing flows and optimal transport — which this monograph will develop extensively — was explored by Villani (2003, 2009) on the mathematical side and by Ruthotto & Haber (2019) and Finlay et al. (2020) on the computational side.

The survey by **Papamakarios, Nalisnick, Rezende, Mohamed & Lakshminarayanan (2021)** in *JMLR* remains the definitive reference for this era.

---

## 1.10 The Discrete Precursor: ResNets as Euler Steps

There is a moment in the history of a field when a change of perspective reveals that two apparently different things are secretly the same. The identification of ResNets with ODE discretizations was such a moment. It did not immediately solve any open problem, but it reframed the architectural design question — *how many layers, how much residual scaling?* — as a question about numerical analysis: *what step size, what integrator?* And that reframing opened a door that led, within a few years, directly to flow matching.

The preceding sections developed normalizing flows and described their limitations: tractable Jacobians require architectural structure, and architectural structure means coordinate ordering, and coordinate ordering means inductive bias not present in the data. The question is whether there is an escape — a way to compute likelihoods exactly without restricting the architecture. The answer, arrived at independently by several groups in 2018, is that you can trade the architectural constraint for a dynamical one: instead of building the transport map $T_\theta$ as a composition of finitely many structured layers, build it as the time-1 map of a *differential equation*. The Jacobian determinant then satisfies its own differential equation, one that requires no special structure to be tractable.

The observation that launched this line of research was made explicit by Weinan E (2017) and by Lu et al. (2017, "Beyond Finite Layer Neural Networks: Bridging Deep Architectures and Numerical Differential Equations"), but it had been implicit in the residual network literature since He et al. (2016). A residual network layer is:

$$x^{(\ell+1)} = x^{(\ell)} + f_\theta(x^{(\ell)}, \ell),$$

where $f_\theta$ is a sub-network parameterized by the layer index $\ell$. Written this way, $x^{(\ell)}$ is a sequence, $x^{(\ell+1)} - x^{(\ell)} = f_\theta(x^{(\ell)}, \ell)$ is the increment, and the update rule is precisely **Euler's method** for the ODE:

$$\frac{dx}{d\ell} = f_\theta(x, \ell),$$

with a step size of 1 and the continuous variable $\ell$ playing the role of time. A ResNet with $L$ layers is the Euler discretization of this ODE on the interval $[0, L]$ with unit step size.

This realization suggests a natural question: what happens in the limit as the step size goes to zero and the number of layers goes to infinity, keeping the total "depth" $L$ fixed? The answer is the **Neural ODE** (Chen, Rubanova, Bettencourt, Duvenaud, NeurIPS 2018) — the most cited paper of NeurIPS 2018 and one of the most influential in the decade.

> [!definition] 1.3 — Neural ODE
> A **Neural ODE** is a model whose hidden state $h(t) \in \mathbb{R}^d$ evolves according to:
>
> $$\frac{dh}{dt} = f_\theta(h(t), t), \qquad t \in [0, T],$$
>
> where $f_\theta : \mathbb{R}^d \times [0,T] \to \mathbb{R}^d$ is a neural network. The output $h(T)$ of the model is the solution at time $T$, computed by a numerical ODE solver. Gradients with respect to $\theta$ are computed via the **adjoint method**, without backpropagating through the solver's internal states.


The conceptual shift is significant. In a standard deep network, the "architecture" — the number of layers, the sequence of transformations — is a fixed design choice made before training. In a Neural ODE, the "architecture" is the continuous-time vector field $f_\theta$, and the number of solver steps taken at inference is a run-time parameter, not a training-time decision. Deep networks discretize dynamics at a fixed resolution; Neural ODEs represent the dynamics directly and discretize adaptively.

## 1.11 The Adjoint Method: Training Without Backpropagating Through the Solver

The immediate practical challenge for Neural ODEs is computing gradients of a loss $\mathcal{L} = \mathcal{L}(h(T))$ with respect to the parameters $\theta$. Naively, one would store all intermediate states of the ODE solver (the forward pass) and backpropagate through them. But this requires $O(N)$ memory for $N$ solver steps and makes the gradient computation dependent on the solver's internal implementation — incompatible with adaptive solvers that vary the step count.

The **adjoint method** resolves both issues. Define the **adjoint state** $a(t) = d\mathcal{L}/dh(t)$. It satisfies a *backward* ODE:

> [!equation] 1.10
> $$\frac{da}{dt} = -a(t)^\top \frac{\partial f_\theta}{\partial h}(h(t), t),$$


initialized at $a(T) = d\mathcal{L}/dh(T)$ and integrated backward in time. The gradient with respect to $\theta$ is then:

$$\frac{d\mathcal{L}}{d\theta} = -\int_0^T a(t)^\top \frac{\partial f_\theta}{\partial \theta}(h(t), t)\, dt.$$

The key insight is that the adjoint ODE can be solved *simultaneously* with the backward reconstruction of $h(t)$, using $O(d)$ memory at any point in time — independent of the number of solver steps. This makes Neural ODEs memory-efficient even for very fine discretizations.

> [!remark] 1.10
> The adjoint method for ODEs has a long history in control theory (Pontryagin's maximum principle, 1956) and numerical optimization (Pironneau, 1974). Chen et al.'s contribution was to recognize its applicability to neural network training and to implement it in a way compatible with automatic differentiation frameworks. The paper also showed that Neural ODEs define reversible dynamics — running the ODE backward recovers the initial state exactly — making them naturally suited for flow-based generative modeling.


## 1.12 Continuous Normalizing Flows

The application of Neural ODEs to generative modeling is immediate. If $f_\theta$ defines the dynamics of a particle $h(t)$ starting from noise $h(0) \sim p_0 = \mathcal{N}(0, I_d)$, then $h(1)$ is a sample from some distribution $p_1 = p_\theta$. We want $p_\theta \approx p_{\text{data}}$.

For this to be useful, we need to evaluate $\log p_\theta(x)$ for any data point $x$. The starting point is the change-of-variables formula — but now the map $T_\theta : h(0) \mapsto h(1)$ is defined by the flow of the ODE, not a composition of discrete layers. The challenge is computing the log-determinant of the Jacobian of this flow map.

The answer is the **instantaneous change-of-variables formula**, due to Liouville (1838) in the deterministic mechanics context and Chen et al. (2018) in the neural network context:

> [!theorem] 1.1 — Instantaneous Change of Variables
> Let $f_\theta$ be a continuously differentiable vector field with flow map $\phi_t$ (so $\phi_t(x)$ is the position at time $t$ of the trajectory starting at $x$). If $p_t$ is the density of the pushforward $(\phi_t)_\# p_0$, then $p_t$ satisfies:
>
> $$\frac{d}{dt} \log p_t(\phi_t(x)) = -\nabla \cdot f_\theta(\phi_t(x), t) = -\mathrm{tr}\!\left(\frac{\partial f_\theta}{\partial h}(\phi_t(x), t)\right).$$
>
> Integrating, the log-density at time 1 is:
>
> $$\log p_1(x_1) = \log p_0(x_0) - \int_0^1 \nabla \cdot f_\theta(x_t, t)\, dt,$$
>
> where $x_0 = \phi_0^{-1}(x_1)$ is the initial condition of the trajectory ending at $x_1$.


This is the continuous-time analogue of the coupling-layer log-determinant. Instead of summing the log-diagonal entries of a triangular Jacobian, we integrate the *trace* of the full Jacobian over time. The crucial difference: the trace involves all $d^2$ partial derivatives, not just $d$ diagonal ones. For a $d$-dimensional vector field, computing the full trace naively costs $O(d^2)$ — still better than $O(d^3)$ for the full determinant, but still quadratic in dimension.

**The resulting model — a neural network trained as the right-hand side of an ODE, with likelihoods computed via the trace integral — is called a Continuous Normalizing Flow (CNF)**. The name honors the fact that this is the continuous-time limit of the normalizing flow architecture from the preceding sections, achieving exact likelihoods with no architectural constraints: $f_\theta$ can be any neural network whatsoever.

## 1.13 FFJORD: Free-Form Jacobian via Hutchinson's Estimator (2018)

The $O(d^2)$ cost of the full trace was reduced to $O(d)$ by Grathwohl, Chen, Bettencourt, Sutskever, and Duvenaud in **FFJORD** (Free-Form Jacobian of Reversible Dynamics, 2018), using a classical trick from numerical linear algebra: **Hutchinson's trace estimator** (Hutchinson, 1989).

For any square matrix $A \in \mathbb{R}^{d \times d}$, if $\epsilon \sim \mathcal{N}(0, I_d)$ or $\epsilon \sim \mathrm{Uniform}(\{-1,+1\}^d)$, then:

$$\mathrm{tr}(A) = \mathbb{E}_\epsilon[\epsilon^\top A \epsilon].$$

This identity holds exactly in expectation. Applied to the Jacobian of $f_\theta$:

$$\nabla \cdot f_\theta(x, t) = \mathrm{tr}\!\left(\frac{\partial f_\theta}{\partial x}\right) = \mathbb{E}_\epsilon\!\left[\epsilon^\top \frac{\partial f_\theta}{\partial x} \epsilon\right].$$

The matrix-vector product $(\partial f_\theta / \partial x)\, \epsilon$ can be computed via a single backward pass (vector-Jacobian product), costing the same as one forward pass of $f_\theta$. The trace estimator therefore requires $O(d)$ operations per time step, with variance that can be reduced by averaging over multiple $\epsilon$ samples.

> [!equation] 1.11
> $$\log p_1(x_1) = \log p_0(x_0) - \int_0^1 \mathbb{E}_\epsilon\!\left[\epsilon^\top \frac{\partial f_\theta}{\partial x}(x_t, t)\, \epsilon\right] dt.$$


In practice, the stochastic trace estimator introduces some variance into the log-likelihood, but this can be controlled by using a fixed $\epsilon$ per data point (the same noise vector throughout the ODE integration for a given $x$). The variance decreases as the vector field becomes "more diagonal" in the basis aligned with $\epsilon$ — which, interestingly, is precisely the kind of decorrelated dynamics that the training procedure tends to encourage.

FFJORD was a significant practical breakthrough. A CNF with a standard transformer or MLP as $f_\theta$, trained via maximum likelihood with the Hutchinson trace estimator, could in principle model *any* continuous distribution — no coupling layers, no coordinate ordering, no autoregressive structure. It achieved state-of-the-art density estimation results on tabular datasets (UCI benchmarks) and produced cleaner samples than normalizing flows on some image datasets.

## 1.14 The Architecture Freedom and Its Costs

The freedom from architectural constraints that CNFs promised was real, and researchers quickly demonstrated capabilities that coupling-layer flows could not match. Without the coupling structure, a CNF could:

- Model distributions with complex topological structure (multi-modal distributions with non-trivially connected support)
- Apply naturally to data lying on lower-dimensional manifolds embedded in $\mathbb{R}^d$
- Use architectures designed for the data domain — U-Nets for images, transformers for sequences — as the vector field, without modification

But this freedom came with a cost that would take several years to fully appreciate: **training a CNF requires simulating the ODE**. And this simulation is expensive in exactly the ways that matter for large-scale training.

Consider what maximum likelihood training of a CNF on a mini-batch of $B$ data points requires:

1. **Forward ODE solve**: for each $x_1$ in the mini-batch, numerically integrate the ODE backward from $x_1$ to find $x_0 = \phi_0(x_1)$. This requires $N_{\text{forward}}$ evaluations of $f_\theta$.
2. **Trace integration**: simultaneously integrate $\int_0^1 \nabla \cdot f_\theta(x_t, t)\, dt$ along each backward trajectory — $N_{\text{forward}}$ more evaluations of the Jacobian-vector product.
3. **Adjoint backward solve**: integrate the adjoint ODE (1.10) to compute gradients — another $N_{\text{backward}}$ evaluations of $f_\theta$ and its Jacobian.

Total cost per gradient step: $O(B \cdot (N_{\text{forward}} + N_{\text{backward}}) \cdot C_f)$, where $C_f$ is the cost of one $f_\theta$ evaluation. Compare this with a normalizing flow, where the gradient step costs $O(B \cdot C_f)$ — just one forward pass per data point.

The number of solver steps $N$ is not a free parameter: it is determined by the stiffness of the ODE, which is in turn determined by the complexity of $f_\theta$. For a highly nonlinear $f_\theta$ — which is exactly what's needed to model complex distributions — $N$ can be in the hundreds or thousands. A training step that is $100\times$ more expensive than the normalizing flow baseline is not unusual.

> [!remark] 1.11
> There is a further subtlety: the adjoint method (Section 1.11) avoids storing intermediate states but requires *re-running the forward ODE* during the backward pass, effectively doubling the forward computation. For some architectures, storing the forward trajectory (checkpointing) is cheaper, accepting $O(N)$ memory for $O(1)$ forward computation — the standard autograd approach. Modern CNF training typically uses the adjoint method for memory efficiency and accepts the $2\times$ computation overhead.


## 1.15 The Simulation Bottleneck

The combination of ODE simulation during training and exact likelihood computation during evaluation defined the state-of-the-art CNF of 2018–2020. These models were mathematically beautiful and practically useful for small-scale problems, but the simulation requirement created a hard wall against scaling.

To understand why, consider the training pipeline for an image CNF at the scale of ImageNet ($d = 3 \times 256 \times 256 \approx 200{,}000$ dimensions):

- Each training step requires solving an ODE in $\mathbb{R}^{200{,}000}$, with $O(100)$ function evaluations.
- Each function evaluation of $f_\theta$ costs as much as a forward pass of a large neural network.
- The Hutchinson trace estimator adds another backward pass per step.
- For a mini-batch of 256 images, the total cost is roughly $256 \times 100 \times 3 \approx 76{,}800$ neural network forward passes per gradient update.

Compare this with a diffusion model of the same scale: each training step requires exactly *one* forward pass of the neural network (predicting the noise), regardless of the sequence length or data dimension. The asymmetry is not a constant factor — it grows with the complexity of the distribution, because more complex distributions require more solver steps to integrate accurately.

This is the **simulation bottleneck**: the exact likelihoods that CNFs provide come at the cost of ODE simulation during training, and this cost scales adversarially with the complexity of the target distribution. The very property that makes CNFs difficult to train — complex, rapidly-changing vector fields — is also the property that makes them expressive.

> [!definition] 1.4 — Number of Function Evaluations (NFE)
> The **number of function evaluations** (NFE) is the total number of times the vector field $f_\theta$ is evaluated by the ODE solver during a single forward pass (generation) or backward pass (training). NFE is the primary computational cost metric for continuous normalizing flows. At inference, a well-trained flow typically requires $10$–$100$ NFE; during training with the adjoint method, the cost is $2$–$3 \times$ higher.


## 1.16 What CNFs Contributed to the Story

Despite — or perhaps because of — the simulation bottleneck, continuous normalizing flows made two contributions that are foundational for the remainder of this monograph.

**The flow map as a first-class object.** The CNF framework establishes the flow map $\phi_t : \mathbb{R}^d \to \mathbb{R}^d$ — the map sending initial conditions at time 0 to positions at time $t$ — as the fundamental object of interest. The flow map is an *interpolant*: it describes a continuous path from the noise distribution $p_0$ to the data distribution $p_1$, passing through a family of intermediate distributions $\{p_t = (\phi_t)_\# p_0\}_{t \in [0,1]}$. This interpolation structure — which is absent from normalizing flows, where there is only an input and an output — becomes the central object in Chapters 3 and 4.

**The instantaneous change-of-variables formula.** Theorem 1.1 establishes that the log-density satisfies its own ODE, driven by the divergence of the vector field. This is not just a computational formula; it is a profound statement about the geometry of probability transport. It says that probability mass is locally conserved: the rate at which a small volume element changes is determined by how much the flow is diverging (expanding) or converging (compressing). This perspective connects directly to the continuity equation of fluid mechanics, which will be the starting point for flow matching.

## 1.17 The Parallel Track: Something Else Was Happening

As CNFs were being developed and refined — Kidger, Morrill, Foster, and Lyons' "Neural Controlled Differential Equations" (2020), Kidger's comprehensive thesis *"On Neural Differential Equations"* (2022) systematizing the whole enterprise — a completely separate and initially unrelated line of research was arriving at what appeared to be a wholly different answer to the generative modeling problem.

This parallel track did not start with transport, or invertible maps, or Jacobians. It started with a much simpler-sounding idea: *destroying structure is easy; can we learn to reverse the destruction?*

Sohl-Dickstein, Weiss, Maheswaranathan, and Ganguli proposed this idea in 2015 — the same year NICE appeared — in a paper titled "Deep Unsupervised Learning using Nonequilibrium Thermodynamics." The paper was largely ignored for five years. And then, in 2019 and 2020, it came back with a force that would reshape the entire field.

That story — the denoising track, its sudden explosion, and its unexpected convergence with the CNF track — is Chapter 2.

## 1.18 Historical Notes (Neural ODEs and CNFs)

**Chen, Rubanova, Bettencourt, and Duvenaud (NeurIPS 2018)** introduced Neural ODEs and the adjoint method for training, establishing continuous-time dynamics as a viable framework for deep learning. **Grathwohl, Chen, Bettencourt, Sutskever, and Duvenaud (ICLR 2019)** introduced FFJORD, making CNF training practical with the Hutchinson trace estimator.

The theoretical foundations draw on a long tradition: the instantaneous change-of-variables formula is Liouville's theorem (1838) for Hamiltonian systems, generalized to arbitrary divergence-free dynamics in the fluid mechanics literature. The adjoint method as used here traces to **Pontryagin's maximum principle** (1956) and was developed for PDE-constrained optimization by Pironneau (1974) and Lions (1971).

The systematic study of the expressivity and theoretical properties of Neural ODEs — including universal approximation, approximation rates, and the relationship to discrete ResNets — is developed in **Kidger's thesis** (*"On Neural Differential Equations"*, 2022), which remains the most comprehensive single reference for this material.

The connection between residual networks and ODE discretizations was developed explicitly by **Weinan E (2017)** ("A Proposal on Machine Learning via Dynamical Systems") and **Lu, Zhong, Li & Dong (2017)** ("Beyond Finite Layer Neural Networks: Bridging Deep Architectures and Numerical Differential Equations"). See also **Haber & Ruthotto (2017)** and **Chang et al. (2018)** for the stability and reversibility perspectives.

## References

- [Goodfellow et al.](https://arxiv.org/abs/1406.2661) — Generative Adversarial Nets (2014)
- [Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114) — Auto-Encoding Variational Bayes
- [Tabak & Vanden-Eijnden (2010)](https://doi.org/10.4310/CMS.2010.v8.n1.a1) — Density Estimation by Dual Ascent of the Log-Likelihood
- [Tabak & Turner (2013)](https://doi.org/10.1002/cpa.21423) — A Family of Nonparametric Density Estimation Algorithms
- [Rezende & Mohamed (ICML 2015)](https://arxiv.org/abs/1505.05770) — Variational Inference with Normalizing Flows
- [Dinh, Krueger, and Bengio](https://arxiv.org/abs/1410.8516) — NICE: Non-linear Independent Components Estimation (2014)
- [Dinh, Sohl-Dickstein, and Bengio](https://arxiv.org/abs/1605.08803) — Density Estimation using Real-valued Non-Volume Preserving Transformations (RealNVP, 2016)
- [Kingma and Dhariwal](https://arxiv.org/abs/1807.03039) — Glow: Generative Flow with Invertible 1×1 Convolutions (NeurIPS 2018)
- [Papamakarios, Murray, and Rainforth](https://arxiv.org/abs/1705.07057) — Masked Autoregressive Flow for Density Estimation (NeurIPS 2017)
- [Kingma et al.](https://arxiv.org/abs/1606.04934) — Improving Variational Inference with Inverse Autoregressive Flow (NeurIPS 2016)
- [Papamakarios, Nalisnick, Rezende, Mohamed & Lakshminarayanan (2021)](https://arxiv.org/abs/1912.02762) — Normalizing Flows for Probabilistic Modeling and Inference (JMLR)
- [Villani (2003)](https://doi.org/10.1090/gsm/058) — Topics in Optimal Transportation
- [Villani (2009)](https://doi.org/10.1007/978-3-540-71050-9) — Optimal Transport: Old and New
- [Ruthotto & Haber (2019)](https://arxiv.org/abs/1904.00804) — Deep Neural Networks Motivated by Partial Differential Equations
- [Finlay et al. (2020)](https://arxiv.org/abs/2002.02798) — How to Train Your Neural ODE: The World of Jacobian and Kinetic Regularization
- [He et al. (2016)](https://arxiv.org/abs/1512.03385) — Deep Residual Learning for Image Recognition
- [Weinan E (2017)](https://arxiv.org/abs/1710.02266) — A Proposal on Machine Learning via Dynamical Systems
- [Lu, Zhong, Li & Dong (2017)](https://arxiv.org/abs/1710.10121) — Beyond Finite Layer Neural Networks: Bridging Deep Architectures and Numerical Differential Equations
- [Sohl-Dickstein, Weiss, Maheswaranathan, and Ganguli (ICML 2015)](https://arxiv.org/abs/1503.03585) — Deep Unsupervised Learning using Nonequilibrium Thermodynamics
- [Chen, Rubanova, Bettencourt, and Duvenaud (NeurIPS 2018)](https://arxiv.org/abs/1806.07366) — Neural Ordinary Differential Equations
- [Grathwohl, Chen, Bettencourt, Sutskever, and Duvenaud (ICLR 2019)](https://arxiv.org/abs/1810.01367) — FFJORD: Free-Form Continuous Dynamics for Scalable Reversible Generative Models
- [Hutchinson (1989)](https://doi.org/10.1080/03610918908812806) — A Stochastic Estimator of the Trace of the Influence Matrix for Laplacian Smoothing Splines
- [Haber & Ruthotto (2017)](https://arxiv.org/abs/1705.03341) — Stable Architectures for Deep Neural Networks
- [Chang et al. (2018)](https://arxiv.org/abs/1710.10348) — Multi-Level Residual Networks from Dynamical Systems View
- [Kidger, Morrill, Foster, and Lyons (2020)](https://arxiv.org/abs/2005.08926) — Neural Controlled Differential Equations for Irregular Time Series
- [Kidger (2022)](https://arxiv.org/abs/2202.02435) — On Neural Differential Equations (PhD thesis)
