---

title: "The Map Before the Flow"
subtitle: "Normalizing flows, invertible architectures, and the birth of explicit generative transport — 2014 to 2018"
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

**Exact likelihoods without simulation.** The change-of-variables formula (1.1) gives exact log-likelihoods in a single forward pass. This is something that VAEs (approximate ELBO), GANs (no likelihood at all), and — as we will see in Chapter 3 — diffusion models (intractable marginal likelihood) cannot offer. For scientific applications where likelihoods are needed for Bayesian inference or hypothesis testing, this matters.

**Latent space structure.** Because the latent code $z = f_\theta(x)$ is a deterministic function of $x$, the latent space of a normalizing flow is the *same space as the data space*. Interpolation in latent space has a clean geometric interpretation: the straight line $tz_1 + (1-t)z_2$ maps, under $T_\theta$, to a geodesic (under the pullback metric) in data space. This is qualitatively different from VAE latent spaces, where the posterior is stochastic and interpolation is less principled.

**The seeds of what comes next.** In retrospect, the coupling layer construction already contains the germ of flow matching. The conditional update $y_{k+1:d} = x_{k+1:d} + m_\theta(x_{1:k})$ is a discrete-time flow of the second half of the coordinates, driven by the first half. The move to continuous time — letting the step size go to zero — is exactly the step taken in Chapter 2.

## 1.8 The State of Affairs in 2018

By the end of 2018, the landscape was this. Normalizing flows were theoretically principled, practically working, and actively being improved. Glow had demonstrated that high-resolution image synthesis was possible with exact likelihoods. RealNVP had shown the multi-scale architecture pattern that would be adopted by subsequent work. Autoregressive flows had revealed the deep connection between density estimation and invertible transformations.

But the field was also beginning to feel the ceiling. Training deep flow stacks was computationally expensive. The best bits-per-dimension results on image datasets still lagged behind autoregressive models like PixelCNN, which could afford much larger architectures because they did not need to compute a Jacobian. And critically, the coupling constraint — the source of tractability — was also the source of a subtle rigidity that limited how well flows could model data with complex, entangled dependencies.

The question forming in several research groups simultaneously was: what if the architectural constraint is unnecessary? What if tractable Jacobians are not the price of exact likelihoods — what if there is another way?

That question had been brewing since 2015 in a completely different corner of the literature. But the most direct path to an answer came not from probability theory but from a deceptively simple observation about deep learning architectures. ResNet layers, written as $x_{t+1} = x_t + f_\theta(x_t)$, look exactly like Euler steps for an ODE. What happens if you take that analogy seriously and let the number of layers go to infinity?

The answer to that question is the subject of Chapter 2.

## 1.9 Historical Notes

The history of normalizing flows as a machine learning tool properly begins with **Tabak & Turner (2013)** and **Tabak & Vanden-Eijnden (2010)**, who introduced the term "normalizing flow" and developed the mathematical framework for density estimation via composition of simple maps. The machine learning community's engagement began with **Rezende & Mohamed (2015)** (ICML), who introduced normalizing flows in the context of variational inference and named the variational normalizing flow objective. The architecture that made flows practical for generative modeling was **NICE** (Dinh, Krueger, Bengio, 2014/ICLR 2015), followed by **RealNVP** (Dinh, Sohl-Dickstein, Bengio, 2016/ICLR 2017), and culminating in **Glow** (Kingma & Dhariwal, NeurIPS 2018). Autoregressive flows were developed as **IAF** (Kingma et al., NeurIPS 2016) and **MAF** (Papamakarios, Murray, Rainforth, NeurIPS 2017). The connection between normalizing flows and optimal transport — which this monograph will develop extensively — was explored by Villani (2003, 2009) on the mathematical side and by Ruthotto & Haber (2019) and Finlay et al. (2020) on the computational side.

The survey by **Papamakarios, Nalisnick, Rezende, Mohamed & Lakshminarayanan (2021)** in *JMLR* remains the definitive reference for this era.
