---

title: "Information Geometry"
subtitle: "The Riemannian geometry of probability distributions"
---



Statistics has its own geometry. The collection of all probability distributions on a given sample space is not merely a set — it is a manifold, and the natural way to measure distances between distributions equips it with a Riemannian metric. That metric is the **Fisher information metric**, and the geometry it induces — information geometry — is the subject of this final chapter. The chapter draws together everything that has come before: smooth manifolds (Chapter 1), Riemannian metrics (Chapter 4), connections (Chapter 5), curvature (Chapter 6), and the duality structures that connect symplectic geometry (Chapter 9) to the dually flat spaces of exponential families. Information geometry is not an appendix to differential geometry but one of its richest applications — and, through its connections to machine learning, optimal transport, and quantum mechanics, one of its most active frontiers.

## Statistical Manifolds

Let $\mathcal{X}$ be a measurable sample space and $\mathcal{M} = \{p_\theta : \theta \in \Theta\}$ a parametric family of probability distributions on $\mathcal{X}$, where $\Theta \subset \mathbb{R}^n$ is an open parameter domain. Under mild regularity conditions (differentiability of the log-likelihood, integrability of derivatives), $\mathcal{M}$ is a smooth manifold with coordinates $\theta = (\theta^1, \ldots, \theta^n)$.

> [!definition] Score Function and Fisher Information Matrix
> For a parametric model $p_\theta$, the **score function** is $\ell_i(\theta; x) = \partial_i \log p_\theta(x)$, where $\partial_i = \partial/\partial\theta^i$. The **Fisher information matrix** is
> $$g_{ij}(\theta) = \mathbb{E}_{x \sim p_\theta}[\partial_i \log p_\theta(x)\, \partial_j \log p_\theta(x)] = -\mathbb{E}_{x \sim p_\theta}[\partial_i \partial_j \log p_\theta(x)].$$


The Fisher information matrix is symmetric and, under identifiability, positive definite. It therefore defines a **Riemannian metric** on the parameter manifold $\mathcal{M}$, called the **Fisher information metric** or **Fisher-Rao metric**.

> [!remark]
> The two expressions for $g_{ij}$ in the definition are equal because $\int p_\theta\, d\mu = 1$ implies $\mathbb{E}[\partial_i \log p_\theta] = 0$ (differentiating the normalization). The first expression (outer product of scores) shows that $g_{ij}$ measures how much information the data carries about the parameters; the second (negative Hessian of log-likelihood) connects to the curvature of the likelihood surface.


The metric $g_{ij}$ is the unique (up to scalar) Riemannian metric on $\mathcal{M}$ that is invariant under sufficient statistics: if $T: \mathcal{X} \to \mathcal{Y}$ is a sufficient statistic for $p_\theta$, then the Fisher metric on the induced model on $\mathcal{Y}$ equals the pullback of the metric on $\mathcal{M}$. This is **Chentsov's theorem** — the Fisher metric is canonically singled out by statistical invariance.

## Exponential Families

The most important class of statistical manifolds is the family of **exponential distributions**, which arise naturally in maximum entropy problems, Bayesian statistics, and machine learning.

> [!definition] Exponential Family
> A family $\{p_\theta\}$ is an **exponential family** if it can be written as
> $$p_\theta(x) = \exp\!\left(\sum_{i=1}^n \theta^i T_i(x) - \psi(\theta) + k(x)\right),$$
> where $T = (T_1, \ldots, T_n)$ are **sufficient statistics**, $\theta = (\theta^1, \ldots, \theta^n)$ are **natural parameters**, $\psi(\theta) = \log \int \exp(\sum_i \theta^i T_i(x))\, d\mu(x)$ is the **log-partition function** (or cumulant-generating function), and $k(x)$ is a base measure.


Examples: Gaussian $\mathcal{N}(\mu, \sigma^2)$, Bernoulli, Poisson, Gamma, Dirichlet, von Mises — all are exponential families. The natural parameters $\theta^i$ are the coordinates; the expectation parameters $\eta_i = \mathbb{E}_{p_\theta}[T_i(x)] = \partial_i \psi(\theta)$ provide dual coordinates.

> [!theorem] Geometry of Exponential Families
> For an exponential family:
> 1. The Fisher information matrix is the Hessian of the log-partition function: $g_{ij}(\theta) = \partial_i \partial_j \psi(\theta)$.
> 2. The map $\theta \mapsto \eta = \nabla\psi(\theta)$ (natural to expectation parameters) is a diffeomorphism; its inverse is $\eta \mapsto \theta = \nabla\varphi(\eta)$, where $\varphi(\eta) = \sup_\theta \{\langle\theta, \eta\rangle - \psi(\theta)\}$ is the **Legendre transform** of $\psi$.
> 3. The pair $(\psi, \varphi)$ is a **Legendre pair**: $\psi(\theta) + \varphi(\eta) = \langle\theta, \eta\rangle$, and the Fisher metric in expectation coordinates is the Hessian of $\varphi$: $g^{ij}(\eta) = \partial^i \partial^j \varphi(\eta)$.


The log-partition function $\psi$ is strictly convex (as a generating function of cumulants), so its Hessian is indeed a metric. The natural and expectation parameters are related by a convex duality analogous to the Legendre duality in mechanics — a pattern that information geometry makes precise through the theory of dual connections.

## Divergences and the KL Geometry

The Fisher metric arises naturally from a divergence — a measure of dissimilarity between distributions that is not necessarily symmetric.

> [!definition] KL Divergence
> The **Kullback-Leibler divergence** from $p$ to $q$ is
> $$D_{\mathrm{KL}}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)}\, d\mu(x) \geq 0,$$
> with equality if and only if $p = q$ almost everywhere.


The KL divergence is not symmetric and does not satisfy the triangle inequality — it is not a distance. However, its second-order Taylor expansion near $p = q$ recovers the Fisher metric: for $q = p_{\theta + d\theta}$ and $p = p_\theta$,

> [!equation]
> $$D_{\mathrm{KL}}(p_\theta \| p_{\theta + d\theta}) = \tfrac{1}{2} g_{ij}(\theta)\, d\theta^i\, d\theta^j + O(|d\theta|^3).$$


This shows that the Fisher metric is the "local divergence" of KL. More generally, any **Bregman divergence** $D_F(p \| q) = F(p) - F(q) - \langle \nabla F(q), p - q\rangle$ for a strictly convex $F$ induces a Riemannian metric (the Hessian of $F$) by the same second-order expansion.

> [!theorem] Pythagorean Relations
> In an exponential family, the KL divergence satisfies:
> 1. **Generalized Pythagorean theorem**: if $C$ is an **$e$-flat** submanifold (affine in $\theta$-coordinates) and $p^* = \arg\min_{q \in C} D_{\mathrm{KL}}(p \| q)$ is the **$m$-projection** of $p$ onto $C$, then for all $q \in C$:
> $$D_{\mathrm{KL}}(p \| q) = D_{\mathrm{KL}}(p \| p^*) + D_{\mathrm{KL}}(p^* \| q).$$
> 2. Dually, if $C$ is **$m$-flat** (affine in $\eta$-coordinates) and $p^* = \arg\min_{q \in C} D_{\mathrm{KL}}(q \| p)$ is the **$e$-projection**, the same identity holds with reversed arguments.


This is the geometric content of the **EM algorithm** (Expectation-Maximization): the E-step is an $m$-projection (updating sufficient statistics), the M-step is an $e$-projection (fitting parameters). Each step decreases the KL divergence, and the interleaving of the two projections converges to the maximum likelihood estimate.

## Dual Connections: the $\alpha$-Geometry

The Fisher metric alone does not capture the full differential-geometric structure of a statistical manifold. The key enrichment is a pair of dual connections.

> [!definition] $\alpha$-Connections
> For $\alpha \in \mathbb{R}$, the **$\alpha$-connection** $\nabla^{(\alpha)}$ on a statistical manifold is defined by its Christoffel symbols:
> $$\Gamma^{(\alpha)}_{ij,k}(\theta) = \mathbb{E}\!\left[\partial_i \partial_j \ell\, \partial_k \ell + \frac{1-\alpha}{2}\, \partial_i \ell\, \partial_j \ell\, \partial_k \ell\right],$$
> where $\ell = \log p_\theta$. The special cases are:
> - $\alpha = 0$: the Levi-Civita connection $\nabla^{(0)}$ of the Fisher metric.
> - $\alpha = 1$: the **$e$-connection** (exponential connection) $\nabla^{(e)}$.
> - $\alpha = -1$: the **$m$-connection** (mixture connection) $\nabla^{(m)}$.


The $\alpha$- and $(-\alpha)$-connections are **dual** with respect to the Fisher metric $g$: for any vector fields $X, Y, Z$,
$$Xg(Y, Z) = g(\nabla^{(\alpha)}_X Y, Z) + g(Y, \nabla^{(-\alpha)}_X Z).$$
This duality is the central structure of information geometry. The pair $(g, \nabla^{(\alpha)}, \nabla^{(-\alpha)})$ is called a **dualistic structure** on the statistical manifold.

> [!theorem] Flatness of Exponential Families
> For an exponential family:
> 1. The $e$-connection $\nabla^{(e)}$ is flat, with globally affine coordinates $\theta$ (natural parameters). The geodesics of $\nabla^{(e)}$ are straight lines in $\theta$-coordinates.
> 2. The $m$-connection $\nabla^{(m)}$ is flat, with globally affine coordinates $\eta$ (expectation parameters). The geodesics of $\nabla^{(m)}$ are straight lines in $\eta$-coordinates.
> 3. The natural and expectation coordinates are related by $\eta = \nabla\psi(\theta)$, and the Fisher metric is the Hessian: $g_{ij} = \partial_i\partial_j\psi$.


A manifold with a pair of dual flat connections is called **dually flat**. The geometry of dually flat spaces — with its Pythagorean theorems, divergences, and projection theorems — is the heart of information geometry as developed by Amari. Exponential families are the canonical example, but dually flat geometry appears also in optimal transport (see below) and in certain neural network parameter spaces.

## Fisher-Rao Geodesics and Statistical Distance

The geodesics of the Levi-Civita connection $\nabla^{(0)}$ — the **Fisher-Rao geodesics** — define the natural notion of distance between distributions:

$$d_{\mathrm{FR}}(p, q) = \inf_\gamma \int_0^1 \sqrt{g_{ij}(\gamma(t))\,\dot\gamma^i(t)\,\dot\gamma^j(t)}\, dt.$$

For the family of Gaussian distributions $\mathcal{N}(\mu, \sigma^2)$ with coordinates $(\mu, \sigma)$, the Fisher metric is $g = \sigma^{-2}(d\mu^2 + 2\, d\sigma^2)$ — proportional to the Poincaré metric of the upper half-plane $\mathbb{H}^2 = \{(\mu, \sigma) : \sigma > 0\}$. The statistical manifold of Gaussians is therefore a hyperbolic space of constant negative curvature $-\frac{1}{2}$.

More generally, the sectional curvatures of statistical manifolds encode geometric properties of the model. For mixtures of exponential families, curvature measures the interaction between components. Flat ($\alpha = \pm 1$) connections identify the most tractable families; curved geometries indicate harder inference problems.

> [!remark]
> The **Rao distance** — the geodesic distance in the Fisher-Rao metric — was introduced by C.R. Rao in 1945 in the same paper that proved the Cramér-Rao bound. Rao recognized that the Fisher information defines a Riemannian structure on parameter space, anticipating by decades the full development of information geometry by Efron (1975) and Amari (1985).


## The Cramér-Rao Bound as a Geometric Inequality

The most famous result in statistical estimation has a clean geometric interpretation.

> [!theorem] Cramér-Rao Bound
> Let $\hat\theta(x)$ be an unbiased estimator of $\theta$. Then the covariance matrix of $\hat\theta$ satisfies
> $$\mathrm{Cov}(\hat\theta) \geq g(\theta)^{-1}$$
> in the sense of positive semi-definite matrices, where $g(\theta)^{-1}$ is the inverse of the Fisher information matrix.


Geometrically, the Cramér-Rao bound is a statement about the **projection** of the estimator onto the tangent space of the statistical manifold. The inequality becomes an equality — the estimator is **efficient** — if and only if the estimator is a function of a sufficient statistic, equivalently when the score function lies in the tangent space of the model. The efficiency of maximum likelihood estimators in regular models follows from this geometric picture.

The bound has a natural interpretation in terms of quantum mechanics: for a quantum system with density matrix $\rho$, the **quantum Fisher information** $\mathcal{F}(\rho, A) = 4\, \mathrm{Var}_\rho(A)$ for a Hermitian observable $A$ satisfies a quantum Cramér-Rao bound, and the corresponding geometric structure on the space of density matrices is the **Bures metric**.

## Natural Gradient and Optimization

A central application of information geometry to machine learning is the **natural gradient**, introduced by Amari (1998).

In standard gradient descent, the parameter update $\theta \leftarrow \theta - \epsilon\, \nabla_\theta \mathcal{L}(\theta)$ treats all parameter directions equally. But the Fisher metric encodes the geometry of the model: moving distance $\epsilon$ in different directions in $\theta$-space corresponds to very different changes in the distribution $p_\theta$. The natural gradient corrects for this by moving in the steepest-descent direction in the Fisher-Rao metric:

> [!equation]
> $$\tilde\nabla_\theta \mathcal{L}(\theta) = g(\theta)^{-1}\, \nabla_\theta \mathcal{L}(\theta).$$


This is the **natural gradient** — the gradient of $\mathcal{L}$ with respect to the Fisher information metric rather than the Euclidean metric on parameter space. Natural gradient descent converges in $O(1)$ steps (rather than $O(\kappa)$ steps, where $\kappa$ is the condition number) for strongly convex losses in dually flat geometries. For neural networks, the natural gradient corresponds to the **K-FAC** (Kronecker-Factored Approximate Curvature) approximation to the Fisher matrix.

> [!remark]
> The natural gradient has a probabilistic interpretation: $\tilde\nabla_\theta \mathcal{L}$ is the direction in distribution space (not parameter space) that most increases $\mathcal{L}$, measured by KL divergence. Algorithms that implement this idea — **natural policy gradient** in reinforcement learning, **TRPO** (Trust Region Policy Optimization), and **KFAC**-preconditioned deep learning — consistently outperform vanilla gradient methods on models with complex parameter geometry.


## Information Geometry and Optimal Transport

A deep connection links information geometry to optimal transport (OT) — the subject of Chapter 9's closing remarks on geometric quantization and, more broadly, of the companion volume on generative flows.

The **Wasserstein distance** $W_2$ between probability measures is the geodesic distance in the $L^2$-Wasserstein space $(\mathcal{P}_2(\mathcal{X}), W_2)$. The space $(\mathcal{P}_2(\mathbb{R}^n), W_2)$ is an infinite-dimensional Riemannian manifold (in the sense of Otto calculus): the tangent space at $\rho$ is the space of square-integrable functions modulo constants, the metric is $\langle s, t\rangle_\rho = \int s\, t\, d\rho$, and the geodesics are displacement interpolants (McCann interpolants).

The Fisher-Rao metric and the Wasserstein metric both live on spaces of probability distributions but measure different things:
- **Fisher-Rao**: infinitesimal distinguishability via statistical experiments — sensitivity to perturbations at every point of $\mathcal{X}$.
- **Wasserstein $W_2$**: cost of transporting mass — sensitive to the geometry of $\mathcal{X}$ but not to pointwise likelihood.

These two metrics on the same space generate very different geometries. The **spherical Hellinger distance** $d_H(p, q) = \|\sqrt{p} - \sqrt{q}\|_{L^2}$ is related to the Fisher-Rao geodesic distance by $d_{\mathrm{FR}} = 2\arccos(1 - d_H^2/2)$, giving the sphere $S^\infty$ of square-root densities its natural geometry.

> [!theorem] Duality via $\Gamma$-Calculus
> On the Wasserstein space $(\mathcal{P}_2(\mathbb{R}^n), W_2)$, the gradient flow of the **entropy functional** $H(\rho) = \int \rho \log \rho\, dx$ is the **heat equation** $\partial_t \rho = \Delta \rho$. The gradient flow of the **KL divergence** $D_{\mathrm{KL}}(\rho \| \pi)$ with respect to the Wasserstein metric is the **Fokker-Planck equation** $\partial_t \rho = \nabla \cdot (\rho \nabla \log(\rho/\pi))$ — the Langevin diffusion toward the target $\pi$. This connects sampling algorithms (Langevin MCMC) to gradient flows in Wasserstein space.


The **de Bruijn identity** links Fisher information to entropy along Gaussian convolution: $\frac{d}{dt} H(\rho * \mathcal{N}(0,t)) = \frac{1}{2} I(\rho * \mathcal{N}(0,t))$, where $I(\rho) = \int \rho |\nabla \log \rho|^2\, dx$ is the **Fisher information functional**. This is the bridge between score-based diffusion models and information geometry: the score function $\nabla \log \rho_t$ at each diffusion time $t$ is precisely the integrand of the Fisher information, and learning the score is equivalent to fitting the Fisher geometry of the diffused distribution.

## Quantum Information Geometry

The geometry of quantum states parallels classical information geometry, with density matrices replacing probability distributions and the trace replacing integration.

The space of quantum states (density matrices) $\mathcal{S} = \{\rho : \rho \geq 0,\, \mathrm{tr}\,\rho = 1\}$ is a manifold. The **quantum Fisher information metric** (Bures metric) is:
$$g^{\mathrm{Bures}}_{ij}(\rho) = \tfrac{1}{2}\,\mathrm{tr}(L_i\, \rho),$$
where $L_i$ is the **symmetric logarithmic derivative** defined implicitly by $\partial_i \rho = \frac{1}{2}(L_i \rho + \rho L_i)$.

The Bures metric generates the **quantum Fisher information**, which governs the precision of quantum parameter estimation via the quantum Cramér-Rao bound $\mathrm{Cov}(\hat\theta) \geq \mathcal{F}(\rho)^{-1}$. This bound is saturated by projective measurements in the eigenbasis of the symmetric logarithmic derivative — the **optimal measurement** in quantum metrology.

Quantum information geometry also admits $\alpha$-connections (Petz's monotone metrics), dual connections, and a quantum analogue of the exponential family (the **$e$-family** of Gibbs states $\rho_\theta \propto e^{\sum_i \theta^i A_i}$). The quantum-classical correspondence is deepest for commuting observables, where quantum information geometry reduces to its classical counterpart.

## Geometry of Neural Networks and Generative Models

Information geometry provides a natural language for modern machine learning.

A **deep neural network** $p_\theta(y|x)$ defines a statistical manifold parameterized by weights $\theta \in \mathbb{R}^d$. The Fisher information matrix of this model is $g(\theta) = \mathbb{E}_{x, y \sim p_\theta}[\nabla_\theta \log p_\theta(y|x) \otimes \nabla_\theta \log p_\theta(y|x)]$. Since $d$ may be tens of billions, computing $g$ exactly is infeasible; K-FAC and related methods approximate it block-diagonally by Kronecker products.

The **information-geometric view of generative models** is particularly illuminating:
- **Normalizing flows** parameterize a push-forward of a base measure through a diffeomorphism $f_\theta$. The Fisher metric on the flow manifold pulls back the Wasserstein metric on the distribution manifold via the push-forward map.
- **Score-based diffusion** learns the score function $\nabla \log p_t$, which is precisely the natural gradient of the entropy $H(\rho_t)$ with respect to the Fisher metric — connecting training objectives to geodesics in the space of distributions.
- **Variational inference** minimizes $D_{\mathrm{KL}}(q_\phi \| p)$ over a variational family $\{q_\phi\}$; the optimal $q^*$ is the $m$-projection of $p$ onto the family. The ELBO gradient is proportional to the natural gradient on the variational manifold.
- **GAN training** corresponds (in idealized form) to minimizing a divergence between the model and data distributions, where the choice of divergence determines the geometry. The Wasserstein GAN uses a geometry closer to $W_1$ than KL; f-GANs correspond to $f$-divergences with different local geometries.

> [!remark]
> The **neural tangent kernel** (NTK), which governs the training dynamics of infinitely wide networks, is the Fisher information matrix of the network in the infinite-width limit. The convergence of gradient descent in the NTK regime is controlled by the eigenspectrum of the Fisher matrix — information geometry predicts the learning dynamics of large neural networks.


## Closing: The Unity of Differential Geometry

This chapter, and this series, ends at the intersection of differential geometry and modern science. The ten chapters have built a single coherent edifice: smooth manifolds provide the stage; tangent bundles and differential forms give the calculus; Riemannian metrics introduce measurement; connections and parallel transport provide the machinery for comparing across fibers; curvature measures the failure of flatness; Lie groups and symmetric spaces encode symmetry; fiber bundles unify local and global structure; symplectic geometry governs Hamiltonian dynamics; and information geometry applies the entire apparatus to the space of probability distributions.

The unity is not accidental. Probability distributions are smooth objects — they live on manifolds, their spaces have curvature, their symmetries form Lie groups, their dynamics satisfy Hamiltonian equations (in the Wasserstein sense), and their estimation theory is a chapter of Riemannian geometry. The tools developed for pure geometry turn out to be precisely those needed for modern statistics, machine learning, and physics. The direction of transfer also runs the other way: Fisher information, KL divergence, and quantum states have generated new geometric structures — dually flat spaces, $\alpha$-connections, Bures geometry — that enrich the discipline from which they emerged.

Differential geometry is not background material for applications. It is the native language in which the applications are most clearly written.

---

*Information geometry was founded by C.R. Rao (1945), who introduced the Fisher-Rao metric, and developed into a systematic theory by Shun-ichi Amari, whose book* Methods of Information Geometry *(with Nagaoka, 2000) remains the standard reference. The dual connection structure was clarified by Amari and Nagaoka; the dually flat geometry of exponential families by Efron (1975) and Amari (1982). The connection between Wasserstein gradient flows and Fokker-Planck equations is due to Jordan, Kinderlehrer, and Otto (1998). Quantum information geometry was systematically developed by Petz (1996). The application of natural gradient to neural network optimization is in Amari (1998); the K-FAC approximation in Martens and Grosse (2015). The neural tangent kernel is introduced in Jacot, Gabriel, and Hongler (2018).*
