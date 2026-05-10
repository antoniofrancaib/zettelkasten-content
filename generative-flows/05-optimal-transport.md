---

title: "The Cost of Moving Mass"
subtitle: "Monge maps, Kantorovich duality, Wasserstein geometry, and the dynamic formulation that is flow matching — 1781 to 2024"
---



Chapter 4 used optimal transport as a practical tool — mini-batch OT to reduce trajectory crossings, the OT coupling to straighten paths, the reflow procedure to approximate optimal maps. But at no point did we define what optimal transport actually *is*, or prove why the OT coupling has the properties we claimed. This was a deliberate omission: the flow matching framework can be stated, understood, and used without OT theory, and the historical development did not require it. But understanding *why* straight paths are optimal, *what* the Wasserstein metric measures, and *how* the Benamou-Brenier formula connects transport to the continuity equation — this is the mathematical foundation on which the entire field implicitly rests. The debt is now repaid.

The story begins in 1781, when Gaspard Monge asked how to move a pile of sand into a hole at minimum cost. It was reformulated in 1942 by Leonid Kantorovich, who relaxed the problem in a way that always admits a solution and, incidentally, founded linear programming. It was given its modern geometric form by Yann Brenier in 1991, who showed that the optimal map for quadratic cost is the gradient of a convex function. And in 2000, Jean-David Benamou and Brenier reformulated the problem in terms of fluid mechanics — the continuity equation and kinetic energy — in a form that turns out to be identical to the flow matching objective of Chapter 4. The connection was not an accident. It was always there, waiting to be recognized.

## 5.1 The Monge Problem

The original formulation is deceptively simple. Given two probability distributions $\mu$ and $\nu$ on $\mathbb{R}^d$ — think of $\mu$ as the distribution of sand and $\nu$ as the shape of the hole — find a map $T : \mathbb{R}^d \to \mathbb{R}^d$ that pushes $\mu$ onto $\nu$ while minimizing the total transport cost:

> [!equation] 5.1
> $$\inf_{T:\; T_\#\mu = \nu} \int_{\mathbb{R}^d} c\bigl(x, T(x)\bigr)\, d\mu(x),$$


where $c(x, y)$ is the cost of moving a unit of mass from $x$ to $y$, and $T_\#\mu = \nu$ is the pushforward constraint: for every measurable set $A$, $\mu(T^{-1}(A)) = \nu(A)$, meaning that all mass arriving at $A$ under $T$ accounts for exactly $\nu(A)$.

The pushforward constraint is precisely the condition from Chapter 1 — a normalizing flow computes a map $T_\theta$ such that $T_{\theta\,\#}\,p_0 = p_{\text{data}}$. The Monge problem asks: among all such maps, which one is cheapest?

The most natural cost is the **quadratic cost** $c(x, y) = \|x - y\|^2$, measuring the squared Euclidean distance each particle travels. With this choice, the Monge problem seeks the transport map that minimizes the total squared displacement.

> [!remark] 5.1
> The Monge problem does not always have a solution. If $\mu = \delta_0$ is a point mass at the origin and $\nu = \frac{1}{2}\delta_{-1} + \frac{1}{2}\delta_1$, then no deterministic map $T$ can push $\mu$ to $\nu$ — a single point cannot be split in two. This failure motivated Kantorovich's relaxation, which allows mass to be split.


## 5.2 The Kantorovich Relaxation

Kantorovich's insight was to replace the deterministic map $T$ with a **coupling** (or transport plan) $\pi$ — a joint probability distribution on $\mathbb{R}^d \times \mathbb{R}^d$ whose marginals are $\mu$ and $\nu$:

> [!definition] 5.1 — Kantorovich Problem
> The **Kantorovich optimal transport problem** with cost $c$ is:
>
> $$W_c(\mu, \nu) = \inf_{\pi \in \Gamma(\mu, \nu)} \int_{\mathbb{R}^d \times \mathbb{R}^d} c(x, y)\, d\pi(x, y),$$
>
> where $\Gamma(\mu, \nu)$ is the set of all couplings — probability measures on $\mathbb{R}^d \times \mathbb{R}^d$ with first marginal $\mu$ and second marginal $\nu$.


A coupling $\pi$ generalizes a map: if $T$ is a transport map, then $\pi = (\mathrm{id} \times T)_\#\mu$ is the coupling that puts all the mass from $x$ onto $T(x)$. But a general coupling can split mass — the conditional $\pi(\cdot | x)$ can be a distribution over multiple destinations, not a point mass. This relaxation always admits a solution, even when the Monge problem does not.

The Kantorovich problem is a linear program: the objective $\int c\, d\pi$ is linear in $\pi$, and the marginal constraints $\pi_1 = \mu$, $\pi_2 = \nu$ are linear. This structure — which Kantorovich developed into the theory of linear programming — guarantees existence of a minimizer and, through duality, provides a powerful characterization of the solution.

> [!theorem] 5.1 — Kantorovich Duality
> The Kantorovich problem with cost $c$ satisfies:
>
> $$\inf_{\pi \in \Gamma(\mu,\nu)} \int c\, d\pi = \sup_{(\varphi,\psi):\, \varphi(x)+\psi(y) \leq c(x,y)} \left[\int \varphi\, d\mu + \int \psi\, d\nu\right],$$
>
> where the supremum is over all pairs of integrable functions $(\varphi, \psi)$ satisfying the constraint $\varphi(x) + \psi(y) \leq c(x,y)$ for all $(x,y)$. At optimality, $\psi = \varphi^c$ is the $c$-transform of $\varphi$.


The dual functions $\varphi$ and $\psi$ are called **Kantorovich potentials**. They have a direct interpretation: $\varphi(x)$ is the "price" of mass at location $x$, and $\psi(y)$ is the "price" at location $y$, such that the total price paid equals the total transport cost. The constraint $\varphi(x) + \psi(y) \leq c(x,y)$ says that the prices are consistent with the cost — it is never cheaper to buy mass at both endpoints than to transport it.

## 5.3 Brenier's Theorem and the Monge Map

For the quadratic cost $c(x,y) = \|x - y\|^2$, Brenier's theorem gives a definitive characterization of the optimal transport map when it exists.

> [!theorem] 5.2 — Brenier's Theorem
> Let $\mu$ and $\nu$ be probability measures on $\mathbb{R}^d$ with finite second moments, and suppose $\mu$ is absolutely continuous with respect to the Lebesgue measure. Then there exists a unique optimal transport map $T^*$ for the quadratic cost, and it is the gradient of a convex function:
>
> $$T^*(x) = \nabla \phi(x),$$
>
> where $\phi : \mathbb{R}^d \to \mathbb{R}$ is convex. The convex function $\phi$ is the Kantorovich potential.


The requirement that $\mu$ be absolutely continuous (no point masses) is precisely the condition under which the Monge problem has a solution for quadratic cost. The result is remarkable: the optimal way to rearrange mass is always a gradient map — a "potential flow" in the language of fluid mechanics — and the potential is convex, meaning the map is monotone. There is no rotation, no folding, no crossing of trajectories. The optimal transport map simply pushes each particle downhill along the convex potential.

> [!remark] 5.2
> Brenier's theorem explains, at a fundamental level, why OT-conditioned flows in Chapter 4 produce non-crossing trajectories. The optimal coupling assigns each noise sample $x_0$ to a data sample $x_1$ via the gradient of a convex function. The straight-line interpolation $x_t = (1-t)x_0 + tT^*(x_0)$ then produces non-crossing paths, because monotone maps preserve the ordering of particles. Independent random coupling, by contrast, is far from the gradient of any convex function, and the resulting trajectories cross freely.


## 5.4 Wasserstein Distances

The optimal transport cost defines a family of distances on the space of probability measures.

> [!definition] 5.2 — Wasserstein Distance
> The **$p$-Wasserstein distance** between probability measures $\mu$ and $\nu$ with finite $p$-th moments is:
>
> $$W_p(\mu, \nu) = \left(\inf_{\pi \in \Gamma(\mu,\nu)} \int \|x - y\|^p\, d\pi(x, y)\right)^{1/p}.$$


$W_p$ is a genuine metric on the space $\mathcal{P}_p(\mathbb{R}^d)$ of probability measures with finite $p$-th moments: it is symmetric, satisfies the triangle inequality, and $W_p(\mu, \nu) = 0$ if and only if $\mu = \nu$. The case $p = 2$ is the most important for our purposes — the **$2$-Wasserstein distance** $W_2$ inherits a Riemannian structure from the underlying Euclidean space.

Otto (2001) showed that $(\mathcal{P}_2(\mathbb{R}^d), W_2)$ can be formally endowed with the structure of an infinite-dimensional Riemannian manifold. The tangent space at a measure $\rho$ consists of velocity fields $v$ satisfying the continuity equation $\partial_t \rho + \nabla \cdot (\rho v) = 0$, and the inner product is:

$$\langle v, w \rangle_\rho = \int_{\mathbb{R}^d} v(x) \cdot w(x)\, \rho(x)\, dx.$$

This is the $L^2(\rho)$ inner product weighted by the density — exactly the kinetic energy that appears in the flow matching objective. Geodesics in this Riemannian structure are optimal transport paths, and the geodesic distance is $W_2$. The space of probability measures is not just a metric space; it is a *curved space* with a rich geometry of its own.

## 5.5 The Benamou-Brenier Formula: Dynamic Optimal Transport

The connection between optimal transport and the continuity equation is made explicit by the Benamou-Brenier formula (2000), which recasts the static OT problem as a dynamic one.

> [!theorem] 5.3 — Benamou-Brenier Formula
> The squared $2$-Wasserstein distance between $\mu$ and $\nu$ equals:
>
> $$W_2^2(\mu, \nu) = \inf_{(\rho_t, v_t)} \int_0^1 \int_{\mathbb{R}^d} \|v_t(x)\|^2\, \rho_t(x)\, dx\, dt,$$
>
> where the infimum is over all smooth paths of densities $\{\rho_t\}_{t \in [0,1]}$ and velocity fields $\{v_t\}$ satisfying:
> 1. The continuity equation: $\partial_t \rho_t + \nabla \cdot (\rho_t v_t) = 0$,
> 2. The boundary conditions: $\rho_0 = \mu$, $\rho_1 = \nu$.


Read this theorem carefully: it says that the squared Wasserstein distance is the minimum kinetic energy needed to transport $\mu$ to $\nu$ through a time-dependent flow governed by the continuity equation. The infimum is over both the density path $\rho_t$ and the velocity field $v_t$ — subject to the continuity equation linking the two.

This is the flow matching problem of Chapter 4. The continuity equation constraint is equation (4.1). The kinetic energy $\int_0^1 \mathbb{E}_{\rho_t}[\|v_t\|^2]\, dt$ is the expected squared norm of the velocity field, integrated over time — the objective that the minimum-norm velocity field minimizes. The Benamou-Brenier theorem says: among all flows transporting $\mu$ to $\nu$, the one with minimum kinetic energy is the optimal transport geodesic, and its kinetic energy is $W_2^2(\mu, \nu)$.

> [!remark] 5.3
> The Benamou-Brenier formula reveals that the flow matching framework has been computing optimal transport all along — or rather, approximating it. The flow matching objective (4.1) trains a velocity field to satisfy the continuity equation; the conditional flow matching objective (4.3) does so efficiently using conditional paths. The minimum-norm velocity field that the objective converges to is the Benamou-Brenier minimizer — the velocity field of the optimal transport geodesic. Flow matching, properly understood, is a simulation-free method for approximating the Wasserstein geodesic between the noise distribution and the data distribution.


## 5.6 Displacement Interpolation

The minimizer of the Benamou-Brenier formula has a beautiful geometric structure discovered by McCann (1997).

> [!definition] 5.3 — McCann's Displacement Interpolation
> Given $\mu$ and $\nu$ with optimal transport map $T^* = \nabla\phi$ (Brenier's theorem), the **displacement interpolation** is the curve of measures:
>
> $$\rho_t = \bigl((1-t)\,\mathrm{id} + t\, T^*\bigr)_\# \mu, \qquad t \in [0,1].$$
>
> Each particle at $x$ moves along the straight line from $x$ to $T^*(x)$ at constant speed.


Displacement interpolation is the geodesic in $(\mathcal{P}_2(\mathbb{R}^d), W_2)$ — the shortest path between $\mu$ and $\nu$ in Wasserstein space. Unlike the **mixture interpolation** $\rho_t^{\text{mix}} = (1-t)\mu + t\nu$, which simply fades between the two distributions, displacement interpolation physically moves the mass. The difference is dramatic: mixture interpolation of two separated Gaussians produces a bimodal intermediate distribution, while displacement interpolation produces a single Gaussian that slides from one to the other.

This is exactly the rectified flow interpolant $x_t = (1-t)x_0 + t\,T^*(x_0)$ from Section 4.6, but with the OT map replacing the random coupling. The rectified flow reflow procedure iteratively approximates the OT map, and with each iteration the interpolation becomes closer to displacement interpolation — the constant-speed geodesic in Wasserstein space.

McCann showed that many important functionals are *convex* along displacement interpolations — the entropy $\int \rho \log \rho$, the internal energy $\int V\, d\rho$, the interaction energy $\iint W(x-y)\, d\rho(x)\, d\rho(y)$ — a property known as **displacement convexity**. This convexity is the geometric reason that the OT path is well-behaved: the objective landscape along the geodesic has no spurious local minima.

## 5.7 From Theory to Computation: Mini-Batch OT

The exact computation of the OT map between two continuous distributions is, in general, an infinite-dimensional optimization problem. For discrete distributions (empirical measures from samples), it reduces to a finite linear program — but the computational cost scales cubically in the number of samples.

**The discrete problem.** Given $n$ source points $\{x_0^{(i)}\}_{i=1}^n$ and $n$ target points $\{x_1^{(j)}\}_{j=1}^n$, each with equal mass $1/n$, the discrete OT problem with quadratic cost is:

> [!equation] 5.2
> $$\min_{\sigma \in S_n} \frac{1}{n}\sum_{i=1}^n \|x_0^{(i)} - x_1^{(\sigma(i))}\|^2,$$


where $S_n$ is the set of permutations. This is the **linear assignment problem**: find the bijection between source and target points that minimizes total squared distance. The Hungarian algorithm solves it in $O(n^3)$ time; for large $n$, the auction algorithm of Bertsekas gives practical $O(n^2 \log n)$ performance.

**Mini-batch OT for flow matching.** At each training step, draw a mini-batch of $n$ noise samples and $n$ data samples. Solve the discrete OT problem (5.2) to obtain the optimal permutation $\sigma^*$. Pair each noise sample $x_0^{(i)}$ with the data sample $x_1^{(\sigma^*(i))}$ and train using the conditional flow matching objective with these paired samples. The cost per step is dominated by the assignment algorithm: $O(n^3)$ for batch size $n$, which is acceptable for $n \leq 1024$ and negligible compared to the neural network forward/backward pass for $n \leq 256$.

**Entropic regularization and Sinkhorn.** An alternative is to replace the linear program with its entropy-regularized version:

> [!equation] 5.3
> $$\min_{\pi \in \Gamma(\mu_n, \nu_n)} \sum_{i,j} c_{ij}\, \pi_{ij} - \varepsilon\, H(\pi),$$


where $H(\pi) = -\sum_{i,j}\pi_{ij}\log\pi_{ij}$ is the entropy of the coupling and $\varepsilon > 0$ is a regularization parameter. The solution has the form $\pi^*_{ij} = a_i\, K_{ij}\, b_j$ where $K_{ij} = e^{-c_{ij}/\varepsilon}$ is the Gibbs kernel, and the vectors $a$ and $b$ are found by alternating normalization — the **Sinkhorn algorithm**. This converges in $O(n^2/\varepsilon)$ time per iteration and produces a smooth coupling that approximates the exact OT solution as $\varepsilon \to 0$. We will develop the Sinkhorn algorithm and its connection to Schrödinger bridges in the next chapter.

## 5.8 The Geometry of Curved and Straight Paths

The practical importance of optimal transport for generative modeling reduces to a single geometric fact: OT paths are straight, and straight paths require fewer solver steps.

To make this precise, consider the linear interpolant $x_t = (1-t)x_0 + tx_1$ under some coupling $\pi(x_0, x_1)$ with marginals $p_0$ and $p_1$. The conditional velocity is $x_1 - x_0$ — constant in time and directed along the straight line from $x_0$ to $x_1$. But the *marginal* velocity field $u_t(x) = \mathbb{E}_\pi[x_1 - x_0 \mid x_t = x]$ is the conditional expectation over all $(x_0, x_1)$ pairs that pass through $x$ at time $t$, and this expectation can vary with $t$, producing a curved marginal field even though the conditional paths are straight.

The curvature of the marginal field is directly determined by the amount of crossing among the conditional paths. If two trajectories $x_0^{(1)} \to x_1^{(1)}$ and $x_0^{(2)} \to x_1^{(2)}$ cross — meaning their straight-line interpolants $x_t^{(1)}$ and $x_t^{(2)}$ coincide at some $(x, t)$ — then the marginal velocity at that point is an average of the two different directions, and this average changes as $t$ varies (the trajectories diverge after the crossing).

> [!proposition] 5.4 — OT Coupling Minimizes Marginal Curvature
> Among all couplings $\pi \in \Gamma(p_0, p_1)$ with the linear interpolant $x_t = (1-t)x_0 + tx_1$, the optimal transport coupling $\pi_{\mathrm{OT}}$ minimizes the expected temporal variation of the marginal velocity field:
>
> $$\pi_{\mathrm{OT}} = \arg\min_{\pi \in \Gamma(p_0, p_1)} \int_0^1 \mathbb{E}_{x \sim p_t}\!\left[\left\|\frac{\partial u_t}{\partial t}(x)\right\|^2\right] dt.$$
>
> In the limit of a continuous base distribution, the OT coupling produces a marginal velocity field with zero temporal variation — the marginal trajectories are themselves straight lines.


The implication for generation is direct. Euler integration with step size $h = 1/N$ has global error $O(h \cdot \text{curvature})$. For a perfectly straight marginal velocity field (zero curvature), a single Euler step is exact — one NFE suffices. For a curved field (independent coupling), many Euler steps are needed to track the curvature. The NFE-quality tradeoff is thus controlled by how close the training coupling is to the OT coupling.

This is the geometric reason that flow matching with OT conditioning generates good samples in 10–50 steps, while DDPM (which uses the VP-SDE interpolant with independent coupling) requires $\sim$1000. The straight-line interpolant under OT coupling is a discrete approximation to the displacement interpolation — the Wasserstein geodesic — and geodesics are the smoothest possible paths between two points.

## 5.9 Historical Notes

The Monge problem was posed in **Monge (1781)** ("Mémoire sur la théorie des déblais et des remblais"), motivated by military engineering — moving earth to build fortifications at minimum effort. **Kantorovich (1942)** reformulated the problem as a linear program in "On the translocation of masses," work that contributed to his Nobel Prize in Economics (1975). **Brenier (1991)** proved the existence and uniqueness of the optimal map for quadratic cost as the gradient of a convex function, connecting optimal transport to convex analysis.

**McCann (1997)** introduced displacement interpolation and displacement convexity, giving the space of probability measures a Riemannian-like structure. **Benamou and Brenier (2000)** established the dynamic formulation (Theorem 5.3), connecting optimal transport to fluid mechanics and the continuity equation. **Otto (2001)** formalized the Riemannian structure of $(\mathcal{P}_2, W_2)$ and showed that many PDEs (heat equation, porous medium equation) are gradient flows in this geometry.

The modern theory is systematized in **Villani (2003)** ("Topics in Optimal Transportation") and **Villani (2008)** ("Optimal Transport: Old and New"), which remain the standard references. **Cuturi (2013)** ("Sinkhorn Distances: Lightspeed Computation of Optimal Transport") brought computational OT to machine learning through entropic regularization and the Sinkhorn algorithm. **Peyré and Cuturi (2019)** ("Computational Optimal Transport") is the standard reference for algorithmic aspects.

The connection to generative modeling was anticipated by **Tabak and Vanden-Eijnden (2010)** and **Tabak and Turner (2013)**, who proposed transport-based density estimation using the continuity equation. The direct application to flow matching was developed by **Pooladian, Ben-Hamu, Domingo-Enrich, Amos, Lipman, and Chen (2023)** ("Multisample Flow Matching"), who demonstrated the practical benefits of mini-batch OT conditioning, and by **Liu (2022)** ("Rectified Flow"), who connected the reflow procedure to iterative OT approximation. The theoretical analysis of path straightness and its relation to NFE efficiency appears in **Liu (2022)** and **Albergo, Boffi, and Vanden-Eijnden (2023)** ("Stochastic Interpolants: A Unifying Framework for Flows and Diffusions"), who connected flow matching to the kinetic energy formulation of Benamou-Brenier.
