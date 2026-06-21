---

title: "Optimal Transport"
subtitle: "Monge maps, Wasserstein geometry, entropy-regularized OT, bridge matching, and the stochastic middle ground — 1781 to 2025"
---



Chapter 3 used optimal transport as a practical tool — mini-batch OT to reduce trajectory crossings, the OT coupling to straighten paths, the reflow procedure to approximate optimal maps. But at no point did we define what optimal transport actually *is*, or prove why the OT coupling has the properties we claimed. This was a deliberate omission: the flow matching framework can be stated, understood, and used without OT theory, and the historical development did not require it. But understanding *why* straight paths are optimal, *what* the Wasserstein metric measures, and *how* the Benamou-Brenier formula connects transport to the continuity equation — this is the mathematical foundation on which the entire field implicitly rests.

The story begins in 1781, when Gaspard Monge asked how to move a pile of sand into a hole at minimum cost. It was reformulated in 1942 by Leonid Kantorovich, who relaxed the problem in a way that always admits a solution. It was given its modern geometric form by Yann Brenier in 1991, who showed that the optimal map for quadratic cost is the gradient of a convex function. And in 2000, Jean-David Benamou and Brenier reformulated the problem in terms of fluid mechanics — the continuity equation and kinetic energy — in a form that turns out to be identical to the flow matching objective of Chapter 3. Then in 1931, Erwin Schrödinger asked a related question about Brownian motion that became, ninety years later, a generative model. This chapter develops both threads and their unification.

## 4.1 The Monge Problem

The original formulation is deceptively simple. Given two probability distributions $\mu$ and $\nu$ on $\mathbb{R}^d$ — think of $\mu$ as the distribution of sand and $\nu$ as the shape of the hole — find a map $T : \mathbb{R}^d \to \mathbb{R}^d$ that pushes $\mu$ onto $\nu$ while minimizing the total transport cost:

> [!equation] 4.1
> $$\inf_{T:\; T_\#\mu = \nu} \int_{\mathbb{R}^d} c\bigl(x, T(x)\bigr)\, d\mu(x),$$


where $c(x, y)$ is the cost of moving a unit of mass from $x$ to $y$, and $T_\#\mu = \nu$ is the pushforward constraint: for every measurable set $A$, $\mu(T^{-1}(A)) = \nu(A)$, meaning that all mass arriving at $A$ under $T$ accounts for exactly $\nu(A)$.

The pushforward constraint is precisely the condition from Chapter 1 — a normalizing flow computes a map $T_\theta$ such that $T_{\theta\,\#}\,p_0 = p_{\text{data}}$. The Monge problem asks: among all such maps, which one is cheapest?

The most natural cost is the **quadratic cost** $c(x, y) = \|x - y\|^2$, measuring the squared Euclidean distance each particle travels. With this choice, the Monge problem seeks the transport map that minimizes the total squared displacement.

> [!remark] 4.1
> The Monge problem does not always have a solution. If $\mu = \delta_0$ is a point mass at the origin and $\nu = \frac{1}{2}\delta_{-1} + \frac{1}{2}\delta_1$, then no deterministic map $T$ can push $\mu$ to $\nu$ — a single point cannot be split in two. This failure motivated Kantorovich's relaxation, which allows mass to be split.


## 4.2 The Kantorovich Relaxation

Kantorovich's insight was to replace the deterministic map $T$ with a **coupling** (or transport plan) $\pi$ — a joint probability distribution on $\mathbb{R}^d \times \mathbb{R}^d$ whose marginals are $\mu$ and $\nu$:

> [!definition] 4.1 — Kantorovich Problem
> The **Kantorovich optimal transport problem** with cost $c$ is:
>
> $$W_c(\mu, \nu) = \inf_{\pi \in \Gamma(\mu, \nu)} \int_{\mathbb{R}^d \times \mathbb{R}^d} c(x, y)\, d\pi(x, y),$$
>
> where $\Gamma(\mu, \nu)$ is the set of all couplings — probability measures on $\mathbb{R}^d \times \mathbb{R}^d$ with first marginal $\mu$ and second marginal $\nu$.


A coupling $\pi$ generalizes a map: if $T$ is a transport map, then $\pi = (\mathrm{id} \times T)_\#\mu$ is the coupling that puts all the mass from $x$ onto $T(x)$. But a general coupling can split mass — the conditional $\pi(\cdot | x)$ can be a distribution over multiple destinations, not a point mass. This relaxation always admits a solution, even when the Monge problem does not.

The Kantorovich problem is a linear program: the objective $\int c\, d\pi$ is linear in $\pi$, and the marginal constraints $\pi_1 = \mu$, $\pi_2 = \nu$ are linear. This structure — which Kantorovich developed into the theory of linear programming — guarantees existence of a minimizer and, through duality, provides a powerful characterization of the solution.

> [!theorem] 4.1 — Kantorovich Duality
> The Kantorovich problem with cost $c$ satisfies:
>
> $$\inf_{\pi \in \Gamma(\mu,\nu)} \int c\, d\pi = \sup_{(\varphi,\psi):\, \varphi(x)+\psi(y) \leq c(x,y)} \left[\int \varphi\, d\mu + \int \psi\, d\nu\right],$$
>
> where the supremum is over all pairs of integrable functions $(\varphi, \psi)$ satisfying the constraint $\varphi(x) + \psi(y) \leq c(x,y)$ for all $(x,y)$. At optimality, $\psi = \varphi^c$ is the $c$-transform of $\varphi$.


The dual functions $\varphi$ and $\psi$ are called **Kantorovich potentials**. They have a direct interpretation: $\varphi(x)$ is the "price" of mass at location $x$, and $\psi(y)$ is the "price" at location $y$, such that the total price paid equals the total transport cost. The constraint $\varphi(x) + \psi(y) \leq c(x,y)$ says that the prices are consistent with the cost — it is never cheaper to buy mass at both endpoints than to transport it.

## 4.3 Brenier's Theorem and the Monge Map

For the quadratic cost $c(x,y) = \|x - y\|^2$, Brenier's theorem gives a definitive characterization of the optimal transport map when it exists.

> [!theorem] 4.2 — Brenier's Theorem
> Let $\mu$ and $\nu$ be probability measures on $\mathbb{R}^d$ with finite second moments, and suppose $\mu$ is absolutely continuous with respect to the Lebesgue measure. Then there exists a unique optimal transport map $T^*$ for the quadratic cost, and it is the gradient of a convex function:
>
> $$T^*(x) = \nabla \phi(x),$$
>
> where $\phi : \mathbb{R}^d \to \mathbb{R}$ is convex. The convex function $\phi$ is the Kantorovich potential.


The requirement that $\mu$ be absolutely continuous (no point masses) is precisely the condition under which the Monge problem has a solution for quadratic cost. The result is remarkable: the optimal way to rearrange mass is always a gradient map — a "potential flow" in the language of fluid mechanics — and the potential is convex, meaning the map is monotone. There is no rotation, no folding, no crossing of trajectories. The optimal transport map simply pushes each particle downhill along the convex potential.

> [!remark] 4.2
> Brenier's theorem explains, at a fundamental level, why OT-conditioned flows in Chapter 3 produce non-crossing trajectories. The optimal coupling assigns each noise sample $x_0$ to a data sample $x_1$ via the gradient of a convex function. The straight-line interpolation $x_t = (1-t)x_0 + tT^*(x_0)$ then produces non-crossing paths, because monotone maps preserve the ordering of particles. Independent random coupling, by contrast, is far from the gradient of any convex function, and the resulting trajectories cross freely.


## 4.4 Wasserstein Distances

The optimal transport cost defines a family of distances on the space of probability measures.

> [!definition] 4.2 — Wasserstein Distance
> The **$p$-Wasserstein distance** between probability measures $\mu$ and $\nu$ with finite $p$-th moments is:
>
> $$W_p(\mu, \nu) = \left(\inf_{\pi \in \Gamma(\mu,\nu)} \int \|x - y\|^p\, d\pi(x, y)\right)^{1/p}.$$


$W_p$ is a genuine metric on the space $\mathcal{P}_p(\mathbb{R}^d)$ of probability measures with finite $p$-th moments: it is symmetric, satisfies the triangle inequality, and $W_p(\mu, \nu) = 0$ if and only if $\mu = \nu$. The case $p = 2$ is the most important for our purposes — the **$2$-Wasserstein distance** $W_2$ inherits a Riemannian structure from the underlying Euclidean space.

Otto (2001) showed that $(\mathcal{P}_2(\mathbb{R}^d), W_2)$ can be formally endowed with the structure of an infinite-dimensional Riemannian manifold. The tangent space at a measure $\rho$ consists of velocity fields $v$ satisfying the continuity equation $\partial_t \rho + \nabla \cdot (\rho v) = 0$, and the inner product is:

$$\langle v, w \rangle_\rho = \int_{\mathbb{R}^d} v(x) \cdot w(x)\, \rho(x)\, dx.$$

This is the $L^2(\rho)$ inner product weighted by the density — exactly the kinetic energy that appears in the flow matching objective. Geodesics in this Riemannian structure are optimal transport paths, and the geodesic distance is $W_2$. The space of probability measures is not just a metric space; it is a *curved space* with a rich geometry of its own.

## 4.5 The Benamou-Brenier Formula: Dynamic Optimal Transport

The connection between optimal transport and the continuity equation is made explicit by the Benamou-Brenier formula (2000), which recasts the static OT problem as a dynamic one.

> [!theorem] 4.3 — Benamou-Brenier Formula
> The squared $2$-Wasserstein distance between $\mu$ and $\nu$ equals:
>
> $$W_2^2(\mu, \nu) = \inf_{(\rho_t, v_t)} \int_0^1 \int_{\mathbb{R}^d} \|v_t(x)\|^2\, \rho_t(x)\, dx\, dt,$$
>
> where the infimum is over all smooth paths of densities $\{\rho_t\}_{t \in [0,1]}$ and velocity fields $\{v_t\}$ satisfying:
> 1. The continuity equation: $\partial_t \rho_t + \nabla \cdot (\rho_t v_t) = 0$,
> 2. The boundary conditions: $\rho_0 = \mu$, $\rho_1 = \nu$.


Read this theorem carefully: it says that the squared Wasserstein distance is the minimum kinetic energy needed to transport $\mu$ to $\nu$ through a time-dependent flow governed by the continuity equation. The infimum is over both the density path $\rho_t$ and the velocity field $v_t$ — subject to the continuity equation linking the two.

This is the flow matching problem of Chapter 3. The continuity equation constraint is equation (3.1). The kinetic energy $\int_0^1 \mathbb{E}_{\rho_t}[\|v_t\|^2]\, dt$ is the expected squared norm of the velocity field, integrated over time — the objective that the minimum-norm velocity field minimizes. The Benamou-Brenier theorem says: among all flows transporting $\mu$ to $\nu$, the one with minimum kinetic energy is the optimal transport geodesic, and its kinetic energy is $W_2^2(\mu, \nu)$.

> [!remark] 4.3
> The Benamou-Brenier formula reveals that the flow matching framework has been computing optimal transport all along — or rather, approximating it. The flow matching objective (3.1) trains a velocity field to satisfy the continuity equation; the conditional flow matching objective (3.3) does so efficiently using conditional paths. The minimum-norm velocity field that the objective converges to is the Benamou-Brenier minimizer — the velocity field of the optimal transport geodesic. Flow matching, properly understood, is a simulation-free method for approximating the Wasserstein geodesic between the noise distribution and the data distribution.


## 4.6 Displacement Interpolation

The minimizer of the Benamou-Brenier formula has a beautiful geometric structure discovered by McCann (1997).

> [!definition] 4.3 — McCann's Displacement Interpolation
> Given $\mu$ and $\nu$ with optimal transport map $T^* = \nabla\phi$ (Brenier's theorem), the **displacement interpolation** is the curve of measures:
>
> $$\rho_t = \bigl((1-t)\,\mathrm{id} + t\, T^*\bigr)_\# \mu, \qquad t \in [0,1].$$
>
> Each particle at $x$ moves along the straight line from $x$ to $T^*(x)$ at constant speed.


Displacement interpolation is the geodesic in $(\mathcal{P}_2(\mathbb{R}^d), W_2)$ — the shortest path between $\mu$ and $\nu$ in Wasserstein space. Unlike the **mixture interpolation** $\rho_t^{\text{mix}} = (1-t)\mu + t\nu$, which simply fades between the two distributions, displacement interpolation physically moves the mass. The difference is dramatic: mixture interpolation of two separated Gaussians produces a bimodal intermediate distribution, while displacement interpolation produces a single Gaussian that slides from one to the other.

This is exactly the rectified flow interpolant $x_t = (1-t)x_0 + t\,T^*(x_0)$ from Section 3.6, but with the OT map replacing the random coupling. The rectified flow reflow procedure iteratively approximates the OT map, and with each iteration the interpolation becomes closer to displacement interpolation — the constant-speed geodesic in Wasserstein space.

McCann showed that many important functionals are *convex* along displacement interpolations — the entropy $\int \rho \log \rho$, the internal energy $\int V\, d\rho$, the interaction energy $\iint W(x-y)\, d\rho(x)\, d\rho(y)$ — a property known as **displacement convexity**. This convexity is the geometric reason that the OT path is well-behaved: the objective landscape along the geodesic has no spurious local minima.

## 4.7 From Theory to Computation: Mini-Batch OT

The exact computation of the OT map between two continuous distributions is, in general, an infinite-dimensional optimization problem. For discrete distributions (empirical measures from samples), it reduces to a finite linear program — but the computational cost scales cubically in the number of samples.

**The discrete problem.** Given $n$ source points $\{x_0^{(i)}\}_{i=1}^n$ and $n$ target points $\{x_1^{(j)}\}_{j=1}^n$, each with equal mass $1/n$, the discrete OT problem with quadratic cost is:

> [!equation] 4.2
> $$\min_{\sigma \in S_n} \frac{1}{n}\sum_{i=1}^n \|x_0^{(i)} - x_1^{(\sigma(i))}\|^2,$$


where $S_n$ is the set of permutations. This is the **linear assignment problem**: find the bijection between source and target points that minimizes total squared distance. The Hungarian algorithm solves it in $O(n^3)$ time; for large $n$, the auction algorithm of Bertsekas gives practical $O(n^2 \log n)$ performance.

**Mini-batch OT for flow matching.** At each training step, draw a mini-batch of $n$ noise samples and $n$ data samples. Solve the discrete OT problem (4.2) to obtain the optimal permutation $\sigma^*$. Pair each noise sample $x_0^{(i)}$ with the data sample $x_1^{(\sigma^*(i))}$ and train using the conditional flow matching objective with these paired samples. The cost per step is dominated by the assignment algorithm: $O(n^3)$ for batch size $n$, which is acceptable for $n \leq 1024$ and negligible compared to the neural network forward/backward pass for $n \leq 256$.

**Entropic regularization and Sinkhorn.** An alternative is to replace the linear program with its entropy-regularized version:

> [!equation] 4.3
> $$\min_{\pi \in \Gamma(\mu_n, \nu_n)} \sum_{i,j} c_{ij}\, \pi_{ij} - \varepsilon\, H(\pi),$$


where $H(\pi) = -\sum_{i,j}\pi_{ij}\log\pi_{ij}$ is the entropy of the coupling and $\varepsilon > 0$ is a regularization parameter. The solution has the form $\pi^*_{ij} = a_i\, K_{ij}\, b_j$ where $K_{ij} = e^{-c_{ij}/\varepsilon}$ is the Gibbs kernel, and the vectors $a$ and $b$ are found by alternating normalization — the **Sinkhorn algorithm**. This converges in $O(n^2/\varepsilon)$ time per iteration and produces a smooth coupling that approximates the exact OT solution as $\varepsilon \to 0$. The Sinkhorn algorithm and its connection to Schrödinger bridges are developed in the following section.

## 4.8 The Geometry of Curved and Straight Paths

The practical importance of optimal transport for generative modeling reduces to a single geometric fact: OT paths are straight, and straight paths require fewer solver steps.

To make this precise, consider the linear interpolant $x_t = (1-t)x_0 + tx_1$ under some coupling $\pi(x_0, x_1)$ with marginals $p_0$ and $p_1$. The conditional velocity is $x_1 - x_0$ — constant in time and directed along the straight line from $x_0$ to $x_1$. But the *marginal* velocity field $u_t(x) = \mathbb{E}_\pi[x_1 - x_0 \mid x_t = x]$ is the conditional expectation over all $(x_0, x_1)$ pairs that pass through $x$ at time $t$, and this expectation can vary with $t$, producing a curved marginal field even though the conditional paths are straight.

The curvature of the marginal field is directly determined by the amount of crossing among the conditional paths. If two trajectories $x_0^{(1)} \to x_1^{(1)}$ and $x_0^{(2)} \to x_1^{(2)}$ cross — meaning their straight-line interpolants $x_t^{(1)}$ and $x_t^{(2)}$ coincide at some $(x, t)$ — then the marginal velocity at that point is an average of the two different directions, and this average changes as $t$ varies (the trajectories diverge after the crossing).

> [!proposition] 4.4 — OT Coupling Minimizes Marginal Curvature
> Among all couplings $\pi \in \Gamma(p_0, p_1)$ with the linear interpolant $x_t = (1-t)x_0 + tx_1$, the optimal transport coupling $\pi_{\mathrm{OT}}$ minimizes the expected temporal variation of the marginal velocity field:
>
> $$\pi_{\mathrm{OT}} = \arg\min_{\pi \in \Gamma(p_0, p_1)} \int_0^1 \mathbb{E}_{x \sim p_t}\!\left[\left\|\frac{\partial u_t}{\partial t}(x)\right\|^2\right] dt.$$
>
> In the limit of a continuous base distribution, the OT coupling produces a marginal velocity field with zero temporal variation — the marginal trajectories are themselves straight lines.


The implication for generation is direct. Euler integration with step size $h = 1/N$ has global error $O(h \cdot \text{curvature})$. For a perfectly straight marginal velocity field (zero curvature), a single Euler step is exact — one NFE suffices. For a curved field (independent coupling), many Euler steps are needed to track the curvature. The NFE-quality tradeoff is thus controlled by how close the training coupling is to the OT coupling.

This is the geometric reason that flow matching with OT conditioning generates good samples in 10–50 steps, while DDPM (which uses the VP-SDE interpolant with independent coupling) requires $\sim$1000. The straight-line interpolant under OT coupling is a discrete approximation to the displacement interpolation — the Wasserstein geodesic — and geodesics are the smoothest possible paths between two points.

## 4.9 Historical Notes (Optimal Transport)

The Monge problem was posed in **Monge (1781)** ("Mémoire sur la théorie des déblais et des remblais"), motivated by military engineering — moving earth to build fortifications at minimum effort. **Kantorovich (1942)** reformulated the problem as a linear program in "On the translocation of masses," work that contributed to his Nobel Prize in Economics (1975). **Brenier (1991)** proved the existence and uniqueness of the optimal map for quadratic cost as the gradient of a convex function, connecting optimal transport to convex analysis.

**McCann (1997)** introduced displacement interpolation and displacement convexity, giving the space of probability measures a Riemannian-like structure. **Benamou and Brenier (2000)** established the dynamic formulation (Theorem 4.3), connecting optimal transport to fluid mechanics and the continuity equation. **Otto (2001)** formalized the Riemannian structure of $(\mathcal{P}_2, W_2)$ and showed that many PDEs (heat equation, porous medium equation) are gradient flows in this geometry.

The modern theory is systematized in **Villani (2003)** ("Topics in Optimal Transportation") and **Villani (2008)** ("Optimal Transport: Old and New"), which remain the standard references. **Cuturi (2013)** ("Sinkhorn Distances: Lightspeed Computation of Optimal Transport") brought computational OT to machine learning through entropic regularization and the Sinkhorn algorithm. **Peyré and Cuturi (2019)** ("Computational Optimal Transport") is the standard reference for algorithmic aspects.

The connection to generative modeling was anticipated by **Tabak and Vanden-Eijnden (2010)** and **Tabak and Turner (2013)**, who proposed transport-based density estimation using the continuity equation. The direct application to flow matching was developed by **Pooladian, Ben-Hamu, Domingo-Enrich, Amos, Lipman, and Chen (2023)** ("Multisample Flow Matching"), who demonstrated the practical benefits of mini-batch OT conditioning, and by **Liu (2022)** ("Rectified Flow"), who connected the reflow procedure to iterative OT approximation. The theoretical analysis of path straightness and its relation to NFE efficiency appears in **Liu (2022)** and **Albergo, Boffi, and Vanden-Eijnden (2023)** ("Stochastic Interpolants: A Unifying Framework for Flows and Diffusions"), who connected flow matching to the kinetic energy formulation of Benamou-Brenier.

---

## 4.10 The Schrödinger Bridge Problem

In 1931, Erwin Schrödinger posed a question that appeared, at the time, to have nothing to do with generative modeling. He considered a large number of particles observed to have distribution $\mu$ at time $0$ and distribution $\nu$ at time $1$, and asked: given that the particles undergo Brownian motion between the two observations, what is the *most likely* stochastic evolution connecting $\mu$ to $\nu$? The answer — the Schrödinger bridge — is optimal transport with an entropy penalty, and it produces stochastic rather than deterministic trajectories.

The central insight is that the Schrödinger bridge is the *natural* stochastic extension of the optimal transport problem of the preceding sections. Where OT finds the deterministic map that moves mass at minimum cost, the Schrödinger bridge finds the stochastic process that moves mass at minimum cost while maintaining a prescribed level of randomness. The regularization parameter $\varepsilon$ controls the tradeoff: at $\varepsilon = 0$ we recover the OT geodesic; at $\varepsilon \to \infty$ we recover unstructured Brownian motion. The generative models of Chapters 2 and 3 — diffusion and flow matching — sit at opposite ends of this spectrum, and Schrödinger bridges provide the unified theory that connects them.

Let $R$ be a **reference process** — a probability measure over paths $\{X_t\}_{t \in [0,1]}$ in $\mathbb{R}^d$. The canonical choice is the Wiener measure (Brownian motion) with diffusion coefficient $\sigma$: $dX_t = \sigma\, dW_t$. The Schrödinger bridge problem asks for the path measure $P^*$ that is closest to $R$ in KL divergence while satisfying the marginal constraints:

> [!definition] 4.4 — Schrödinger Bridge Problem
> Given a reference process $R$ and marginal distributions $\mu$, $\nu$, the **Schrödinger bridge** is:
>
> $$P^* = \arg\min_{P:\, P_0 = \mu,\; P_1 = \nu} \mathrm{KL}(P \| R),$$
>
> where $P_0$ and $P_1$ denote the marginal distributions of $P$ at times $0$ and $1$.


When $R$ is Brownian motion, the Schrödinger bridge $P^*$ is itself a diffusion process $dX_t = f_t^*(X_t)\, dt + \sigma\, dW_t$ — it has the same noise as Brownian motion but acquires a drift $f_t^*$ that steers the particles from $\mu$ to $\nu$. The KL objective penalizes deviations from the reference: the drift is the minimum correction to Brownian motion that achieves the desired marginals. Particles still diffuse randomly, but they diffuse with purpose.

> [!remark] 4.4
> The Schrödinger bridge has a compelling physical interpretation. Imagine $10^{23}$ particles undergoing Brownian motion. At time $0$ they are distributed as $\mu$; at time $1$ they are observed to be distributed as $\nu$. The Schrödinger bridge is the conditional law of the process given both endpoint observations — the most likely interpolation between the two snapshots. It is the answer to: "given what we see at the beginning and end, what most likely happened in between?"


## 4.11 Entropy-Regularized Optimal Transport

The Schrödinger bridge has an equivalent **static** formulation in terms of couplings, which connects it directly to the Kantorovich problem above.

Let $R_{01}$ denote the joint distribution of $(X_0, X_1)$ under the reference process $R$ — for Brownian motion with diffusion $\sigma$, this is $R_{01}(x, y) = \mu(x) \cdot \mathcal{N}(y; x, \sigma^2 I)$. The static Schrödinger bridge seeks the coupling $\pi^*$ closest to $R_{01}$ in KL divergence:

> [!equation] 4.4
> $$\pi^* = \arg\min_{\pi \in \Gamma(\mu, \nu)} \mathrm{KL}(\pi \| R_{01}).$$


Expanding the KL divergence, this becomes:

$$\pi^* = \arg\min_{\pi \in \Gamma(\mu, \nu)} \int \log\frac{\pi(x,y)}{R_{01}(x,y)}\, d\pi(x,y) = \arg\min_{\pi \in \Gamma(\mu,\nu)} \left[\frac{1}{\sigma^2}\int \|x-y\|^2\, d\pi - \sigma^2\, H(\pi)\right] + \text{const},$$

where $H(\pi) = -\int \pi \log \pi$ is the entropy of the coupling. Up to a scaling, this is precisely the **entropy-regularized optimal transport** problem:

> [!definition] 4.5 — Entropy-Regularized OT
> The **entropy-regularized optimal transport** problem with regularization $\varepsilon > 0$ is:
>
> $$\pi^*_\varepsilon = \arg\min_{\pi \in \Gamma(\mu, \nu)} \int \|x - y\|^2\, d\pi(x, y) - \varepsilon\, H(\pi).$$
>
> As $\varepsilon \to 0$, $\pi^*_\varepsilon$ converges to the OT coupling $\pi_{\mathrm{OT}}$. As $\varepsilon \to \infty$, $\pi^*_\varepsilon$ converges to the independent coupling $\mu \otimes \nu$.


The regularization parameter $\varepsilon = \sigma^2$ is the squared diffusion coefficient. Large noise means large entropy regularization, which pushes the coupling toward independence (Brownian motion doesn't care about the target). Small noise means small regularization, which pushes the coupling toward the OT solution (particles take the shortest path). The Schrödinger bridge interpolates between these extremes.

## 4.12 The Sinkhorn Algorithm and Iterative Proportional Fitting

The entropy-regularized problem has a key computational advantage over exact OT: its solution has the factored form $\pi^*(x, y) = a(x)\, K(x, y)\, b(y)$, where $K(x, y) = e^{-\|x-y\|^2/\varepsilon}$ is the Gibbs kernel. The marginal constraints $\pi^*_1 = \mu$, $\pi^*_2 = \nu$ then determine $a$ and $b$ through the coupled equations:

$$a(x) = \frac{\mu(x)}{\int K(x,y)\, b(y)\, dy}, \qquad b(y) = \frac{\nu(y)}{\int a(x)\, K(x,y)\, dx}.$$

> [!definition] 4.6 — Sinkhorn Algorithm
> The **Sinkhorn algorithm** (also called iterative proportional fitting, or IPF) alternately enforces the two marginal constraints:
>
> 1. Initialize $b^{(0)}(y) = 1$.
> 2. For $k = 0, 1, 2, \ldots$:
>    - $a^{(k)}(x) = \mu(x) \,/\, \int K(x,y)\, b^{(k)}(y)\, dy$
>    - $b^{(k+1)}(y) = \nu(y) \,/\, \int a^{(k)}(x)\, K(x,y)\, dx$
>
> The iterates converge to the unique solution $(a^*, b^*)$ of the entropy-regularized OT problem.


For discrete distributions with $n$ support points, each iteration costs $O(n^2)$ — a matrix-vector multiplication with the kernel matrix $K_{ij} = e^{-\|x_i - y_j\|^2/\varepsilon}$. Convergence is linear, with rate depending on $\varepsilon$: smaller $\varepsilon$ (closer to exact OT) requires more iterations. The Sinkhorn algorithm was brought to machine learning by Cuturi (2013), who showed that with GPU-accelerated matrix operations, it provides practical OT distances and couplings at a fraction of the cost of exact solvers.

> [!remark] 4.5
> The Sinkhorn algorithm has a beautiful interpretation as alternating KL projections. Let $\mathcal{C}_1 = \{\pi : \pi_1 = \mu\}$ and $\mathcal{C}_2 = \{\pi : \pi_2 = \nu\}$ be the two marginal constraint sets. Starting from the reference kernel $K$, the Sinkhorn iterations alternate between projecting onto $\mathcal{C}_1$ and $\mathcal{C}_2$ in the sense of KL divergence. Each projection is a rescaling of the coupling, and the fixed point is the unique element of $\mathcal{C}_1 \cap \mathcal{C}_2$ closest to $K$ in KL — the Schrödinger bridge. This is a special case of the iterative proportional fitting procedure of Deming and Stephan (1940), applied to the Schrödinger problem by Fortet (1940) and Beurling (1960).


## 4.13 Dynamic Schrödinger Bridges

The dynamic formulation of the Schrödinger bridge — over path measures rather than couplings — leads to a system of coupled PDEs that generalizes the Benamou-Brenier formula.

The Schrödinger bridge $P^*$ is a Markov diffusion process $dX_t = f_t^*(X_t)\, dt + \sigma\, dW_t$ with drift:

> [!equation] 4.5
> $$f_t^*(x) = \sigma^2\, \nabla \log \hat{p}_t(x),$$


where $\hat{p}_t$ satisfies a forward-backward PDE system called the **Schrödinger system**:

> [!equation] 4.6
> $$\partial_t \hat{p}_t = \tfrac{\sigma^2}{2}\, \Delta \hat{p}_t, \qquad \partial_t \check{p}_t = -\tfrac{\sigma^2}{2}\, \Delta \check{p}_t,$$


with boundary conditions $\hat{p}_0 \cdot \check{p}_0 = \mu$ and $\hat{p}_1 \cdot \check{p}_1 = \nu$. The forward equation for $\hat{p}$ is the heat equation running forward in time; the backward equation for $\check{p}$ is the heat equation running backward. The marginal density of the bridge at time $t$ is their product: $\rho_t = \hat{p}_t \cdot \check{p}_t$.

> [!theorem] 4.4 — Schrödinger Bridge Drift
> The drift of the Schrödinger bridge $f_t^*(x) = \sigma^2 \nabla \log \hat{p}_t(x)$ is the unique drift that makes $dX_t = f_t^*(X_t)\,dt + \sigma\,dW_t$ have marginals satisfying $\rho_0 = \mu$, $\rho_1 = \nu$, while minimizing $\mathrm{KL}(P \| R)$ over all diffusion processes with diffusion coefficient $\sigma$.


The Schrödinger system (4.6) connects directly to the Sinkhorn algorithm: the alternating projections in Sinkhorn correspond to alternately solving the forward and backward heat equations — each projection updates one of $\hat{p}, \check{p}$ while holding the other fixed. This is the continuous-time version of iterative proportional fitting.

## 4.14 Half-Bridges and the Generative Setting

In generative modeling, we typically know the source distribution $\mu = \mathcal{N}(0, I)$ analytically but know the target distribution $\nu = p_{\text{data}}$ only through samples. This asymmetry simplifies the Schrödinger bridge to a **half-bridge**.

If $\mu = \mathcal{N}(0, I)$, the forward half $\hat{p}_0 = \mathcal{N}(0, I) / \check{p}_0$ is known once $\check{p}_0$ is determined. The problem reduces to finding $\check{p}_t$ alone — the backward component — which encodes the information about where the particles need to go. In the generative direction (sampling from $\mu$ and transporting to $\nu$), only the drift $f_t^* = \sigma^2 \nabla \log \hat{p}_t$ is needed, and this drift is what the neural network learns to approximate.

The half-bridge perspective clarifies the relationship to diffusion models. A diffusion model with VP-SDE forward process $dX_t = -\frac{1}{2}\beta_t X_t\, dt + \sqrt{\beta_t}\, dW_t$ defines a reference process $R$ that is *not* pure Brownian motion — it has a mean-reverting drift. The corresponding Schrödinger bridge with this reference process connects $p_{\text{data}}$ at $t=0$ to $\mathcal{N}(0, I)$ at $t=1$, and the reverse-time drift $-\frac{1}{2}\beta_t x + \beta_t \nabla \log p_t(x)$ is the Schrödinger bridge drift for this specific reference. Diffusion models are Schrödinger half-bridges in disguise — they use a non-trivial reference process whose bridge drift happens to involve the score function.

## 4.15 Bridge Matching

Solving the Schrödinger system (4.6) directly is intractable for high-dimensional problems. **Bridge matching** is the simulation-free approach that makes Schrödinger bridges practical for generative modeling.

The key observation is that, given a pair $(x_0, x_1) \sim \pi^*$, the conditional process $\{X_t \mid X_0 = x_0, X_1 = x_1\}$ under the Schrödinger bridge is a **Brownian bridge** — a process whose law is known analytically:

> [!equation] 4.7
> $$X_t \mid X_0 = x_0, X_1 = x_1 \;\sim\; \mathcal{N}\!\left((1-t)\, x_0 + t\, x_1,\; \sigma^2 t(1-t)\, I\right).$$


The conditional drift of this bridge is:

$$f_t^{\text{bridge}}(x \mid x_0, x_1) = \frac{x_1 - x}{1 - t},$$

which is known in closed form — it points from the current position toward $x_1$, with magnitude inversely proportional to the remaining time. This is the Brownian bridge drift: the process is pulled toward its known endpoint.

> [!definition] 4.7 — Bridge Matching Objective
> The **bridge matching objective** trains a neural drift $f_\theta(x, t)$ by regressing against the conditional bridge drift:
>
> $$\mathcal{L}_{\mathrm{BM}}(\theta) = \mathbb{E}_{t,\; (x_0, x_1) \sim \pi,\; x_t \sim p_t(\cdot | x_0, x_1)}\!\left[\left\|f_\theta(x_t, t) - \frac{x_1 - x_t}{1 - t}\right\|^2\right],$$
>
> where $x_t$ is sampled from the Brownian bridge (4.7) and $\pi$ is a coupling between $\mu$ and $\nu$.


The tower property applies exactly as in the conditional flow matching theorem (Theorem 3.1): the conditional and marginal objectives have the same gradient. The bridge matching objective is tractable — sample $(x_0, x_1)$, sample $x_t$ from the known Brownian bridge distribution, compute the closed-form target $(x_1 - x_t)/(1-t)$, and regress.

> [!remark] 4.6
> Compare the bridge matching target $(x_1 - x_t)/(1-t)$ with the flow matching target $x_1 - x_0$ (for the linear interpolant). Both point from the current state toward the data point. The bridge matching target is time-dependent and diverges as $t \to 1$ (the drift must become infinite to hit the exact endpoint), while the flow matching target is constant. The divergence at $t = 1$ is regularized by the noise: the Brownian bridge automatically concentrates at $x_1$ as $t \to 1$, so the effective force is bounded. The stochastic nature of the bridge absorbs the singularity that a deterministic flow would have to handle through the velocity field alone.


## 4.16 I$^2$SB and Iterative Bridge Refinement

A natural question arises: what coupling $\pi$ should be used in bridge matching? The optimal choice is the Schrödinger bridge coupling $\pi^*$ — but this is exactly what we are trying to learn. The resolution is an iterative procedure that alternates between learning the bridge drift and updating the coupling.

**I$^2$SB** (Image-to-Image Schrödinger Bridge, Liu et al. 2023) implements this idea:

1. **Initialize** with the independent coupling $\pi_0 = \mu \otimes \nu$ — pair noise and data randomly.
2. **Learn the drift** $f_\theta^{(k)}$ by bridge matching with coupling $\pi_k$.
3. **Update the coupling**: run the learned SDE $dX_t = f_\theta^{(k)}(X_t)\, dt + \sigma\, dW_t$ forward from $x_0 \sim \mu$ to obtain samples $\hat{x}_1$. The new coupling $\pi_{k+1}$ pairs each $x_0$ with its generated $\hat{x}_1$.
4. **Repeat** from step 2.

Each iteration is a half-IPF step: learning the drift given the coupling, then updating the coupling given the drift. The procedure converges to the Schrödinger bridge, with each iteration reducing the KL divergence to the reference process.

This iterative scheme is the stochastic analogue of the reflow procedure from Section 3.6. Reflow iterates toward the OT map (the deterministic limit); I$^2$SB iterates toward the Schrödinger bridge (the stochastic version with entropy regularization). Both start from an independent coupling and progressively learn the optimal pairing.

## 4.17 The Spectrum from Flow Matching to Diffusion

The Schrödinger bridge provides a unifying framework that places all the generative models of the preceding chapters on a single spectrum controlled by the diffusion coefficient $\sigma$.

At $\sigma = 0$ (no noise): the bridge collapses to the OT geodesic — the deterministic flow of the preceding sections. The trajectories are straight lines under the Brenier map. The model is a flow matching model with OT coupling. Generation is deterministic: each noise sample maps to exactly one output.

At intermediate $\sigma$: the bridge is a genuine Schrödinger bridge. Trajectories are stochastic — the same starting point can produce different outputs on different runs. The bridge drift steers particles toward the target while respecting the entropy budget set by $\sigma$. The coupling is softer than OT — mass splitting is allowed — and the paths are smoother in the sense that the drift is better behaved near boundaries.

At large $\sigma$: the bridge approaches pure Brownian motion with marginal constraints. The drift becomes weak relative to the noise, and the process resembles a noisy diffusion that only weakly respects the target. This is the over-regularized regime where the model generates diverse but low-quality samples.

> [!remark] 4.7
> The Schrödinger bridge perspective resolves a longstanding puzzle: why do diffusion models (which use SDEs) sometimes generate better samples than flow matching models (which use ODEs), despite flow matching having straighter paths and fewer NFE? The answer is that the SDE's noise acts as an implicit regularizer — a corrector that prevents trajectories from drifting into low-density regions where the learned velocity field is unreliable. The ODE trajectory is a razor's edge: any error in the velocity field accumulates without correction. The SDE trajectory self-corrects: noise pushes the particle back toward high-density regions where the drift is well-learned. The optimal noise level $\sigma^*$ trades off the cost of noise (more NFE, more variance) against the benefit of correction (robustness to model error).


## 4.18 Historical Notes (Schrödinger Bridges)

**Schrödinger (1931)** ("Über die Umkehrung der Naturgesetze") posed the original bridge problem as a thought experiment about the time-reversal of diffusion. **Fortet (1940)** and **Beurling (1960)** established the connection to iterative proportional fitting. The modern mathematical theory was developed by **Föllmer (1988)** and systematized by **Léonard (2014)** ("A survey of the Schrödinger problem and some of its connections with optimal transport"), who clarified the convergence of Schrödinger bridges to OT maps as $\varepsilon \to 0$.

**Cuturi (2013)** ("Sinkhorn Distances: Lightspeed Computation of Optimal Transport") introduced the Sinkhorn algorithm to machine learning, enabling efficient computation of entropy-regularized OT. **Peyré and Cuturi (2019)** provide the comprehensive computational treatment.

The application to generative modeling began with **De Bortoli, Thornton, Heng, and Doucet (2021)** ("Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling"), who showed that Schrödinger bridges can be learned by alternating score matching and diffusion simulation. **Vargas, Nüsken, and Richter (2021)** ("Solving Schrödinger Bridges via Maximum Likelihood") proposed a maximum-likelihood approach. **Chen, Conforti, and Georgiou (2021)** developed the stochastic control perspective. **Liu, Gat, Polyak, et al. (2023)** ("I$^2$SB: Image-to-Image Schrödinger Bridge") demonstrated practical image-to-image translation using iterative bridge refinement. **Shi, De Bortoli, Campbell, and Doucet (2024)** ("Diffusion Schrödinger Bridge Matching") introduced the bridge matching framework that makes training tractable. **Peluchetti (2023)** ("Diffusion Bridge Mixture Transports, Schrödinger Bridge Problems and Generative Modeling") clarified the theoretical connections between bridges, diffusion models, and flow matching.

The connection between Schrödinger bridges and iterative proportional fitting was formalized by **Ruschendorf (1995)** and brought to the generative modeling setting by **De Bortoli et al. (2021)**. The interpretation of diffusion models as Schrödinger half-bridges appears in **Huang, Jiao, Shi, and Kang (2021)** and was further developed by **Chen and Lipman (2024)**.

## References

- [Cuturi (2013)](https://arxiv.org/abs/1306.0895) — Sinkhorn Distances: Lightspeed Computation of Optimal Transport
- [Peyré and Cuturi (2019)](https://arxiv.org/abs/1803.00567) — Computational Optimal Transport
- [De Bortoli, Thornton, Heng, and Doucet (2021)](https://arxiv.org/abs/2106.01357) — Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling
- [Vargas, Nüsken, and Richter (2021)](https://arxiv.org/abs/2106.04379) — Solving Schrödinger Bridges via Maximum Likelihood
- [Liu, Gat, Polyak, et al. (2023)](https://arxiv.org/abs/2303.07845) — I²SB: Image-to-Image Schrödinger Bridge
- [Shi, De Bortoli, Campbell, and Doucet (2024)](https://arxiv.org/abs/2309.01099) — Diffusion Schrödinger Bridge Matching
