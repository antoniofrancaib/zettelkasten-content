---

title: "The Most Likely Path"
subtitle: "Entropy-regularized optimal transport, iterative proportional fitting, bridge matching, and the stochastic middle ground — 1931 to 2025"
---



In 1931, Erwin Schrödinger posed a question that appeared, at the time, to have nothing to do with generative modeling. He considered a large number of particles observed to have distribution $\mu$ at time $0$ and distribution $\nu$ at time $1$, and asked: given that the particles undergo Brownian motion between the two observations, what is the *most likely* stochastic evolution connecting $\mu$ to $\nu$? The answer — the Schrödinger bridge — is optimal transport with an entropy penalty. It produces stochastic rather than deterministic trajectories, and it interpolates continuously between two extremes: pure Brownian motion (maximum entropy, no transport structure) and the deterministic OT map (minimum entropy, maximum transport efficiency). For ninety years this was a curiosity of mathematical physics. Then, in 2021, it became a generative model.

The central insight of this chapter is that the Schrödinger bridge is the *natural* stochastic extension of the optimal transport problem of Chapter 5. Where OT finds the deterministic map that moves mass at minimum cost, the Schrödinger bridge finds the stochastic process that moves mass at minimum cost while maintaining a prescribed level of randomness. The regularization parameter $\varepsilon$ controls the tradeoff: at $\varepsilon = 0$ we recover the OT geodesic; at $\varepsilon \to \infty$ we recover unstructured Brownian motion. The generative models of Chapters 3 and 4 — diffusion and flow matching — sit at opposite ends of this spectrum, and Schrödinger bridges provide the unified theory that connects them.

## 6.1 The Schrödinger Bridge Problem

Let $R$ be a **reference process** — a probability measure over paths $\{X_t\}_{t \in [0,1]}$ in $\mathbb{R}^d$. The canonical choice is the Wiener measure (Brownian motion) with diffusion coefficient $\sigma$: $dX_t = \sigma\, dW_t$. The Schrödinger bridge problem asks for the path measure $P^*$ that is closest to $R$ in KL divergence while satisfying the marginal constraints:

> [!definition] 6.1 — Schrödinger Bridge Problem
> Given a reference process $R$ and marginal distributions $\mu$, $\nu$, the **Schrödinger bridge** is:
>
> $$P^* = \arg\min_{P:\, P_0 = \mu,\; P_1 = \nu} \mathrm{KL}(P \| R),$$
>
> where $P_0$ and $P_1$ denote the marginal distributions of $P$ at times $0$ and $1$.


When $R$ is Brownian motion, the Schrödinger bridge $P^*$ is itself a diffusion process $dX_t = f_t^*(X_t)\, dt + \sigma\, dW_t$ — it has the same noise as Brownian motion but acquires a drift $f_t^*$ that steers the particles from $\mu$ to $\nu$. The KL objective penalizes deviations from the reference: the drift is the minimum correction to Brownian motion that achieves the desired marginals. Particles still diffuse randomly, but they diffuse with purpose.

> [!remark] 6.1
> The Schrödinger bridge has a compelling physical interpretation. Imagine $10^{23}$ particles undergoing Brownian motion. At time $0$ they are distributed as $\mu$; at time $1$ they are observed to be distributed as $\nu$. The Schrödinger bridge is the conditional law of the process given both endpoint observations — the most likely interpolation between the two snapshots. It is the answer to: "given what we see at the beginning and end, what most likely happened in between?"


## 6.2 Entropy-Regularized Optimal Transport

The Schrödinger bridge has an equivalent **static** formulation in terms of couplings, which connects it directly to the Kantorovich problem of Chapter 5.

Let $R_{01}$ denote the joint distribution of $(X_0, X_1)$ under the reference process $R$ — for Brownian motion with diffusion $\sigma$, this is $R_{01}(x, y) = \mu(x) \cdot \mathcal{N}(y; x, \sigma^2 I)$. The static Schrödinger bridge seeks the coupling $\pi^*$ closest to $R_{01}$ in KL divergence:

> [!equation] 6.1
> $$\pi^* = \arg\min_{\pi \in \Gamma(\mu, \nu)} \mathrm{KL}(\pi \| R_{01}).$$


Expanding the KL divergence, this becomes:

$$\pi^* = \arg\min_{\pi \in \Gamma(\mu, \nu)} \int \log\frac{\pi(x,y)}{R_{01}(x,y)}\, d\pi(x,y) = \arg\min_{\pi \in \Gamma(\mu,\nu)} \left[\frac{1}{\sigma^2}\int \|x-y\|^2\, d\pi - \sigma^2\, H(\pi)\right] + \text{const},$$

where $H(\pi) = -\int \pi \log \pi$ is the entropy of the coupling. Up to a scaling, this is precisely the **entropy-regularized optimal transport** problem:

> [!definition] 6.2 — Entropy-Regularized OT
> The **entropy-regularized optimal transport** problem with regularization $\varepsilon > 0$ is:
>
> $$\pi^*_\varepsilon = \arg\min_{\pi \in \Gamma(\mu, \nu)} \int \|x - y\|^2\, d\pi(x, y) - \varepsilon\, H(\pi).$$
>
> As $\varepsilon \to 0$, $\pi^*_\varepsilon$ converges to the OT coupling $\pi_{\mathrm{OT}}$. As $\varepsilon \to \infty$, $\pi^*_\varepsilon$ converges to the independent coupling $\mu \otimes \nu$.


The regularization parameter $\varepsilon = \sigma^2$ is the squared diffusion coefficient. Large noise means large entropy regularization, which pushes the coupling toward independence (Brownian motion doesn't care about the target). Small noise means small regularization, which pushes the coupling toward the OT solution (particles take the shortest path). The Schrödinger bridge interpolates between these extremes.

## 6.3 The Sinkhorn Algorithm and Iterative Proportional Fitting

The entropy-regularized problem has a key computational advantage over exact OT: its solution has the factored form $\pi^*(x, y) = a(x)\, K(x, y)\, b(y)$, where $K(x, y) = e^{-\|x-y\|^2/\varepsilon}$ is the Gibbs kernel. The marginal constraints $\pi^*_1 = \mu$, $\pi^*_2 = \nu$ then determine $a$ and $b$ through the coupled equations:

$$a(x) = \frac{\mu(x)}{\int K(x,y)\, b(y)\, dy}, \qquad b(y) = \frac{\nu(y)}{\int a(x)\, K(x,y)\, dx}.$$

> [!definition] 6.3 — Sinkhorn Algorithm
> The **Sinkhorn algorithm** (also called iterative proportional fitting, or IPF) alternately enforces the two marginal constraints:
>
> 1. Initialize $b^{(0)}(y) = 1$.
> 2. For $k = 0, 1, 2, \ldots$:
>    - $a^{(k)}(x) = \mu(x) \,/\, \int K(x,y)\, b^{(k)}(y)\, dy$
>    - $b^{(k+1)}(y) = \nu(y) \,/\, \int a^{(k)}(x)\, K(x,y)\, dx$
>
> The iterates converge to the unique solution $(a^*, b^*)$ of the entropy-regularized OT problem.


For discrete distributions with $n$ support points, each iteration costs $O(n^2)$ — a matrix-vector multiplication with the kernel matrix $K_{ij} = e^{-\|x_i - y_j\|^2/\varepsilon}$. Convergence is linear, with rate depending on $\varepsilon$: smaller $\varepsilon$ (closer to exact OT) requires more iterations. The Sinkhorn algorithm was brought to machine learning by Cuturi (2013), who showed that with GPU-accelerated matrix operations, it provides practical OT distances and couplings at a fraction of the cost of exact solvers.

> [!remark] 6.2
> The Sinkhorn algorithm has a beautiful interpretation as alternating KL projections. Let $\mathcal{C}_1 = \{\pi : \pi_1 = \mu\}$ and $\mathcal{C}_2 = \{\pi : \pi_2 = \nu\}$ be the two marginal constraint sets. Starting from the reference kernel $K$, the Sinkhorn iterations alternate between projecting onto $\mathcal{C}_1$ and $\mathcal{C}_2$ in the sense of KL divergence. Each projection is a rescaling of the coupling, and the fixed point is the unique element of $\mathcal{C}_1 \cap \mathcal{C}_2$ closest to $K$ in KL — the Schrödinger bridge. This is a special case of the iterative proportional fitting procedure of Deming and Stephan (1940), applied to the Schrödinger problem by Fortet (1940) and Beurling (1960).


## 6.4 Dynamic Schrödinger Bridges

The dynamic formulation of the Schrödinger bridge — over path measures rather than couplings — leads to a system of coupled PDEs that generalizes the Benamou-Brenier formula.

The Schrödinger bridge $P^*$ is a Markov diffusion process $dX_t = f_t^*(X_t)\, dt + \sigma\, dW_t$ with drift:

> [!equation] 6.2
> $$f_t^*(x) = \sigma^2\, \nabla \log \hat{p}_t(x),$$


where $\hat{p}_t$ satisfies a forward-backward PDE system called the **Schrödinger system**:

> [!equation] 6.3
> $$\partial_t \hat{p}_t = \tfrac{\sigma^2}{2}\, \Delta \hat{p}_t, \qquad \partial_t \check{p}_t = -\tfrac{\sigma^2}{2}\, \Delta \check{p}_t,$$


with boundary conditions $\hat{p}_0 \cdot \check{p}_0 = \mu$ and $\hat{p}_1 \cdot \check{p}_1 = \nu$. The forward equation for $\hat{p}$ is the heat equation running forward in time; the backward equation for $\check{p}$ is the heat equation running backward. The marginal density of the bridge at time $t$ is their product: $\rho_t = \hat{p}_t \cdot \check{p}_t$.

> [!theorem] 6.1 — Schrödinger Bridge Drift
> The drift of the Schrödinger bridge $f_t^*(x) = \sigma^2 \nabla \log \hat{p}_t(x)$ is the unique drift that makes $dX_t = f_t^*(X_t)\,dt + \sigma\,dW_t$ have marginals satisfying $\rho_0 = \mu$, $\rho_1 = \nu$, while minimizing $\mathrm{KL}(P \| R)$ over all diffusion processes with diffusion coefficient $\sigma$.


The Schrödinger system (6.3) connects directly to the Sinkhorn algorithm: the alternating projections in Sinkhorn correspond to alternately solving the forward and backward heat equations — each projection updates one of $\hat{p}, \check{p}$ while holding the other fixed. This is the continuous-time version of iterative proportional fitting.

## 6.5 Half-Bridges and the Generative Setting

In generative modeling, we typically know the source distribution $\mu = \mathcal{N}(0, I)$ analytically but know the target distribution $\nu = p_{\text{data}}$ only through samples. This asymmetry simplifies the Schrödinger bridge to a **half-bridge**.

If $\mu = \mathcal{N}(0, I)$, the forward half $\hat{p}_0 = \mathcal{N}(0, I) / \check{p}_0$ is known once $\check{p}_0$ is determined. The problem reduces to finding $\check{p}_t$ alone — the backward component — which encodes the information about where the particles need to go. In the generative direction (sampling from $\mu$ and transporting to $\nu$), only the drift $f_t^* = \sigma^2 \nabla \log \hat{p}_t$ is needed, and this drift is what the neural network learns to approximate.

The half-bridge perspective clarifies the relationship to diffusion models. A diffusion model with VP-SDE forward process $dX_t = -\frac{1}{2}\beta_t X_t\, dt + \sqrt{\beta_t}\, dW_t$ defines a reference process $R$ that is *not* pure Brownian motion — it has a mean-reverting drift. The corresponding Schrödinger bridge with this reference process connects $p_{\text{data}}$ at $t=0$ to $\mathcal{N}(0, I)$ at $t=1$, and the reverse-time drift $-\frac{1}{2}\beta_t x + \beta_t \nabla \log p_t(x)$ is the Schrödinger bridge drift for this specific reference. Diffusion models are Schrödinger half-bridges in disguise — they use a non-trivial reference process whose bridge drift happens to involve the score function.

## 6.6 Bridge Matching

Solving the Schrödinger system (6.3) directly is intractable for high-dimensional problems. **Bridge matching** (also called score-based Schrödinger bridge matching) is the simulation-free approach that makes Schrödinger bridges practical for generative modeling.

The key observation is that, given a pair $(x_0, x_1) \sim \pi^*$, the conditional process $\{X_t \mid X_0 = x_0, X_1 = x_1\}$ under the Schrödinger bridge is a **Brownian bridge** — a process whose law is known analytically:

> [!equation] 6.4
> $$X_t \mid X_0 = x_0, X_1 = x_1 \;\sim\; \mathcal{N}\!\left((1-t)\, x_0 + t\, x_1,\; \sigma^2 t(1-t)\, I\right).$$


The conditional drift of this bridge is:

$$f_t^{\text{bridge}}(x \mid x_0, x_1) = \frac{x_1 - x}{1 - t},$$

which is known in closed form — it points from the current position toward $x_1$, with magnitude inversely proportional to the remaining time. This is the Brownian bridge drift: the process is pulled toward its known endpoint.

> [!definition] 6.4 — Bridge Matching Objective
> The **bridge matching objective** trains a neural drift $f_\theta(x, t)$ by regressing against the conditional bridge drift:
>
> $$\mathcal{L}_{\mathrm{BM}}(\theta) = \mathbb{E}_{t,\; (x_0, x_1) \sim \pi,\; x_t \sim p_t(\cdot | x_0, x_1)}\!\left[\left\|f_\theta(x_t, t) - \frac{x_1 - x_t}{1 - t}\right\|^2\right],$$
>
> where $x_t$ is sampled from the Brownian bridge (6.4) and $\pi$ is a coupling between $\mu$ and $\nu$.


The tower property applies exactly as in the conditional flow matching theorem (Theorem 4.1): the conditional and marginal objectives have the same gradient. The bridge matching objective is tractable — sample $(x_0, x_1)$, sample $x_t$ from the known Brownian bridge distribution, compute the closed-form target $(x_1 - x_t)/(1-t)$, and regress.

> [!remark] 6.3
> Compare the bridge matching target $(x_1 - x_t)/(1-t)$ with the flow matching target $x_1 - x_0$ (for the linear interpolant). Both point from the current state toward the data point. The bridge matching target is time-dependent and diverges as $t \to 1$ (the drift must become infinite to hit the exact endpoint), while the flow matching target is constant. The divergence at $t = 1$ is regularized by the noise: the Brownian bridge automatically concentrates at $x_1$ as $t \to 1$, so the effective force is bounded. The stochastic nature of the bridge absorbs the singularity that a deterministic flow would have to handle through the velocity field alone.


## 6.7 I$^2$SB and Iterative Bridge Refinement

A natural question arises: what coupling $\pi$ should be used in bridge matching? The optimal choice is the Schrödinger bridge coupling $\pi^*$ — but this is exactly what we are trying to learn. The resolution is an iterative procedure that alternates between learning the bridge drift and updating the coupling.

**I$^2$SB** (Image-to-Image Schrödinger Bridge, Liu et al. 2023) implements this idea:

1. **Initialize** with the independent coupling $\pi_0 = \mu \otimes \nu$ — pair noise and data randomly.
2. **Learn the drift** $f_\theta^{(k)}$ by bridge matching with coupling $\pi_k$.
3. **Update the coupling**: run the learned SDE $dX_t = f_\theta^{(k)}(X_t)\, dt + \sigma\, dW_t$ forward from $x_0 \sim \mu$ to obtain samples $\hat{x}_1$. The new coupling $\pi_{k+1}$ pairs each $x_0$ with its generated $\hat{x}_1$.
4. **Repeat** from step 2.

Each iteration is a half-IPF step: learning the drift given the coupling, then updating the coupling given the drift. The procedure converges to the Schrödinger bridge, with each iteration reducing the KL divergence to the reference process.

This iterative scheme is the stochastic analogue of the reflow procedure from Section 4.6. Reflow iterates toward the OT map (the deterministic limit); I$^2$SB iterates toward the Schrödinger bridge (the stochastic version with entropy regularization). Both start from an independent coupling and progressively learn the optimal pairing.

## 6.8 The Spectrum from Flow Matching to Diffusion

The Schrödinger bridge provides a unifying framework that places all the generative models of the preceding chapters on a single spectrum controlled by the diffusion coefficient $\sigma$.

At $\sigma = 0$ (no noise): the bridge collapses to the OT geodesic — the deterministic flow of Chapter 5. The trajectories are straight lines under the Brenier map. The model is a flow matching model with OT coupling. Generation is deterministic: each noise sample maps to exactly one output.

At intermediate $\sigma$: the bridge is a genuine Schrödinger bridge. Trajectories are stochastic — the same starting point can produce different outputs on different runs. The bridge drift steers particles toward the target while respecting the entropy budget set by $\sigma$. The coupling is softer than OT — mass splitting is allowed — and the paths are smoother in the sense that the drift is better behaved near boundaries.

At large $\sigma$: the bridge approaches pure Brownian motion with marginal constraints. The drift becomes weak relative to the noise, and the process resembles a noisy diffusion that only weakly respects the target. This is the over-regularized regime where the model generates diverse but low-quality samples.

> [!remark] 6.4
> The Schrödinger bridge perspective resolves a longstanding puzzle: why do diffusion models (which use SDEs) sometimes generate better samples than flow matching models (which use ODEs), despite flow matching having straighter paths and fewer NFE? The answer is that the SDE's noise acts as an implicit regularizer — a corrector that prevents trajectories from drifting into low-density regions where the learned velocity field is unreliable. The ODE trajectory is a razor's edge: any error in the velocity field accumulates without correction. The SDE trajectory self-corrects: noise pushes the particle back toward high-density regions where the drift is well-learned. The optimal noise level $\sigma^*$ trades off the cost of noise (more NFE, more variance) against the benefit of correction (robustness to model error).


## 6.9 Historical Notes

**Schrödinger (1931)** ("Über die Umkehrung der Naturgesetze") posed the original bridge problem as a thought experiment about the time-reversal of diffusion. **Fortet (1940)** and **Beurling (1960)** established the connection to iterative proportional fitting. The modern mathematical theory was developed by **Föllmer (1988)** and systematized by **Léonard (2014)** ("A survey of the Schrödinger problem and some of its connections with optimal transport"), who clarified the convergence of Schrödinger bridges to OT maps as $\varepsilon \to 0$.

**Cuturi (2013)** ("Sinkhorn Distances: Lightspeed Computation of Optimal Transport") introduced the Sinkhorn algorithm to machine learning, enabling efficient computation of entropy-regularized OT. **Peyré and Cuturi (2019)** provide the comprehensive computational treatment.

The application to generative modeling began with **De Bortoli, Thornton, Heng, and Doucet (2021)** ("Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling"), who showed that Schrödinger bridges can be learned by alternating score matching and diffusion simulation. **Vargas, Nüsken, and Richter (2021)** ("Solving Schrödinger Bridges via Maximum Likelihood") proposed a maximum-likelihood approach. **Chen, Conforti, and Georgiou (2021)** developed the stochastic control perspective. **Liu, Gat, Polyak, et al. (2023)** ("I$^2$SB: Image-to-Image Schrödinger Bridge") demonstrated practical image-to-image translation using iterative bridge refinement. **Shi, De Bortoli, Campbell, and Doucet (2024)** ("Diffusion Schrödinger Bridge Matching") introduced the bridge matching framework that makes training tractable. **Peluchetti (2023)** ("Diffusion Bridge Mixture Transports, Schrödinger Bridge Problems and Generative Modeling") clarified the theoretical connections between bridges, diffusion models, and flow matching.

The connection between Schrödinger bridges and iterative proportional fitting was formalized by **Ruschendorf (1995)** and brought to the generative modeling setting by **De Bortoli et al. (2021)**. The interpretation of diffusion models as Schrödinger half-bridges appears in **Huang, Jiao, Shi, and Kang (2021)** and was further developed by **Chen and Lipman (2024)**.
