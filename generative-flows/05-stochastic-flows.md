---

title: "Stochastic Flows"
subtitle: "When and why stochasticity helps — Langevin correctors, drift-based generation, and the design of noise — 2020 to 2025"
---



Chapter 2 introduced stochastic differential equations as the mathematical foundation of diffusion models: the forward SDE destroys structure, the reverse SDE recovers it, and the score function is the bridge between the two. Chapter 3 showed that for generation one can often replace the SDE with a deterministic ODE — the probability flow ODE — that shares the same marginal distributions but requires no random noise. Chapter 4 placed both on a spectrum via the Schrödinger bridge, parameterized by the diffusion coefficient. This chapter develops the question that all three raise but none fully answer: given that both stochastic and deterministic dynamics can generate from the same distribution, *when should you use noise, and how much*?

The answer is more subtle than "ODEs are faster, SDEs are more robust." The duality between SDEs and ODEs is exact at the population level — they share marginals — but approximate in practice, where the vector field is imperfectly learned and the integration is discretized. In this practical regime, noise plays three distinct roles: it corrects errors in the learned field, it provides mode coverage in multimodal distributions, and it acts as a regularizer that smooths the sampling path. Understanding these roles, and designing the noise schedule accordingly, is the subject of this chapter.

## 5.1 Forward SDEs: Noise as a Design Choice

The general forward SDE on $\mathbb{R}^d$ takes the form:

> [!equation] 5.1
> $$dX_t = f_t(X_t)\, dt + g_t\, dW_t, \qquad X_0 \sim p_0,$$


where $f_t : \mathbb{R}^d \to \mathbb{R}^d$ is the **drift**, $g_t \in \mathbb{R}$ is the **diffusion coefficient**, and $W_t$ is a standard Wiener process. The marginal density $p_t$ of $X_t$ evolves according to the **Fokker-Planck equation**:

> [!equation] 5.2
> $$\frac{\partial p_t}{\partial t} = -\nabla \cdot (f_t\, p_t) + \frac{g_t^2}{2}\, \Delta p_t.$$


The two terms have complementary roles. The drift term $-\nabla \cdot (f_t p_t)$ is the continuity equation — it transports probability mass along $f_t$ without creating or destroying it. The diffusion term $\frac{g_t^2}{2}\Delta p_t$ is the heat operator — it spreads probability mass, smoothing sharp features and filling in gaps. Together they produce a probability path $\{p_t\}$ that is both transported and smoothed.

The drift $f_t$ and diffusion coefficient $g_t$ are *design choices*, not fixed by the problem. Different choices produce different probability paths, different training objectives, and different sampling properties:

- **Pure drift** ($g_t = 0$): the SDE reduces to an ODE $\dot{X}_t = f_t(X_t)$. No noise, no smoothing. This is the flow matching setting.
- **Pure diffusion** ($f_t = 0$): the SDE is Brownian motion. Maximum noise, no directed transport. This is the reference process of the Schrödinger bridge.
- **VP-SDE** ($f_t(x) = -\frac{1}{2}\beta_t x$, $g_t = \sqrt{\beta_t}$): the Ornstein-Uhlenbeck process that drives diffusion models. The drift pushes $X_t$ toward the origin while the noise inflates it — the balance produces a process that converges to $\mathcal{N}(0, I)$.
- **VE-SDE** ($f_t = 0$, $g_t = \sqrt{d[\sigma^2(t)]/dt}$): variance-exploding noise that progressively corrupts the signal without shifting the mean.

> [!remark] 5.1
> The design space of $(f_t, g_t)$ is enormous — any pair that produces a well-defined process and ends near the target distribution is a valid choice. The challenge is not finding *a* valid choice but finding the *best* one for a given application. "Best" depends on the evaluation metric: fewest NFE at inference, lowest FID, highest log-likelihood, fastest training convergence. The VP-SDE and VE-SDE were not derived from an optimality principle — they were chosen by analogy with physics (Ornstein-Uhlenbeck, heat diffusion) and shown empirically to work well. Whether they are optimal in any formal sense remains open.


## 5.2 The Reverse-Time SDE

The fundamental result that makes SDE-based generation possible is Anderson's theorem (1982): every forward SDE (5.1) has a time-reversed counterpart whose marginals trace the same path $\{p_t\}$ in reverse.

> [!theorem] 5.1 — Anderson's Reverse-Time SDE
> If $\{X_t\}_{t \in [0,1]}$ satisfies the forward SDE (5.1), then the time-reversed process $\{Y_t\}$ with $Y_t = X_{1-t}$ satisfies:
>
> $$dY_t = \left[-f_{1-t}(Y_t) + g_{1-t}^2\, \nabla \log p_{1-t}(Y_t)\right] dt + g_{1-t}\, d\bar{W}_t,$$
>
> where $\bar{W}_t$ is a standard Wiener process (independent of $W_t$) and $p_t$ is the marginal density at time $t$.


The reverse drift has two components: $-f_{1-t}$ reverses the original drift, and $g_{1-t}^2 \nabla \log p_{1-t}$ is the **score correction** — a drift toward high-density regions of $p_t$ that compensates for the entropy production of the forward diffusion. The diffusion coefficient $g_{1-t}$ is unchanged — the same amount of noise is injected forward and backward.

The score function $\nabla \log p_t(x)$ is the only quantity that needs to be learned; everything else ($f_t$, $g_t$) is known by design. This is the key structural insight: the forward process defines the corruption, the reverse process defines the generation, and the score is the only learned component.

## 5.3 The Probability Flow ODE: A Deterministic Mirror

For every SDE (5.1), there exists a deterministic ODE whose marginals $\{p_t\}$ are identical:

> [!theorem] 5.2 — Probability Flow ODE
> The ODE
>
> $$\frac{dx}{dt} = f_t(x) - \frac{g_t^2}{2}\, \nabla \log p_t(x)$$
>
> has the same marginal densities $\{p_t\}$ as the SDE (5.1). That is, if $X_0 \sim p_0$ and $x_0 \sim p_0$, and $X_t$ evolves by (5.1) while $x_t$ evolves by the ODE, then $X_t \sim p_t$ and $x_t \sim p_t$ for all $t$.


The proof is direct: substitute the ODE velocity field $v_t(x) = f_t(x) - \frac{g_t^2}{2}\nabla \log p_t(x)$ into the continuity equation $\partial_t p_t + \nabla \cdot (p_t v_t) = 0$ and verify, using the identity $\nabla \cdot (p_t \nabla \log p_t) = \Delta p_t$, that it reduces to the Fokker-Planck equation (5.2). The diffusion term $\frac{g_t^2}{2}\Delta p_t$ is absorbed into the drift through the score.

> [!remark] 5.2
> The probability flow ODE is not unique: there are infinitely many ODEs with the same marginals (any divergence-free perturbation of the velocity field preserves the marginals). The probability flow ODE is the specific choice obtained by "determinizing" the SDE — replacing the stochastic noise with its expected effect on the density. It is the minimum-norm velocity field consistent with the Fokker-Planck evolution, and for this reason it produces the straightest deterministic trajectories compatible with the SDE's marginal path.


## 5.4 When Stochasticity Helps: Three Mechanisms

The SDE and its probability flow ODE share marginals, but their *trajectories* are fundamentally different. The ODE maps each initial point $x_0$ to a unique output $x_1 = \phi_1(x_0)$ via a deterministic map. The SDE maps $x_0$ to a distribution over outputs — different noise realizations produce different $x_1$. This difference has three practical consequences.

**Error correction.** In practice, the score $\nabla \log p_t$ is approximated by a neural network $s_\theta(x, t)$. The learned field has errors, especially in low-density regions where training data is sparse. Under the ODE, errors accumulate: a small velocity error at time $t$ displaces the trajectory, which then encounters a different (possibly worse) velocity field, compounding the error. Under the SDE, noise provides a correction mechanism: even if the drift pushes the particle into a low-density region, the diffusion can nudge it back toward high-density regions where the score is reliable.

> [!equation] 5.3
> $$\text{ODE error:} \quad \|x_1^{\text{approx}} - x_1^{\text{exact}}\| \sim O(\delta \cdot e^{L \cdot T}), \qquad \text{SDE: mixing provides $O(\sqrt{\delta})$ corrections}$$


where $\delta$ is the local score error and $L$ is the Lipschitz constant of the velocity field. The ODE error grows exponentially with the Lipschitz constant (a measure of trajectory sensitivity), while the SDE error is moderated by the mixing effect of diffusion.

**Mode coverage.** For multimodal target distributions, the ODE must route every trajectory to the correct mode — a precise assignment from noise to data that the velocity field must encode exactly. If the field is slightly wrong near a mode boundary, a trajectory destined for mode $A$ can end up in mode $B$. The SDE has a softer requirement: the noise allows late-stage mode switching. Even after the drift has committed to one mode, the Brownian motion can push the particle to a neighboring mode. This produces better mode coverage at the cost of slightly less precise samples within each mode.

**Trajectory smoothness.** The ODE velocity field must do all the work of transporting particles — including resolving fine-scale structure in the target distribution. This can create regions of high curvature in the velocity field, requiring many ODE steps. The SDE offloads part of this work to the noise: the drift handles the large-scale transport, and the diffusion handles the fine-scale detail. The result is a smoother drift that is easier to integrate with fewer steps, even though each step is stochastic.

## 5.5 Stochastic Interpolants with Non-Zero Diffusion

The stochastic interpolant framework of Section 3.5 can be extended to produce SDEs rather than ODEs. Recall that the interpolant $x_t = \alpha_t x_0 + \beta_t x_1 + \sigma_t z$ with $z \sim \mathcal{N}(0, I)$ defines a probability path $\{p_t\}$. When $\sigma_t = 0$, the path is deterministic and the target velocity field is the conditional expectation of $\dot{x}_t$. When $\sigma_t > 0$, the path has a stochastic component, and the natural dynamics are an SDE.

> [!definition] 5.1 — Stochastic Interpolant SDE
> Given a stochastic interpolant $x_t = \alpha_t x_0 + \beta_t x_1 + \sigma_t z$ with $\sigma_t > 0$ for $t \in (0,1)$, the associated SDE is:
>
> $$dX_t = b_t(X_t)\, dt + \eta_t\, dW_t,$$
>
> where $b_t$ is the **drift** and $\eta_t$ is the **diffusion coefficient**, related to the interpolant parameters by:
>
> $$\eta_t^2 = 2\sigma_t \dot{\sigma}_t + \eta_t^2, \qquad b_t(x) = \mathbb{E}[\dot{\alpha}_t x_0 + \dot{\beta}_t x_1 + \dot{\sigma}_t z \mid x_t = x] + \frac{\eta_t^2 - 2\sigma_t\dot{\sigma}_t}{2}\nabla\log p_t(x).$$


The drift decomposes into two parts: a **transport term** (the conditional expectation, which is what the ODE velocity field would be) and a **score term** (proportional to $\nabla \log p_t$, which accounts for the additional diffusion). The diffusion coefficient $\eta_t$ can be chosen independently of the interpolant noise $\sigma_t$, giving an extra degree of freedom: even for the same probability path $\{p_t\}$, different choices of $\eta_t$ produce different SDEs with different trajectory properties.

> [!remark] 5.3
> The decomposition of the drift into transport and score terms is a general fact: any SDE with a given marginal path can be written as the probability flow ODE drift plus a score correction. The transport term moves mass according to the path; the score term compensates for the discrepancy between the actual diffusion coefficient and what the path requires. Training objective: regress $b_\theta(x_t, t)$ against the conditional drift $\dot{\alpha}_t x_0 + \dot{\beta}_t x_1 + \dot{\sigma}_t z$ plus the score correction — or equivalently, jointly learn the transport and score using a combined loss.


## 5.6 Langevin Correctors

The simplest way to inject stochasticity into a deterministic sampler is the **predictor-corrector** scheme: alternate between ODE prediction steps (which advance along the learned velocity field) and Langevin correction steps (which draw samples toward the true distribution).

> [!definition] 5.2 — Langevin Corrector Step
> Given a current sample $x$ at time $t$ with marginal density $p_t$, a **Langevin corrector step** with step size $\epsilon$ updates:
>
> $$x \leftarrow x + \epsilon\, \nabla \log p_t(x) + \sqrt{2\epsilon}\, z, \qquad z \sim \mathcal{N}(0, I).$$
>
> This is one step of the (unadjusted) Langevin algorithm targeting $p_t$. In the limit of many steps with $\epsilon \to 0$, the iterates converge to an exact sample from $p_t$.


The predictor-corrector (PC) scheme of Song et al. (2021) works as follows: at each time step $t_k$, first advance from $t_k$ to $t_{k+1}$ using the probability flow ODE (the predictor), then run $M$ Langevin correction steps at $t_{k+1}$ (the corrector). The predictor moves the sample to the right neighborhood; the corrector refines it to match the local density.

The corrector does not need to converge fully — a few steps suffice to reduce the accumulated error from the predictor. The effective number of function evaluations per time step is $1 + M$: one for the ODE step and $M$ for the Langevin steps. The tradeoff is clear: more corrector steps mean higher sample quality but more NFE.

**Metropolis-adjusted Langevin (MALA).** The unadjusted Langevin corrector has a bias — it only converges to $p_t$ in the limit of vanishing step size. The Metropolis-adjusted version adds an accept-reject step:

1. Propose $x' = x + \epsilon\, \nabla \log p_t(x) + \sqrt{2\epsilon}\, z$.
2. Accept with probability $\min\!\left(1,\; \frac{p_t(x')\, q(x|x')}{p_t(x)\, q(x'|x)}\right)$, where $q$ is the proposal kernel.

MALA has no bias regardless of step size, at the cost of requiring $p_t(x)$ up to a normalizing constant and one additional score evaluation per step. In practice, the acceptance rate is high (>90%) for well-chosen $\epsilon$, and the bias correction is worth the extra cost.

## 5.7 Generative Modeling via Drifting

A complementary perspective inverts the usual relationship: instead of starting with a probability path and deriving the SDE, start with a *drift function* and let the SDE define the path.

> [!definition] 5.3 — Drift-Based Generation
> In **drift-based generation**, the generative process is defined by an SDE:
>
> $$dX_t = f_\theta(X_t, t)\, dt + g_t\, dW_t, \qquad X_0 \sim p_0,$$
>
> where the drift $f_\theta$ is a neural network and the diffusion schedule $g_t$ is fixed by design. The model is trained so that the terminal distribution $p_1$ of this SDE matches the target $p_{\text{data}}$.


This is the most direct formulation of the generative problem as an SDE: learn the drift, fix the noise, simulate to generate. The training objective can be derived from several principles:

**Score matching.** If we require $p_1 = p_{\text{data}}$, we can train $f_\theta$ to match the time-reversed drift $-f_t + g_t^2 \nabla \log p_t$. This reduces to learning the score $\nabla \log p_t$ at each time — the standard diffusion model training.

**Bridge matching.** If paired data $(x_0, x_1)$ is available, we can use the bridge matching objective (Definition 4.4) to regress $f_\theta$ against the conditional bridge drift.

**Annealed drift.** A design choice specific to drift-based generation is to *anneal* the diffusion coefficient from high to low during generation:

$$g_t = g_{\max}(1-t) + g_{\min}\, t,$$

starting with high noise (broad exploration) and ending with low noise (precise targeting). This annealing schedule means the SDE explores widely at early times — visiting multiple modes, covering the full support of $p_{\text{data}}$ — and then narrows down to precise samples as $t \to 1$. The drift $f_\theta$ adapts to the noise level: at high noise it performs coarse transport; at low noise it performs fine refinement.

> [!remark] 5.4
> Drift-based generation with noise annealing is closely related to the annealed Langevin dynamics of NCSN (Song and Ermon, 2019), but formulated as a continuous SDE rather than a sequence of discrete Langevin chains. The continuous formulation has two advantages: it admits a well-defined reverse SDE (enabling exact log-likelihood computation via the Fokker-Planck equation) and it can be integrated with adaptive ODE/SDE solvers that choose the step size automatically. The NCSN approach, which runs separate Langevin chains at each noise level, is a coarse discretization of this continuous SDE.


## 5.8 The Design Space: Choosing Between ODE and SDE

The choice between ODE and SDE generation — and the choice of diffusion coefficient within the SDE family — is a design decision with quantifiable tradeoffs.

**Sample quality vs. NFE.** For a fixed number of function evaluations $N$, the ODE typically produces better samples at high $N$ (e.g., $N \geq 50$) because each step makes full use of the learned velocity field. At low $N$ (e.g., $N \leq 10$), the SDE often wins because the noise compensates for the large discretization error of the ODE.

**Mode coverage vs. sample precision.** The SDE generates more diverse samples (better mode coverage) but with slightly worse individual sample quality (more noise per sample). The ODE generates more precise samples but may miss modes entirely if the velocity field is imperfect near mode boundaries.

**Log-likelihood.** The ODE admits exact log-likelihood computation via the instantaneous change-of-variables formula (Chapter 1). The SDE does not — the noise makes the map stochastic, so there is no deterministic change of variables. If exact log-likelihoods are needed (e.g., for model comparison, anomaly detection, or variational inference), the ODE is the only choice.

**Computational cost.** ODE steps and SDE steps have the same computational cost per step (one neural network evaluation each). The predictor-corrector scheme costs $1 + M$ evaluations per step, where $M$ is the number of corrector steps. Langevin correctors typically use $M = 1$–$5$, a modest overhead.

> [!proposition] 5.3 — The Interpolation Family
> For any probability path $\{p_t\}$ and any diffusion schedule $g_t \geq 0$, the SDE
>
> $$dX_t = \left[v_t(X_t) + \frac{g_t^2}{2}\nabla\log p_t(X_t)\right] dt + g_t\, dW_t$$
>
> has marginals $\{p_t\}$, where $v_t$ is the probability flow ODE velocity field (the minimum-norm velocity satisfying the continuity equation). This provides a one-parameter family of SDEs indexed by $g_t$, all sharing the same marginals, interpolating between $g_t = 0$ (the probability flow ODE) and any desired noise level.


This family makes the design choice explicit: given a trained velocity field $v_\theta$ and a trained score $s_\theta$, you can sample from *any* member of the interpolation family at inference time, tuning $g_t$ as a hyperparameter. Low $g_t$ for speed (fewer steps, ODE-like), high $g_t$ for robustness (more correction, SDE-like). The choice can even vary over time — use the ODE for the initial coarse transport, switch to the SDE for the final fine-scale refinement — without retraining.

## 5.9 Historical Notes

The reverse-time SDE was derived by **Anderson (1982)** ("Reverse-Time Diffusion Equation Models"), building on earlier work by **Nelson (1967)** on stochastic mechanics. The probability flow ODE was introduced by **Song, Sohl-Dickstein, Kingma, Kumar, Ermon, and Poole (2021)** in the Score SDE paper, which unified VP-SDE, VE-SDE, and the probability flow ODE in a single framework. The predictor-corrector sampling scheme was also introduced in this paper.

The systematic analysis of the ODE-SDE tradeoff was carried out by **Karras, Aittala, Aila, and Laine (2022)** ("Elucidating the Design Space of Diffusion-Based Generative Models"), who showed that the choice of noise schedule, time discretization, and sampling procedure have larger effects on generation quality than the choice of SDE. **Berner, Richter, and Ullrich (2022)** provided theoretical analysis of discretization error in both ODE and SDE sampling, showing that SDE samplers have favorable error scaling in high dimensions.

Stochastic interpolants with non-zero diffusion were developed by **Albergo, Boffi, and Vanden-Eijnden (2023)** ("Stochastic Interpolants: A Unifying Framework for Flows and Diffusions"), who showed how to jointly learn the drift and score from the interpolant and derived the optimal diffusion coefficient that minimizes the variance of the training objective. **Ma, Davis, Azizzadenesheli, Anandkumar, and colleagues (2024)** scaled stochastic interpolant SDEs with the Scalable Interpolant Transformer (SiT).

Langevin dynamics as a generative tool was pioneered by **Song and Ermon (2019)** (NCSN) and refined in **Song and Ermon (2020)** (NCSNv2). The Metropolis-adjusted Langevin algorithm has a long history in computational statistics, surveyed in **Roberts and Rosenthal (1998)**. Its use as a corrector for diffusion models was explored by **Song et al. (2021)** and further analyzed by **Wu, Trippe, Naesseth, Blei, and Cunningham (2020)** in the context of annealed importance sampling.

The drift-based perspective — learning a drift for a fixed-noise SDE — connects to the optimal control literature, where the Schrödinger bridge drift (Chapter 4) is the solution to a stochastic control problem. **Tzen and Raginsky (2019)** ("Neural Stochastic Differential Equations: Deep Latent Gaussian Models in the Diffusion Limit") and **Li, Wong, Chen, and Duvenaud (2020)** explored neural SDEs from the latent variable perspective. The annealing approach to noise scheduling was formalized by **Dockhorn, Vahdat, and Kreis (2022)** ("Score-Based Generative Modeling with Critically-Damped Langevin Diffusion"), who introduced underdamped Langevin dynamics with momentum as an alternative to the standard overdamped formulation.

## References

- [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, and Poole (2021)](https://arxiv.org/abs/2011.13456) — Score-Based Generative Modeling through Stochastic Differential Equations
- [Karras, Aittala, Aila, and Laine (2022)](https://arxiv.org/abs/2206.00364) — Elucidating the Design Space of Diffusion-Based Generative Models
- [Albergo, Boffi, and Vanden-Eijnden (2023)](https://arxiv.org/abs/2303.08797) — Stochastic Interpolants: A Unifying Framework for Flows and Diffusions
- [Tzen and Raginsky (2019)](https://arxiv.org/abs/1905.09883) — Neural Stochastic Differential Equations: Deep Latent Gaussian Models in the Diffusion Limit
- [Dockhorn, Vahdat, and Kreis (2022)](https://arxiv.org/abs/2112.07068) — Score-Based Generative Modeling with Critically-Damped Langevin Diffusion
