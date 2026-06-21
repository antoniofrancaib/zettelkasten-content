---

title: "The Speed Problem"
subtitle: "Consistency models, stochastic flow maps, distillation, and the race to one step — 2021 to 2026"
---



There is a structural irony at the heart of flow matching. The entire motivation for conditional flow matching — the thesis of Chapter 3 — was to eliminate the need for simulation during training. The original continuous normalizing flows required integrating an ODE at every gradient step; the simulation bottleneck made training prohibitively expensive. Flow matching resolved this by replacing ODE simulation with regression against a known conditional velocity field, reducing training cost to a single forward pass per gradient step. The simulation bottleneck was declared solved.

It was solved for training. At inference, it reappears.

Generating a sample from a trained flow model requires integrating the learned ODE from $t=0$ to $t=1$ — numerically, using some discretization of the time interval. The Euler method, the simplest choice, takes steps of size $\Delta t$ and accumulates error of order $O(\Delta t)$ per step; a total error of $O(\varepsilon)$ requires $1/\varepsilon$ steps. For image generation at competitive quality, this typically means 50 to 1000 function evaluations (NFE) — 50 to 1000 forward passes through the neural network. For a model with hundreds of millions of parameters, each forward pass may take seconds on consumer hardware. A 1000-step generation takes minutes. For drug screening workflows that need to evaluate millions of molecules, this is a fatal bottleneck.

The quest for faster inference — fewer NFE without quality degradation — has been one of the most active areas in generative modeling since 2021. This chapter traces the arc from DDIM's first insight (stochasticity is optional) through the theory of flow maps and the consistency model framework, to the shortcut models of 2025 that train a single network capable of generating in as few as one step. The story is one of increasing conceptual clarity: each method, at first glance a clever engineering trick, turns out to be a principled consequence of the geometry of probability transport.

## 8.1 The Return of the Simulation Bottleneck

To understand why inference is expensive, it helps to be precise about where the error comes from.

A flow model defines an ODE $\dot{x} = v_\theta(x, t)$ on $[0,1]$ with $x_0 \sim p_0$. The exact solution at $t=1$ is the sample $x_1$. Any numerical integrator approximates this solution by evaluating $v_\theta$ at finitely many times. The **Euler method** with $T$ steps takes:

> [!equation] 8.1
> $$x_{t_{n+1}} = x_{t_n} + \Delta t \cdot v_\theta(x_{t_n}, t_n), \qquad \Delta t = \frac{1}{T}, \quad n = 0, 1, \ldots, T-1.$$


The local truncation error at each step is $O(\Delta t^2)$; the global error after $T$ steps accumulates to $O(\Delta t) = O(1/T)$. To halve the error, one must double the number of steps — and double the NFE.

The error is amplified by the curvature of the learned trajectories. If the ODE flow is a straight line — $x_t = x_0 + t(x_1 - x_0)$ — then $v_\theta(x, t) = x_1 - x_0$ is constant, and the Euler method is exact in a single step, regardless of $\Delta t$. If trajectories are curved, successive Euler updates accumulate systematic bias proportional to the second derivative of $x_t$ with respect to $t$. The fundamental difficulty is that, under a naive linear interpolant with independent coupling — matching random noise samples to random data points — the optimal velocity field $u_t(x) = \mathbb{E}[x_1 - x_0 | x_t = x]$ is the average of many different straight-line directions, and their average defines a curved marginal trajectory even when each individual conditional trajectory is straight.

> [!remark] 8.1
> The connection between trajectory curvature and NFE requirement can be made precise. If $\kappa = \sup_{t \in [0,1]} \|\partial_t^2 x_t\|$ is the maximum curvature of the ODE trajectory, then the Euler global error satisfies $\|x_T - x_1\| = O(\kappa / T)$. Reducing curvature — making trajectories straighter — is therefore equivalent to reducing the NFE required to achieve a given accuracy. This is the geometric core of every fast-sampling method: they are, at bottom, all trying to straighten the path from noise to data.


## 8.2 DDIM: Determinism as an Accelerant

The first practical breakthrough came not from the flow matching community but from the diffusion community — and it consisted of a simple observation: the stochasticity in DDPM sampling is unnecessary.

Recall from Chapter 2 that the DDPM reverse process is a Markov chain:
$$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I).$$

The added noise $\sigma_t\, \epsilon$ ensures that the chain maintains the correct marginal distribution at each step, but it also forces the process to take small steps: if you skip steps, the accumulated noise pushes the sample off the data manifold. **Denoising Diffusion Implicit Models** (Song, Meng, and Ermon, ICLR 2021) showed that the noise term can be removed entirely — or scaled by an arbitrary parameter $\eta \in [0,1]$ — while leaving the marginal distributions $q(x_t)$ unchanged:

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat{x}_0(x_t, t) + \sqrt{1-\bar\alpha_{t-1} - \eta^2\sigma_t^2}\,\frac{x_t - \sqrt{\bar\alpha_t}\,\hat{x}_0}{\sqrt{1-\bar\alpha_t}} + \eta\,\sigma_t\,\epsilon,$$

where $\hat{x}_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta)/ \sqrt{\bar\alpha_t}$ is the predicted clean sample. With $\eta = 0$, the process is **deterministic**: the same initial noise $x_T$ always produces the same final sample $x_0$. The deterministic reverse process is exactly the **probability flow ODE** of Chapter 2 (equation 2.8), integrated backwards.

Determinism enables a crucial speedup: a deterministic ODE can be integrated with larger steps and adaptive step-size control. Song et al. showed that DDIM with 50 steps achieves comparable sample quality to DDPM with 1000 steps — a 20× reduction in NFE. This was the opening result of the fast-sampling era.

## 8.3 Higher-Order Solvers and Exponential Integrators

DDIM's insight — that the sampling process is an ODE, not a Markov chain — opened the door to the machinery of numerical ODE solving. The probability flow ODE for the VP-SDE takes the form:

$$\frac{dx}{dt} = f(t)\, x - \frac{g(t)^2}{2}\, s_\theta(x, t),$$

where $f(t) = -\beta(t)/2$ and $g(t)^2 = \beta(t)$ for the standard DDPM noise schedule. This is a **semilinear ODE** — linear in $x$ with a non-linear forcing term $-\frac{g(t)^2}{2} s_\theta(x, t)$. Semilinear ODEs admit **exponential integrators**: solvers that treat the linear part exactly (by analytical computation of the matrix exponential $e^{\int f(t)\, dt}$) and approximate only the non-linear forcing.

**DPM-Solver** (Lu, Zhou, Bao, Chen, Li, and Zhu, NeurIPS 2022) exploits this structure. Introducing the log-SNR reparametrization $\lambda(t) = \log(\alpha_t/\sigma_t)$, the probability flow ODE transforms into an equation for which the linear part integrates analytically. The DPM-Solver update from $t_i$ to $t_{i+1}$ is:

> [!equation] 8.2
> $$x_{t_{i+1}} = \frac{\alpha_{t_{i+1}}}{\alpha_{t_i}}\, x_{t_i} - \alpha_{t_{i+1}}\!\int_{\lambda(t_i)}^{\lambda(t_{i+1})} e^{-\lambda}\, \hat{\epsilon}_\theta(x_\lambda,\, \lambda)\, d\lambda,$$


where $\hat{\epsilon}_\theta$ is the noise prediction. The linear prefactor $\alpha_{t_{i+1}}/\alpha_{t_i}$ is computed analytically; the integral of $\hat{\epsilon}_\theta$ is approximated by quadrature (first-order DPM-Solver approximates the integral as a rectangle; higher-order versions use Taylor expansion of $\hat{\epsilon}_\theta$ in $\lambda$). The key gain: the truncation error of DPM-Solver is due only to the approximation of the $\hat\epsilon_\theta$ integral — the linear part contributes zero error. This allows second- and third-order accuracy with 2–4 NFE per step, achieving competitive sample quality in as few as 10–20 total NFE.

For **flow matching** ODEs — where $\dot{x} = v_\theta(x, t)$ has no special linear structure — DPM-Solver does not apply directly. The appropriate alternatives are higher-order Runge-Kutta methods. The **midpoint method** (second order, 2 NFE per step) evaluates:

$$x_{t+\Delta t/2} = x_t + \tfrac{\Delta t}{2}\, v_\theta(x_t, t), \qquad x_{t+\Delta t} = x_t + \Delta t\, v_\theta\!\left(x_{t+\Delta t/2},\, t+\tfrac{\Delta t}{2}\right).$$

**Heun's method** (second order, 2 NFE per step, smaller constant) and **Runge-Kutta 4** (fourth order, 4 NFE per step) provide further improvements. For flow matching models trained with OT-conditioned straight trajectories, the improvement from higher-order methods is substantial: as trajectories become straighter, Heun with 10 steps can match Euler with 100 steps.

## 8.4 Rectified Flow and the Reflow Procedure

Higher-order solvers reduce the NFE required to integrate a given ODE accurately. A complementary approach modifies the ODE itself — at training time — to be easier to integrate.

The central observation of **Rectified Flow** (Liu, Gong, and Liu, ICLR 2023) is the following. Under the standard linear interpolant with independent coupling, the optimal velocity field $u_t(x)$ is curved because it averages over different source-target pairs: a point $x_t$ in the middle of the data manifold was generated by many different $(x_0, x_1)$ pairs with different directions, and $u_t(x_t)$ is their conditional average. The individual conditional trajectories $x_t = tx_1 + (1-t)x_0$ are straight lines — but the *marginal* flow, which is what the model actually learns, is curved wherever these lines cross.

**Reflow** is a procedure to uncross them. Given a trained model $v_\theta$, generate a dataset of coupled pairs:

$$\{(x_0^{(i)},\, x_1^{(i)})\}_{i=1}^M, \quad x_0^{(i)} \sim p_0,\quad x_1^{(i)} = \text{ODE}_\theta(x_0^{(i)}),$$

where $\text{ODE}_\theta(x_0)$ denotes integrating the trained ODE from $x_0$ to produce a generated sample. Train a new model $v_{\theta'}$ using the flow matching objective on the coupled pairs:

> [!equation] 8.3
> $$\mathcal{L}_\mathrm{reflow}(\theta') = \mathbb{E}_{i,\, t}\!\left[\left\|v_{\theta'}\!\left(t\,x_1^{(i)} + (1-t)\,x_0^{(i)},\; t\right) - \!\left(x_1^{(i)} - x_0^{(i)}\right)\right\|^2\right].$$


The conditional velocity target is the constant vector $x_1^{(i)} - x_0^{(i)}$ — not averaged over multiple targets, because the source-target pairing is now fixed by the ODE coupling. Since the paired $(x_0^{(i)}, x_1^{(i)})$ lie on the same ODE trajectory, the straight-line interpolant between them approximates the actual trajectory of the original ODE; minimizing (8.3) trains a model whose trajectories are approximately straight.

Iterating the procedure — "1-Rectified Flow" trains on random pairs, "2-Rectified Flow" reflows 1-RF's output, and so on — progressively straightens trajectories. Liu et al. showed that 2-Rectified Flow with a 1-step Euler integrator achieves sample quality comparable to 1-Rectified Flow with 100 steps on standard image generation benchmarks. The cost is generating the coupled dataset $\{(x_0^{(i)}, x_1^{(i)})\}$ — which itself requires 100+ NFE per sample from the original model.

> [!remark] 8.2
> The reflow procedure has a clean geometric interpretation in terms of the theory developed in Chapter 3. Mini-batch OT conditioning (Section 3.5) produces straight trajectories *at training time* by minimizing the transport cost between noise and data samples within each batch. Reflow achieves the same effect *after training* by using the trained ODE to generate better-coupled pairs. Both methods are trying to solve the same problem — reduce trajectory crossings — using different interventions: OT at training, reflow at post-training. In practice, OT-conditioned training combined with reflow gives straighter trajectories than either method alone.


## 8.5 The Consistency Function

All the methods discussed so far are trying to make the ODE easier to integrate. **Consistency models** (Song, Dhariwal, Chen, and Sutskever, ICML 2023) take a different approach: rather than integrating an ODE at all, learn a function that maps directly from any point on a trajectory to its endpoint.

> [!definition] 8.1 — Consistency Function
> Given an ODE $\dot{x} = v_\theta(x, t)$ on $[0,1]$ with flow map $\Psi_{t \to s} : \mathcal{X} \to \mathcal{X}$ (the map that takes $x_t$ to $x_s$ by exact integration), the **consistency function** $f : \mathcal{X} \times [0,1] \to \mathcal{X}$ is defined by:
>
> $$f(x_t, t) = \Psi_{t \to 1}(x_t).$$
>
> It maps any point $x_t$ on a trajectory, at any time $t$, directly to the trajectory's endpoint $x_1 \in p_1$.


The consistency function satisfies a fundamental self-consistency property: for any two times $0 \leq s \leq t \leq 1$ and any $x_t$ on a trajectory,

$$f(x_t, t) = f(x_s, s), \qquad \text{where } x_s = \Psi_{t \to s}(x_t).$$

Any two points on the same trajectory map to the same endpoint. This is the **consistency property** — the defining constraint of the function.

Given a learned consistency function $f_\theta$, one-step generation is immediate: draw $x_0 \sim p_0$, apply $x_1 = f_\theta(x_0, 0)$. No ODE integration, no time-stepping, one forward pass. Multi-step generation — trading NFE for quality — proceeds by alternating between applying $f_\theta$ to jump to $t = 1$, adding noise to move back to some $t < 1$, and applying $f_\theta$ again. Each additional "denoise-and-remoise" cycle corrects errors introduced by the imperfect consistency function.

> [!theorem] 8.1 — One-Step Generation from Consistency Models
> If $f_\theta$ exactly satisfies the consistency property — $f_\theta(x_t, t) = f_\theta(x_s, s)$ for all $s \leq t$ on any trajectory — then $f_\theta(x_0, 0) \sim p_1$ for all $x_0 \sim p_0$, and generation is exact in one step.


The proof is trivial: the consistency property at $t=0$ gives $f_\theta(x_0, 0) = f_\theta(x_1, 1)$ where $x_1$ is the ODE endpoint. Since the consistency function at $t=1$ is the identity ($f(x_1, 1) = x_1$), the single application of $f_\theta$ recovers the data distribution exactly.

## 8.6 Consistency Distillation and Training

There are two ways to learn a consistency function. The first — **consistency distillation** (CD) — assumes a pre-trained ODE model is available and uses it as a teacher.

Given adjacent time points $t_n < t_{n+1}$ in a discrete time schedule, and a data point $x_{t_{n+1}} \sim q(x_{t_{n+1}})$ (drawn from the noisy data distribution at time $t_{n+1}$), compute a one-step ODE approximation:

$$\hat{x}_{t_n} = x_{t_{n+1}} - (t_n - t_{n+1})\, v_\phi(x_{t_{n+1}},\, t_{n+1}),$$

where $v_\phi$ is the pre-trained teacher velocity field. The consistency distillation loss enforces that $f_\theta$ agrees on the two points:

> [!equation] 8.4
> $$\mathcal{L}_\mathrm{CD}(\theta,\, \theta^-) = \mathbb{E}_{n,\, x_{t_{n+1}}}\!\left[d\!\left(f_\theta(x_{t_{n+1}},\, t_{n+1}),\; f_{\theta^-}(\hat{x}_{t_n},\, t_n)\right)\right],$$


where $d$ is a distance function ($\ell^2$, LPIPS, or the Pseudo-Huber loss), and $\theta^- = \mathrm{sg}(\mathrm{EMA}(\theta))$ is a stop-gradient exponential moving average of $\theta$. The EMA target stabilizes training: $f_{\theta^-}$ is a slowly changing reference that the current $f_\theta$ is trained to match.

The second approach — **consistency training** (CT) — trains from scratch without a pre-trained ODE teacher. The key observation is that the consistency property $f(x_{t_{n+1}}, t_{n+1}) = f(x_{t_n}, t_n)$ can be enforced even when $\hat{x}_{t_n}$ comes from the *consistency function itself* (evaluated with EMA parameters), rather than from an ODE teacher. This makes CD and CT formally equivalent in structure: both minimize the discrepancy between $f_\theta(x_{t_{n+1}}, t_{n+1})$ and $f_{\theta^-}(\hat{x}_{t_n}, t_n)$; they differ only in how $\hat{x}_{t_n}$ is obtained — from a pre-trained ODE (CD) or from two previous evaluations of $f_\theta$ itself (CT).

**Improved consistency models** (Song and Dhariwal, NeurIPS 2023) resolved several training instabilities in the original formulation: improved discretization schedules for $\{t_n\}$, the Pseudo-Huber loss, adaptive EMA rates, and a careful treatment of the boundary condition $f(x_1, 1) = x_1$ at the noise end. These changes reduced the generation quality of 1-step consistency models on CIFAR-10 from ~5 to ~2.8 FID — comparable to multi-step diffusion with 50 NFE.

## 8.7 Progressive Distillation

An earlier distillation approach — conceptually simpler than consistency models — is **progressive distillation** (Salimans and Ho, ICLR 2022).

The idea is straightforward. Given a teacher model that generates in $T$ steps, train a student model that generates in $T/2$ steps, where each student step matches the combined effect of two teacher steps. Then treat the student as the new teacher, train a new student with $T/4$ steps, and repeat until reaching 1 step.

Formally, let $v_\phi$ be a teacher that generates in $T$ steps. The student $v_\theta$ is trained to match the two-step output of $v_\phi$ in a single step. Starting from $x_t$, compute:

$$\tilde{x}_{t - 2\Delta t} = \text{2-step-Euler}_\phi(x_t, t), \qquad \Delta t = 1/T,$$

and define the student target as the velocity that would move $x_t$ to $\tilde{x}_{t-2\Delta t}$ in one step of size $2\Delta t$:

> [!equation] 8.5
> $$\mathcal{L}_\mathrm{PD}(\theta) = \mathbb{E}_{t,\, x_t}\!\left[\left\|v_\theta(x_t,\, t) - \frac{\tilde{x}_{t-2\Delta t} - x_t}{2\Delta t}\right\|^2\right].$$


Each distillation stage reduces NFE by 2×; after $\log_2 T$ stages, the student generates in a single step. Salimans and Ho demonstrated that this procedure, applied to a diffusion model trained to generate in 1024 steps, produces a competitive 1-step generator — with some quality degradation at the final stage that subsequent improvements have reduced.

The connection to reflow is close: progressive distillation trains a student model whose velocity field points directly from $x_t$ to the two-step ODE output, rather than to the data point; reflow trains a model whose velocity field points from noise to the ODE output at $t=1$. Both are straightening trajectories, but at different scales — progressive distillation at the per-step level, reflow at the global level.

## 8.8 Flow Maps and the General Framework

Consistency functions and progressive distillation are both instances of a more general concept: the **flow map** of the ODE, trained as a neural network directly.

> [!definition] 8.2 — Flow Map
> The **flow map** $\Psi_{t \to s}^\theta : \mathcal{X} \to \mathcal{X}$ of a parametric ODE $\dot{x} = v_\theta(x, t)$ is the function that maps the state at time $t$ to the state at time $s$ by exact integration:
>
> $$\Psi_{t \to s}^\theta(x) = x + \int_t^s v_\theta(x_r, r)\, dr.$$
>
> A **learned flow map** $F_\theta(x_t, t, s) \approx \Psi_{t \to s}^\theta(x_t)$ approximates this map directly as a neural network, parametrized by both the current state $x_t$ and the source and target times $(t, s)$.


Every fast-sampling method corresponds to a specific choice of which flow maps are learned and how:

- **DDIM**: implicitly learns $\Psi_{T \to 0}$ as the composition of deterministic DDPM steps.
- **Reflow**: trains the velocity field so that $\Psi_{0 \to 1}$ is approximately linear.
- **Progressive distillation**: trains $F_\theta(x_t, t, t-2\Delta t)$ from $F_\phi(x_t, t, t-\Delta t)$.
- **Consistency models**: trains $F_\theta(x_t, t, 1)$ — the full-time flow map — from any $t$.

The unified view is: all fast-sampling methods are learning, in different ways, to represent the ODE flow map in fewer evaluations than direct integration requires.

## 8.9 Shortcut Models

The most flexible realization of the flow map framework is **shortcut models** (Frans, Hafner, Levine, and Abbeel, 2025), which train a single neural network capable of predicting any flow map $F_\theta(x_t, t, \Delta t)$ — parametrized by the step size $\Delta t$ — without a separate distillation phase.

The shortcut model augments the standard velocity field with a step-size input: $s_\theta : \mathcal{X} \times [0,1] \times (0,1] \to \mathcal{X}$, where $s_\theta(x_t, t, d)$ is a "direction" such that:

$$x_{t+d} \approx x_t + d \cdot s_\theta(x_t, t, d).$$

For small $d$, $s_\theta(x_t, t, d) \approx v_\theta(x_t, t)$ — the standard velocity. For large $d$, $s_\theta$ makes a "shortcut" prediction that effectively integrates many small velocity steps in one.

Training uses two losses. For small $d$ (sampled uniformly near 0): the standard conditional flow matching loss, ensuring that $s_\theta(\cdot, \cdot, d)$ behaves like the correct velocity field. For large $d$: a **self-consistency loss** that enforces the composition property of flow maps:

> [!equation] 8.6
> $$\mathcal{L}_\mathrm{SC}(\theta) = \mathbb{E}_{t,\, d,\, x_t}\!\left[\left\|s_\theta(x_t,\, t,\, d) - \frac{1}{d}\bigl(x' - x_t\bigr)\right\|^2\right],$$


where $x' = \mathrm{sg}\!\left(x_t + \tfrac{d}{2}\, s_\theta\!\left(x_t, t, \tfrac{d}{2}\right) + \tfrac{d}{2}\, s_\theta\!\left(x_t + \tfrac{d}{2}\,s_\theta\!\left(x_t, t, \tfrac{d}{2}\right),\; t+\tfrac{d}{2},\; \tfrac{d}{2}\right)\right)$

is the result of two half-steps of size $d/2$ using the model itself, with stop-gradient $\mathrm{sg}(\cdot)$. The consistency property here is: two steps of size $d/2$ should equal one step of size $d$. The stop-gradient on the target prevents gradient loops from the two half-step evaluations.

The shortcut model achieves several desirable properties simultaneously. It generates in one step by calling $s_\theta(x_0, 0, 1)$. It generates in $T$ steps by calling $s_\theta(x_t, t, 1/T)$ at each step, recovering standard flow matching behavior. For intermediate step counts, it interpolates — using the consistency objective to ensure that shorter chains compose into longer ones. A single model covers the entire speed-quality curve, without requiring separate distillation stages.

> [!remark] 8.3
> Shortcut models resolve an aesthetic problem with consistency distillation and progressive distillation: both require a separate distillation phase, producing a second model that cannot be updated when the teacher is retrained. Shortcut models train the fast-sampling capability jointly with the standard flow matching capability, in a single procedure. The self-consistency loss plays the same role as the consistency property of Bregman divergences in the generator matching theorem of Chapter 6: it replaces an intractable "marginal" target (the exact flow map) with a "conditional" self-referential target (two half-steps), exploiting the composition law of ODEs.


## 8.10 The NFE-Quality Frontier

The practical outcome of this body of work is a substantially extended Pareto frontier between sample quality and NFE.

In 2020, DDPM required 1000 NFE to achieve competitive FID on standard benchmarks. In 2021, DDIM reduced this to 50. In 2022, DPM-Solver reached competitive quality in 10-20 NFE. In 2023, improved consistency models achieved ~3 FID on CIFAR-10 in a single step. By 2025, shortcut models and related approaches produce competitive image and molecular samples in 1-4 NFE across a range of resolutions and domains.

The quality floor — the best achievable quality at exactly 1 NFE — has been steadily falling. For image generation at $256 \times 256$ resolution, 1-step consistency models currently achieve FID scores of 4-6, compared to ~2 for 50-step flow matching. The gap narrows but has not closed: at 4 steps, consistency models are competitive with 50-step generation; at 1 step, there remains a visible gap on complex high-resolution data.

For scientific applications, the picture is different. In protein structure prediction and molecular docking, the relevant metric is not FID but physical validity — bond length constraints, clashscore, designability. Here, 1-step consistency models often perform surprisingly well: the physical constraints are less demanding than perceptual ones, and the space of valid protein conformations is smoother than the space of natural images. FoldFlow-2 with OT conditioning achieves competitive designability scores in 10 steps; shortcut variants reduce this to 2-4 steps with minimal quality loss.

## 8.11 Historical Notes

**DDIM** is the foundation of the fast-sampling literature. **Song, Meng, and Ermon (ICLR 2021)** showed that the DDPM reverse process is a deterministic ODE — the probability flow ODE of Chapter 2 — and that 50-step integration suffices for competitive quality. This paper established the conceptual framework for all subsequent work: the sampling process is an ODE, not a Markov chain, and ODE numerics apply.

**DPM-Solver** and **DPM-Solver++** (**Lu, Zhou, Bao, Chen, Li, and Zhu, NeurIPS 2022**) developed the exponential integrator approach for the semilinear diffusion ODE, achieving 10–20 NFE. **PNDM** (**Liu, Ren, Lin, and Zhao, ICLR 2022**) took a different approach — pseudo-numerical methods that exploit the smoothness of the score function in diffusion time — achieving similar speedups from a different mathematical direction.

**Progressive distillation** (**Salimans and Ho, ICLR 2022**) introduced the distillation paradigm: train a student to match multiple teacher steps in one. This was the first practical 1-step generation method, albeit with quality degradation.

**Rectified Flow / Reflow** (**Liu, Gong, and Liu, ICLR 2023**) approached the problem from the training side: straighten trajectories by retraining on ODE-coupled pairs. The 2-Rectified Flow with 1-step Euler demonstrated that near-one-step generation is possible without distillation, at the cost of a second training run. Rectified Flow was later adopted as the technical foundation for **Stable Diffusion 3** (Esser, Kulal, Blattmann, Entezari, Müller, Lorenz, Zhang, Mathieu, Rombauer, Omer, Dockhorn, Boesel, Kramel, Birk, Levi, Kreis, Torsten, and Vahdat, 2024) and **Flux** (Black Forest Labs, 2024), two prominent production text-to-image models, establishing rectified flow matching as the de facto standard for high-quality large-scale generation.

**Consistency models** (**Song, Dhariwal, Chen, and Sutskever, ICML 2023**) introduced the consistency function framework — learn the full-time flow map directly. **Improved consistency training** (**Song and Dhariwal, NeurIPS 2023**) resolved training instabilities and achieved ~3 FID in 1 step on CIFAR-10.

**Shortcut models** (**Frans, Hafner, Levine, and Abbeel, 2025**) unified fast and slow sampling in a single model parametrized by step size, eliminating the need for separate distillation. Concurrent work on **consistency flow matching** (**Yang, Tang, and Gong, 2025**) developed a version of the consistency framework directly for flow matching (rather than diffusion) ODEs, obtaining tighter theoretical connections to the generator matching framework of Chapter 6.

The arc of this literature — from DDIM's empirical discovery to the theoretical framework of flow maps and consistency functions — is a pattern that recurs throughout this monograph: a practical speedup motivates a mathematical formalization, which reveals the underlying principle, which enables new methods. The principle here is the composition law of ODE flows: $\Psi_{t \to s} \circ \Psi_{s \to r} = \Psi_{t \to r}$. Every fast-sampling method exploits this law, in one form or another.
