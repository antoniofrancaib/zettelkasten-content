---

title: "The Convergence"
subtitle: "Simulation-free training, linear interpolants, and the unification of two tracks — 2022 to 2023"
---



There is a moment in the history of every major field when two lines of investigation, developed by different communities using different tools, turn out to be the same thing. Sometimes it takes a generation to recognize. In machine learning in 2022, it happened within months, and was acknowledged almost immediately — a rare convergence that arrived with the force of inevitability.

The transport track of Chapters 1 and 2 culminated in the simulation bottleneck: continuous normalizing flows required ODE simulation at every training step, and this cost grew adversarially with the complexity of the target distribution. The denoising track of Chapter 3 abolished that bottleneck — DDPM trains with a single forward pass per step — but replaced it with a sampling bottleneck: generating one sample required a thousand sequential network evaluations. Two approaches, two bottlenecks, no complete solution.

The resolution came from asking a question that seems almost too simple: if you could design the path from noise to data yourself — not as the output of a diffusion SDE, but as a freely chosen interpolation — what is the cheapest possible training procedure? The answer turned out to be regression: sample a point on the path, regress the vector field at that point against the known path velocity, repeat. No ODE simulation. No score functions. No adjoint method. Just supervised learning on a moving target.

Three essentially simultaneous papers arrived at this answer in 2022. Lipman, Chen, Ben-Hamu, Nickel, and Le came from the CNF direction; Albergo and Vanden-Eijnden from stochastic analysis; Liu, Gong, and Liu from straight-line interpolation. Each contributed a different aspect of the same insight, and together they define the modern theory of generative flow.

## 4.1 The Continuity Equation: Probability as Fluid

The mathematical bridge between probability paths and vector fields is not new — it is the **continuity equation**, the fundamental conservation law of fluid mechanics, adapted to probability distributions.

If $\{p_t\}_{t \in [0,1]}$ is a smooth family of probability densities evolving in time, and $u_t : \mathbb{R}^d \to \mathbb{R}^d$ is a velocity field, then the statement that probability mass is neither created nor destroyed as it flows with velocity $u_t$ is:

> [!equation] 4.1
> $$\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t\, u_t) = 0.$$


This is the continuity equation. It is simultaneously a statement about fluid mechanics (density is conserved under flow) and about probability (total probability is conserved under the ODE $\dot{x} = u_t(x)$). It is the continuous-time generalization of the change-of-variables formula: Theorem 2.1 (the instantaneous change-of-variables formula) is simply (4.1) rewritten along characteristics.

The continuity equation expresses a one-to-many relationship: given a probability path $\{p_t\}$, there are infinitely many velocity fields $u_t$ consistent with it. Any two solutions differ by a divergence-free field — a field $w_t$ with $\nabla \cdot (p_t w_t) = 0$. Among all consistent fields, the one with minimum kinetic energy $\mathbb{E}_{p_t}[\|u_t(x)\|^2]$ is called the **minimum-norm velocity field**, and it plays a special role: it corresponds to the displacement interpolation of optimal transport theory, following straight-line trajectories in probability space.

> [!remark] 4.1
> The continuity equation (4.1) is the cornerstone of the entire theory. In fluid mechanics, it governs the conservation of mass. In kinetic theory, it governs the evolution of particle distributions. In the context of generative modeling, it says precisely what is required of a velocity field: at every point and time, the velocity must be consistent with how the probability density is changing. Every normalizing flow, every diffusion model, and every flow matching method is implicitly solving the continuity equation — they differ only in how they parametrize the solution.


## 4.2 Flow Matching: Design Your Own Path

The central insight of flow matching is a complete inversion of the training philosophy of CNFs. In FFJORD, the vector field $v_\theta$ is learned, and the probability path $p_t$ is an implicit consequence of what the learned ODE does to the noise distribution. You train $v_\theta$ by maximum likelihood, which requires simulating the ODE to evaluate the likelihood, which is the bottleneck.

**Flow matching inverts this.** Choose a desired probability path $\{p_t\}_{t \in [0,1]}$ first — explicitly, deliberately, with $p_0 = \mathcal{N}(0, I)$ and $p_1 = p_{\text{data}}$. Then derive the target vector field $u_t$ from the continuity equation (4.1), and train $v_\theta$ to match it by regression. No ODE simulation is needed during training: evaluating the regression target requires only sampling from the path, not integrating the ODE.

> [!definition] 4.1 — Flow Matching Objective
> Given a target probability path $\{p_t\}$ with associated velocity field $u_t$, the **flow matching objective** is:
>
> $$\mathcal{L}_{\mathrm{FM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\; x \sim p_t}\!\left[\|v_\theta(x, t) - u_t(x)\|^2\right].$$
>
> The minimizer $v_\theta^* = u_t$ is the exact velocity field of the target path. At convergence, the ODE $\dot{x} = v_{\theta^*}(x, t)$ transports $p_0$ to $p_1 = p_{\text{data}}$.


The problem: $u_t(x)$ is the **marginal** velocity field, defined as the minimum-norm solution of (4.1). For an arbitrary $p_t$, this requires computing a conditional expectation over all data points that could have contributed to the density at $x$ — a high-dimensional integral that is no more tractable than the likelihood computation it was meant to replace.

> [!equation] 4.2
> $$u_t(x) = \mathbb{E}_{p_1(x_1 | x_t = x)}\!\left[u_t(x \mid x_1)\right],$$


where $u_t(x|x_1)$ is the conditional velocity field at $x$, given that the trajectory will end at $x_1$. To evaluate (4.2), you would need to invert the flow at every point — again, an ODE solve. The marginal flow matching objective is formally correct but computationally inert.

## 4.3 The Conditional Flow Matching Theorem

The resolution is elegant, and it is the technical heart of the Lipman et al. paper. The intractable marginal objective $\mathcal{L}_{\mathrm{FM}}$ can be replaced by a tractable **conditional** objective without changing the gradient with respect to $\theta$.

The construction begins by replacing the marginal path $p_t(x)$ with a family of conditional paths $p_t(x|x_1)$ — one for each data point $x_1 \sim p_{\text{data}}$. Each conditional path goes from some base distribution $p_0(x|x_1) \approx \mathcal{N}(0, I)$ (independent of $x_1$, or approximately so) to a point mass concentrated at $x_1$. The marginal path is recovered by integrating over the data distribution:

$$p_t(x) = \int p_t(x \mid x_1)\, p_1(x_1)\, dx_1.$$

Similarly, the marginal velocity field is the expectation of the conditional velocity fields under the data posterior:

$$u_t(x) = \mathbb{E}_{p_1(x_1|x_t = x)}\!\left[u_t(x \mid x_1)\right].$$

Now define the **conditional flow matching objective**, which trains $v_\theta$ to match the conditional velocity fields rather than the marginal one:

> [!equation] 4.3
> $$\mathcal{L}_{\mathrm{CFM}}(\theta) = \mathbb{E}_{t,\; x_1 \sim p_1,\; x \sim p_t(\cdot|x_1)}\!\left[\|v_\theta(x, t) - u_t(x \mid x_1)\|^2\right].$$


This objective is tractable: it requires sampling $x_1$ from the dataset, sampling $x$ from the known conditional path $p_t(\cdot|x_1)$, and evaluating the known conditional velocity $u_t(x|x_1)$ — all operations that cost a constant per training step.

> [!theorem] 4.1 — Conditional Flow Matching Equivalence
> The marginal and conditional flow matching objectives have the same gradient with respect to the model parameters: $\nabla_\theta \mathcal{L}_{\mathrm{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\mathrm{CFM}}(\theta)$.
>
> Consequently, $\mathcal{L}_{\mathrm{FM}}$ and $\mathcal{L}_{\mathrm{CFM}}$ have the same minimizer, and minimizing the tractable conditional objective (4.3) produces a vector field that exactly transports $p_0$ to $p_1$.


The proof is a one-line application of the tower property of conditional expectation. Expanding $\mathcal{L}_{\mathrm{FM}}$ and $\mathcal{L}_{\mathrm{CFM}}$ and subtracting, the difference is:

$$\mathcal{L}_{\mathrm{CFM}}(\theta) - \mathcal{L}_{\mathrm{FM}}(\theta) = \mathbb{E}_{t,\, x_1,\, x}\!\left[\|u_t(x \mid x_1)\|^2\right] - \mathbb{E}_{t,\, x}\!\left[\|u_t(x)\|^2\right],$$

which depends only on the target velocity fields and not on $v_\theta$ at all. Taking $\nabla_\theta$ of both sides gives zero — the difference is a constant with respect to $\theta$.

This theorem is the technical lever that makes flow matching work. It says: you can replace the intractable marginal regression problem with a tractable conditional regression problem, at zero cost. The key requirement is that you can:
1. Design a conditional path $p_t(x|x_1)$ that you can sample from in closed form.
2. Derive the corresponding conditional velocity field $u_t(x|x_1)$ analytically.

Both are achievable for a large family of paths.

## 4.4 Gaussian Conditional Paths and Their Velocity Fields

The simplest — and most practically important — choice of conditional path is a Gaussian:

$$p_t(x \mid x_1) = \mathcal{N}\!\left(x;\; \mu_t(x_1),\; \sigma_t(x_1)^2 I\right),$$

where $\mu_t : \mathbb{R}^d \to \mathbb{R}^d$ is a time-dependent mean schedule and $\sigma_t : \mathbb{R}^d \to \mathbb{R}_{>0}$ is a time-dependent standard deviation schedule, both satisfying the boundary conditions $\mu_0(x_1) = 0$, $\sigma_0(x_1) = 1$ (start at noise) and $\mu_1(x_1) = x_1$, $\sigma_1(x_1) = \sigma_{\min} \approx 0$ (end concentrated at the data point).

For this family, samples from $p_t(\cdot|x_1)$ are simply $x_t = \mu_t(x_1) + \sigma_t(x_1)\, \epsilon$ for $\epsilon \sim \mathcal{N}(0, I)$, and the conditional velocity field has the closed form:

> [!equation] 4.4
> $$u_t(x \mid x_1) = \dot{\mu}_t(x_1) + \frac{\dot{\sigma}_t(x_1)}{\sigma_t(x_1)}\,\bigl(x - \mu_t(x_1)\bigr),$$


where dots denote time derivatives. This follows from the continuity equation applied to a Gaussian whose mean and variance evolve according to $\dot{\mu}_t$ and $\dot{\sigma}_t$. The two terms are interpretable: the first, $\dot{\mu}_t$, is the velocity of the mean — how fast the center of the conditional distribution moves toward $x_1$. The second, $(\dot{\sigma}_t / \sigma_t)(x - \mu_t)$, is the relative velocity of a point displaced from the mean — how fast the distribution contracts or expands around its center.

**The linear interpolant.** The simplest schedule satisfying the boundary conditions is:

$$\mu_t(x_1) = t\, x_1, \qquad \sigma_t = 1 - (1-\sigma_{\min})t \approx 1 - t.$$

With $\sigma_{\min} \to 0$, we have $x_t = t\, x_1 + (1-t)\, \epsilon$ and, substituting into (4.4):

$$u_t(x_t \mid x_1) = \frac{x_1 - \epsilon}{1} = x_1 - \epsilon.$$

The conditional velocity field at the sample point $x_t$ is exactly $x_1 - \epsilon$ — the vector from the initial noise to the target data point. It is **constant in time** and **constant along the trajectory**: every sampled point on the linear path has the same velocity, pointing directly from $\epsilon$ to $x_1$. The conditional trajectories are straight lines.

> [!equation] 4.5
> $$\mathcal{L}_{\mathrm{CFM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\; x_1 \sim p_{\text{data}},\; \epsilon \sim \mathcal{N}(0,I)}\!\left[\|v_\theta\!\left(t\, x_1 + (1-t)\,\epsilon,\; t\right) - (x_1 - \epsilon)\|^2\right].$$


Compare this with the DDPM objective (3.5): sample a data point $x_1$, sample noise $\epsilon$, form a noisy mixture $x_t$, regress the network output against a target involving $x_1$ and $\epsilon$. The structure is identical. The difference is the interpolant: DDPM uses $x_t = \sqrt{\bar\alpha_t}\, x_1 + \sqrt{1-\bar\alpha_t}\, \epsilon$ (a curved, VP-SDE-driven path), while flow matching uses $x_t = t\, x_1 + (1-t)\, \epsilon$ (a straight line). Both are simulation-free. Both cost one forward pass per gradient step. The path curvature is the only distinction — and it turns out to matter enormously for how many steps are needed at inference.

> [!remark] 4.2
> The training objective (4.5) has the same structure as every supervised learning problem in this monograph: sample an input, compute a target, minimize squared error. The input is the noisy intermediate $x_t$, the target is the displacement $x_1 - \epsilon$, and the model $v_\theta$ is any neural network. Unlike DDPM, where the target $-\epsilon/\sqrt{1-\bar\alpha_t}$ is a scaled noise prediction, the flow matching target $x_1 - \epsilon$ has a direct geometric interpretation: it is the velocity of the straight-line path from $\epsilon$ to $x_1$. The network learns to point toward the destination.


## 4.5 Stochastic Interpolants: The General Framework

Albergo and Vanden-Eijnden's "Building Normalizing Flows with Stochastic Interpolants" (2022) arrived at the same construction from a different direction, and in doing so identified the most general possible version of the idea.

A **stochastic interpolant** is a random variable of the form:

> [!equation] 4.6
> $$x_t = \alpha_t\, x_0 + \beta_t\, x_1 + \sigma_t\, z, \qquad (x_0, x_1, z) \sim p_0 \times p_1 \times \mathcal{N}(0, I),$$


where $\alpha_t, \beta_t, \sigma_t : [0,1] \to \mathbb{R}$ are smooth schedules satisfying:
- $\alpha_0 = 1, \beta_0 = 0$ (start from $p_0$ at $t = 0$)
- $\alpha_1 = 0, \beta_1 = 1$ (arrive at $p_1$ at $t = 1$)
- $\sigma_0 = \sigma_1 = 0$ (no injected noise at endpoints)

The triple $(x_0, x_1, z)$ is sampled independently: $x_0$ from the base (noise) distribution, $x_1$ from the data distribution, and $z$ from a standard Gaussian. The interpolant defines a stochastic process $\{x_t\}$ whose marginal at $t=0$ is $p_0$ and at $t=1$ is $p_1$.

The velocity of $x_t$ along a realization is:

$$\dot{x}_t = \dot{\alpha}_t\, x_0 + \dot{\beta}_t\, x_1 + \dot{\sigma}_t\, z.$$

This is a random variable. The target velocity field — the function that a deterministic ODE must learn to reproduce the marginals $\{p_t\}$ of the interpolant — is the conditional expectation:

$$u_t(x) = \mathbb{E}\!\left[\dot{\alpha}_t\, x_0 + \dot{\beta}_t\, x_1 + \dot{\sigma}_t\, z \;\Big|\; x_t = x\right].$$

Training proceeds by regressing $v_\theta(x_t, t)$ against $\dot{x}_t$:

$$\mathcal{L}_{\mathrm{SI}}(\theta) = \mathbb{E}_{t,\, x_0,\, x_1,\, z}\!\left[\|v_\theta(x_t, t) - (\dot{\alpha}_t\, x_0 + \dot{\beta}_t\, x_1 + \dot{\sigma}_t\, z)\|^2\right],$$

an objective with the same tower-property justification as Theorem 4.1 — the regression target is an unbiased sample of the conditional expectation $u_t(x_t)$.

> [!definition] 4.2 — Stochastic Interpolant
> A **stochastic interpolant** is a parametric family of probability paths $\{p_t\}$ defined by (4.6). The family subsumes:
> - **Linear flow matching**: $\alpha_t = 1-t$, $\beta_t = t$, $\sigma_t = 0$
> - **DDPM (VP-SDE)**: $\alpha_t = \sqrt{\bar\alpha_t}$, $\beta_t = 0$, $\sigma_t = \sqrt{1-\bar\alpha_t}$ (with $x_1 \leftrightarrow$ data, $x_0 \leftrightarrow$ noise)
> - **Trigonometric schedules**: $\alpha_t = \cos(\pi t/2)$, $\beta_t = \sin(\pi t/2)$, $\sigma_t = 0$ (used in Stable Diffusion 3)
> - **General**: any smooth $(\alpha_t, \beta_t, \sigma_t)$ satisfying the boundary conditions


The stochastic interpolant framework answers a question that had been implicit throughout the preceding chapters: what is the space of all valid interpolations between $p_0$ and $p_1$? The answer is: any smooth schedule satisfying the boundary conditions gives a valid interpolant, and every such interpolant can be learned by the same regression procedure. The choice of schedule determines the geometry of the paths — their curvature, their spread, the effective number of solver steps needed at inference — but not the validity or training cost of the method.

## 4.6 Rectified Flow: Straight Lines and Reflow

Liu, Gong, and Liu's "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (2022) approached the same idea from yet another direction — and contributed an insight about what happens when the training pairs $(x_0, x_1)$ are not independent.

**The basic objective.** Rectified flow defines the interpolant as $x_t = (1-t)x_0 + t\, x_1$ — a straight line between $x_0 \sim p_0$ and $x_1 \sim p_1$ — and trains $v_\theta$ to match the constant velocity $x_1 - x_0$:

$$\mathcal{L}_{\mathrm{RF}}(\theta) = \mathbb{E}_{t,\, (x_0, x_1) \sim \pi_0}\!\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right],$$

where $\pi_0 = p_0 \times p_1$ is the independent coupling — each noise sample paired randomly with each data sample. This is the same objective as (4.5) with $\epsilon = x_0$ and $x_1 = $ data. At this level, rectified flow and linear flow matching are identical.

The distinction comes from the **reflow procedure**. Even though the individual conditional paths $t \mapsto (1-t)x_0 + tx_1$ are straight lines (constant velocity), the marginal vector field $u_t(x) = \mathbb{E}[x_1 - x_0 | x_t = x]$ is *not* straight — because different $(x_0, x_1)$ pairs with $x_t = x$ will have different velocities, and the expectation curves the effective dynamics. This marginal curvature is exactly what makes multi-step ODE integration necessary at inference.

**Reflow** iterates: after training $v_\theta^{(1)}$ with the independent coupling $p_0 \times p_1$, use the learned flow to pair each noise sample $x_0 \sim p_0$ with its generated output $\hat{x}_1 = \phi_1(x_0)$ under $v_\theta^{(1)}$. The new coupling $\pi_1 = \{(x_0, \hat{x}_1)\}$ is correlated — noise samples are now matched with specific data-like outputs — and training on this coupled dataset with the same linear interpolant produces a new flow $v_\theta^{(2)}$ whose trajectories are straighter. In the limit of repeated reflowing, trajectories converge to straight lines in the marginal sense, enabling accurate one-step generation.

> [!remark] 4.3
> The reflow procedure is related to optimal transport but distinct from it. OT coupling minimizes the expected transport cost $\mathbb{E}[\|x_0 - x_1\|^2]$ over all couplings $\pi(x_0, x_1)$ with marginals $p_0$ and $p_1$. Reflow iteratively decreases this cost — each round of reflow produces a coupling whose marginals match and whose transport cost is lower than the previous — but it does not solve the OT problem exactly. The difference is in the computational cost: solving OT exactly requires solving a linear program, while reflow only requires training a neural network.


## 4.7 Optimal Transport Conditioning

The connection between flow matching and optimal transport goes deeper than the reflow approximation. The key observation — due independently to Lipman et al. (2022) and Pooladian et al. (2023) — is that the curvature of the marginal vector field is entirely determined by the coupling used to define the conditional paths.

With the independent coupling $\pi_0 = p_0 \times p_1$, the conditional paths are straight lines but the marginal paths curve — trajectories from different pairs cross each other, and the marginal velocity field must account for all these crossings. With the **optimal transport coupling** $\pi_{\mathrm{OT}}$ — the coupling that minimizes $\mathbb{E}[\|x_0 - x_1\|^2]$ — the conditional paths are still straight lines, but now they do not cross. In the absence of crossings, the conditional and marginal paths coincide, and the marginal vector field is itself straight-line.

In practice, the exact OT coupling is computationally expensive to compute for large datasets. **Mini-batch OT** is the practical substitute: for each training mini-batch of $N$ noise samples $\{x_0^{(i)}\}$ and $N$ data samples $\{x_1^{(j)}\}$, solve the discrete OT problem (a linear assignment on an $N \times N$ cost matrix) and pair each noise sample with its assigned data sample. The cost matrix entry is $\|x_0^{(i)} - x_1^{(j)}\|^2$; the assignment is solved in $O(N^3)$ time by the Hungarian algorithm or $O(N^2/\epsilon)$ time by Sinkhorn regularization.

Training with mini-batch OT conditioning reduces the curvature of the marginal vector field even within a single round of training, without requiring reflow. The result is fewer NFE needed at inference — flows trained with OT conditioning generate competitive samples with 10–50 function evaluations rather than the 1000 required by DDPM.

> [!proposition] 4.2 — OT Conditioning and Path Straightness
> Among all couplings $\pi(x_0, x_1)$ with marginals $p_0 = \mathcal{N}(0,I)$ and $p_1 = p_{\text{data}}$, the optimal transport coupling $\pi_{\mathrm{OT}}$ produces the straightest linear interpolant: the marginal vector field $u_t(x) = \mathbb{E}_{\pi_{\mathrm{OT}}}[x_1 - x_0 \mid x_t = x]$ has minimum expected change over time, $\mathbb{E}[\|\dot{u}_t(x_t)\|^2] \to 0$. Straighter paths require fewer ODE steps to integrate accurately.


## 4.8 Diffusion Models as a Special Case

The unification theorem — that diffusion models are a special case of flow matching — is now straightforward to state. The probability flow ODE from Chapter 3 (equation 3.8):

$$\frac{dx}{dt} = f(x, t) - \tfrac{1}{2}g(t)^2\, \nabla_x \log p_t(x),$$

is a flow with a specific vector field $u_t(x) = f(x,t) - \frac{1}{2}g(t)^2\nabla_x\log p_t(x)$. This vector field satisfies the continuity equation (4.1) for the marginals $\{p_t\}$ of the forward diffusion SDE. It is therefore a valid marginal velocity field in the sense of flow matching — one that arises from a particular choice of probability path.

What probability path? For the VP-SDE, the marginal at time $t$ is $p_t(x) = \int p_{\text{data}}(x_1)\mathcal{N}(x;\sqrt{\bar\alpha_t}x_1, (1-\bar\alpha_t)I)dx_1$ — a Gaussian convolution of the data distribution that shrinks the signal and inflates the noise. This is exactly the stochastic interpolant (4.6) with $\alpha_t = \sqrt{\bar\alpha_t}$, $\beta_t = 0$, $\sigma_t = \sqrt{1-\bar\alpha_t}$.

> [!remark] 4.4
> The diffusion interpolant has $\beta_t = 0$ — it contains no direct component of $x_1$ in the definition of $x_t$. The data signal enters only through the *noise* level: at low noise, $x_t \approx \sqrt{\bar\alpha_t} x_1$ is a scaled version of the data; at high noise, $x_t$ is pure Gaussian. By contrast, the linear flow matching interpolant $x_t = tx_1 + (1-t)\epsilon$ always contains a direct mixture of signal and noise. This structural difference — not just a schedule difference — is what makes diffusion paths curved and flow matching paths straight.


The VP-SDE probability flow ODE can therefore be trained with the conditional flow matching objective (4.3), using the Gaussian conditional path $p_t(x|x_1) = \mathcal{N}(x; \sqrt{\bar\alpha_t}x_1, (1-\bar\alpha_t)I)$ and its associated conditional velocity field. The resulting training objective is, after substituting the Gaussian path, equivalent to DDPM training — the objectives match up to reparametrization. This is not a coincidence; it is the theorem. The entire denoising track, from NCSN to DDPM to the Score SDE, is recoverable from the conditional flow matching construction by choosing the VP-SDE interpolant.

What diffusion models chose — implicitly, without framing it as a choice — was the VP-SDE path. Flow matching reveals that this is just one member of a large family. The most natural other member is the linear interpolant, and as we have seen, it has strictly shorter trajectories.

## 4.9 The Practical Consequences

The convergence of the two tracks is not merely a theoretical satisfaction — it has immediate practical consequences for how generative models are trained and deployed.

**Training cost is equalized.** Both DDPM and flow matching train with a single forward-backward pass per gradient step. The training cost asymmetry that separated the denoising track from CNFs is eliminated. Whatever architecture the designer chooses — U-Net, DiT, transformer — the same network can be trained under either regime with the same wall-clock cost per step.

**Inference cost is not.** DDPM requires $\sim$1000 steps; linear flow matching with OT conditioning generates competitive samples in 10–50 steps; with exact OT or reflow, this drops to 1–4 steps for many distributions. The difference is entirely in path curvature: straighter paths integrate more accurately with fewer ODE steps. Flow matching makes the inference cost a *design choice* controlled by the interpolant and the coupling, not a fixed property of the method.

**The architecture is the bottleneck.** With both tracks converged to the same training procedure, the remaining question is: what neural network architecture should parametrize the vector field? The answer — which has evolved rapidly from U-Nets to DiTs to flow transformers — is now the primary practical axis along which progress happens. The mathematical framework is settled; the engineering is not.

**Stable Diffusion 3 and beyond.** The practical validation of flow matching at scale came with Esser, Kulal, Blattmann, Rombach et al. (2024): Stable Diffusion 3 (SD3) used a rectified flow objective with a trigonometric noise schedule and a transformer architecture (the Multimodal Diffusion Transformer, MMDiT), and produced significantly better samples with fewer steps than its diffusion-based predecessors. Meta's Movie Gen (2024) and OpenAI's Sora (2024) use related flow-based objectives. The shift is industry-wide: as of 2024, flow matching has largely displaced pure diffusion objectives for new large-scale model training.

## 4.10 What the Convergence Reveals

The convergence of two tracks into one framework has a deeper significance than the practical improvements it enables. It reveals the shape of the entire landscape.

**The transport problem is solved, modulo approximation.** The question that opened Chapter 1 — can we find a learnable map $T_\theta$ that transports $p_0$ to $p_{\text{data}}$? — is answered affirmatively and constructively by flow matching. Given any data distribution from which we can sample, and any interpolant $\{p_t\}$ connecting it to a Gaussian, we can train a vector field by regression to transport one to the other, at the cost of a single forward pass per step, with no approximation other than the finite capacity of the neural network.

**The score function is not fundamental.** The denoising track — NCSN, DDPM, the Score SDE — presented the score $\nabla_x \log p_t(x)$ as the central object. The probability flow ODE is expressed in terms of the score. Flow matching reveals that the score is only *one* way to parametrize the velocity field: for the VP-SDE path, the score determines the vector field; for the linear path, the vector field is determined directly by the interpolant velocity $x_1 - \epsilon$ without any reference to the score. The score is a derived quantity, not a primitive one.

**The space of valid paths is large.** Every smooth interpolant satisfying the boundary conditions defines a valid flow matching problem. Different interpolants give different tradeoffs between path curvature, training difficulty, and inference speed. The optimal interpolant — in the sense of minimizing expected NFE at a given approximation accuracy — is an open research question, with evidence pointing toward the straight-line interpolant under OT coupling as optimal among scale-invariant families.

**The remaining chapters build on this foundation.** Part II develops the transport-theoretic underpinnings: Chapter 5 gives the full optimal transport theory behind path straightening, Chapter 6 develops Schrödinger bridges and entropic transport, and Chapter 7 explores the SDE–ODE duality and drift-based generation. Part III then extends the framework: Chapter 8 subsumes everything under the generator matching framework, Chapter 9 extends to discrete and Riemannian spaces, and Chapter 10 handles joint generation on product spaces. Part IV addresses the remaining challenges of efficient sampling, Monte Carlo correction, and guided generation.

## 4.11 Historical Notes

The three founding papers of flow matching appeared as preprints in late 2022 and as conference papers at ICLR 2023:

**Lipman, Chen, Ben-Hamu, Nickel, and Le (ICLR 2023)** ("Flow Matching for Generative Modeling") introduced the conditional flow matching theorem (Theorem 4.1) and Gaussian conditional paths, establishing the general framework. **Albergo and Vanden-Eijnden (ICLR 2023)** ("Building Normalizing Flows with Stochastic Interpolants") introduced the stochastic interpolant framework (Definition 4.2), emphasizing the role of the coupling between $x_0$ and $x_1$ and the connection to stochastic analysis. **Liu, Gong, and Liu (ICLR 2023)** ("Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow") introduced the reflow procedure and established the connection between coupling straightness and inference efficiency.

A closely related concurrent work — **Albergo, Boffi, and Vanden-Eijnden (2023)** ("Stochastic Interpolants: A Unifying Framework for Flows and Diffusions") — extended the interpolant framework to include stochastic trajectories and made the connection to SDEs explicit. Ma, Gong, and colleagues ("Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers", 2024) demonstrated scaled-up performance across image benchmarks.

The optimal transport connection was developed by **Pooladian, Ben-Hamu, Domingo-Enrich, Amos, Lipman, and Chen (2023)** ("Multisample flow matching: Straightening flows with minibatch couplings") and **Tong, Malkin, Huguet, Zhang, Rector-Brooks, Fatras, Wolf, and Bengio (2024)** ("Improving and Generalizing Flow-Matching for Source-Free Domain Adaptation"), who showed that mini-batch OT conditioning substantially reduces NFE requirements.

The large-scale validation came with **Stable Diffusion 3** (Esser et al., 2024, "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis") which used a cosine-schedule rectified flow trained with a transformer architecture, achieving state-of-the-art image quality. The theoretical analysis connecting flow matching to the kinetic optimal transport formulation of Brenier (1991) is developed in **Albergo, Boffi, and Vanden-Eijnden (2023)** and **Liu (2022)** ("Rectified Flow: A Marginal Preserving Approach to Optimal Transport").

The continuity equation and its role in generative modeling was discussed in the CNF literature from the beginning (Chen et al., 2018), but its role as the *definition* of a generative model — separate from any training procedure — was first made explicit in **Lipman et al. (2023)**. The observation that diffusion models are a special case of flow matching appears in all three founding papers and was also noted independently in **Song (2023)** ("Consistency Models") and **Karras et al. (2022)** ("Elucidating the Design Space of Diffusion-Based Generative Models"), which systematically explored the space of diffusion schedules and presaged the flow matching perspective.
