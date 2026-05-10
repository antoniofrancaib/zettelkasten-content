---

title: "When Architecture Becomes Dynamics"
subtitle: "Neural ODEs, continuous normalizing flows, and the simulation bottleneck — 2018 to 2019"
---



There is a moment in the history of a field when a change of perspective reveals that two apparently different things are secretly the same. The identification of ResNets with ODE discretizations was such a moment. It did not immediately solve any open problem, but it reframed the architectural design question — *how many layers, how much residual scaling?* — as a question about numerical analysis: *what step size, what integrator?* And that reframing opened a door that led, within a few years, directly to flow matching.

Chapter 1 left us with a working theory of normalizing flows and a precise description of their limitations: tractable Jacobians require architectural structure, and architectural structure means coordinate ordering, and coordinate ordering means inductive bias not present in the data. The question is whether there is an escape — a way to compute likelihoods exactly without restricting the architecture. The answer, arrived at independently by several groups in 2018, is that you can trade the architectural constraint for a dynamical one: instead of building the transport map $T_\theta$ as a composition of finitely many structured layers, build it as the time-1 map of a *differential equation*. The Jacobian determinant then satisfies its own differential equation, one that requires no special structure to be tractable.

## 2.1 The Discrete Precursor: ResNets as Euler Steps

The observation that launched this line of research was made explicit by Weinan E (2017) and by Lu et al. (2017, "Beyond Finite Layer Neural Networks: Bridging Deep Architectures and Numerical Differential Equations"), but it had been implicit in the residual network literature since He et al. (2016). A residual network layer is:

$$x^{(\ell+1)} = x^{(\ell)} + f_\theta(x^{(\ell)}, \ell),$$

where $f_\theta$ is a sub-network parameterized by the layer index $\ell$. Written this way, $x^{(\ell)}$ is a sequence, $x^{(\ell+1)} - x^{(\ell)} = f_\theta(x^{(\ell)}, \ell)$ is the increment, and the update rule is precisely **Euler's method** for the ODE:

$$\frac{dx}{d\ell} = f_\theta(x, \ell),$$

with a step size of 1 and the continuous variable $\ell$ playing the role of time. A ResNet with $L$ layers is the Euler discretization of this ODE on the interval $[0, L]$ with unit step size.

This realization suggests a natural question: what happens in the limit as the step size goes to zero and the number of layers goes to infinity, keeping the total "depth" $L$ fixed? The answer is the **Neural ODE** (Chen, Rubanova, Bettencourt, Duvenaud, NeurIPS 2018) — the most cited paper of NeurIPS 2018 and one of the most influential in the decade.

> [!definition] 2.1 — Neural ODE
> A **Neural ODE** is a model whose hidden state $h(t) \in \mathbb{R}^d$ evolves according to:
>
> $$\frac{dh}{dt} = f_\theta(h(t), t), \qquad t \in [0, T],$$
>
> where $f_\theta : \mathbb{R}^d \times [0,T] \to \mathbb{R}^d$ is a neural network. The output $h(T)$ of the model is the solution at time $T$, computed by a numerical ODE solver. Gradients with respect to $\theta$ are computed via the **adjoint method**, without backpropagating through the solver's internal states.


The conceptual shift is significant. In a standard deep network, the "architecture" — the number of layers, the sequence of transformations — is a fixed design choice made before training. In a Neural ODE, the "architecture" is the continuous-time vector field $f_\theta$, and the number of solver steps taken at inference is a run-time parameter, not a training-time decision. Deep networks discretize dynamics at a fixed resolution; Neural ODEs represent the dynamics directly and discretize adaptively.

## 2.2 The Adjoint Method: Training Without Backpropagating Through the Solver

The immediate practical challenge for Neural ODEs is computing gradients of a loss $\mathcal{L} = \mathcal{L}(h(T))$ with respect to the parameters $\theta$. Naively, one would store all intermediate states of the ODE solver (the forward pass) and backpropagate through them. But this requires $O(N)$ memory for $N$ solver steps and makes the gradient computation dependent on the solver's internal implementation — incompatible with adaptive solvers that vary the step count.

The **adjoint method** resolves both issues. Define the **adjoint state** $a(t) = d\mathcal{L}/dh(t)$. It satisfies a *backward* ODE:

> [!equation] 2.1
> $$\frac{da}{dt} = -a(t)^\top \frac{\partial f_\theta}{\partial h}(h(t), t),$$


initialized at $a(T) = d\mathcal{L}/dh(T)$ and integrated backward in time. The gradient with respect to $\theta$ is then:

$$\frac{d\mathcal{L}}{d\theta} = -\int_0^T a(t)^\top \frac{\partial f_\theta}{\partial \theta}(h(t), t)\, dt.$$

The key insight is that the adjoint ODE can be solved *simultaneously* with the backward reconstruction of $h(t)$, using $O(d)$ memory at any point in time — independent of the number of solver steps. This makes Neural ODEs memory-efficient even for very fine discretizations.

> [!remark] 2.1
> The adjoint method for ODEs has a long history in control theory (Pontryagin's maximum principle, 1956) and numerical optimization (Pironneau, 1974). Chen et al.'s contribution was to recognize its applicability to neural network training and to implement it in a way compatible with automatic differentiation frameworks. The paper also showed that Neural ODEs define reversible dynamics — running the ODE backward recovers the initial state exactly — making them naturally suited for flow-based generative modeling.


## 2.3 Continuous Normalizing Flows

The application of Neural ODEs to generative modeling is immediate. If $f_\theta$ defines the dynamics of a particle $h(t)$ starting from noise $h(0) \sim p_0 = \mathcal{N}(0, I_d)$, then $h(1)$ is a sample from some distribution $p_1 = p_\theta$. We want $p_\theta \approx p_{\text{data}}$.

For this to be useful, we need to evaluate $\log p_\theta(x)$ for any data point $x$. The starting point is the change-of-variables formula — but now the map $T_\theta : h(0) \mapsto h(1)$ is defined by the flow of the ODE, not a composition of discrete layers. The challenge is computing the log-determinant of the Jacobian of this flow map.

The answer is the **instantaneous change-of-variables formula**, due to Liouville (1838) in the deterministic mechanics context and Chen et al. (2018) in the neural network context:

> [!theorem] 2.1 — Instantaneous Change of Variables
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

**The resulting model — a neural network trained as the right-hand side of an ODE, with likelihoods computed via the trace integral — is called a Continuous Normalizing Flow (CNF)**. The name honors the fact that this is the continuous-time limit of the normalizing flow architecture from Chapter 1, achieving exact likelihoods with no architectural constraints: $f_\theta$ can be any neural network whatsoever.

## 2.4 FFJORD: Free-Form Jacobian via Hutchinson's Estimator (2018)

The $O(d^2)$ cost of the full trace was reduced to $O(d)$ by Grathwohl, Chen, Bettencourt, Sutskever, and Duvenaud in **FFJORD** (Free-Form Jacobian of Reversible Dynamics, 2018), using a classical trick from numerical linear algebra: **Hutchinson's trace estimator** (Hutchinson, 1989).

For any square matrix $A \in \mathbb{R}^{d \times d}$, if $\epsilon \sim \mathcal{N}(0, I_d)$ or $\epsilon \sim \mathrm{Uniform}(\{-1,+1\}^d)$, then:

$$\mathrm{tr}(A) = \mathbb{E}_\epsilon[\epsilon^\top A \epsilon].$$

This identity holds exactly in expectation. Applied to the Jacobian of $f_\theta$:

$$\nabla \cdot f_\theta(x, t) = \mathrm{tr}\!\left(\frac{\partial f_\theta}{\partial x}\right) = \mathbb{E}_\epsilon\!\left[\epsilon^\top \frac{\partial f_\theta}{\partial x} \epsilon\right].$$

The matrix-vector product $(\partial f_\theta / \partial x)\, \epsilon$ can be computed via a single backward pass (vector-Jacobian product), costing the same as one forward pass of $f_\theta$. The trace estimator therefore requires $O(d)$ operations per time step, with variance that can be reduced by averaging over multiple $\epsilon$ samples.

> [!equation] 2.2
> $$\log p_1(x_1) = \log p_0(x_0) - \int_0^1 \mathbb{E}_\epsilon\!\left[\epsilon^\top \frac{\partial f_\theta}{\partial x}(x_t, t)\, \epsilon\right] dt.$$


In practice, the stochastic trace estimator introduces some variance into the log-likelihood, but this can be controlled by using a fixed $\epsilon$ per data point (the same noise vector throughout the ODE integration for a given $x$). The variance decreases as the vector field becomes "more diagonal" in the basis aligned with $\epsilon$ — which, interestingly, is precisely the kind of decorrelated dynamics that the training procedure tends to encourage.

FFJORD was a significant practical breakthrough. A CNF with a standard transformer or MLP as $f_\theta$, trained via maximum likelihood with the Hutchinson trace estimator, could in principle model *any* continuous distribution — no coupling layers, no coordinate ordering, no autoregressive structure. It achieved state-of-the-art density estimation results on tabular datasets (UCI benchmarks) and produced cleaner samples than normalizing flows on some image datasets.

## 2.5 The Architecture Freedom and Its Costs

The freedom from architectural constraints that CNFs promised was real, and researchers quickly demonstrated capabilities that coupling-layer flows could not match. Without the coupling structure, a CNF could:

- Model distributions with complex topological structure (multi-modal distributions with non-trivially connected support)
- Apply naturally to data lying on lower-dimensional manifolds embedded in $\mathbb{R}^d$
- Use architectures designed for the data domain — U-Nets for images, transformers for sequences — as the vector field, without modification

But this freedom came with a cost that would take several years to fully appreciate: **training a CNF requires simulating the ODE**. And this simulation is expensive in exactly the ways that matter for large-scale training.

Consider what maximum likelihood training of a CNF on a mini-batch of $B$ data points requires:

1. **Forward ODE solve**: for each $x_1$ in the mini-batch, numerically integrate the ODE backward from $x_1$ to find $x_0 = \phi_0(x_1)$. This requires $N_{\text{forward}}$ evaluations of $f_\theta$.
2. **Trace integration**: simultaneously integrate $\int_0^1 \nabla \cdot f_\theta(x_t, t)\, dt$ along each backward trajectory — $N_{\text{forward}}$ more evaluations of the Jacobian-vector product.
3. **Adjoint backward solve**: integrate the adjoint ODE (2.1) to compute gradients — another $N_{\text{backward}}$ evaluations of $f_\theta$ and its Jacobian.

Total cost per gradient step: $O(B \cdot (N_{\text{forward}} + N_{\text{backward}}) \cdot C_f)$, where $C_f$ is the cost of one $f_\theta$ evaluation. Compare this with a normalizing flow, where the gradient step costs $O(B \cdot C_f)$ — just one forward pass per data point.

The number of solver steps $N$ is not a free parameter: it is determined by the stiffness of the ODE, which is in turn determined by the complexity of $f_\theta$. For a highly nonlinear $f_\theta$ — which is exactly what's needed to model complex distributions — $N$ can be in the hundreds or thousands. A training step that is $100\times$ more expensive than the normalizing flow baseline is not unusual.

> [!remark] 2.2
> There is a further subtlety: the adjoint method (Section 2.2) avoids storing intermediate states but requires *re-running the forward ODE* during the backward pass, effectively doubling the forward computation. For some architectures, storing the forward trajectory (checkpointing) is cheaper, accepting $O(N)$ memory for $O(1)$ forward computation — the standard autograd approach. Modern CNF training typically uses the adjoint method for memory efficiency and accepts the $2\times$ computation overhead.


## 2.6 The Simulation Bottleneck

The combination of ODE simulation during training and exact likelihood computation during evaluation defined the state-of-the-art CNF of 2018–2020. These models were mathematically beautiful and practically useful for small-scale problems, but the simulation requirement created a hard wall against scaling.

To understand why, consider the training pipeline for an image CNF at the scale of ImageNet ($d = 3 \times 256 \times 256 \approx 200{,}000$ dimensions):

- Each training step requires solving an ODE in $\mathbb{R}^{200{,}000}$, with $O(100)$ function evaluations.
- Each function evaluation of $f_\theta$ costs as much as a forward pass of a large neural network.
- The Hutchinson trace estimator adds another backward pass per step.
- For a mini-batch of 256 images, the total cost is roughly $256 \times 100 \times 3 \approx 76{,}800$ neural network forward passes per gradient update.

Compare this with a diffusion model of the same scale: each training step requires exactly *one* forward pass of the neural network (predicting the noise), regardless of the sequence length or data dimension. The asymmetry is not a constant factor — it grows with the complexity of the distribution, because more complex distributions require more solver steps to integrate accurately.

This is the **simulation bottleneck**: the exact likelihoods that CNFs provide come at the cost of ODE simulation during training, and this cost scales adversarially with the complexity of the target distribution. The very property that makes CNFs difficult to train — complex, rapidly-changing vector fields — is also the property that makes them expressive.

> [!definition] 2.2 — Number of Function Evaluations (NFE)
> The **number of function evaluations** (NFE) is the total number of times the vector field $f_\theta$ is evaluated by the ODE solver during a single forward pass (generation) or backward pass (training). NFE is the primary computational cost metric for continuous normalizing flows. At inference, a well-trained flow typically requires $10$–$100$ NFE; during training with the adjoint method, the cost is $2$–$3 \times$ higher.


## 2.7 What CNFs Contributed to the Story

Despite — or perhaps because of — the simulation bottleneck, continuous normalizing flows made two contributions that are foundational for the remainder of this monograph.

**The flow map as a first-class object.** The CNF framework establishes the flow map $\phi_t : \mathbb{R}^d \to \mathbb{R}^d$ — the map sending initial conditions at time 0 to positions at time $t$ — as the fundamental object of interest. The flow map is an *interpolant*: it describes a continuous path from the noise distribution $p_0$ to the data distribution $p_1$, passing through a family of intermediate distributions $\{p_t = (\phi_t)_\# p_0\}_{t \in [0,1]}$. This interpolation structure — which is absent from normalizing flows, where there is only an input and an output — becomes the central object in Chapters 4 and 5.

**The instantaneous change-of-variables formula.** Theorem 2.1 establishes that the log-density satisfies its own ODE, driven by the divergence of the vector field. This is not just a computational formula; it is a profound statement about the geometry of probability transport. It says that probability mass is locally conserved: the rate at which a small volume element changes is determined by how much the flow is diverging (expanding) or converging (compressing). This perspective connects directly to the continuity equation of fluid mechanics, which will be the starting point for flow matching.

## 2.8 The Parallel Track: Something Else Was Happening

As CNFs were being developed and refined — Kidger, Morrill, Foster, and Lyons' "Neural Controlled Differential Equations" (2020), Kidger's comprehensive thesis *"On Neural Differential Equations"* (2022) systematizing the whole enterprise — a completely separate and initially unrelated line of research was arriving at what appeared to be a wholly different answer to the generative modeling problem.

This parallel track did not start with transport, or invertible maps, or Jacobians. It started with a much simpler-sounding idea: *destroying structure is easy; can we learn to reverse the destruction?*

Sohl-Dickstein, Weiss, Maheswaranathan, and Ganguli proposed this idea in 2015 — the same year NICE appeared — in a paper titled "Deep Unsupervised Learning using Nonequilibrium Thermodynamics." The paper was largely ignored for five years. And then, in 2019 and 2020, it came back with a force that would reshape the entire field.

That story — the denoising track, its sudden explosion, and its unexpected convergence with the CNF track you have just read — is Chapter 3.

## 2.9 Historical Notes

**Chen, Rubanova, Bettencourt, and Duvenaud (NeurIPS 2018)** introduced Neural ODEs and the adjoint method for training, establishing continuous-time dynamics as a viable framework for deep learning. **Grathwohl, Chen, Bettencourt, Sutskever, and Duvenaud (ICLR 2019)** introduced FFJORD, making CNF training practical with the Hutchinson trace estimator.

The theoretical foundations draw on a long tradition: the instantaneous change-of-variables formula is Liouville's theorem (1838) for Hamiltonian systems, generalized to arbitrary divergence-free dynamics in the fluid mechanics literature. The adjoint method as used here traces to **Pontryagin's maximum principle** (1956) and was developed for PDE-constrained optimization by Pironneau (1974) and Lions (1971).

The systematic study of the expressivity and theoretical properties of Neural ODEs — including universal approximation, approximation rates, and the relationship to discrete ResNets — is developed in **Kidger's thesis** (*"On Neural Differential Equations"*, 2022), which remains the most comprehensive single reference for this material.

The connection between residual networks and ODE discretizations was developed explicitly by **Weinan E (2017)** ("A Proposal on Machine Learning via Dynamical Systems") and **Lu, Zhong, Li & Dong (2017)** ("Beyond Finite Layer Neural Networks: Bridging Deep Architectures and Numerical Differential Equations"). See also **Haber & Ruthotto (2017)** and **Chang et al. (2018)** for the stability and reversibility perspectives.
