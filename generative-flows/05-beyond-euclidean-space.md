---

title: "The Geometry of Data"
subtitle: "Discrete flows via CTMCs, Riemannian flow matching, and flows on the simplex — 2023 to 2025"
---



The flow matching framework of Chapter 4 is formulated for data in $\mathbb{R}^d$ — the Euclidean space where the linear interpolant $x_t = tx_1 + (1-t)\epsilon$ is defined by scalar multiplication, where the velocity is a vector, and where the continuity equation takes its canonical form. For a large fraction of the most scientifically important data modalities, this is the wrong space.

Protein structures are not points in $\mathbb{R}^d$. They are sequences of rigid frames, each living in $SE(3) = \mathbb{R}^3 \times SO(3)$ — the group of translations and rotations in three-dimensional space. Molecular conformations are constrained to the torus — the product of circles parametrizing dihedral bond angles. DNA and protein sequences are discrete: their natural space is a Cartesian product of finite alphabets, not a continuum. The probability simplex — the home of categorical distributions over tokens — is a bounded manifold with an intrinsic geometric structure that Euclidean interpolation violates at its boundary. RNA structures, crystal lattices, point clouds on surfaces: the spaces of scientific data are curved, discrete, bounded, or combinatorially structured in ways that straight-line interpolation simply does not see.

Part II of this monograph — beginning here — is the story of extending the framework of Chapters 1–4 to these non-Euclidean settings. The extensions are not afterthoughts. They are the developments that have made generative modeling genuinely useful across molecular biology, structural chemistry, and genomics, and they have forced a clarification of what the framework is really about. The answer — which Chapter 6 will make precise — is that the continuity equation and its discrete analogue are the fundamental objects, and Euclidean flow matching is just one instantiation of a much more general principle.

## 5.1 What Fails in Non-Euclidean Settings

It is worth being precise about what breaks when the data space is not $\mathbb{R}^d$, because different structures break in different ways.

**Linearity fails on manifolds.** The interpolant $x_t = tx_1 + (1-t)x_0$ requires that convex combinations of data points be valid. On a sphere, the midpoint of two unit vectors is not a unit vector — it has norm strictly less than 1. On a rotation group, the average of two rotation matrices is not a rotation matrix. On the simplex, the boundary is a barrier: a point with $x_i = 0$ is valid, but a convex combination $tx_1 + (1-t)x_0$ can easily place probability mass outside $[0,1]$ when $x_0$ and $x_1$ have different support. Every interpolant that relies on linear combination fails on manifolds.

**Vector fields cannot leave the tangent bundle.** On a manifold, the velocity of a trajectory must be tangent to the manifold at every point — it must lie in the tangent space $T_x\mathcal{M}$. A neural network parametrizing a velocity field in $\mathbb{R}^d$ will generically produce vectors that point off the manifold, requiring an explicit projection or a change of parametrization.

**Euclidean noise is the wrong base distribution.** The standard Gaussian $\mathcal{N}(0,I)$ is the natural reference distribution on $\mathbb{R}^d$. On a sphere, the natural reference is the uniform distribution. On $SO(3)$, it is the Haar measure. On the simplex, it is the symmetric Dirichlet distribution. Each manifold carries its own canonical "uninformative" distribution, and the noise-to-data interpolant must connect the data distribution to this reference, not to a Gaussian.

**Discreteness requires a different calculus.** Continuous vector fields and ODEs have no analogue on a discrete state space. A point in $\{A, C, G, T\}^L$ — a DNA sequence of length $L$ — cannot be continuously deformed: there are no intermediate states between two sequences. The appropriate language is probability theory on graphs: Markov chains rather than ODEs, rate matrices rather than vector fields, the Kolmogorov equation rather than the continuity equation.

## 5.2 Continuous-Time Markov Chains: The Discrete Backbone

The right language for generative modeling on discrete state spaces is **continuous-time Markov chains** (CTMCs). These are the natural discrete-space analogue of the SDEs of Chapter 3 and the ODEs of Chapter 4 — stochastic processes that jump between discrete states at random times, with instantaneous transition rates that can depend on the current state and time.

Let $\mathcal{X}$ be a finite (or countable) state space — a vocabulary, a set of graph structures, a space of discrete tokens. A CTMC on $\mathcal{X}$ is defined by a time-dependent **rate matrix** $R_t$:

> [!definition] 5.1 — Rate Matrix and CTMC
> A **rate matrix** $R_t$ at time $t$ is a matrix with entries $R_t(x, y)$ for $x, y \in \mathcal{X}$ satisfying:
> - $R_t(x, y) \geq 0$ for all $x \neq y$ (non-negative off-diagonal entries),
> - $R_t(x, x) = -\sum_{y \neq x} R_t(x, y)$ (rows sum to zero).
>
> The entry $R_t(x, y)$ is the instantaneous rate of jumping from state $x$ to state $y$ at time $t$. A **continuous-time Markov chain** (CTMC) with rate matrix $R_t$ is a stochastic process $\{X_t\}$ on $\mathcal{X}$ in which, in a small interval $[t, t+dt)$, the probability of transitioning from $X_t = x$ to $X_{t+dt} = y \neq x$ is $R_t(x, y)\, dt + O(dt^2)$.


The marginal distribution $p_t(x) = \mathbb{P}(X_t = x)$ of a CTMC evolves according to the **Kolmogorov forward equation** — the discrete analogue of the continuity equation (4.1):

> [!equation] 5.1
> $$\frac{d}{dt} p_t(x) = \sum_{y \in \mathcal{X}} p_t(y)\, R_t(y, x) = \sum_{y \neq x}\!\left[p_t(y)\, R_t(y, x) - p_t(x)\, R_t(x, y)\right].$$


The two terms in the last expression have a clean interpretation: the first is the probability flux *into* state $x$ from all other states; the second is the probability flux *out of* $x$ to all other states. Equation (5.1) says that the rate of change of $p_t(x)$ is the net inflow of probability mass — precisely the discrete analogue of conservation of probability mass in the continuity equation.

> [!remark] 5.1
> The analogy between the Kolmogorov equation (5.1) and the continuity equation (4.1) is exact in the following sense: both express conservation of total probability ($\int p_t\, dx = 1$ or $\sum_x p_t(x) = 1$) under a generative process. The vector field $u_t$ in (4.1) is replaced by the rate matrix $R_t$ in (5.1). The divergence $\nabla \cdot (p_t u_t)$ is replaced by the generator applied to $p_t$. This parallelism is not coincidental — it is the discrete-space instance of the general Markov generator framework that Chapter 6 will make precise.


## 5.3 Discrete Flow Matching

The discrete analogue of flow matching is immediate: instead of designing a probability path $\{p_t\}$ on $\mathbb{R}^d$ and deriving its vector field, design a probability path $\{p_t\}$ on $\mathcal{X}$ and derive its rate matrix from the Kolmogorov equation.

Given a target path with $p_0 = p_{\text{noise}}$ (a simple reference distribution on $\mathcal{X}$, often uniform) and $p_1 = p_{\text{data}}$, the **minimum-norm rate matrix** consistent with the Kolmogorov equation is:

$$R_t(x, y) = \frac{\left[\partial_t p_t(y)\right]_+}{p_t(x)}, \qquad x \neq y,$$

where $[\,\cdot\,]_+$ denotes the positive part, and $\partial_t p_t(y)$ is the time derivative of the marginal at $y$. This is the exact discrete analogue of the minimum-norm velocity field from Section 4.1. The problem, as before, is that computing this marginal rate matrix requires integrating over all data points that could generate a given $x_t$ — an intractable marginalization.

The conditional flow matching theorem extends verbatim to discrete spaces. Define conditional paths $p_t(x|x_1)$ for each data point $x_1 \sim p_1$, with known conditional rate matrices $R_t(x, y|x_1)$ satisfying the Kolmogorov equation for $p_t(\cdot|x_1)$. Then define the **discrete conditional flow matching objective**:

> [!equation] 5.2
> $$\mathcal{L}_{\mathrm{dCFM}}(\theta) = \mathbb{E}_{t,\; x_1 \sim p_1,\; x \sim p_t(\cdot|x_1)}\!\left[\sum_{y \neq x} \left|R_t^\theta(x, y) - R_t(x, y \mid x_1)\right|^2 \cdot p_t(x|x_1)\right].$$


As in the continuous case, the conditional and marginal objectives share the same minimizer — the tower-property argument applies identically. Minimizing (5.2) requires only sampling $x_1$ from data, sampling $x$ from the known conditional path, and evaluating the known conditional rate. No CTMC simulation during training. One forward pass per gradient step.

## 5.4 The Masking Interpolant

The canonical conditional path for discrete flow matching is the **masking interpolant** — a construction that turns out to be intimately connected to masked language models.

Augment the state space $\mathcal{X}$ with a special absorbing state $\text{[M]}$ (the mask token). Define the conditional path:

$$p_t(x \mid x_1) = \begin{cases} 1 - t & x = \text{[M]} \\ t & x = x_1 \\ 0 & \text{otherwise.}\end{cases}$$

At time $t=0$, the token is always masked. At time $t=1$, it is always the true token $x_1$. The forward process sends data to masks; the reverse process denoises masks into tokens.

The conditional rate matrix follows from the Kolmogorov equation applied to $p_t(\cdot|x_1)$. Since $\partial_t p_t(\text{[M]}|x_1) = -1$ and the masked state can only transition to the true token, the unique consistent conditional rate is:

> [!equation] 5.3
> $$R_t\!\left(\text{[M]},\, x_1\;\Big|\; x_1\right) = \frac{1}{1-t}, \qquad R_t(x_1, y \mid x_1) = 0 \text{ for all } y.$$


The rate $1/(1-t)$ diverges as $t \to 1$, forcing rapid unmasking near the end of the process — intuitive, since at $t=1$ the token must be fully revealed. The true token, once revealed, is absorbing: it never transitions back to the mask.

The marginal rate matrix (averaging over $x_1$) is:

$$R_t\!\left(\text{[M]}, y\right) = \mathbb{E}_{p_1(x_1|\text{[M]})}\!\left[R_t(\text{[M]}, y \mid x_1)\right] = \frac{p_1(y)}{1-t}.$$

The rate of unmasking to token $y$ is proportional to the prior probability of $y$ in the data, scaled by $1/(1-t)$.

> [!proposition] 5.1 — Masking is Masked Language Modeling
> Training a rate model $R_t^\theta(\text{[M]}, y)$ by the discrete conditional flow matching objective (5.2) with the masking interpolant (5.3) is equivalent to training a weighted masked language model: given a masked token at time $t$, predict the true token $x_1$ with cross-entropy loss weighted by $1/(1-t)$.


This theorem — observed independently by Gat et al. and Campbell et al. in 2024 — provides a precise theoretical grounding for the empirical success of BERT-style masked language models as generative models. The masking interpolant is the canonical discrete flow; masked language model training is the corresponding conditional flow matching objective. The connection extends to the multi-token case: for a sequence $x_1 = (x_1^1, \ldots, x_1^L)$, masking each token independently at rate proportional to $t$ gives a joint masking process, and the discrete flow matching objective reduces to a weighted sum of per-token cross-entropy losses — precisely what BERT is trained with.

## 5.5 Beyond Masking: Uniform and Absorbing Noise Processes

The masking interpolant is not the only choice. Two other canonical forward processes have complementary properties.

**Uniform noise.** The reference $p_0 = \text{Uniform}(\mathcal{X})$ is the maximum-entropy distribution over tokens. The conditional path interpolates between uniform noise and the true token: $p_t(x|x_1) = (1-t)/|\mathcal{X}| + t\, \delta_{x,x_1}$. This is the discrete analogue of the Gaussian interpolant: the token is not absent (as with masking) but corrupted — replaced with a random token. The conditional rate matrix sends every token to $x_1$ at rate $1/((1-t))$, and from $x_1$ to others at a noise-injection rate.

The uniform interpolant generalizes naturally to continuous-time corruptions: for each step, a token is replaced with a uniform random token, or left unchanged. D3PM (Austin et al., 2021) introduced this framework under the name "multinomial diffusion" in the discrete diffusion context; its connection to discrete flow matching was later made explicit.

**Absorbing at data.** Alternatively, one can run the interpolant in the other direction: start with the data token $x_1$ and corrupted it toward a reference by progressively absorbing into a noise state. This produces a "data-to-noise" path that is the time-reverse of the masking interpolant, and corresponds to a forward process where tokens are unmasked at the start and masked at the end — the direction used in masked diffusion models.

> [!remark] 5.2
> There is a precise analogy between discrete and continuous flow matching at the level of the training objective. In continuous FM (Chapter 4), the target for the vector field regression is $u_t(x_t|x_1) = (x_1 - x_t)/(1-t)$ — the velocity pointing from the current noisy state toward the clean data, amplified as $t \to 1$. In discrete FM with masking, the target for the rate model is $R_t(\text{[M]}, x_1|x_1) = 1/(1-t)$ — the rate of transitioning from masked to clean, also amplified as $t \to 1$. The structure is identical: learn to de-corrupt at an increasing rate as you approach the data distribution.


## 5.6 Riemannian Manifolds: A Working Primer

The second major extension addresses data that is continuous but lives on a curved space. The minimal mathematical machinery required is a Riemannian manifold — a smooth space equipped with a notion of distance and angle.

A **Riemannian manifold** $(\mathcal{M}, g)$ consists of a smooth manifold $\mathcal{M}$ and a Riemannian metric $g$: a family of positive definite inner products $g_x : T_x\mathcal{M} \times T_x\mathcal{M} \to \mathbb{R}$, one for each point $x \in \mathcal{M}$, varying smoothly with $x$. The tangent space $T_x\mathcal{M}$ is the space of all velocity vectors of smooth curves through $x$; it is a vector space of the same dimension as $\mathcal{M}$.

The two maps that make computation on Riemannian manifolds tractable are:

> [!definition] 5.2 — Exponential and Logarithm Maps
> The **exponential map** $\exp_x : T_x\mathcal{M} \to \mathcal{M}$ maps a tangent vector $v \in T_x\mathcal{M}$ to the point $\exp_x(v) \in \mathcal{M}$ reached by following the unique geodesic starting at $x$ with initial velocity $v$ for unit time.
>
> The **logarithm map** $\log_x : \mathcal{M} \to T_x\mathcal{M}$ (defined on a neighborhood of $x$) is the inverse: $\log_x(y)$ is the unique tangent vector at $x$ such that $\exp_x(\log_x(y)) = y$. Its magnitude $\|\log_x(y)\|_{g_x}$ equals the geodesic distance $d_\mathcal{M}(x, y)$.


**Geodesics** — the shortest paths on $\mathcal{M}$ — generalize straight lines. The geodesic from $x$ to $y$ is the curve $\gamma_t = \exp_x(t\, \log_x(y))$, $t \in [0,1]$, traversing the shortest path at constant speed. On the unit sphere $S^{d-1}$, geodesics are great circle arcs. On $SO(3)$, geodesics are rotations along a fixed axis. On the simplex with the Fisher-Rao metric, geodesics are arcs in the corresponding spherical orthant.

These maps are computable in closed form for several important manifolds:

- **Sphere** $S^{d-1}$: $\exp_x(v) = \cos(\|v\|)x + \sin(\|v\|)\hat{v}$, where $\hat{v} = v/\|v\|$.
- **$SO(3)$**: $\exp_R(V) = R \cdot \mathrm{expm}(R^\top V)$, where $\mathrm{expm}$ is the matrix exponential, and $V \in \mathfrak{so}(3)$ is a skew-symmetric matrix.
- **Symmetric positive definite matrices**: $\exp_X(V) = X^{1/2} \mathrm{expm}(X^{-1/2} V X^{-1/2}) X^{1/2}$.
- **Hyperbolic space** $\mathbb{H}^d$: explicit closed form via hyperbolic trigonometry.

## 5.7 Riemannian Flow Matching

Chen and Lipman's "Flow Matching on General Geometries" (ICLR 2024, outstanding paper) extended the conditional flow matching framework to arbitrary Riemannian manifolds, establishing a training procedure that is simulation-free, geometry-respecting, and computationally tractable for all manifolds with closed-form exponential and logarithm maps.

The conditional path on $\mathcal{M}$ is the **geodesic interpolant**: for source $x_0 \sim p_0$ (the reference distribution on $\mathcal{M}$, e.g. uniform or wrapped Gaussian) and target $x_1 \sim p_1 = p_{\text{data}}$:

> [!equation] 5.4
> $$x_t = \exp_{x_0}\!\left(t\, \log_{x_0}(x_1)\right), \qquad t \in [0, 1].$$


This is the geodesic from $x_0$ to $x_1$, evaluated at time $t$. It reduces to $x_t = x_0 + t(x_1 - x_0)$ in flat space, recovering the Euclidean linear interpolant. The conditional velocity field is the derivative of (5.4) with respect to $t$, which at time $t$ points along the geodesic toward $x_1$:

$$u_t(x_t \mid x_1) = \frac{\log_{x_t}(x_1)}{1 - t}.$$

This formula says: at any intermediate point $x_t$ on the geodesic, the target velocity points toward $x_1$ along the remaining geodesic arc, at a speed that increases as $t \to 1$ (the $1/(1-t)$ factor, matching the Euclidean formula). The conditional flow matching objective on $\mathcal{M}$ is then:

> [!equation] 5.5
> $$\mathcal{L}_{\mathrm{RFM}}(\theta) = \mathbb{E}_{t,\; x_1 \sim p_1,\; x_0 \sim p_0}\!\left[\left\|v_\theta(x_t, t) - \frac{\log_{x_t}(x_1)}{1-t}\right\|_{g(x_t)}^2\right],$$


where the squared norm is taken in the Riemannian metric at $x_t$, and $v_\theta(x_t, t) \in T_{x_t}\mathcal{M}$ is the learned velocity — a tangent vector at $x_t$.

> [!theorem] 5.2 — Riemannian Conditional Flow Matching
> On any Riemannian manifold $(\mathcal{M}, g)$ with computable exponential and logarithm maps, the conditional flow matching objective (5.5) is:
> 1. Simulation-free: each training step samples $(x_0, x_1, t)$, computes $x_t$ via (5.4) and the target via $\log_{x_t}(x_1)/(1-t)$, and takes one gradient step — no ODE solve required.
> 2. Equivalent to the marginal flow matching objective: the conditional and marginal objectives share the same minimizer.
> 3. Geometrically consistent: the learned flow stays on $\mathcal{M}$ by construction, since the velocity field $v_\theta$ is parametrized as a tangent-space-valued function.


The only computational requirement — computable $\exp$ and $\log$ maps — is satisfied in closed form for all manifolds arising in molecular and structural biology: spheres, tori, $SO(3)$, $SE(3)$, hyperbolic spaces, and the Stiefel manifold of orthonormal frames.

> [!remark] 5.3
> The formula $u_t(x_t|x_1) = \log_{x_t}(x_1)/(1-t)$ has a simple geometric interpretation: it is the "geodesic shooting" velocity that, if the ODE $\dot{x} = v_\theta(x,t)$ is solved exactly, would transport $x_t$ to $x_1$ along the remaining geodesic in the remaining time $1-t$. This is the Riemannian generalization of the Euclidean identity $u_t = (x_1 - x_t)/(1-t)$, which transports $x_t$ to $x_1$ along a straight line in time $1-t$. The Riemannian metric determines what "straight" means.


## 5.8 The Probability Simplex and Categorical Geometry

The probability simplex $\Delta^{d-1} = \{x \in \mathbb{R}^d_{\geq 0} : \sum_{i=1}^d x_i = 1\}$ deserves special treatment. It arises naturally in language modeling — where $x$ represents a distribution over a vocabulary — in population genetics, in mixture models, and in any setting where the object of interest is a probability vector over $d$ categories.

Euclidean flow matching is problematic on $\Delta^{d-1}$ for a fundamental reason: the linear interpolant $x_t = tx_1 + (1-t)x_0$ stays on the simplex (convex combinations of probability vectors are probability vectors), but the boundary of the simplex — where some $x_i = 0$ — is a fixed point of many dynamics, creating pathologies. More importantly, the Euclidean metric on $\Delta^{d-1}$ is not the natural one.

The natural metric on $\Delta^{d-1}$ is the **Fisher-Rao metric**: $g_{ij}^{\text{FR}}(x) = \delta_{ij}/x_i$. Under this metric, the simplex has a beautiful geometric structure: the map $\phi : \Delta^{d-1} \to S^{d-1}_+$ defined by $\phi(x)_i = \sqrt{x_i}$ is an isometry between $(\Delta^{d-1}, g^{\text{FR}})$ and the positive orthant of the unit sphere $S^{d-1}_+ = \{z \in S^{d-1} : z_i \geq 0\}$ with the round metric. Geodesics on the simplex correspond to great circle arcs on $S^{d-1}_+$ under this identification.

This isometry has immediate consequences for flow matching on the simplex:

**The canonical reference distribution** is the uniform distribution on $\Delta^{d-1}$ (symmetric Dirichlet $\mathrm{Dir}(\alpha = 1)$), which corresponds under $\phi$ to the uniform distribution on $S^{d-1}_+$.

**Geodesic interpolation** on the simplex is simply spherical interpolation of the square-root vectors: $\gamma_t = \phi^{-1}(\exp_{\phi(x_0)}(t\, \log_{\phi(x_0)}(\phi(x_1))))$. This avoids the boundary issues of Euclidean interpolation — great circle arcs on the positive orthant naturally stay in the interior when both endpoints are interior points.

**Dirichlet Flow Matching** (Stark, Jing, et al., 2024) exploits this structure for DNA sequence generation: represent each position in the sequence as a probability vector over $\{A, C, G, T\}$, use the Fisher-Rao geodesic as the interpolant, and train with the Riemannian flow matching objective (5.5) on the spherical parametrization. The trained flow generates DNA sequences by integrating the learned ODE from a symmetric Dirichlet sample to a one-hot vector, and the resulting sequences match the statistical properties of real genomic data better than models trained with Euclidean noise schedules.

## 5.9 SE(3): Flows for Proteins and Molecules

The application that most forcefully demonstrated the value of Riemannian flow matching was protein backbone generation — the task of generating the three-dimensional structure of a protein, residue by residue.

A protein backbone is a sequence of $N$ residues, each represented by a **frame** $(t_i, R_i) \in SE(3)$: a translation $t_i \in \mathbb{R}^3$ (the position of the $\alpha$-carbon) and a rotation $R_i \in SO(3)$ (the local orientation of the residue). The space of all protein backbones is therefore $SE(3)^N$ — a product of $N$ copies of the special Euclidean group.

$SE(3) = \mathbb{R}^3 \rtimes SO(3)$ is a Lie group (a manifold with a compatible group structure). Its geometry decomposes naturally: the translation component lives in Euclidean space ($\mathbb{R}^3$) and is handled by standard flow matching; the rotation component lives in $SO(3)$ and requires Riemannian methods.

**Geodesics on $SO(3)$.** The Lie algebra $\mathfrak{so}(3)$ consists of $3 \times 3$ skew-symmetric matrices, identifiable with $\mathbb{R}^3$ via the hat map: for $\omega \in \mathbb{R}^3$, the matrix $\hat\omega$ has $\hat\omega_{ij} = -\epsilon_{ijk}\omega_k$. The exponential map is:

$$\exp_I(\hat\omega) = I + \frac{\sin\|\omega\|}{\|\omega\|}\hat\omega + \frac{1-\cos\|\omega\|}{\|\omega\|^2}\hat\omega^2 \qquad \text{(Rodrigues' formula)},$$

and the logarithm map inverts this for $\|\omega\| < \pi$. Geodesics on $SO(3)$ are rotations about a fixed axis at constant angular velocity.

**FoldFlow** (Bose, Akhound-Sadegh, Huguet, Wolf, Rekesh, Su, Zhang, Bengio, Bhatt, Lee, Tong, Hamilton, and Bronstein, NeurIPS 2023) was the first model to apply flow matching on $SE(3)$ to protein backbone generation, using geodesic interpolation on each residue frame and a transformer architecture on the sequence of frames. The training is entirely simulation-free: sample a noise structure $\{(t_i^0, R_i^0)\}$ from a standard distribution on $SE(3)^N$, a target structure $\{(t_i^1, R_i^1)\}$ from the PDB, interpolate geodesically to get $\{(t_i^t, R_i^t)\}$, and regress the velocity field against $\{(\log_{t_i^t}(t_i^1)/(1-t), \log_{R_i^t}(R_i^1)/(1-t))\}$ using a single forward pass.

FoldFlow-2 (Huguet, Vuckovic, Fatras, Tong, Sun, Wolf, Bronstein, Tazi, and Bengio, 2024) scaled this architecture with OT conditioning on $SE(3)$: matching noise frames to data frames by a Riemannian OT cost on $SE(3)$, producing straighter geodesic paths and requiring fewer integration steps at inference. The resulting model generates protein backbones with designable, physically plausible structures at a quality competitive with — and in some metrics exceeding — diffusion-based predecessors like FrameDiff and RFdiffusion.

> [!remark] 5.4
> The separate treatment of translations (Euclidean) and rotations (Riemannian) in $SE(3)$ is justified by the product structure of the group, but it ignores the non-commutativity of rotations and translations in $SE(3)$. More principled approaches use the full Lie group geometry of $SE(3)$, where geodesics couple the translational and rotational components via the Maurer-Cartan form. For most practical protein generation tasks, the approximate product-space treatment produces excellent results; the theoretically exact Lie group treatment is relevant for applications requiring precise equivariance guarantees, such as docking and co-design.


## 5.10 Torus Flows and Dihedral Angles

One more manifold deserves mention: the torus $\mathbb{T}^k = (S^1)^k$ — the product of $k$ circles. It is the natural home of dihedral angles, the bond-angle degrees of freedom that determine molecular conformation.

A molecule with $k$ rotatable bonds has a conformation space that is a subset of $\mathbb{T}^k$: each dihedral angle $\phi_j \in (-\pi, \pi]$ parametrizes a $S^1$ factor. The metric on $\mathbb{T}^k$ is flat (no curvature), and geodesics are straight lines modulo $2\pi$ — angle interpolation. The exponential and logarithm maps are just angular addition and subtraction, with wrapping.

TorsionalDiffusion (Jing, Corso, Chang, Barzilay, Jaakkola, ICLR 2023) used torus diffusion for small-molecule conformation generation; FoldFlow's successor models extended the approach to protein side-chain dihedral angles in addition to the backbone frames. The combination of $SE(3)$ flow matching for backbones and $\mathbb{T}^k$ flow matching for side chains gives a complete conformational flow model with no Euclidean approximations.

## 5.11 The Emerging Picture

The extensions of this chapter share a common structure. In every setting — discrete tokens, spheres, $SO(3)$, the simplex, the torus — the framework is:

1. Choose a state space $\mathcal{X}$ appropriate to the data.
2. Choose a reference distribution $p_0$ on $\mathcal{X}$ (uniform on finite spaces, wrapped Gaussian or uniform on manifolds).
3. Define a conditional path $\{p_t(\cdot|x_1)\}$ from $p_0$ toward the data point $x_1$, using the natural structure of $\mathcal{X}$ (geodesics on manifolds, masking on discrete spaces).
4. Derive the conditional rate matrix or velocity field from the path.
5. Train by regression against this conditional target — one forward pass per step, no simulation.

The training cost is universal: one forward pass per gradient step, regardless of whether the space is Euclidean, curved, or discrete. The difference is entirely in the geometry of the path and the form of the regression target.

This universality suggests that there is a single abstract principle underlying all these cases — and indeed there is. The Kolmogorov equation (5.1) and the continuity equation (4.1) are both special cases of the same object: the **Kolmogorov forward equation for a Markov generator** on a general state space. The vector field $u_t$ and the rate matrix $R_t$ are both instances of a Markov generator: the infinitesimal description of a Markov process. The conditional flow matching theorem holds in full generality for any Markov generator, any state space, and any conditional path.

This is the subject of Chapter 6: a single theorem that contains normalizing flows, diffusion models, flow matching, discrete flows, and Riemannian flows as special cases, stated at the level of a general Markov generator.

## 5.12 Historical Notes

**Discrete diffusion.** The systematic study of discrete diffusion models begins with **Austin, Johnson, Ho, Kingma, and Song (NeurIPS 2021)** ("Structured Denoising Diffusion Models in Discrete State-Spaces"), which introduced D3PM — a framework for discrete diffusion with absorbing, uniform, and token-proximity forward processes. **Hoogeboom, Nielsen, Jaini, Forré, and Welling (NeurIPS 2021)** ("Argmax Flows and Multinomial Diffusion") developed the multinomial diffusion model from the normalizing flow direction. **Lou, Meng, and Ermon (ICML 2024)** ("Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution") derived a score-matching analog for discrete spaces using ratio estimators.

**Discrete flow matching.** The CTMC-based flow matching framework was developed simultaneously by **Gat, Remez, Shaul, Kreuk, Chen, Synnaeve, Adi, and Lipman (NeurIPS 2024)** ("Discrete Flow Matching") and **Campbell, Yim, Barzilay, Jaakkola, and Gat (NeurIPS 2024)** ("Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design"). The connection between the masking interpolant and masked language models (Theorem 5.1) is established in both papers. A related concurrent framework is **Shi, Holderrieth, Haber, Schmidt, and Holderrieth (2024)** ("Simplified and Generalized Masked Diffusion for Discrete Data").

**Riemannian diffusion.** The Riemannian extension of score-based models was developed by **De Bortoli, Thornton, Heng, and Doucet (NeurIPS 2022)** ("Riemannian Score-Based Generative Modelling") and **Huang, Aghajohari, Bose, Panangaden, and Courville (NeurIPS 2022)** ("Riemannian Diffusion Models"), establishing that the score SDE framework extends to Riemannian manifolds with appropriate modifications to the Fokker-Planck equation.

**Riemannian flow matching.** The simulation-free extension is **Chen and Lipman (ICLR 2024, outstanding paper)** ("Flow Matching on General Geometries"), which introduced the geodesic conditional path (5.4), the formula $u_t = \log_{x_t}(x_1)/(1-t)$ (5.5), and the premetric approximation for manifolds without closed-form geodesics. A contemporaneous independent development is **Pooladian, Ben-Hamu, Domingo-Enrich, Amos, Lipman, and Chen (2023)**, which extended OT conditioning to Riemannian settings.

**Protein structure generation.** The SE(3) diffusion approach for protein backbones was introduced by **Yim, Trippe, De Bortoli, Mathieu, Doucet, Barzilay, and Jaakkola (ICML 2023)** ("SE(3) Diffusion Model with Application to Protein Backbone Generation"). The flow matching version — **FoldFlow** — is due to **Bose, Akhound-Sadegh, Huguet, Wolf, Rekesh, Su, Zhang, Bengio, Bhatt, Lee, Tong, Hamilton, and Bronstein (NeurIPS 2023)**, with the scaled-up version **FoldFlow-2 (Huguet et al., 2024)** incorporating OT conditioning on $SE(3)$. The AlphaFold-3 (Abramson et al., Nature 2024) architecture uses a diffusion process on atom coordinates without explicit $SE(3)$ structure, while FrameDiff (Yim et al.) and RFdiffusion (Watson, Juergens, et al., Nature 2023) use explicit frame representations.

**Simplex flows.** The Dirichlet flow matching framework for genomic sequences is developed in **Stark, Jing, Wang, Geiger, Barzilay, Jaakkola (ICML 2024)** ("Dirichlet Flow Matching with Applications to DNA Sequence Design"), which establishes the isometry between the simplex with Fisher-Rao metric and the spherical orthant and applies it to promoter sequence design. **Campbell, Benton, De Bortoli, Rainforth, Deligiannidis, and Doucet (NeurIPS 2022)** ("A Continuous Time Framework for Discrete Denoising Models") provided the CTMC theoretical foundation for these approaches.
