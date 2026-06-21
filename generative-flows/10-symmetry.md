---

title: "Symmetry"
subtitle: "Equivariant architectures, divergence-free fields, neural conservation laws, and why the right constraints make learning easier — 2020 to 2025"
---



Every generative model encodes assumptions about its target distribution — through the choice of interpolant, the architecture of the neural network, the training loss, and the sampling procedure. When those assumptions are wrong, the model wastes capacity learning structure that could have been provided for free. When they are right, the model learns faster, generalizes better, and respects constraints that no amount of data could teach reliably. This chapter is about the right assumptions: the symmetries, conservation laws, and geometric constraints that are known *a priori* from physics or from the structure of the data, and the architectural techniques for baking them into the generative model rather than hoping the network will discover them from examples.

The principle is ancient in physics — Noether's theorem (1918) established that every continuous symmetry of a physical system corresponds to a conserved quantity. The application to neural networks is more recent but equally powerful: if the target distribution is invariant under a group of transformations, then the vector field that generates it should be equivariant under the same group; if the dynamics conserve a quantity (mass, energy, charge), then the learned field should conserve it exactly, not approximately. The gap between "approximately learned" and "exactly enforced" is the gap between a model that works on the training distribution and one that works everywhere.

## 10.1 The Principle of Equivariance

A function $f : X \to Y$ is **equivariant** with respect to a group $G$ acting on $X$ and $Y$ if it commutes with the group action: applying a transformation $g \in G$ to the input and then computing $f$ gives the same result as computing $f$ first and then applying $g$ to the output.

> [!definition] 10.1 — Equivariance
> Let $G$ be a group acting on spaces $X$ and $Y$ via representations $\rho_X$ and $\rho_Y$. A function $f: X \to Y$ is **$G$-equivariant** if:
>
> $$f(\rho_X(g) \cdot x) = \rho_Y(g) \cdot f(x) \qquad \text{for all } g \in G,\; x \in X.$$
>
> When $\rho_Y$ is the trivial representation ($\rho_Y(g) = \mathrm{id}$ for all $g$), equivariance reduces to **invariance**: $f(g \cdot x) = f(x)$.


For generative flows, the relevant function is the vector field $v_\theta(x, t)$ that defines the ODE $\dot{x} = v_\theta(x, t)$. If the target distribution $p_{\text{data}}$ is invariant under $G$ — meaning $p_{\text{data}}(g \cdot x) = p_{\text{data}}(x)$ for all $g \in G$ — then the velocity field that transports $p_0$ to $p_{\text{data}}$ should be equivariant: $v_\theta(g \cdot x, t) = g \cdot v_\theta(x, t)$. This ensures that the flow itself is equivariant: if you transform the input, the output is transformed in the same way. The generated distribution is then automatically $G$-invariant, regardless of the quality of training.

> [!remark] 10.1
> Equivariance is a *hard constraint*: an equivariant network satisfies it exactly, for every input, not just on average over the training distribution. This is strictly stronger than *data augmentation*, which encourages approximate equivariance by training on transformed copies of the data. Data augmentation requires the network to learn the symmetry from examples; equivariant architecture provides it for free. The capacity saved by not learning the symmetry is redirected to learning the remaining structure — the part of the distribution that is *not* determined by symmetry.


## 10.2 Group Actions on Molecular Data

The most important symmetry group for molecular generative models is the **Euclidean group** $E(3) = \mathbb{R}^3 \rtimes O(3)$, consisting of translations, rotations, and reflections of three-dimensional space. A molecule's properties do not depend on where it is placed or how it is oriented — only on the relative positions of its atoms. This means:

- The **energy** $U(r_1, \ldots, r_N)$ is $E(3)$-invariant: $U(Rr_1 + t, \ldots, Rr_N + t) = U(r_1, \ldots, r_N)$ for any rotation $R \in O(3)$ and translation $t \in \mathbb{R}^3$.
- The **force field** $F_i = -\nabla_{r_i} U$ is $E(3)$-equivariant: $F_i(Rr + t) = R \cdot F_i(r)$.
- The **velocity field** of a generative flow on molecular coordinates should be $E(3)$-equivariant: $v_\theta(Rr + t, t) = Rv_\theta(r, t) + t'$ (where $t'$ accounts for the translation of the velocity).

For systems with chirality (molecules that are distinct from their mirror images), the relevant group is $SE(3) = \mathbb{R}^3 \rtimes SO(3)$ — translations and *proper* rotations only, excluding reflections. For systems with identical particles (atoms of the same element), there is an additional **permutation symmetry**: the distribution is invariant under relabeling of identical atoms, and the velocity field should be equivariant under the corresponding permutation of coordinates.

## 10.3 Equivariant Neural Architectures

Building equivariant neural networks requires that every layer preserve the symmetry. The key construction is the **equivariant linear map**: for a group $G$ acting on input and output feature spaces, the space of $G$-equivariant linear maps is characterized by Schur's lemma and can be parameterized explicitly.

**Message-passing neural networks (MPNNs).** For molecular systems represented as graphs with atom positions $r_i$ and features $h_i$, equivariant message passing updates each atom's features using messages from neighbors:

> [!equation] 10.1
> $$h_i^{(\ell+1)} = \phi\!\left(h_i^{(\ell)},\; \sum_{j \in \mathcal{N}(i)} \psi(h_i^{(\ell)}, h_j^{(\ell)}, \|r_i - r_j\|, \hat{r}_{ij})\right),$$


where $\hat{r}_{ij} = (r_j - r_i)/\|r_j - r_i\|$ is the unit direction vector. The key design choice is how to handle the directional information $\hat{r}_{ij}$:

- **Invariant networks** (SchNet, DimeNet): use only distances $\|r_i - r_j\|$ and angles between triplets. The output is $E(3)$-invariant. Suitable for predicting scalar properties (energy) but cannot predict vector quantities (forces, velocities) directly.
- **Equivariant networks** (EGNN, PaiNN, MACE): propagate both scalar features $s_i$ and vector features $\vec{v}_i \in \mathbb{R}^3$, updating them with operations that preserve equivariance. Vector features transform as $\vec{v}_i \mapsto R\vec{v}_i$ under rotation $R$.
- **Tensor field networks** (TFN, NequIP, Allegro): use irreducible representations of $SO(3)$ — spherical harmonics of order $\ell = 0, 1, 2, \ldots$ — to represent features of arbitrary angular momentum. The tensor product of representations, projected via Clebsch-Gordan coefficients, provides equivariant nonlinearities.

For flow matching on molecular coordinates, the velocity field $v_\theta(r, t) \in \mathbb{R}^{3N}$ must be $SE(3)$-equivariant. This is achieved by using an equivariant backbone to produce per-atom vector outputs $\vec{v}_i \in \mathbb{R}^3$, which assemble into the full velocity field. The flow ODE $\dot{r}_i = \vec{v}_i(r, t)$ is then automatically equivariant, and the generated distribution inherits $SE(3)$-invariance.

## 10.4 Neural Conservation Laws

Beyond symmetry, many physical systems satisfy **conservation laws** — quantities that are preserved exactly by the dynamics. If the generative flow is meant to model a physical system, these conservation laws should be satisfied by construction.

> [!definition] 10.2 — Neural Conservation Law
> A vector field $v : \mathbb{R}^d \to \mathbb{R}^d$ satisfies a **conservation law** for the quantity $Q(x) = \int \rho(x) \, q(x) \, dx$ if the flow $\dot{x} = v(x)$ preserves $Q$ along every trajectory:
>
> $$\frac{d}{dt} Q(x_t) = \nabla Q(x_t) \cdot v(x_t) = 0 \qquad \text{for all } x_t.$$
>
> Equivalently, $v$ lies in the null space of $\nabla Q^\top$ at every point: the velocity is always tangent to the level sets of $Q$.


The most fundamental conservation law for probability flows is **conservation of mass** — the continuity equation $\partial_t \rho + \nabla \cdot (\rho v) = 0$ itself. This is automatically satisfied by any ODE flow (the pushforward of a measure under a diffeomorphism preserves total mass). But stronger conservation laws — conservation of linear momentum $\sum_i m_i v_i$, angular momentum $\sum_i r_i \times m_i v_i$, or center of mass $\sum_i m_i r_i / \sum_i m_i$ — are not automatic and must be enforced architecturally.

**Zero center-of-mass velocity.** For molecular systems, the center of mass should not drift during generation. This requires $\sum_{i=1}^N v_i(r, t) = 0$. The simplest enforcement is **projection**: compute the unconstrained velocity $\tilde{v}_i$ from the network, then subtract the mean:

> [!equation] 10.2
> $$v_i(r, t) = \tilde{v}_i(r, t) - \frac{1}{N}\sum_{j=1}^N \tilde{v}_j(r, t).$$


This projection is differentiable, preserves equivariance (if the unconstrained network is equivariant), and costs nothing at runtime. It guarantees that the generated molecules have their center of mass at the origin, without requiring the network to learn this trivial constraint.

## 10.5 Divergence-Free and Curl-Free Decompositions

The **Helmholtz decomposition** states that any smooth vector field on $\mathbb{R}^d$ can be uniquely decomposed into a curl-free (irrotational) component and a divergence-free (solenoidal) component:

> [!theorem] 10.1 — Helmholtz Decomposition
> Any sufficiently smooth vector field $v : \mathbb{R}^d \to \mathbb{R}^d$ with appropriate decay conditions can be written as:
>
> $$v = \nabla \phi + \nabla \times A,$$
>
> where $\phi$ is a scalar potential (the curl-free part) and $A$ is a vector potential (the divergence-free part). The decomposition is unique under suitable boundary conditions.


Each component has a distinct physical meaning for generative flows:

- The **curl-free component** $\nabla \phi$ transports probability without rotation. It is the gradient of a potential and corresponds to the optimal transport direction — the Brenier map is a pure gradient flow. This component changes the density $\rho_t$ (it has nonzero divergence in general).
- The **divergence-free component** $\nabla \times A$ rotates probability without changing the density. It satisfies $\nabla \cdot (\nabla \times A) = 0$, so it is a volume-preserving flow — the density at every point is unchanged. This component can reduce transport cost by rearranging trajectories without affecting the marginal evolution.

> [!remark] 10.2
> For generative modeling, the curl-free component does the work of transforming $p_0$ into $p_1$ (it must change the density), while the divergence-free component is "free" — it can be added without affecting the marginal distributions. This is the formal statement of the observation from Chapter 3 that the continuity equation admits infinitely many velocity fields: they differ by divergence-free perturbations. The minimum-norm velocity field (the Benamou-Brenier minimizer) has zero divergence-free component — it is pure gradient flow.


**Volume-preserving flows.** A special case arises when the generative flow should preserve volume — that is, the Jacobian determinant of the flow map should be 1 everywhere. This requires $\nabla \cdot v = 0$ identically, meaning the velocity field is purely divergence-free. Volume-preserving flows cannot change the density — they can only rearrange it. They are useful when the source and target distributions have the same support and the same density (e.g., sampling from a uniform distribution on a manifold), or as components of a larger architecture (e.g., the coupling layers in NICE, which are volume-preserving by construction).

A neural divergence-free vector field in $\mathbb{R}^3$ can be constructed as $v = \nabla \times \Psi_\theta$, where $\Psi_\theta : \mathbb{R}^3 \to \mathbb{R}^3$ is any neural network. The curl of any smooth field is automatically divergence-free: $\nabla \cdot (\nabla \times \Psi) = 0$. In $d > 3$ dimensions, the construction generalizes using antisymmetric matrices: $v_i = \sum_j \partial_j A_{ij}^\theta$ where $A_{ij}^\theta = -A_{ji}^\theta$ is a neural antisymmetric matrix field.

## 10.6 Symplectic Flows and Hamiltonian Structure

When the state space has **Hamiltonian structure** — a phase space $(q, p)$ with positions $q$ and momenta $p$ — the natural flows are symplectic: they preserve the symplectic form $\omega = \sum_i dq_i \wedge dp_i$ and, consequently, phase-space volume (Liouville's theorem).

> [!definition] 10.3 — Symplectic Flow
> A diffeomorphism $\phi : \mathbb{R}^{2n} \to \mathbb{R}^{2n}$ is **symplectic** if it preserves the symplectic form: $\phi^*\omega = \omega$. Equivalently, the Jacobian $J = D\phi$ satisfies:
>
> $$J^\top \Omega\, J = \Omega, \qquad \text{where } \Omega = \begin{pmatrix} 0 & I_n \\ -I_n & 0 \end{pmatrix}.$$


Hamiltonian flows — solutions of $\dot{q} = \partial H / \partial p$, $\dot{p} = -\partial H / \partial q$ — are automatically symplectic. For generative models on phase space, this structure can be enforced by parametrizing the velocity field as a Hamiltonian vector field:

> [!equation] 10.3
> $$v_\theta(q, p, t) = \Omega\, \nabla_{(q,p)} H_\theta(q, p, t) = \begin{pmatrix} \nabla_p H_\theta \\ -\nabla_q H_\theta \end{pmatrix},$$


where $H_\theta$ is a neural network representing the Hamiltonian. This construction guarantees that the flow is symplectic, preserves phase-space volume, and satisfies the Hamiltonian conservation laws — all without any regularization or soft penalty. The only learned quantity is the scalar function $H_\theta$; the vector field structure comes from the symplectic gradient.

## 10.7 Constraint Manifolds and Projected Flows

Many physical systems live on constraint manifolds — subsets of $\mathbb{R}^d$ defined by equality constraints $c(x) = 0$. Proteins have fixed bond lengths and bond angles; rigid bodies have fixed shape; molecules in a simulation box have periodic boundary conditions. A generative flow on the ambient space $\mathbb{R}^d$ will generally violate these constraints; the flow must be **projected** onto the constraint manifold.

Given a constraint $c : \mathbb{R}^d \to \mathbb{R}^k$ defining the manifold $\mathcal{M} = \{x : c(x) = 0\}$, a vector field $v(x)$ is tangent to $\mathcal{M}$ if and only if $Dc(x) \cdot v(x) = 0$, where $Dc$ is the Jacobian of the constraint. The **projected velocity field** is:

> [!equation] 10.4
> $$v^{\mathcal{M}}(x) = \left(I - Dc(x)^\top(Dc(x)\, Dc(x)^\top)^{-1} Dc(x)\right) v(x),$$


which removes the component of $v$ normal to the constraint surface. This projection can be applied to any unconstrained neural velocity field $v_\theta$, producing a flow that remains on $\mathcal{M}$ exactly (up to numerical integration error). The projection is differentiable and can be incorporated into the training loop, so the network learns to produce fields that are approximately tangent — reducing the magnitude of the projection correction and improving integration accuracy.

## 10.8 When to Constrain and When to Learn

Architectural constraints are not free — they restrict the function class of the neural network, which can hurt performance if the constraint is wrong or too restrictive. The decision of what to bake in and what to learn is a model-design question with no universal answer, but several principles apply:

**Exact symmetries should be enforced.** If the target distribution is exactly invariant under a group (e.g., $SE(3)$ for molecular coordinates), the velocity field should be exactly equivariant. The cost of equivariance is typically small (the equivariant layers are slightly more expensive than unconstrained layers), and the benefit is large (the model cannot generate symmetry-violating samples, regardless of the training data or training procedure).

**Approximate symmetries should be learned.** If the symmetry is only approximate (e.g., a protein that is nearly but not exactly symmetric), hard enforcement can hurt — the model is forced to respect a symmetry that the data violates. In this case, data augmentation or soft penalties are more appropriate.

**Conservation laws depend on the application.** For molecular systems where conservation of momentum or energy is physically required, hard enforcement (projection, Hamiltonian parametrization) is appropriate. For image generation, where there are no conservation laws, unconstrained architectures are preferred.

**Dimensionality reduction is always valuable.** Symmetry constraints reduce the effective dimensionality of the learning problem. An $SE(3)$-equivariant network on $N$ atoms in $\mathbb{R}^3$ has effectively $3N - 6$ degrees of freedom (removing 3 translations and 3 rotations), compared to $3N$ for an unconstrained network. For $N = 100$ atoms, this is a 2% reduction — modest. But the reduction in the space of *functions* the network must represent is much larger: the equivariant network only needs to learn one representative per orbit, while the unconstrained network must learn the same function repeated across all orientations and positions. This multiplicative savings in function complexity is the real benefit of equivariance.

> [!remark] 10.3
> The history of deep learning offers a clear lesson: convolutional neural networks succeeded not because convolutions are computationally cheap (they are not), but because translation equivariance is the right inductive bias for natural images. The same principle applies to molecular generative models: $SE(3)$-equivariant architectures succeed not because they are fast, but because Euclidean symmetry is the right inductive bias for three-dimensional molecular structure. The next frontier — which remains largely open — is identifying and enforcing the relevant symmetries for non-molecular domains: gauge symmetries for physical fields, conformal symmetry for scale-invariant data, and information-geometric symmetries for probability distributions themselves.


## 10.9 Historical Notes

The theory of equivariant neural networks was systematized by **Cohen and Welling (2016)** ("Group Equivariant Convolutional Networks"), who generalized CNNs to arbitrary symmetry groups. **Thomas, Smidt, Kearnes, Yang, Li, Kohlhoff, and Riley (2018)** introduced tensor field networks using irreducible representations of $SO(3)$. **Satorras, Hoogeboom, and Welling (2021)** introduced E(n)-equivariant graph neural networks (EGNN), providing a simple and effective architecture for molecular systems.

For molecular generative flows, **Köhler, Klein, and Noé (2020)** ("Equivariant Flows: Exact Likelihood Generative Learning for Symmetric Densities") demonstrated $E(n)$-equivariant normalizing flows. **Hoogeboom, Satorras, Vignac, and Welling (2022)** ("Equivariant Diffusion for Molecule Generation in 3D") applied equivariant diffusion to molecular generation. **Bose, Akhound-Sadegh, Huguet, et al. (2024)** ("SE(3)-Stochastic Flow Matching for Protein Backbone Generation") developed $SE(3)$-equivariant flow matching for proteins.

Neural conservation laws and divergence-free fields were explored by **Greydanus, Dzamba, and Sotelo (2019)** ("Hamiltonian Neural Networks"), who parameterized the Hamiltonian to enforce energy conservation. **Cranmer, Greydanus, Hoyer, et al. (2020)** ("Lagrangian Neural Networks") took the Lagrangian approach. **Finzi, Wang, and Wilson (2020)** studied general constraint satisfaction in equivariant networks. The Helmholtz decomposition for neural vector fields was applied to generative models by **Rozen, Grover, Nickel, and Lipman (2021)**.

The projection approach to constraint manifolds builds on the SHAKE and RATTLE algorithms from molecular dynamics (**Ryckaert, Ciccotti, and Berendsen, 1977**; **Andersen, 1983**), adapted to the neural ODE setting by **Lou, Lim, Katsman, Huang, Jiang, Lim, and De Sa (2020)** ("Neural Manifold Ordinary Differential Equations"). Symplectic neural networks were introduced by **Chen, Zhang, Arjovsky, and Bottou (2020)** ("Symplectic Recurrent Neural Networks") and extended to generative flows by **Toth, Rezende, Jaegle, Racanière, Botev, and Higgins (2020)**.
