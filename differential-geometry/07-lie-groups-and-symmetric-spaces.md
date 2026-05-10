---

title: "Lie Groups and Symmetric Spaces"
subtitle: "Lie groups, Lie algebras, the exponential map revisited, and spaces with maximal symmetry"
---



Symmetry and geometry are inseparable. The most important manifolds in mathematics and physics — spheres, hyperbolic spaces, Grassmannians, the space of positive definite matrices — all carry large groups of isometries that act transitively, meaning any point can be moved to any other by a symmetry. These are the **homogeneous spaces**, and their geometry is entirely controlled by the symmetry group. When the symmetry group is itself a smooth manifold with a group structure — a **Lie group** — the interplay becomes exceptionally rich and computable.

Lie groups encode continuous symmetry. They appear wherever a physical or mathematical system has a continuous family of transformations that preserve some structure: rotations in space, unitary transformations in quantum mechanics, gauge transformations in field theory, diffeomorphisms in general relativity. The Lie algebra — the tangent space at the identity, equipped with the Lie bracket — is the infinitesimal version of the group, and the exponential map connects the two. On Riemannian manifolds with a transitive isometry group, the geometry simplifies dramatically: everything can be computed from Lie algebraic data alone.

## Lie Groups

> [!definition] 7.1 — Lie Group
> A **Lie group** is a smooth manifold $G$ that is also a group, with the group operations
>
> $$\mu: G \times G \to G, \quad (g, h) \mapsto gh \qquad \text{(multiplication)}$$
> $$\iota: G \to G, \quad g \mapsto g^{-1} \qquad \text{(inversion)}$$
>
> both smooth. A **Lie group homomorphism** is a smooth map $\phi: G \to H$ between Lie groups that is also a group homomorphism.


The canonical examples:

- **$\mathbb{R}^n$** under addition: a Lie group of dimension $n$, abelian.
- **$S^1 = U(1) = \{z \in \mathbb{C} : |z| = 1\}$** under multiplication: the circle group, the simplest compact Lie group.
- **$\mathrm{GL}(n, \mathbb{R}) = \{A \in \mathbb{R}^{n \times n} : \det A \neq 0\}$**: the general linear group, an open submanifold of $\mathbb{R}^{n^2}$ of dimension $n^2$.
- **$\mathrm{SL}(n, \mathbb{R}) = \{A \in \mathbb{R}^{n \times n} : \det A = 1\}$**: dimension $n^2 - 1$, defined by one equation. By the regular value theorem (Chapter 1), this is a smooth submanifold.
- **$\mathrm{O}(n) = \{A \in \mathbb{R}^{n \times n} : A^T A = I\}$**: the orthogonal group, dimension $n(n-1)/2$. The condition $A^T A = I$ imposes $n(n+1)/2$ equations on $n^2$ entries, but they are not independent — the rank of the differential drops. The connected component of the identity is $\mathrm{SO}(n) = \{A \in \mathrm{O}(n) : \det A = 1\}$.
- **$\mathrm{U}(n) = \{A \in \mathbb{C}^{n \times n} : A^\dagger A = I\}$**: the unitary group, a compact Lie group of real dimension $n^2$.
- **$\mathrm{SU}(n) = \{A \in \mathrm{U}(n) : \det A = 1\}$**: dimension $n^2 - 1$. In particular, $\mathrm{SU}(2) \cong S^3$ as smooth manifolds.
- **$\mathrm{Sp}(2n, \mathbb{R}) = \{A \in \mathbb{R}^{2n \times 2n} : A^T J A = J\}$** where $J = \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix}$: the symplectic group, of dimension $n(2n+1)$, central to Chapter 9.

All matrix groups are Lie groups by the closed subgroup theorem: a closed subgroup of $\mathrm{GL}(n, \mathbb{R})$ is automatically a smooth submanifold and hence a Lie group.

## The Lie Algebra

The tangent space at the identity of a Lie group carries a canonical Lie algebra structure, the infinitesimal shadow of the group.

> [!definition] 7.2 — Left-Invariant Vector Fields and the Lie Algebra
> For $g \in G$, define **left translation** $L_g: G \to G$ by $L_g(h) = gh$. A vector field $X \in \mathfrak{X}(G)$ is **left-invariant** if $(L_g)_* X = X$ for all $g$, i.e.,
>
> $$d(L_g)_h(X_h) = X_{gh} \quad \text{for all } g, h \in G.$$
>
> The space of left-invariant vector fields is a Lie subalgebra of $\mathfrak{X}(G)$: closed under the Lie bracket (since $(L_g)_*[X,Y] = [(L_g)_*X, (L_g)_*Y]$). It is isomorphic to $T_eG$ (the tangent space at the identity $e$) via evaluation: $X \mapsto X_e$. The **Lie algebra** of $G$ is
>
> $$\mathfrak{g} = T_e G,$$
>
> equipped with the Lie bracket $[v, w] = [X_v, X_w]_e$, where $X_v, X_w$ are the left-invariant extensions of $v, w \in T_eG$.


For matrix groups, the Lie algebra is the corresponding space of matrices with the commutator bracket:

| Group | Lie Algebra | Bracket |
|---|---|---|
| $\mathrm{GL}(n,\mathbb{R})$ | $\mathfrak{gl}(n,\mathbb{R}) = \mathbb{R}^{n\times n}$ | $[A,B] = AB - BA$ |
| $\mathrm{SL}(n,\mathbb{R})$ | $\mathfrak{sl}(n,\mathbb{R}) = \{A : \mathrm{tr}\,A = 0\}$ | $[A,B] = AB - BA$ |
| $\mathrm{SO}(n)$ | $\mathfrak{so}(n) = \{A : A + A^T = 0\}$ | $[A,B] = AB - BA$ |
| $\mathrm{U}(n)$ | $\mathfrak{u}(n) = \{A : A + A^\dagger = 0\}$ | $[A,B] = AB - BA$ |
| $\mathrm{SU}(n)$ | $\mathfrak{su}(n) = \{A : A + A^\dagger = 0,\, \mathrm{tr}\,A = 0\}$ | $[A,B] = AB - BA$ |

The Lie algebra encodes all local structure of the group. Two Lie groups with the same Lie algebra are locally isomorphic — they differ only in their global topology (fundamental group). For instance, $\mathrm{SO}(3)$ and $\mathrm{SU}(2)$ have the same Lie algebra $\mathfrak{su}(2) \cong \mathfrak{so}(3)$ but different global topology: $\mathrm{SU}(2) \cong S^3$ is simply connected while $\pi_1(\mathrm{SO}(3)) = \mathbb{Z}/2$.

## The Exponential Map for Lie Groups

The exponential map of Chapter 4 — defined via geodesics of the Levi-Civita connection — specializes to an algebraically defined map for Lie groups.

> [!definition] 7.3 — Lie Group Exponential Map
> For a Lie group $G$ with Lie algebra $\mathfrak{g}$, the **exponential map** $\exp: \mathfrak{g} \to G$ is defined by
>
> $$\exp(v) = \gamma_v(1),$$
>
> where $\gamma_v: \mathbb{R} \to G$ is the unique one-parameter subgroup with $\gamma_v'(0) = v$ — that is, $\gamma_v$ satisfies $\gamma_v(s+t) = \gamma_v(s)\gamma_v(t)$ and $\gamma_v'(0) = v \in T_eG = \mathfrak{g}$.


For matrix groups, the exponential map is literally the matrix exponential:

$$\exp(A) = e^A = \sum_{k=0}^\infty \frac{A^k}{k!} = I + A + \frac{A^2}{2} + \frac{A^3}{6} + \cdots$$

This series converges for all $A \in \mathfrak{gl}(n,\mathbb{R})$ and maps to the corresponding group: $e^A \in \mathrm{GL}(n,\mathbb{R})$, and if $A$ is skew-symmetric then $e^A \in \mathrm{SO}(n)$, if $A$ is skew-Hermitian then $e^A \in \mathrm{U}(n)$, and so on. The **Baker-Campbell-Hausdorff formula** expresses $\log(\exp(A)\exp(B))$ as a series in $A$, $B$, and their iterated brackets:

$$\log(e^A e^B) = A + B + \tfrac{1}{2}[A,B] + \tfrac{1}{12}[A,[A,B]] - \tfrac{1}{12}[B,[A,B]] + \cdots$$

This formula encodes how the non-commutativity of the group multiplication is captured entirely by the Lie bracket in the algebra.

The exponential map is a local diffeomorphism near $0 \in \mathfrak{g}$ (by the inverse function theorem, since $d(\exp)_0 = \mathrm{id}_\mathfrak{g}$), but need not be globally injective or surjective. For compact connected Lie groups, $\exp$ is surjective — every group element is a matrix exponential. For non-compact groups like $\mathrm{SL}(2,\mathbb{R})$, some elements are not in the image of $\exp$.

## The Adjoint Representation

Every Lie group acts on its own Lie algebra by conjugation, giving the adjoint representation.

> [!definition] 7.4 — Adjoint Representation
> The **conjugation** map $C_g: G \to G$, $C_g(h) = ghg^{-1}$, fixes the identity, so its differential at $e$ gives a linear map $\mathrm{Ad}_g = d(C_g)_e: \mathfrak{g} \to \mathfrak{g}$. The resulting group homomorphism
>
> $$\mathrm{Ad}: G \to \mathrm{GL}(\mathfrak{g}), \quad g \mapsto \mathrm{Ad}_g,$$
>
> is the **adjoint representation** of $G$. Its derivative at $e$ is the **adjoint representation of the Lie algebra**:
>
> $$\mathrm{ad}: \mathfrak{g} \to \mathfrak{gl}(\mathfrak{g}), \quad \mathrm{ad}_v(w) = [v, w].$$
>
> The **Killing form** is the symmetric bilinear form $B: \mathfrak{g} \times \mathfrak{g} \to \mathbb{R}$ defined by $B(v,w) = \mathrm{tr}(\mathrm{ad}_v \circ \mathrm{ad}_w)$.


For matrix groups, $\mathrm{Ad}_g(v) = gvg^{-1}$. The Killing form is central to the classification of semisimple Lie algebras (Cartan's criterion: a Lie algebra is semisimple if and only if its Killing form is non-degenerate) and determines the canonical bi-invariant metric on compact semisimple Lie groups: $\langle v, w \rangle = -B(v,w)$ is positive definite for compact semisimple $\mathfrak{g}$.

## Bi-Invariant Metrics and Geodesics

A Riemannian metric $g$ on $G$ is **bi-invariant** if it is invariant under both left and right translations: $(L_h)^*g = g$ and $(R_h)^*g = g$ for all $h \in G$. Bi-invariant metrics exist on every compact Lie group (average any left-invariant metric over the group using the Haar measure). For compact semisimple groups, the negative Killing form gives a canonical one.

On a Lie group with a bi-invariant metric, the geometry is completely explicit:

> [!theorem] 7.1 — Geometry of Bi-Invariant Metrics
> Let $G$ be a Lie group with bi-invariant metric $\langle \cdot, \cdot \rangle$. Then:
>
> 1. **Geodesics through $e$** are exactly the one-parameter subgroups: $\gamma(t) = \exp(tv)$ for $v \in \mathfrak{g}$.
> 2. **The Levi-Civita connection** satisfies $\nabla_X Y = \frac{1}{2}[X,Y]$ for left-invariant fields $X, Y$.
> 3. **The Riemann curvature** satisfies $R(X,Y)Z = \frac{1}{4}[[X,Y],Z]$ for left-invariant fields.
> 4. **The sectional curvature** of the plane spanned by orthonormal left-invariant fields $X, Y$ is $K(X,Y) = \frac{1}{4}|[X,Y]|^2 \geq 0$.


Point (4) shows compact Lie groups with bi-invariant metrics always have non-negative sectional curvature — a strong geometric consequence of compactness and the non-negativity of $|[X,Y]|^2$.

For $\mathrm{SO}(n)$ with the metric $\langle A, B \rangle = -\frac{1}{2}\mathrm{tr}(AB)$, geodesics through $I$ are $t \mapsto e^{tA}$ for skew-symmetric $A$. The distance between two rotation matrices $R_1, R_2 \in \mathrm{SO}(n)$ is

$$d(R_1, R_2) = \frac{1}{\sqrt{2}}\|\log(R_1^T R_2)\|_F,$$

where $\|\cdot\|_F$ is the Frobenius norm. This formula is used directly in equivariant neural networks and Riemannian flow matching on rotation groups.

## Homogeneous Spaces

A Lie group acts on a manifold by symmetries; when the action is transitive (every point can be reached from any other), the manifold is a homogeneous space.

> [!definition] 7.5 — Homogeneous Space
> Let $G$ be a Lie group and $H \leq G$ a closed subgroup. The **coset space** $G/H = \{gH : g \in G\}$ with the quotient topology is a smooth manifold of dimension $\dim G - \dim H$, called a **homogeneous space**.
>
> $G$ acts on $G/H$ by left multiplication: $g \cdot (kH) = (gk)H$. This action is smooth, transitive (any coset can be moved to any other), and the stabilizer of the coset $eH$ is exactly $H$.
>
> Conversely, any manifold $M$ on which $G$ acts smoothly and transitively is diffeomorphic to $G/H$ where $H$ is the stabilizer of any chosen point $p \in M$.


The fundamental examples unify a large portion of geometry:

- **Spheres**: $S^n \cong \mathrm{SO}(n+1)/\mathrm{SO}(n)$. The group $\mathrm{SO}(n+1)$ acts on $S^n \subset \mathbb{R}^{n+1}$ by rotation; the stabilizer of the north pole is $\mathrm{SO}(n)$ (rotations fixing the last coordinate).
- **Hyperbolic space**: $\mathbb{H}^n \cong \mathrm{SO}(n,1)/\mathrm{SO}(n)$. Replace the orthogonal group with the Lorentz group.
- **Real Grassmannian**: $\mathrm{Gr}(k, n) \cong \mathrm{O}(n)/(\mathrm{O}(k) \times \mathrm{O}(n-k))$. The space of $k$-dimensional subspaces of $\mathbb{R}^n$.
- **Stiefel manifold**: $V_k(\mathbb{R}^n) \cong \mathrm{O}(n)/\mathrm{O}(n-k)$. The space of orthonormal $k$-frames in $\mathbb{R}^n$.
- **Positive definite matrices**: $\mathcal{P}(n) = \mathrm{GL}(n,\mathbb{R})/\mathrm{O}(n)$. The space of symmetric positive definite $n \times n$ matrices.
- **Complex projective space**: $\mathbb{CP}^n \cong \mathrm{SU}(n+1)/\mathrm{U}(n)$.

In each case, the $G$-invariant Riemannian metric on $M \cong G/H$ is determined by a choice of $\mathrm{Ad}(H)$-invariant inner product on $\mathfrak{g}/\mathfrak{h}$ (the "horizontal" tangent space at $eH$). Geodesics, curvature, and the exponential map are all computable purely from the Lie algebra data.

## Symmetric Spaces

Symmetric spaces are homogeneous spaces with an additional reflexive symmetry at each point — they are the most regular Riemannian manifolds beyond space forms.

> [!definition] 7.6 — Riemannian Symmetric Space
> A Riemannian manifold $(M, g)$ is a **symmetric space** if for every point $p \in M$ there exists an isometry $s_p: M \to M$ (the **geodesic symmetry at $p$**) satisfying:
>
> 1. $s_p(p) = p$ (fixes $p$)
> 2. $d(s_p)_p = -\mathrm{id}_{T_pM}$ (reverses all tangent vectors at $p$)
>
> Equivalently, $s_p$ is the map that reverses geodesics through $p$: $s_p(\exp_p(v)) = \exp_p(-v)$.


The geodesic symmetry is a generalization of the antipodal map on $S^n$. Its existence forces the Riemann curvature tensor to be parallel: $\nabla R = 0$. This is an extraordinarily strong condition — it means the curvature does not change from point to point in any parallel-transported sense, making the geometry completely homogeneous.

Symmetric spaces are classified by Élie Cartan's complete classification (1926), organized by the sign of their curvature:

**Type I (Compact, $K \geq 0$)**: Compact simple Lie groups $G$ themselves, and compact homogeneous spaces $G/K$ where $K$ is the fixed-point set of an involution. Examples: $S^n$, $\mathbb{CP}^n$, $\mathrm{Gr}(k,n)$, $\mathrm{SO}(n)$, $\mathrm{U}(n)$, $\mathrm{SU}(n)/\mathrm{SO}(n)$ (space of real structures in $\mathbb{C}^n$).

**Type II (Non-compact, $K \leq 0$)**: Non-compact duals of Type I spaces, obtained by replacing the compact group with a non-compact real form. Examples: $\mathbb{H}^n$, $\mathrm{GL}(n,\mathbb{R})/\mathrm{O}(n) \cong \mathcal{P}(n)$ (positive definite matrices), $\mathrm{SL}(n,\mathbb{R})/\mathrm{SO}(n)$ (the Siegel upper half-space in the $n=2$ case).

**Type III (Flat)**: Euclidean spaces and flat tori.

The duality between compact and non-compact symmetric spaces — Type I and Type II — is one of Cartan's deepest observations. They have the same complexification and the same local algebraic structure, but opposite curvature signs.

> [!remark]
> The space of symmetric positive definite matrices $\mathcal{P}(n) = \mathrm{GL}(n)/\mathrm{O}(n)$ is a non-compact symmetric space of Type II with non-positive curvature. It appears throughout statistics and machine learning as the natural space for covariance matrices. Its Riemannian metric is the **affine-invariant metric** $\langle U, V \rangle_P = \mathrm{tr}(P^{-1}UP^{-1}V)$, and its geodesics are $\gamma(t) = P^{1/2}(P^{-1/2}QP^{-1/2})^t P^{1/2}$. The Fréchet mean on $\mathcal{P}(n)$ is the geometric mean of positive definite matrices, computable by Riemannian gradient descent on this symmetric space.


## The Cartan Decomposition

The algebraic backbone of symmetric spaces is a splitting of the Lie algebra.

> [!theorem] 7.2 — Cartan Decomposition
> Let $M = G/K$ be a Riemannian symmetric space, and let $\sigma: G \to G$ be the involution for which $K = G^\sigma$ (the fixed-point subgroup). Then the derivative $d\sigma_e: \mathfrak{g} \to \mathfrak{g}$ is an involution on the Lie algebra, giving a decomposition
>
> $$\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m},$$
>
> where $\mathfrak{k} = \{v \in \mathfrak{g} : d\sigma_e(v) = v\}$ is the Lie algebra of $K$, and $\mathfrak{m} = \{v \in \mathfrak{g} : d\sigma_e(v) = -v\}$ is the orthogonal complement. These satisfy the bracket relations:
>
> $$[\mathfrak{k}, \mathfrak{k}] \subseteq \mathfrak{k}, \qquad [\mathfrak{k}, \mathfrak{m}] \subseteq \mathfrak{m}, \qquad [\mathfrak{m}, \mathfrak{m}] \subseteq \mathfrak{k}.$$
>
> The tangent space $T_{eK}(G/K) \cong \mathfrak{m}$. Geodesics through $eK$ are $t \mapsto \exp(tv)K$ for $v \in \mathfrak{m}$. The curvature at $eK$ is $R(X,Y)Z = -[[X,Y],Z]$ for $X,Y,Z \in \mathfrak{m}$.


The bracket relation $[\mathfrak{m}, \mathfrak{m}] \subseteq \mathfrak{k}$ (rather than back into $\mathfrak{m}$) is the defining algebraic condition — it encodes the geodesic symmetry $s_p$. The curvature formula $R(X,Y)Z = -[[X,Y],Z]$ is determined entirely by the Lie bracket in $\mathfrak{g}$: the sectional curvature of the plane $(X, Y)$ is $K(X,Y) = -|[X,Y]_\mathfrak{k}|^2 / (|X|^2|Y|^2 - \langle X,Y\rangle^2)$, which is $\leq 0$ for non-compact symmetric spaces and $\geq 0$ for compact ones — consistent with the Type I/II classification.

## Lie Groups in Machine Learning and Geometry

Lie groups and symmetric spaces are the natural domains for geometric machine learning. Several direct connections:

**Rotation groups.** $\mathrm{SO}(3)$ and $\mathrm{SO}(n)$ are the natural configuration spaces for orientation in 3D — for molecular conformations, rigid body dynamics, and pose estimation. Riemannian flow matching on $\mathrm{SO}(3)$ uses the geodesic structure (one-parameter subgroups) and the exponential map (matrix exponential of skew-symmetric matrices) to define simulation-free training objectives. The Riemannian metric is the bi-invariant one from the negative Killing form.

**Positive definite matrices.** $\mathcal{P}(n)$ as a symmetric space provides the correct geometry for covariance estimation, brain connectivity matrices, and diffusion tensor imaging. The Fréchet mean on $\mathcal{P}(n)$ (the geometric mean) avoids the distortions that arise from treating covariance matrices as Euclidean vectors.

**Homogeneous spaces as data manifolds.** Grassmannians $\mathrm{Gr}(k,n)$ model subspaces — the natural domain for principal components, subspace clustering, and subspace-valued data. The Stiefel manifold $V_k(\mathbb{R}^n)$ is the domain of orthonormal frames, appearing in matrix factorizations and multi-task learning. Flows and diffusions on these spaces require the Riemannian structure of their symmetric space geometry.

**The exponential and logarithm maps.** On any symmetric space, the exponential map $\exp_p: T_pM \to M$ and its inverse $\log_p: M \to T_pM$ (defined on a neighborhood of $p$) are explicitly computable from the matrix exponential and logarithm. This makes Riemannian gradient descent, Riemannian Gaussian processes, and geodesic interpolation practically implementable — the differential geometry reduces to linear algebra.

**Equivariant architectures.** A neural network $f: M \to N$ is equivariant with respect to a group action $G \circlearrowleft M$ if $f(g \cdot x) = g \cdot f(x)$. Designing equivariant architectures for molecular geometry (where $G = \mathrm{SE}(3)$, the group of rigid motions) or for data on the sphere (where $G = \mathrm{SO}(3)$) requires understanding the representation theory of $G$ — which is determined entirely by the Lie algebra $\mathfrak{g}$ and its decomposition into irreducible representations.

---

*The classical reference for Lie groups and symmetric spaces is Helgason's* Differential Geometry, Lie Groups, and Symmetric Spaces *(1978), encyclopedic and authoritative. For a shorter treatment integrated with Riemannian geometry, see O'Neill's* Semi-Riemannian Geometry *(1983) and do Carmo's* Riemannian Geometry *(Chapter 8). The Baker-Campbell-Hausdorff formula and its applications to matrix groups are in Hall's* Lie Groups, Lie Algebras, and Representations *(2nd ed., 2015). For the machine learning applications — Riemannian optimization on Lie groups and symmetric spaces — see Absil, Mahony, and Sepulchre's* Optimization Algorithms on Matrix Manifolds *(2008).*
