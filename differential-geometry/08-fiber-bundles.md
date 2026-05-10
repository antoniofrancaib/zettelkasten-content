---

title: "Fiber Bundles"
subtitle: "The global geometry of local symmetry"
---



A manifold can carry global structure that is invisible locally. A Möbius band and a cylinder look the same in any small neighborhood — both are locally a rectangle — yet they are topologically distinct. The difference lies not in the individual fibers but in how those fibers are assembled as you travel around the base. Fiber bundle theory is the systematic language for this kind of global-from-local structure, and it underlies virtually all of modern differential geometry: connections are forms on principal bundles, gauge fields are curvatures, characteristic classes are obstructions to triviality, and spinors live in associated bundles. This chapter develops the theory from first principles through to the computational toolkit used in geometry, topology, and physics.

## The Bundle Concept

The tangent bundle $TM$ already showed us the key idea: over each point $p \in M$ sits a copy of $\mathbb{R}^n$, these copies are glued together smoothly, and yet globally $TM$ need not be a product $M \times \mathbb{R}^n$. Fiber bundle theory abstracts and vastly generalizes this picture.

> [!definition] Fiber Bundle
> A **fiber bundle** is a tuple $(E, B, \pi, F)$ where $E$ (total space), $B$ (base), and $F$ (fiber) are smooth manifolds, $\pi: E \to B$ is a smooth surjection, and there exists an open cover $\{U_\alpha\}$ of $B$ with **local trivializations**: diffeomorphisms
> $$\phi_\alpha: \pi^{-1}(U_\alpha) \xrightarrow{\;\sim\;} U_\alpha \times F$$
> such that $\mathrm{pr}_1 \circ \phi_\alpha = \pi$. Each fiber $E_p := \pi^{-1}(p)$ is diffeomorphic to $F$.


The local trivializations are the charts of bundle theory. On overlaps $U_\alpha \cap U_\beta$ we get **transition functions**:
$$g_{\alpha\beta}: U_\alpha \cap U_\beta \longrightarrow \mathrm{Diff}(F), \qquad g_{\alpha\beta}(p) = \phi_\alpha|_p \circ \phi_\beta|_p^{-1}.$$
These satisfy the **cocycle condition** $g_{\alpha\beta}\, g_{\beta\gamma} = g_{\alpha\gamma}$ on triple overlaps, which ensures compatibility. Conversely, any collection of smooth maps satisfying the cocycle condition determines a bundle (up to isomorphism). This shows that fiber bundles are entirely encoded by their transition data.

> [!remark]
> Two bundles over $B$ with fiber $F$ are **isomorphic** if there is a fiber-preserving diffeomorphism $E \to E'$. A bundle is **trivial** if it is isomorphic to the product $B \times F$, equivalently if it admits a global trivialization. Triviality is equivalent to all transition functions being simultaneously conjugate to the identity — an obstruction measured by characteristic classes.


## Vector Bundles

When the fiber is a vector space and the transition functions act linearly, the bundle inherits a linear structure on each fiber.

> [!definition] Vector Bundle
> A **rank-$k$ vector bundle** over $B$ is a fiber bundle with fiber $F = \mathbb{R}^k$ (or $\mathbb{C}^k$) and transition functions $g_{\alpha\beta}: U_\alpha \cap U_\beta \to GL(k, \mathbb{R})$. Each fiber $E_p$ is a $k$-dimensional vector space, and the vector space structure varies smoothly with $p$.


The principal examples in differential geometry are:
- **Tangent bundle** $TM$: rank-$n$ real vector bundle, transition functions are Jacobians.
- **Cotangent bundle** $T^*M$: dual, with transitions $(g^T)^{-1}$.
- **Tensor bundles** $T^{(r,s)}M$: transition functions are tensor products of Jacobians.
- **Exterior bundles** $\Lambda^k T^*M$: differential $k$-forms live in sections of this bundle.
- **Normal bundle** $NM$ of a submanifold: the orthogonal complement of $TM$ in $T\mathbb{R}^N$.

A **section** of a vector bundle $E$ is a smooth map $s: B \to E$ with $\pi \circ s = \mathrm{id}_B$. Sections of $TM$ are vector fields; sections of $\Lambda^k T^*M$ are $k$-forms. The space of sections $\Gamma(E)$ is a module over $C^\infty(B)$.

> [!definition] Bundle Map and Sub-bundle
> A **bundle map** over $B$ is a smooth map $\phi: E \to E'$ commuting with projections and linear on each fiber. A **sub-bundle** $S \subset E$ assigns to each $p$ a subspace $S_p \subset E_p$ that varies smoothly. An exact sequence of bundles $0 \to S \to E \to Q \to 0$ (where $Q = E/S$) splits when $E \cong S \oplus Q$.


## Principal Bundles

While vector bundles carry the linear algebra of geometry, **principal bundles** carry the symmetry. They are the natural home for connections and gauge theory.

> [!definition] Principal $G$-Bundle
> Let $G$ be a Lie group. A **principal $G$-bundle** is a fiber bundle $\pi: P \to B$ with a free, proper, fiber-preserving right action of $G$ on $P$ such that $B = P/G$. Local trivializations have the form $\pi^{-1}(U_\alpha) \cong U_\alpha \times G$, and transition functions $g_{\alpha\beta}: U_\alpha \cap U_\beta \to G$ act by right multiplication in $G$.


The action being **free** (no fixed points other than identity) and **proper** (orbits close up nicely) ensures that $P/G$ is indeed a manifold. The fibers of $P$ are all isomorphic to $G$ as right $G$-spaces, but there is no canonical identity element in each fiber — that is what makes $P$ a principal bundle rather than a trivial product.

Key examples:
- **Frame bundle** $\mathrm{Fr}(M)$: over each $p \in M$, the fiber is the set of all ordered bases of $T_pM$, on which $GL(n,\mathbb{R})$ acts freely and transitively by change of basis. A choice of section (global frame field) trivializes the bundle and is equivalent to a global coordinate system.
- **Orthonormal frame bundle** $\mathrm{OFr}(M)$: restrict to orthonormal bases, structure group reduces to $O(n)$.
- **Hopf fibration** $\pi: S^3 \to S^2$: principal $U(1)$-bundle, fibers are great circles on $S^3$.

> [!remark]
> A principal bundle $P \to B$ admits a global section if and only if it is trivial. This is the precise sense in which "choosing a gauge" in physics corresponds to choosing a section of the frame bundle. The Hopf fibration is non-trivial — there is no continuous global section — which reflects the fact that $S^2$ is not parallelizable.


## Associated Bundles

From a principal $G$-bundle $P \to B$ and any left $G$-action on a space $F$, we can construct an associated fiber bundle with fiber $F$.

> [!definition] Associated Bundle
> Given $\pi: P \to B$ a principal $G$-bundle and $\rho: G \to \mathrm{Diff}(F)$ a left action, the **associated bundle** is
> $$P \times_G F := (P \times F) / \sim, \qquad (p \cdot g, f) \sim (p, \rho(g) f).$$
> The projection sends $[(p, f)]$ to $\pi(p)$, and the fiber over each point is diffeomorphic to $F$.


This construction is extraordinarily powerful:
- Take $F = \mathbb{R}^k$ with $\rho: GL(n) \to GL(\mathbb{R}^k)$ a representation: recover vector bundles.
- Take $F = G/H$ for a closed subgroup: recover homogeneous fiber bundles.
- Take $F$ itself a $G$-space: spinor bundles, associated to the frame bundle with spinor representation of $\mathrm{Spin}(n)$.

The transition functions of the associated bundle are $\rho(g_{\alpha\beta})$, so the entire bundle theory reduces to the representation theory of $G$.

## Connections on Principal Bundles

A **connection** on a principal bundle is the natural generalization of a Levi-Civita connection. It provides a rule for parallel transport — a way to lift curves in $B$ to curves in $P$ that is equivariant with respect to the $G$-action.

At each $p \in P$, the **vertical subspace** $V_p = \ker(d\pi_p) \subset T_pP$ is canonically defined as the tangent to the $G$-orbit. A connection specifies a complementary horizontal distribution.

> [!definition] Ehresmann Connection
> A **connection** on $P \to B$ is a $G$-equivariant smooth distribution $H \subset TP$ with $T_pP = H_p \oplus V_p$ at every $p$. Equivalently, it is a $\mathfrak{g}$-valued 1-form $\omega \in \Omega^1(P; \mathfrak{g})$ such that:
> 1. $\omega(\tilde{A}) = A$ for all $A \in \mathfrak{g}$, where $\tilde{A}$ is the fundamental vector field of $A$,
> 2. $(R_g)^* \omega = \mathrm{Ad}_{g^{-1}} \circ \omega$ (equivariance under right $G$-action).


The horizontal subspace at $p$ is $H_p = \ker(\omega_p)$. Given a curve $\gamma$ in $B$ and a starting point $p_0 \in P$ over $\gamma(0)$, the unique horizontal lift through $p_0$ is the curve $\tilde\gamma$ with $\pi \circ \tilde\gamma = \gamma$ and $\dot{\tilde\gamma}(t) \in H_{\tilde\gamma(t)}$.

The **curvature** of the connection is the $\mathfrak{g}$-valued 2-form on $P$:
> [!equation]
>   $$\Omega = d\omega + \tfrac{1}{2}[\omega \wedge \omega] \in \Omega^2(P; \mathfrak{g}).$$

This is the **Cartan structure equation**. Flatness $\Omega = 0$ means that horizontal transport is path-independent — the bundle is locally trivial via parallel sections.

On local trivializations, the connection pulls back to a **local connection form** (gauge potential) $A_\alpha \in \Omega^1(U_\alpha; \mathfrak{g})$, and the curvature to $F_\alpha = dA_\alpha + \frac{1}{2}[A_\alpha \wedge A_\alpha]$. Under a gauge change $g_{\alpha\beta}$:
$$A_\beta = g_{\alpha\beta}^{-1} A_\alpha\, g_{\alpha\beta} + g_{\alpha\beta}^{-1} d g_{\alpha\beta}.$$
This is the **gauge transformation law** familiar from Yang-Mills theory.

> [!remark]
> The Levi-Civita connection of Chapter 5 corresponds to a connection on the orthonormal frame bundle $\mathrm{OFr}(M)$, with structure group $O(n)$ and connection form the matrix of Christoffel symbols. The Riemann curvature tensor is the local curvature 2-form of this connection.


## Characteristic Classes

Characteristic classes are the primary tool for distinguishing non-isomorphic bundles. They are cohomology classes in $H^*(B)$ that are naturally assigned to a bundle and invariant under bundle isomorphism.

The key idea is that any invariant polynomial $P$ on the Lie algebra $\mathfrak{g}$ — a polynomial function $P: \mathfrak{g} \to \mathbb{R}$ invariant under the adjoint action — produces a closed differential form $P(\Omega)$ on the base, whose de Rham cohomology class is independent of the choice of connection. This is the **Chern-Weil homomorphism**.

> [!definition] Chern-Weil Homomorphism
> Let $P \to B$ be a principal $G$-bundle with connection $\omega$ and curvature $\Omega$. For any $\mathrm{Ad}$-invariant polynomial $P \in I^k(\mathfrak{g})$ of degree $k$, the form $P(\Omega) \in \Omega^{2k}(B)$ is closed, and its de Rham class $[P(\Omega)] \in H^{2k}(B; \mathbb{R})$ is independent of the choice of connection. The resulting map $I^*(\mathfrak{g}) \to H^{2*}(B; \mathbb{R})$ is the **Chern-Weil homomorphism**.


The most important invariant polynomials for $G = GL(n, \mathbb{C})$ are the **elementary symmetric polynomials** of the eigenvalues of $\frac{i}{2\pi}\Omega$:

$$c_k(E) = \left[\sigma_k\!\left(\tfrac{i}{2\pi}\Omega\right)\right] \in H^{2k}(B; \mathbb{Z}),$$

these are the **Chern classes** of a complex vector bundle $E$. For real bundles with $G = GL(n, \mathbb{R})$, the analogous classes are the **Pontryagin classes** $p_k(E) \in H^{4k}(B; \mathbb{Z})$.

> [!theorem] Properties of Chern Classes
> The Chern classes $c_k(E) \in H^{2k}(B;\mathbb{Z})$ satisfy:
> - **Naturality**: $f^* c_k(E) = c_k(f^*E)$ for any smooth map $f: B' \to B$.
> - **Whitney product formula**: $c(E \oplus F) = c(E) \cup c(F)$, where $c = 1 + c_1 + c_2 + \cdots$ is the total Chern class.
> - **Normalization**: For the tautological line bundle $\gamma^1$ over $\mathbb{CP}^1$, $c_1(\gamma^1)$ generates $H^2(\mathbb{CP}^1; \mathbb{Z}) \cong \mathbb{Z}$.
> - **Vanishing**: $c_k(E) = 0$ for $k > \mathrm{rank}(E)$.


The total Chern class encodes the full topological type of a complex bundle over $B$. The first Chern class $c_1(L) \in H^2(B;\mathbb{Z})$ classifies complex line bundles $L \to B$ completely, giving a group isomorphism $\mathrm{Vect}^1_\mathbb{C}(B) \cong H^2(B;\mathbb{Z})$.

> [!remark]
> For the tangent bundle $TM$ of a complex manifold, the Chern classes $c_k(TM)$ are intrinsic invariants. The top class $c_n(TM) \in H^{2n}(M;\mathbb{Z})$ integrates to the Euler characteristic: $\int_M c_n(TM) = \chi(M)$. This is the Gauss-Bonnet-Chern theorem, generalizing the result of Chapter 6.


## The Euler Class and Obstruction Theory

The **Euler class** $e(E) \in H^n(B;\mathbb{Z})$ of an oriented rank-$n$ real vector bundle measures the obstruction to a nowhere-zero section. It satisfies $e(E) \cup e(E) = p_{n/2}(E)$ when $n$ is even and $2e(E) = 0$ when $n$ is odd.

For the tangent bundle $TM$ of a closed oriented $n$-manifold: $\langle e(TM), [M] \rangle = \chi(M)$. The Poincaré-Hopf theorem — the sum of indices of a vector field equals the Euler characteristic — is the geometric realization of this identity. A manifold admits a nowhere-zero vector field if and only if $\chi(M) = 0$.

> [!theorem] Hairy Ball and Classification
> The tangent bundle $TS^{2k}$ of an even-dimensional sphere is non-trivial: there is no continuous nowhere-zero vector field on $S^{2k}$. For $S^1$, $S^3$, and $S^7$, the tangent bundle is trivial (these spheres are parallelizable, with $S^1 \cong U(1)$, $S^3 \cong SU(2) \cong Sp(1)$, and $S^7$ related to the octonions).


## Gauge Theory Perspective

In physics, a **gauge theory** is precisely a theory of connections on a principal $G$-bundle over spacetime $B$. The bundle $P$ is the gauge bundle, sections of associated bundles are matter fields, and the connection $A$ is the gauge potential. Under a local gauge transformation (a local section of $P$), $A$ transforms by the gauge transformation law above.

The **Yang-Mills functional** is:
$$\mathcal{YM}(A) = \int_B |F_A|^2 \, \mathrm{vol}_g,$$
where $F_A = dA + \frac{1}{2}[A \wedge A]$ is the curvature 2-form. Critical points satisfy the **Yang-Mills equations** $d_A^* F_A = 0$, with Bianchi identity $d_A F_A = 0$ automatic. **Anti-self-dual** instantons satisfy $F_A = -*F_A$ on a 4-manifold and are absolute minima of $\mathcal{YM}$. Donaldson's theorem (1983) used the moduli space of instantons on a simply-connected 4-manifold to prove deep results about smooth 4-manifold topology.

The **Chern-Simons form** is a 3-form on a 3-manifold whose integral is a gauge-invariant quantity related to the linking number of curves. It appears as the level in 3-dimensional Chern-Simons theory, controls anomalies in 4-dimensional gauge theory, and is the holographic dual of conformal field theories on the boundary.

## Holonomy Revisited

The holonomy group $\mathrm{Hol}_p(M) \subseteq O(n)$ from Chapter 5 is now seen as the holonomy group of the Levi-Civita connection on the frame bundle. More generally, for any principal $G$-bundle with connection, the **holonomy group** $\mathrm{Hol}_p \subseteq G$ is the subgroup of $G$ arising from parallel transport around loops based at $p$.

> [!theorem] Ambrose-Singer
> The Lie algebra of the holonomy group $\mathrm{Hol}_p$ is spanned by the values $\Omega_q(X, Y)$ as $q$ ranges over all points connected to $p$ by a horizontal curve and $X, Y$ range over horizontal tangent vectors at $q$.


This theorem shows that the curvature generates the infinitesimal holonomy. A flat connection ($\Omega = 0$) has discrete holonomy group; parallel transport depends only on the homotopy class of the loop, giving a representation $\pi_1(B) \to G$. This is the precise relationship between flat connections and representations of the fundamental group, fundamental to both geometric topology and the theory of local systems.

## Splittings and Extensions

A short exact sequence of vector bundles $0 \to S \to E \to Q \to 0$ always splits (non-canonically) — this follows from the existence of bundle metrics. For principal bundles, the analogous question is more subtle: an extension $1 \to N \to G \to H \to 1$ of Lie groups leads to a lifting problem: given a principal $H$-bundle $P$, does it lift to a principal $G$-bundle? The obstruction lives in a characteristic class in $H^*(B)$.

The most important example: an oriented Riemannian manifold $(M, g)$ has structure group $SO(n)$. A **spin structure** is a lift of the frame bundle to a principal $\mathrm{Spin}(n)$-bundle, where $\mathrm{Spin}(n)$ is the simply-connected double cover of $SO(n)$. The obstruction to the existence of a spin structure is the **second Stiefel-Whitney class** $w_2(TM) \in H^2(M; \mathbb{Z}/2)$. Manifolds admitting spin structures — spin manifolds — are the natural home of Dirac operators and Atiyah-Singer index theory.

The **index theorem** of Atiyah and Singer expresses the analytical index (dimension of kernel minus cokernel) of a differential operator on a bundle as a topological integral of characteristic classes over $M$. For the Dirac operator on a spin manifold:
$$\mathrm{ind}(D) = \int_M \hat{A}(TM),$$
where $\hat{A} = 1 - \frac{1}{24}p_1 + \cdots$ is the $\hat{A}$-genus, a polynomial in the Pontryagin classes. The index theorem unifies the Gauss-Bonnet-Chern theorem, Hirzebruch signature theorem, and Riemann-Roch theorem as special cases.

Fiber bundles, then, are not merely a technical formalism — they are the organizing principle of modern geometry. The next chapter moves to symplectic geometry, where the fiber is replaced by the structure of a phase space, and where the interplay of topology and dynamics generates a rich and distinct branch of geometry.

---

*The modern theory of fiber bundles was developed by Norman Steenrod (1951), whose* The Topology of Fibre Bundles *remains a foundational reference. The gauge-theoretic perspective was crystallized in the work of Yang and Mills (1954) and formalized geometrically by Atiyah, Bott, and others. Characteristic classes were introduced by Chern, Pontryagin, and Stiefel-Whitney; the Chern-Weil homomorphism provides their unified construction. The Atiyah-Singer index theorem (1963) is one of the great theorems of twentieth-century mathematics, linking analysis, topology, and geometry through the language of bundles.*
