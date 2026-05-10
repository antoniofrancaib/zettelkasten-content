---

title: "Symplectic Geometry"
subtitle: "The geometry of phase space"
---



Classical mechanics is, at its heart, geometry. The configuration of a mechanical system — the positions of particles, the angles of a pendulum, the shape of a molecule — forms a manifold. But dynamics requires not just positions but momenta, and it is in the cotangent bundle $T^*M$ that the full structure of Hamiltonian mechanics lives. The key geometric object is not a metric but a closed, non-degenerate 2-form: the symplectic form. Symplectic geometry is the study of manifolds equipped with this structure, and its theorems are statements about the conservation laws, integrability, and obstructions that govern Hamiltonian systems. It is also the natural geometric framework for quantum mechanics, geometric quantization, and the moment map constructions that appear throughout modern mathematics.

## Symplectic Manifolds

> [!definition] Symplectic Form
> A **symplectic form** on a smooth manifold $M$ is a 2-form $\omega \in \Omega^2(M)$ that is:
> - **Closed**: $d\omega = 0$,
> - **Non-degenerate**: for every $p \in M$ and every nonzero $v \in T_pM$, there exists $w \in T_pM$ with $\omega_p(v, w) \neq 0$.
>
> A **symplectic manifold** is a pair $(M, \omega)$.


Non-degeneracy forces $M$ to be even-dimensional: if $\dim M = 2n$, then $\omega^n = \omega \wedge \cdots \wedge \omega$ ($n$ times) is a volume form, so symplectic manifolds are always orientable. Closedness is the key dynamical condition — it encodes conservation of energy and Liouville's theorem.

The canonical example is $\mathbb{R}^{2n}$ with coordinates $(q^1, \ldots, q^n, p_1, \ldots, p_n)$ and:

> [!equation]
> $$\omega_{\mathrm{can}} = \sum_{i=1}^n dq^i \wedge dp_i.$$


In matrix form, $\omega$ pairs the $q$-directions against the $p$-directions with the antisymmetric matrix $J = \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix}$.

More generally, the cotangent bundle $T^*M$ of any manifold carries a canonical symplectic form. The **tautological 1-form** (Liouville form) $\lambda \in \Omega^1(T^*M)$ is defined by $\lambda_{(q,p)}(v) = p(d\pi\, v)$ for $v \in T_{(q,p)}(T^*M)$, and $\omega = -d\lambda$ is a symplectic form. In local coordinates this recovers $\omega_{\mathrm{can}}$. Phase space in classical mechanics is always a cotangent bundle.

Beyond cotangent bundles, compact symplectic manifolds abound:
- **Kähler manifolds**: a complex manifold $M$ with a Hermitian metric $h$ has $\omega = \mathrm{Im}(h)$, which is symplectic whenever $\partial\bar\partial$-closed (i.e., $\omega$ is Kähler). All projective complex varieties are Kähler.
- **Coadjoint orbits** $\mathcal{O} \subset \mathfrak{g}^*$: equipped with the Kirillov-Kostant-Souriau (KKS) form, these are the natural symplectic leaves of Lie-Poisson structures.
- **Surfaces**: every oriented surface carries a symplectic structure (take any area form).

> [!remark]
> Symplectic manifolds carry no local invariants beyond dimension: by Darboux's theorem (below), every symplectic manifold looks locally like $(\mathbb{R}^{2n}, \omega_{\mathrm{can}})$. This is in stark contrast to Riemannian geometry, where curvature is a local invariant. All of symplectic geometry is therefore **global** — it is a subject of global topology and group actions.


## Darboux's Theorem

The absence of local invariants is encoded in the following foundational result.

> [!theorem] Darboux
> Let $(M, \omega)$ be a symplectic manifold of dimension $2n$ and $p \in M$. Then there exist local coordinates $(q^1, \ldots, q^n, p_1, \ldots, p_n)$ near $p$ — called **Darboux coordinates** — in which
> $$\omega = \sum_{i=1}^n dq^i \wedge dp_i.$$


The proof uses Moser's trick: connect $\omega$ to a constant-coefficient form by a path $\omega_t = (1-t)\omega_0 + t\omega_1$ and construct a time-dependent vector field $X_t$ whose flow isotopes one to the other. The Moser trick is enormously useful throughout symplectic topology.

A consequence: any two symplectic manifolds of the same dimension are locally diffeomorphic. There is no symplectic analogue of Riemannian curvature. The global topology is what distinguishes them.

## Hamiltonian Mechanics

Given a smooth function $H: M \to \mathbb{R}$ (the **Hamiltonian**), non-degeneracy of $\omega$ defines a unique vector field $X_H$ by:

> [!equation]
> $$\iota_{X_H}\omega = dH, \qquad \text{i.e.,} \quad \omega(X_H, \cdot) = dH.$$


The vector field $X_H$ is called the **Hamiltonian vector field** of $H$.

> [!definition] Hamiltonian Flow
> The flow $\phi_t^H$ of $X_H$ is the **Hamiltonian flow** of $H$. In Darboux coordinates, Hamilton's equations take the classical form:
> $$\dot{q}^i = \frac{\partial H}{\partial p_i}, \qquad \dot{p}_i = -\frac{\partial H}{\partial q^i}.$$


Closedness of $\omega$ ensures that the Hamiltonian flow preserves $\omega$: $(\phi_t^H)^*\omega = \omega$. This is **Liouville's theorem** — the symplectic form (and hence the volume $\omega^n/n!$) is preserved by Hamiltonian flow, so phase space volume is conserved. It is the geometric origin of the second law of thermodynamics in the Liouville picture.

The **Poisson bracket** of two functions $f, g \in C^\infty(M)$ is:
$$\{f, g\} = \omega(X_f, X_g) = df(X_g).$$
In Darboux coordinates: $\{f, g\} = \sum_i \frac{\partial f}{\partial q^i}\frac{\partial g}{\partial p_i} - \frac{\partial f}{\partial p_i}\frac{\partial g}{\partial q^i}$.

> [!theorem] Conservation and the Lie Algebra of Observables
> For any Hamiltonian $H$:
> 1. $\mathcal{L}_{X_H} f = \{H, f\}$ — the time evolution of $f$ along the flow is its Poisson bracket with $H$.
> 2. $f$ is a **first integral** (conserved quantity) if and only if $\{H, f\} = 0$.
> 3. The map $f \mapsto X_f$ is a Lie algebra homomorphism: $X_{\{f,g\}} = [X_f, X_g]$.
> 4. $C^\infty(M)$ with the Poisson bracket is a Lie algebra, and the Jacobi identity $\{f, \{g, h\}\} + \{g, \{h, f\}\} + \{h, \{f, g\}\} = 0$ follows from $d\omega = 0$.


The correspondence between symmetries and conservation laws — **Noether's theorem** — has its sharpest form in symplectic geometry via moment maps.

## Lagrangian Submanifolds

The correct notion of "submanifold" in symplectic geometry is not a Riemannian submanifold but a Lagrangian one.

> [!definition] Lagrangian Submanifold
> A submanifold $L \subset (M, \omega)$ of dimension $n = \frac{1}{2}\dim M$ is **Lagrangian** if $\omega|_L = 0$, i.e., $\omega(v, w) = 0$ for all $v, w \in TL$.


Examples:
- In $T^*M$: the **zero section** $M \hookrightarrow T^*M$ and the **graph** of any closed 1-form $\alpha$ (closed because $d\alpha = \omega|_{\mathrm{graph}\,\alpha}$, so $\omega|_{\mathrm{graph}} = 0$ iff $d\alpha = 0$).
- In $(\mathbb{R}^{2n}, \omega_{\mathrm{can}})$: any $n$-dimensional subspace on which $J$ vanishes, e.g., the $q$-plane $\{p = 0\}$.
- **Tori**: a flat $n$-torus in an integrable system's phase space (see below).

> [!theorem] Weinstein Lagrangian Neighborhood
> Any Lagrangian submanifold $L \subset (M, \omega)$ has a tubular neighborhood symplectomorphic to a neighborhood of the zero section in $(T^*L, \omega_{\mathrm{can}})$. In particular, Lagrangian submanifolds are rigid: they cannot be deformed symplectically without remaining Lagrangian.


Lagrangian submanifolds are the geometric objects that Hamiltonian flows generate: if $L$ is Lagrangian, so is $\phi_t^H(L)$ for any $t$.

## Symplectomorphisms and Symplectic Topology

A **symplectomorphism** is a diffeomorphism $\phi: (M, \omega) \to (M', \omega')$ with $\phi^*\omega' = \omega$. The group $\mathrm{Symp}(M, \omega)$ of symplectomorphisms is an infinite-dimensional Lie group whose Lie algebra is the space of symplectic vector fields (vector fields $X$ with $\mathcal{L}_X\omega = 0$, equivalently $d(\iota_X\omega) = 0$).

Hamiltonian vector fields are symplectic; the converse holds locally (since $\iota_X\omega$ is closed, it is locally exact). The **flux** measures the global obstruction: $\mathrm{flux}(X) \in H^1(M; \mathbb{R})$.

A foundational result distinguishing symplectic topology from volume-preserving topology:

> [!theorem] Gromov Non-Squeezing
> A symplectic ball $B^{2n}(r)$ of radius $r$ in $(\mathbb{R}^{2n}, \omega_{\mathrm{can}})$ can be symplectically embedded into the cylinder $Z^{2n}(R) = B^2(R) \times \mathbb{R}^{2n-2}$ if and only if $r \leq R$.


Gromov proved this in 1985 using his theory of **pseudoholomorphic curves** — maps from Riemann surfaces to $(M, J)$ for an almost complex structure $J$ compatible with $\omega$. The non-squeezing theorem shows that symplectic topology is genuinely richer than volume-preserving topology: a ball cannot be squeezed through a cylinder of smaller radius, even though both have the same volume for appropriate $r < R$. The symplectic width — the infimum of radii of symplectic balls that embed — is a symplectic invariant.

> [!remark]
> Pseudoholomorphic curves have become one of the central tools of symplectic topology. They led to Floer homology (a Morse theory for the symplectic action functional on the loop space), Gromov-Witten invariants (counts of holomorphic curves), and string topology. The non-squeezing theorem is the first indication that symplectic geometry has genuine rigidity.


## Moment Maps and Symplectic Reduction

When a Lie group $G$ acts on $(M, \omega)$ by symplectomorphisms, the moment map encodes the conserved charges.

> [!definition] Moment Map
> Let $G$ act on $(M, \omega)$ by symplectomorphisms, with infinitesimal action $\xi_M$ for $\xi \in \mathfrak{g}$. A **moment map** is a smooth equivariant map $\mu: M \to \mathfrak{g}^*$ such that for all $\xi \in \mathfrak{g}$:
> $$d\langle \mu, \xi \rangle = \iota_{\xi_M}\omega,$$
> where $\langle \mu, \xi \rangle: M \to \mathbb{R}$ is the component of $\mu$ in the direction $\xi$.


Each component $\mu^\xi = \langle \mu, \xi \rangle$ is a Hamiltonian function for the vector field $\xi_M$. Noether's theorem takes the form: if $H$ is $G$-invariant, then $\mu$ is conserved along the flow of $X_H$, i.e., $\{H, \mu^\xi\} = 0$ for all $\xi$.

Examples:
- $G = \mathbb{R}^n$ acting on $T^*\mathbb{R}^n$ by translation: $\mu(q, p) = p$ — momentum.
- $G = SO(3)$ acting on $T^*\mathbb{R}^3$: $\mu(q, p) = q \times p$ — angular momentum.
- $G = U(n)$ acting on $\mathbb{C}^n$: $\mu(z) = \frac{i}{2}z\bar{z}^T$ — Hermitian square.
- Coadjoint action of $G$ on $\mathfrak{g}^*$: the moment map is the identity, and the symplectic leaves are coadjoint orbits with KKS form.

> [!theorem] Marsden-Weinstein-Meyer Symplectic Reduction
> Let $G$ act freely and properly on $(M, \omega)$ with moment map $\mu: M \to \mathfrak{g}^*$. For a regular value $\lambda \in \mathfrak{g}^*$, the **reduced space**
> $$M_\lambda := \mu^{-1}(\lambda) / G_\lambda,$$
> where $G_\lambda$ is the stabilizer of $\lambda$ under the coadjoint action, carries a unique symplectic form $\omega_\lambda$ satisfying $\pi^*\omega_\lambda = \iota^*\omega$, where $\iota: \mu^{-1}(\lambda) \hookrightarrow M$ and $\pi: \mu^{-1}(\lambda) \to M_\lambda$.


Symplectic reduction is the geometric mechanism behind gauge fixing in physics (reducing by gauge symmetry), behind the Kähler quotient in algebraic geometry, and behind integrable systems (reducing to action-angle coordinates). The reduced space $M_\lambda$ is lower-dimensional: $\dim M_\lambda = \dim M - 2\dim G$.

## Integrable Systems

A Hamiltonian system on a $2n$-dimensional symplectic manifold $(M, \omega)$ is **completely integrable** if it possesses $n$ functionally independent, Poisson-commuting first integrals $F_1 = H, F_2, \ldots, F_n$: $\{F_i, F_j\} = 0$ and $dF_1 \wedge \cdots \wedge dF_n \neq 0$ on an open dense set.

> [!theorem] Arnold-Liouville
> Let $(M^{2n}, \omega, H)$ be a completely integrable system. For a regular value $c \in \mathbb{R}^n$, the level set $M_c = \{F_1 = c_1, \ldots, F_n = c_n\}$ is a Lagrangian submanifold. If $M_c$ is compact and connected, it is diffeomorphic to an $n$-torus $\mathbb{T}^n$. Near $M_c$ there exist **action-angle coordinates** $(\theta^1, \ldots, \theta^n, I_1, \ldots, I_n)$ in which $\omega = \sum_i d\theta^i \wedge dI_i$ and $H = H(I)$ depends only on the actions $I_i$.


In action-angle coordinates, the equations of motion are $\dot\theta^i = \partial H/\partial I_i$ (constant frequencies), $\dot I_i = 0$. The motion is quasiperiodic on invariant tori. Integrable systems are the exactly solvable ones: the Kepler problem, the harmonic oscillator, the Euler top, the Toda lattice, the KdV equation (infinitely many commuting flows on an infinite-dimensional phase space).

The **KAM theorem** (Kolmogorov-Arnold-Moser) shows that for nearly integrable systems, most invariant tori survive small Hamiltonian perturbations, though the resonant ones are destroyed. The surviving tori are characterized by Diophantine frequency conditions.

## Contact Geometry and Odd-Dimensional Companions

Symplectic geometry has a natural odd-dimensional counterpart. A **contact structure** on a $(2n+1)$-dimensional manifold $M$ is a maximally non-integrable hyperplane distribution $\xi = \ker\alpha$ for a 1-form $\alpha$ satisfying $\alpha \wedge (d\alpha)^n \neq 0$ everywhere. The form $d\alpha|_\xi$ is a symplectic form on each hyperplane.

Contact manifolds arise as:
- Unit sphere bundles $SM$ of Riemannian manifolds, with the canonical contact form from geodesic flow.
- Boundaries of symplectic domains: if $M \subset (W, \omega)$ is a compact hypersurface with $\omega|_M$ having a one-dimensional kernel, $M$ inherits a contact structure.
- Energy hypersurfaces $H^{-1}(E)$ in Hamiltonian systems (the Reeb vector field on a contact manifold is the analogue of Hamiltonian flow).

The **Reeb vector field** $R$ on $(M, \alpha)$ is uniquely defined by $\iota_R d\alpha = 0$ and $\alpha(R) = 1$. Closed Reeb orbits are the analogue of periodic orbits. The **Weinstein conjecture** (proved by Taubes for $\dim M = 3$) asserts that any Reeb flow on a compact contact 3-manifold has at least one closed orbit.

## Symplectic Capacities and Rigidity

The non-squeezing theorem is the first example of a **symplectic capacity**: a functor $c$ from symplectic manifolds to $[0, \infty]$ that is monotone under symplectic embeddings, conformally covariant ($c(M, \lambda\omega) = \lambda\, c(M, \omega)$), and normalized by $c(B^{2n}(r)) = c(Z^{2n}(r)) = \pi r^2$.

The existence of any symplectic capacity with these properties implies the non-squeezing theorem. Capacities measure the "symplectic size" of a domain — they are intermediate between volume (which is $n$-dimensional) and the full symplectomorphism type. The **Ekeland-Hofer capacities** and the **Hofer-Zehnder capacity** (related to periodic orbits) provide a rich hierarchy.

> [!remark]
> The Hofer metric on $\mathrm{Ham}(M, \omega)$ — the group of Hamiltonian diffeomorphisms — gives it the structure of an infinite-dimensional Riemannian manifold. The geodesics correspond to flows of time-independent Hamiltonians. The resulting geometry of the group of Hamiltonians, developed by Hofer, Lalonde, and Polterovich, shows that the group of Hamiltonian diffeomorphisms has infinite diameter and is more rigid than the group of volume-preserving diffeomorphisms.


## Geometric Quantization

A central motivation for symplectic geometry is **quantization**: the passage from classical mechanics (functions on a symplectic manifold) to quantum mechanics (operators on a Hilbert space). The Poisson algebra $(C^\infty(M), \{\ ,\ \})$ should map to an algebra of self-adjoint operators with commutators $[\hat f, \hat g] = -i\hbar\widehat{\{f, g\}}$.

Geometric quantization provides a systematic (though not fully canonical) approach:
1. **Pre-quantization**: choose a Hermitian line bundle $L \to M$ with connection $\nabla$ of curvature $\frac{i}{\hbar}\omega$ (this requires $[\omega/2\pi\hbar] \in H^2(M; \mathbb{Z})$ — the **pre-quantization condition** or integrality of $\omega$). The Hilbert space of pre-quantum states is $\Gamma(L)$.
2. **Polarization**: choose a Lagrangian distribution $\mathcal{P} \subset T_\mathbb{C}M$ (a complex Lagrangian foliation) and take sections covariantly constant along $\mathcal{P}$. For $T^*M$ with vertical polarization this recovers wave functions as functions of $q$ alone.
3. **Metaplectic correction**: adjust for the half-form bundle $|\Lambda^{1/2}\mathcal{P}|$ to get the correct inner product and quantization of $H = p^2/2m$.

For compact $M$, the quantization condition forces $\omega \in H^2(M; \mathbb{Z})$ up to scaling, which is exactly the condition for $M$ to be a Kähler manifold with integral Kähler class — the Kodaira embedding theorem says such $M$ embeds in projective space. Geometric quantization thus connects the integrality of cohomology classes to the discreteness of quantum spectra.

The coadjoint orbit picture is especially clean: quantizing the coadjoint orbit $\mathcal{O}_\lambda \subset \mathfrak{g}^*$ of a compact Lie group $G$ at the integral weight $\lambda$ produces the irreducible representation $V_\lambda$. This is the orbit method of Kirillov, Kostant, and Souriau, which classifies irreducible representations of nilpotent and compact Lie groups geometrically.

Symplectic geometry thus closes a remarkable loop: it starts as the geometry of classical mechanics, reveals itself as the foundation of Lie group representation theory, and opens onto quantum mechanics through the integrality conditions that link cohomology to spectral theory. The next chapter turns to information geometry, where the fiber is a probability simplex and the metric is the Fisher information, drawing together Riemannian geometry, statistics, and the same duality structures we have seen throughout.

---

*The modern formulation of symplectic geometry owes much to Jean-Marie Souriau, Vladimir Arnold (whose* Mathematical Methods of Classical Mechanics *remains the definitive reference), and Alan Weinstein. Mikhail Gromov's 1985 paper introducing pseudoholomorphic curves transformed the subject. Symplectic reduction was developed by Marsden, Weinstein, and Meyer independently. The Atiyah-Guillemin-Sternberg convexity theorem, Floer homology, and Fukaya categories have since made symplectic geometry one of the most active fields in mathematics.*
