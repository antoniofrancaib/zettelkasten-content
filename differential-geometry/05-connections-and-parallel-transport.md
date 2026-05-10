---

title: "Connections and Parallel Transport"
subtitle: "The Levi-Civita connection, covariant derivatives, parallel transport, and holonomy"
---



On a manifold, the tangent spaces at different points are distinct vector spaces with no canonical identification between them. To say that a vector field is "constant" — that it does not change as you move through the manifold — requires first deciding what "does not change" means when each step moves you to a different tangent space. In $\mathbb{R}^n$, this is trivial: identify all tangent spaces with $\mathbb{R}^n$ itself via the global coordinate frame, and differentiate componentwise. On a curved manifold, there is no such global frame, and naive componentwise differentiation produces results that depend on the coordinate choice rather than the geometry.

A **connection** is the structure that resolves this. It provides a rule for comparing tangent vectors at neighboring points — a systematic way to "connect" the tangent spaces, giving meaning to phrases like "the rate of change of a vector field along a curve." The Riemannian metric of Chapter 4 determines a unique connection — the Levi-Civita connection — that is both compatible with the metric and free of a certain asymmetry called torsion. This connection underlies the geodesic equation, the notion of parallel transport, and the curvature of a Riemannian manifold.

## The Problem with Naive Differentiation

Consider a vector field $X$ on $M$ and a tangent direction $v \in T_pM$. We want to define the derivative of $X$ in the direction $v$ at $p$. The most naive attempt: choose a curve $\gamma$ with $\gamma(0) = p$ and $\gamma'(0) = v$, and set

$$\nabla_v X \stackrel{?}{=} \lim_{t \to 0} \frac{X_{\gamma(t)} - X_{\gamma(0)}}{t}.$$

But $X_{\gamma(t)} \in T_{\gamma(t)}M$ and $X_{\gamma(0)} \in T_pM$ are vectors in different spaces — their difference is not defined. In local coordinates, one can write $X = \sum_i X^i \partial/\partial x^i$ and differentiate componentwise:

$$\left(\nabla_v X\right)^k \stackrel{?}{=} \sum_i v^i \frac{\partial X^k}{\partial x^i}.$$

This is well-defined in a single chart, but the result transforms incorrectly under coordinate changes: it does not define a tangent vector at $p$ in a coordinate-free sense. The correction term that restores correct transformation behavior under coordinate changes is exactly the Christoffel symbol contribution:

$$\left(\nabla_v X\right)^k = \sum_i v^i \frac{\partial X^k}{\partial x^i} + \sum_{i,j} \Gamma^k_{ij} v^i X^j,$$

where $\Gamma^k_{ij}$ appeared already in the geodesic equation of Chapter 4. The connection axioms formalize what structure this formula must satisfy.

## Affine Connections

> [!definition] 5.1 — Affine Connection
> An **affine connection** (or **linear connection**) on a smooth manifold $M$ is a map
>
> $$\nabla: \mathfrak{X}(M) \times \mathfrak{X}(M) \to \mathfrak{X}(M), \quad (X, Y) \mapsto \nabla_X Y,$$
>
> satisfying, for all $X, Y, Z \in \mathfrak{X}(M)$ and $f \in C^\infty(M)$:
>
> 1. **$C^\infty(M)$-linearity in the first slot**: $\nabla_{fX + Z} Y = f\nabla_X Y + \nabla_Z Y$
> 2. **$\mathbb{R}$-linearity in the second slot**: $\nabla_X(Y + Z) = \nabla_X Y + \nabla_X Z$
> 3. **Leibniz rule in the second slot**: $\nabla_X(fY) = (Xf)\, Y + f\nabla_X Y$
>
> The vector field $\nabla_X Y$ is called the **covariant derivative** of $Y$ in the direction $X$.


The first axiom says the connection is tensorial in $X$: the value of $\nabla_X Y$ at a point $p$ depends only on the value of $X$ at $p$, not on how $X$ varies near $p$. This allows $\nabla_v Y$ to be defined for a single tangent vector $v \in T_pM$ (not a full vector field), by choosing any $X$ with $X_p = v$. The second and third axioms say $\nabla_X$ is a derivation on vector fields: it is linear and satisfies the Leibniz rule.

In local coordinates $(x^i)$, a connection is completely determined by the $n^3$ smooth functions $\Gamma^k_{ij}$ defined by

$$\nabla_{\partial/\partial x^i} \frac{\partial}{\partial x^j} = \sum_k \Gamma^k_{ij} \frac{\partial}{\partial x^k}.$$

These are the **Christoffel symbols** of the connection in the given chart. The covariant derivative of a general vector field $Y = \sum_j Y^j \partial/\partial x^j$ in the direction $X = \sum_i X^i \partial/\partial x^i$ is then

$$\nabla_X Y = \sum_k \left( X(Y^k) + \sum_{i,j} \Gamma^k_{ij} X^i Y^j \right) \frac{\partial}{\partial x^k}.$$

The Christoffel symbols are not the components of a tensor — they transform with an extra inhomogeneous term under coordinate changes, exactly canceling the non-tensorial part of the componentwise derivative to produce a genuine $(1,1)$-tensor $\nabla Y$.

Connections are not unique. The space of connections on a manifold is an affine space: if $\nabla$ and $\tilde\nabla$ are two connections, their difference $\nabla - \tilde\nabla$ is a $(1,2)$-tensor field (a tensor because the non-tensorial transformation terms cancel). Conversely, adding any $(1,2)$-tensor $A$ to a connection gives another connection. This freedom is what makes the additional conditions of metric compatibility and torsion-freeness so powerful — they single out a unique connection.

## Torsion and the Symmetry Condition

> [!definition] 5.2 — Torsion Tensor
> The **torsion tensor** of a connection $\nabla$ is the $(1,2)$-tensor field
>
> $$T(X, Y) = \nabla_X Y - \nabla_Y X - [X, Y].$$
>
> A connection is **torsion-free** (or **symmetric**) if $T \equiv 0$, equivalently if $\nabla_X Y - \nabla_Y X = [X, Y]$ for all $X, Y$.


In coordinates, torsion-freeness is equivalent to the symmetry $\Gamma^k_{ij} = \Gamma^k_{ji}$. The torsion tensor measures how much the connection fails to commute with the Lie bracket: on $\mathbb{R}^n$ with the standard connection, the Christoffel symbols vanish and $[X,Y]$ exactly captures the antisymmetric part of the derivative, so torsion vanishes. The geometric meaning of torsion is subtle — it measures a kind of "twisting" of geodesics relative to the frame — and its vanishing is a naturality condition rather than a deep geometric restriction.

## Metric Compatibility

> [!definition] 5.3 — Metric-Compatible Connection
> A connection $\nabla$ on a Riemannian manifold $(M, g)$ is **metric-compatible** (or **compatible with the metric**) if
>
> $$X\bigl(g(Y, Z)\bigr) = g(\nabla_X Y, Z) + g(Y, \nabla_X Z)$$
>
> for all $X, Y, Z \in \mathfrak{X}(M)$. Equivalently, $\nabla g = 0$ — the metric is parallel with respect to $\nabla$.


Metric compatibility says that the connection preserves inner products: if you parallel-transport two vectors along any curve while keeping them "constant" according to the connection, their inner product does not change. This is the correct notion of a connection that respects the geometry of a Riemannian manifold.

In coordinates, metric compatibility is expressed as

$$\frac{\partial g_{jk}}{\partial x^i} = \sum_\ell \left( \Gamma^\ell_{ij} g_{\ell k} + \Gamma^\ell_{ik} g_{j\ell} \right).$$

## The Levi-Civita Connection

The two conditions — torsion-free and metric-compatible — together uniquely determine the connection.

> [!theorem] 5.1 — Fundamental Theorem of Riemannian Geometry
> On any Riemannian manifold $(M, g)$, there exists a unique affine connection $\nabla$ that is:
> - **torsion-free**: $\nabla_X Y - \nabla_Y X = [X,Y]$ for all $X,Y$, and
> - **metric-compatible**: $\nabla g = 0$.
>
> This connection is called the **Levi-Civita connection** of $(M, g)$. Its Christoffel symbols in any coordinate chart are
>
> $$\Gamma^k_{ij} = \frac{1}{2}\sum_\ell g^{k\ell}\!\left(\frac{\partial g_{i\ell}}{\partial x^j} + \frac{\partial g_{j\ell}}{\partial x^i} - \frac{\partial g_{ij}}{\partial x^\ell}\right).$$


The proof is a beautiful piece of algebra. Metric compatibility gives three equations (by cyclically permuting $X, Y, Z$ in the defining identity); torsion-freeness lets the Lie brackets be replaced by commutators of the connection; combining these three equations and solving for $g(\nabla_X Y, Z)$ yields the **Koszul formula**:

> [!equation] 5.1
> $$2\,g(\nabla_X Y, Z) = X\bigl(g(Y,Z)\bigr) + Y\bigl(g(X,Z)\bigr) - Z\bigl(g(X,Y)\bigr) - g([X,Z],Y) - g([Y,Z],X) - g([X,Y],Z).$$


The right side depends only on $g$ and the Lie bracket — no connection appears. Since $g$ is non-degenerate, this uniquely determines $\nabla_X Y$. Taking $X = \partial/\partial x^i$, $Y = \partial/\partial x^j$, $Z = \partial/\partial x^\ell$ (where all Lie brackets vanish) and solving gives the Christoffel symbol formula stated in the theorem.

The Levi-Civita connection is **the** natural connection on a Riemannian manifold. When a differential geometer says "the covariant derivative" without qualification, they mean the Levi-Civita covariant derivative. The Christoffel symbols of Chapter 4 were precisely the Christoffel symbols of this connection.

## Covariant Derivative Along a Curve

The connection enables differentiation not just of vector fields along other vector fields, but of vector fields along curves.

> [!definition] 5.4 — Covariant Derivative Along a Curve
> Let $\gamma: I \to M$ be a smooth curve. A **vector field along $\gamma$** is a smooth map $V: I \to TM$ with $V(t) \in T_{\gamma(t)}M$ for all $t$. The **covariant derivative of $V$ along $\gamma$** is the vector field along $\gamma$ given in local coordinates by
>
> $$\frac{DV}{dt} = \sum_k \left(\frac{dV^k}{dt} + \sum_{i,j} \Gamma^k_{ij}(\gamma(t))\, \dot\gamma^i(t)\, V^j(t)\right)\frac{\partial}{\partial x^k}\bigg|_{\gamma(t)}.$$
>
> This is also written $\nabla_{\dot\gamma} V$ when $V$ extends to a vector field in a neighborhood of $\gamma$.


The geodesic equation of Chapter 4 now has a clean reformulation: a curve $\gamma$ is a **geodesic** if and only if its velocity vector field is parallel along itself,

$$\frac{D\dot\gamma}{dt} = 0.$$

This is the intrinsic statement that a geodesic has zero acceleration — it does not turn relative to the connection. In coordinates, expanding $D\dot\gamma/dt = 0$ immediately gives the geodesic equation with Christoffel symbols.

## Parallel Transport

The covariant derivative along a curve enables the definition of "keeping a vector constant as you move along a curve."

> [!definition] 5.5 — Parallel Vector Field and Parallel Transport
> A vector field $V$ along a curve $\gamma: [a,b] \to M$ is **parallel** (along $\gamma$) if
>
> $$\frac{DV}{dt} = 0 \quad \text{for all } t \in [a, b].$$
>
> Given an initial vector $V_0 \in T_{\gamma(a)}M$, there exists a unique parallel vector field $V$ along $\gamma$ with $V(a) = V_0$. The resulting linear map
>
> $$P_\gamma: T_{\gamma(a)}M \to T_{\gamma(b)}M, \quad V_0 \mapsto V(b),$$
>
> is called **parallel transport** along $\gamma$.


Existence and uniqueness of the parallel vector field follow from the ODE theory applied to $DV/dt = 0$: it is a linear first-order system in the components of $V$, with smooth coefficients determined by $\Gamma^k_{ij}$ along $\gamma$. The linearity of the equation makes $P_\gamma$ a linear isomorphism.

Metric compatibility has an immediate consequence: parallel transport is an **isometry** between tangent spaces. If $\nabla g = 0$ and $V, W$ are both parallel along $\gamma$, then

$$\frac{d}{dt} g(V, W) = g\!\left(\frac{DV}{dt}, W\right) + g\!\left(V, \frac{DW}{dt}\right) = 0.$$

So inner products are preserved: $g_{\gamma(b)}(P_\gamma V_0, P_\gamma W_0) = g_{\gamma(a)}(V_0, W_0)$. Parallel transport sends orthonormal frames to orthonormal frames.

The dependence of $P_\gamma$ on the path $\gamma$ — not just the endpoints — is a fundamental feature of curved spaces. In $\mathbb{R}^n$ with the Euclidean connection, parallel transport is trivial: it is the identity (just translate the vector). On a curved manifold, different paths from $p$ to $q$ can give different isometries $T_pM \to T_qM$.

## Holonomy

The most striking manifestation of path-dependence is holonomy: what happens when you parallel-transport a vector around a closed loop.

> [!definition] 5.6 — Holonomy Group
> Let $(M, g, \nabla)$ be a Riemannian manifold with Levi-Civita connection, and $p \in M$. For each piecewise smooth loop $\gamma: [0,1] \to M$ based at $p$ (i.e., $\gamma(0) = \gamma(1) = p$), parallel transport gives a linear isometry $P_\gamma: T_pM \to T_pM$.
>
> The **holonomy group** at $p$ is
>
> $$\mathrm{Hol}_p(M) = \{P_\gamma \in O(T_pM) : \gamma \text{ is a loop based at } p\} \subseteq O(n).$$
>
> This is indeed a subgroup of $O(n)$ (under composition of loops), and for a connected manifold the holonomy groups at different base points are conjugate subgroups of $O(n)$.


The holonomy group measures the "non-triviality" of the connection. A connection is **flat** — locally equivalent to the standard connection on $\mathbb{R}^n$ — if and only if the holonomy group is trivial: every loop gives the identity transformation.

The classic example is the sphere $S^2$. Take a tangent vector at the north pole $N$ pointing toward the prime meridian. Parallel-transport it:

1. South along the prime meridian to the equator — the vector points east.
2. East along the equator by angle $\theta$ — the vector still points east (it is parallel along a great circle).
3. North along the $\theta$-meridian back to $N$ — the vector has rotated by angle $\theta$ relative to where it started.

A vector transported around a spherical triangle with interior angles $\alpha, \beta, \gamma$ returns rotated by the **spherical excess** $\alpha + \beta + \gamma - \pi$ — which by the Gauss-Bonnet theorem equals the area of the triangle divided by $R^2$ (the square of the radius). This is the first direct glimpse of the connection between holonomy and curvature: the failure of parallel transport to return a vector to itself is measured by the curvature integrated over the enclosed region.

For $S^2$ with the round metric, $\mathrm{Hol}(S^2) = SO(2)$ — all rotations of the tangent plane are achievable by parallel transport around some loop. For $\mathbb{R}^n$, $\mathrm{Hol}(\mathbb{R}^n) = \{I\}$. Berger's classification (1955) lists all possible holonomy groups of simply connected irreducible Riemannian manifolds: $SO(n)$, $U(n/2)$ (Kähler manifolds), $SU(n/2)$ (Calabi-Yau manifolds), $Sp(n/4)$ (hyperkähler manifolds), $Sp(n/4) \cdot Sp(1)$ (quaternionic Kähler), and two exceptional cases $G_2$ and $Spin(7)$.

> [!remark]
> Holonomy has a direct physical interpretation: it is the **Berry phase** in quantum mechanics. When the Hamiltonian of a quantum system is slowly varied around a closed loop in parameter space, the state vector acquires a phase equal to the holonomy of a connection on the bundle of ground states over parameter space. The connection here is not the Levi-Civita connection but the Berry connection — the natural connection on a Hermitian line bundle — and the curvature of this connection is the Berry curvature, which integrates to the topologically protected Chern number.


## The Connection on Tensor Bundles

The Levi-Civita connection extends canonically from the tangent bundle to all tensor bundles, via the Leibniz rule and the requirement that it commute with contractions.

For a $(r, s)$-tensor field $T$, the covariant derivative $\nabla_X T$ is defined by demanding:

- $\nabla_X f = Xf$ for smooth functions ($r = s = 0$)
- $\nabla_X$ agrees with the Levi-Civita connection on vector fields ($r = 1, s = 0$)
- $\nabla_X(\xi)(Y) = X(\xi(Y)) - \xi(\nabla_X Y)$ for 1-forms $\xi$ ($r = 0, s = 1$)
- $\nabla_X(T \otimes S) = (\nabla_X T) \otimes S + T \otimes (\nabla_X S)$ (Leibniz rule for tensor products)

In coordinates, the covariant derivative of a $(1,1)$-tensor $T^i{}_j$ in the direction $\partial/\partial x^k$ is:

$$(\nabla_k T)^i{}_j = \frac{\partial T^i{}_j}{\partial x^k} + \Gamma^i_{k\ell}\, T^\ell{}_j - \Gamma^\ell_{kj}\, T^i{}_\ell.$$

Each upper index contributes a $+\Gamma$ term; each lower index contributes a $-\Gamma$ term. The pattern is fully determined by the Leibniz rule and the connection on vectors and covectors.

The **covariant differential** $\nabla T$ is the $(r, s+1)$-tensor obtained by adding one covariant index, so that $(\nabla T)(X, \ldots) = (\nabla_X T)(\ldots)$. A tensor $T$ with $\nabla T = 0$ is called **parallel** — it is "constant" with respect to the connection. The metric $g$ is parallel ($\nabla g = 0$) by metric compatibility, and so is $g^{-1}$ and the volume form $\mathrm{vol}_g$.

## Second Covariant Derivatives and Commutators

On $\mathbb{R}^n$, partial derivatives commute: $\partial^2 f / \partial x^i \partial x^j = \partial^2 f / \partial x^j \partial x^i$. The analogous statement for covariant derivatives — $\nabla_X \nabla_Y - \nabla_Y \nabla_X$ — fails on a curved manifold. This failure is not a defect but the defining measurement of curvature.

For vector fields $X, Y, Z$, define the operator

$$R(X, Y)Z = \nabla_X \nabla_Y Z - \nabla_Y \nabla_X Z - \nabla_{[X,Y]} Z.$$

This turns out to be tensorial in all three arguments — a $(1,3)$-tensor field called the **Riemann curvature tensor** — because all the derivative terms cancel, leaving only algebraic operations in the components of $X$, $Y$, $Z$ at each point. The curvature tensor is the central object of the next chapter.

The present point is structural: the covariant derivative, defined to solve the problem of differentiating vector fields in a coordinate-free way, contains the seeds of curvature. In trying to differentiate twice, we discover that the order of differentiation matters, and the measure of how much it matters is the curvature. This is not unlike how the commutator of two flows — the Lie bracket — measures the non-commutativity of the corresponding vector fields. Connection, parallel transport, holonomy, and curvature are all facets of the same geometric truth: on a curved manifold, there is no canonical way to identify tangent spaces at different points, and the connection is our best approximation to such an identification. The extent to which it falls short of being a true identification is exactly what curvature quantifies.

## Connections and Geodesics Revisited

With the Levi-Civita connection in hand, the geodesic equation of Chapter 4 finds its proper conceptual home.

> [!theorem] 5.2 — Geodesics as Auto-Parallel Curves
> A smooth curve $\gamma: I \to M$ is a geodesic of the Levi-Civita connection if and only if $\frac{D\dot\gamma}{dt} = 0$ — its velocity vector is parallel along itself. The geodesic equation in coordinates is equivalent to this condition and follows directly from expanding $D\dot\gamma/dt = 0$ using the Christoffel symbols.
>
> Moreover, for any $p \in M$ and $v \in T_pM$, the unique geodesic $\gamma_v$ with $\gamma_v(0) = p$ and $\dot\gamma_v(0) = v$ has constant speed: $|\dot\gamma_v(t)|_g = |v|_g$ for all $t$.


Constant speed follows from $\frac{d}{dt}|\dot\gamma|^2 = 2g(D\dot\gamma/dt, \dot\gamma) = 0$ — the connection is metric-compatible, so the inner product of a parallel vector field with itself is constant. This confirms that geodesics are parametrized by arc-length (up to a constant rescaling), and that the exponential map $\exp_p(v)$ represents the point at arc-length distance $|v|_g$ from $p$ along the geodesic in direction $v/|v|_g$.

The connection thus unifies three previously separate ideas: the covariant derivative (how to differentiate vector fields), parallel transport (how to compare vectors at different points along a curve), and geodesics (the natural notion of "straight" curves). All three arise from the single data of the Levi-Civita connection, which is itself uniquely determined by the Riemannian metric.

---

*The axiomatic treatment of connections is in Kobayashi and Nomizu's* Foundations of Differential Geometry *(Vol. 1, 1963), the definitive reference. A more accessible modern treatment is in Lee's* Introduction to Riemannian Manifolds *(Chapters 4–5). Holonomy and Berger's classification are beautifully surveyed in Besse's* Einstein Manifolds *(Chapter 10). The connection to Berry's phase is developed in Simon (1983) and Shapere and Wilczek's* Geometric Phases in Physics *(1989).*
