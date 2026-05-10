---

title: "The Riemannian Metric"
subtitle: "Lengths, angles, geodesics, the exponential map, and what it means to measure on a curved space"
---



The smooth manifolds of Part I are purely topological objects — they know which maps are continuous and which are smooth, but they have no notion of distance. Two points on the sphere could be a millimeter apart or on opposite sides, and the smooth structure cannot tell the difference. Differential forms gave us a language for integration, but the natural measure on a Riemannian manifold — its volume form — has not yet appeared, because it requires knowing how to measure lengths.

A **Riemannian metric** is the additional structure that turns a smooth manifold into a geometric space: an inner product on each tangent space, varying smoothly from point to point. Once a metric is chosen, lengths, angles, areas, volumes, and distances all become well-defined. The shortest paths through the space — geodesics — emerge as solutions to an ODE, and the exponential map converts infinitesimal data at a point into finite motion on the manifold. This is the chapter where differential geometry becomes genuinely geometric.

## Riemannian Metrics

> [!definition] 4.1 — Riemannian Metric and Riemannian Manifold
> A **Riemannian metric** on a smooth manifold $M$ is a smooth symmetric $(0,2)$-tensor field $g$ such that for every $p \in M$, the bilinear form
>
> $$g_p: T_pM \times T_pM \to \mathbb{R}$$
>
> is an **inner product** — symmetric, bilinear, and positive definite: $g_p(v, v) > 0$ for all $v \neq 0$.
>
> A **Riemannian manifold** is a pair $(M, g)$ of a smooth manifold with a Riemannian metric. A smooth map $F: (M, g) \to (N, h)$ is an **isometry** if it is a diffeomorphism satisfying $F^*h = g$, i.e., $h_{F(p)}(dF_p\, u,\, dF_p\, v) = g_p(u, v)$ for all $p \in M$ and $u, v \in T_pM$.


In a local chart $(U, x^i)$, the metric is written

$$g = \sum_{i,j} g_{ij}(x)\, dx^i \otimes dx^j,$$

where the **metric coefficients** $g_{ij} = g(\partial/\partial x^i, \partial/\partial x^j)$ are smooth functions on $U$ forming a positive definite symmetric matrix at every point. The notation $ds^2 = g_{ij}\, dx^i\, dx^j$ (using Einstein summation) is common in physics and classical differential geometry.

Under a change of coordinates $x \mapsto \tilde x$, the metric coefficients transform as

$$\tilde g_{k\ell} = \sum_{i,j} g_{ij} \frac{\partial x^i}{\partial \tilde x^k} \frac{\partial x^j}{\partial \tilde x^\ell}.$$

This is the covariant transformation law — both indices transform by the transpose Jacobian — consistent with $g$ being a $(0,2)$-tensor.

**Existence.** Every smooth manifold admits a Riemannian metric. The proof uses partitions of unity: choose any atlas $\{(U_\alpha, \varphi_\alpha)\}$ and pull back the Euclidean metric via each chart to get a metric $g_\alpha$ on $U_\alpha$, then set $g = \sum_\alpha \rho_\alpha g_\alpha$ where $\{\rho_\alpha\}$ is a partition of unity subordinate to the cover. The sum is a positive definite bilinear form because a positive combination of positive definite forms is positive definite, and $\sum_\alpha \rho_\alpha = 1$ with $\rho_\alpha \geq 0$ and at least one $\rho_\alpha(p) > 0$ at each $p$.

The metric is far from unique: the space of Riemannian metrics on a manifold is an infinite-dimensional space. Much of Riemannian geometry studies which properties of $(M, g)$ are intrinsic to the metric — determined by the metric alone — versus which depend on the particular embedding of $M$ in some ambient Euclidean space. Gauss's *theorema egregium* (a special case of the broader curvature theory) established that curvature is intrinsic: two surfaces that are isometric have the same curvature at corresponding points, even if one looks curved and the other flat from the outside.

## Fundamental Examples

The **Euclidean metric** on $\mathbb{R}^n$ is $g = \sum_i (dx^i)^2 = \delta_{ij}\, dx^i\, dx^j$. The metric coefficients are $g_{ij} = \delta_{ij}$ in Cartesian coordinates. This is the flat prototype.

The **round metric on $S^n$** is induced by the Euclidean metric on $\mathbb{R}^{n+1}$: for $p \in S^n \subset \mathbb{R}^{n+1}$ and tangent vectors $u, v \in T_pS^n \subset \mathbb{R}^{n+1}$, set $g_p(u,v) = u \cdot v$ (the Euclidean dot product). In spherical coordinates $(\theta, \phi)$ on $S^2$, this gives $g = d\theta^2 + \sin^2\!\theta\, d\phi^2$, confirming that the metric is not flat: the coefficient of $d\phi^2$ degenerates at the poles.

The **hyperbolic metric** on the upper half-plane $\mathbb{H}^2 = \{(x, y) \in \mathbb{R}^2 : y > 0\}$ is $g = (dx^2 + dy^2)/y^2$. This gives $\mathbb{H}^2$ constant sectional curvature $-1$, the model space for negatively curved geometry. The same metric arises on the Poincaré disk $D^2 = \{z \in \mathbb{C} : |z| < 1\}$ via a conformal transformation.

**Product metrics.** If $(M, g_M)$ and $(N, g_N)$ are Riemannian manifolds, the **product metric** on $M \times N$ is $g_{M \times N} = \pi_M^* g_M + \pi_N^* g_N$, where $\pi_M, \pi_N$ are the projections. Geometrically, the metric at $(p, q)$ measures vectors by splitting them into their $M$ and $N$ components and measuring each separately.

## Lengths, Angles, and the Riemannian Distance

The metric assigns a length to every tangent vector: $|v|_g = \sqrt{g_p(v,v)}$ for $v \in T_pM$. From this, lengths of curves and a distance function on $M$ follow.

> [!definition] 4.2 — Length of a Curve and Riemannian Distance
> The **length** of a piecewise smooth curve $\gamma: [a, b] \to M$ is
>
> $$L(\gamma) = \int_a^b |\gamma'(t)|_g\, dt = \int_a^b \sqrt{g_{\gamma(t)}(\gamma'(t), \gamma'(t))}\, dt.$$
>
> The **Riemannian distance** between two points $p, q \in M$ is
>
> $$d(p, q) = \inf_\gamma L(\gamma),$$
>
> where the infimum is over all piecewise smooth curves from $p$ to $q$. This turns $(M, d)$ into a metric space, and the metric space topology coincides with the manifold topology.


Length is reparametrization-invariant: it depends only on the image of $\gamma$, not on how fast $\gamma$ traverses it. A curve is **unit-speed** (or **arc-length parametrized**) if $|\gamma'(t)|_g = 1$ for all $t$, so that $L(\gamma|_{[a,b]}) = b - a$. Any regular curve can be reparametrized to unit speed by defining the arc-length function $s(t) = \int_a^t |\gamma'(\tau)|_g\, d\tau$ and composing with its inverse.

## Geodesics

Among all curves connecting two points, which are "straightest" — the intrinsic analogues of straight lines? In Euclidean space, straight lines are both length-minimizing and the curves with zero acceleration. On a Riemannian manifold, these two characterizations diverge: locally length-minimizing curves (called **minimizing geodesics**) exist between nearby points, but globally distinct geodesics can connect the same pair of points without either being globally minimizing. The notion of "zero acceleration" requires a connection — the subject of Chapter 5 — but the variational characterization of geodesics as critical points of the length functional is already accessible.

> [!definition] 4.3 — Geodesic
> A smooth curve $\gamma: I \to M$ is a **geodesic** if it is locally length-minimizing and unit-speed, or equivalently if its arc-length parametrized version satisfies the **geodesic equation**
>
> $$\frac{d^2 \gamma^k}{dt^2} + \sum_{i,j} \Gamma^k_{ij}(\gamma(t))\, \frac{d\gamma^i}{dt}\, \frac{d\gamma^j}{dt} = 0, \quad k = 1, \ldots, n,$$
>
> where $\Gamma^k_{ij}$ are the **Christoffel symbols** of the metric:
>
> $$\Gamma^k_{ij} = \frac{1}{2} \sum_\ell g^{k\ell} \left(\frac{\partial g_{i\ell}}{\partial x^j} + \frac{\partial g_{j\ell}}{\partial x^i} - \frac{\partial g_{ij}}{\partial x^\ell}\right).$$
>
> Here $(g^{k\ell})$ denotes the inverse of the matrix $(g_{ij})$.


The geodesic equation is a second-order ODE. By the Picard–Lindelöf theorem, for any point $p \in M$ and any tangent vector $v \in T_pM$, there exists a unique geodesic $\gamma_v: (-\varepsilon, \varepsilon) \to M$ with $\gamma_v(0) = p$ and $\gamma_v'(0) = v$.

The Christoffel symbols encode the metric's "directional variation" — how the inner product changes from point to point. They are not the components of a tensor (they transform with an extra term under coordinate changes) but are the local coordinates of a geometric object — the Levi-Civita connection — that we will define properly in Chapter 5.

In Euclidean space, $g_{ij} = \delta_{ij}$ and all $\Gamma^k_{ij} = 0$, so geodesics satisfy $\gamma'' = 0$: they are straight lines. On $S^2$ with the round metric, geodesics are great circles — the intersection of the sphere with planes through the origin. On $\mathbb{H}^2$ with the hyperbolic metric, geodesics are vertical lines and semicircles orthogonal to the $x$-axis.

> [!remark]
> The geodesic equation can be derived by computing the Euler–Lagrange equations for the energy functional $E(\gamma) = \frac{1}{2}\int_a^b g(\gamma', \gamma')\, dt$. Minimizing energy is equivalent to minimizing length among unit-speed curves, but is technically more convenient because the integrand is smooth (without the square root). The Euler–Lagrange equations for $E$ are precisely the geodesic equation above.


## The Exponential Map

The existence and uniqueness of geodesics with prescribed initial data defines a central object: the exponential map.

> [!definition] 4.4 — Exponential Map
> For $p \in M$, let $\mathcal{D}_p \subseteq T_pM$ be the set of vectors $v$ for which the geodesic $\gamma_v$ is defined at least on $[0, 1]$. The **exponential map at $p$** is
>
> $$\exp_p: \mathcal{D}_p \to M, \qquad \exp_p(v) = \gamma_v(1).$$
>
> The domain $\mathcal{D}_p$ is star-shaped about the origin in $T_pM$: if $v \in \mathcal{D}_p$ then $tv \in \mathcal{D}_p$ for all $t \in [0, 1]$.


The exponential map converts directions and distances at $p$ (encoded as tangent vectors, with $|v|_g$ as the distance) into points of $M$ reachable from $p$ along geodesics. Key properties:

- $\exp_p(0) = p$ (the zero vector maps to the base point)
- $\exp_p(tv) = \gamma_v(t)$ for $t \in [0,1]$ (scaling the vector scales how far along the geodesic you go)
- $d(\exp_p)_0 = \mathrm{id}_{T_pM}$ (the differential at the origin is the identity)

The last property — proved by differentiating $\exp_p(tv)$ with respect to $t$ at $t=0$ — implies by the inverse function theorem that $\exp_p$ is a local diffeomorphism near $0 \in T_pM$. There exists a radius $r > 0$ such that $\exp_p$ restricts to a diffeomorphism on the open ball $B_r(0) \subseteq T_pM$.

> [!definition] 4.5 — Normal Coordinates and Injectivity Radius
> The diffeomorphism $\exp_p: B_r(0) \to \exp_p(B_r(0))$ defines a chart on $M$ called **normal coordinates** centered at $p$. In normal coordinates, the metric satisfies $g_{ij}(p) = \delta_{ij}$ and $\partial g_{ij}/\partial x^k|_p = 0$ — the metric is flat to first order at $p$, and all Christoffel symbols vanish at $p$.
>
> The **injectivity radius** at $p$, denoted $\mathrm{inj}(p)$, is the largest $r$ for which $\exp_p|_{B_r(0)}$ is injective. The global injectivity radius of $(M, g)$ is $\mathrm{inj}(M) = \inf_{p \in M} \mathrm{inj}(p)$.


Normal coordinates are extraordinarily useful: any computation at $p$ can be done in coordinates where the metric looks Euclidean to first order, eliminating first-derivative terms in the metric and all Christoffel symbols at $p$. This is the Riemannian analogue of choosing an inertial frame in general relativity.

The injectivity radius measures "how curved" the manifold is globally. For $\mathbb{R}^n$, the injectivity radius is $+\infty$ — geodesics (straight lines) never cross. For $S^n$ of radius $R$, the injectivity radius is $\pi R$ — two geodesics from a point first meet at the antipodal point. For a very pinched surface, the injectivity radius can be arbitrarily small.

## Completeness and the Hopf-Rinow Theorem

The question of whether geodesics can be extended indefinitely distinguishes compact manifolds and those with "edges."

> [!definition] 4.6 — Geodesic Completeness
> A Riemannian manifold $(M, g)$ is **geodesically complete** if every geodesic $\gamma: [0, a) \to M$ can be extended to a geodesic $\gamma: [0, \infty) \to M$. Equivalently, the exponential map $\exp_p$ is defined on all of $T_pM$ for every $p$.


> [!theorem] 4.1 — Hopf-Rinow Theorem
> For a connected Riemannian manifold $(M, g)$, the following are equivalent:
>
> 1. $(M, g)$ is geodesically complete.
> 2. $(M, d)$ is a complete metric space (every Cauchy sequence converges).
> 3. Every closed bounded subset of $M$ is compact.
>
> Moreover, any of these conditions implies: for every pair of points $p, q \in M$, there exists a minimizing geodesic from $p$ to $q$ — a geodesic $\gamma$ with $L(\gamma) = d(p,q)$.


The Hopf-Rinow theorem is the Riemannian analogue of the Heine-Borel theorem. It says that geodesic completeness (a differential-geometric condition), metric completeness (a topological condition), and the existence of minimizing geodesics (a variational condition) are all equivalent. Every compact Riemannian manifold is complete, and every complete simply connected Riemannian manifold of non-positive curvature is diffeomorphic to $\mathbb{R}^n$ (the Cartan-Hadamard theorem).

## The Volume Form

A Riemannian metric on an oriented manifold canonically determines a volume form.

> [!definition] 4.7 — Riemannian Volume Form
> Let $(M, g)$ be an oriented Riemannian manifold of dimension $n$. The **Riemannian volume form** is the unique positively oriented $n$-form $\mathrm{vol}_g$ such that $\mathrm{vol}_g(e_1, \ldots, e_n) = 1$ for every positively oriented orthonormal basis $(e_1, \ldots, e_n)$ of $T_pM$.
>
> In local positively oriented coordinates $(x^1, \ldots, x^n)$,
>
> $$\mathrm{vol}_g = \sqrt{\det(g_{ij})}\, dx^1 \wedge \cdots \wedge dx^n.$$


The factor $\sqrt{\det(g_{ij})}$ is the Jacobian of the change-of-basis from the coordinate frame to an orthonormal frame at each point. The Riemannian volume of a region $U \subseteq M$ is $\mathrm{Vol}(U) = \int_U \mathrm{vol}_g$, which by Stokes' theorem and the theory of Chapter 3 is well-defined and independent of coordinates.

For $\mathbb{R}^n$ with the Euclidean metric, $\sqrt{\det(\delta_{ij})} = 1$ and $\mathrm{vol}_g = dx^1 \wedge \cdots \wedge dx^n$ — the ordinary Lebesgue measure. For $S^2$ with the round metric in spherical coordinates, $\sqrt{\det g} = \sin\theta$, so $\mathrm{vol}_{S^2} = \sin\theta\, d\theta \wedge d\phi$ — the standard area element.

## Isometries and the Symmetry Group

The natural automorphisms of a Riemannian manifold are the isometries.

> [!definition] 4.8 — Isometry Group and Killing Fields
> The **isometry group** $\mathrm{Isom}(M, g)$ is the group of all smooth isometries $F: M \to M$. It is always a Lie group (Myers-Steenrod theorem).
>
> A vector field $X \in \mathfrak{X}(M)$ is a **Killing field** if the flow $\{\theta_t\}$ of $X$ consists of isometries — equivalently, if $\mathcal{L}_X g = 0$. In local coordinates, this is the **Killing equation**:
>
> $$\nabla_i X_j + \nabla_j X_i = 0,$$
>
> where $\nabla$ denotes the covariant derivative (defined in Chapter 5) and $X_j = g_{jk} X^k$.


Killing fields are the infinitesimal generators of symmetries. For $\mathbb{R}^n$ with the Euclidean metric, the Killing fields are the constant vector fields (infinitesimal translations) and the skew-symmetric combinations $x^i \partial/\partial x^j - x^j \partial/\partial x^i$ (infinitesimal rotations), reflecting the full isometry group $\mathrm{Isom}(\mathbb{R}^n) = \mathbb{R}^n \rtimes O(n)$ of translations and rotations/reflections.

For $S^n$ with the round metric, the isometry group is $O(n+1)$ — all orthogonal transformations of the ambient $\mathbb{R}^{n+1}$. This is the maximal possible: no manifold of dimension $n$ can have an isometry group of dimension greater than $n(n+1)/2$ (the dimension of $O(n+1)$ acting on $S^n$), and spaces achieving this maximum are called **space forms**.

## Tensors on Riemannian Manifolds

The metric provides two canonical isomorphisms between the tangent and cotangent bundles, called **musical isomorphisms** after the notation.

> [!definition] 4.9 — Musical Isomorphisms
> The metric $g$ defines bundle isomorphisms:
>
> - **Flat** ($\flat$): $T_pM \to T_p^*M$, given by $v^\flat(w) = g_p(v, w)$. In coordinates: $(v^\flat)_i = g_{ij} v^j$.
> - **Sharp** ($\sharp$): $T_p^*M \to T_pM$, the inverse of $\flat$. In coordinates: $(\xi^\sharp)^i = g^{ij} \xi_j$.
>
> These isomorphisms extend to arbitrary tensor bundles, allowing indices to be raised and lowered with the metric.


The musical isomorphisms mean that on a Riemannian manifold, there is no fundamental distinction between tangent and cotangent vectors — the metric converts freely between them. The **gradient** of a smooth function $f$ is $\mathrm{grad}\, f = (df)^\sharp$ — the tangent vector corresponding to the covector $df$ under $\sharp$. In coordinates, $(\mathrm{grad}\, f)^i = g^{ij} \partial f / \partial x^j$. In Euclidean coordinates, this is the ordinary gradient; in other coordinates, the $g^{ij}$ factors correct for the non-orthonormality of the coordinate frame.

The **Laplace-Beltrami operator** $\Delta: C^\infty(M) \to C^\infty(M)$ is defined as $\Delta f = \mathrm{div}(\mathrm{grad}\, f)$, where the divergence is computed using the volume form. In local coordinates:

$$\Delta f = \frac{1}{\sqrt{\det g}} \sum_{i,j} \frac{\partial}{\partial x^i}\!\left(\sqrt{\det g}\; g^{ij} \frac{\partial f}{\partial x^j}\right).$$

This is the canonical second-order elliptic operator on any Riemannian manifold, generalizing the ordinary Laplacian $\sum_i \partial^2 f / \partial (x^i)^2$ on $\mathbb{R}^n$.

## What the Metric Does Not Yet Provide

The Riemannian metric gives us lengths, angles, volumes, distances, and geodesics. But one fundamental operation is still missing: a way to differentiate vector fields. On $\mathbb{R}^n$, differentiating a vector field $X$ in the direction $v$ is straightforward — take the componentwise directional derivative $D_v X = \sum_i v(X^i) \partial/\partial x^i$. On a manifold, this formula is coordinate-dependent: the result in one chart does not transform as a vector in another chart, because the basis vectors $\partial/\partial x^i$ themselves change from point to point.

The Riemannian metric, it turns out, does determine a canonical way to differentiate vector fields: the **Levi-Civita connection**, the unique connection that is compatible with the metric and torsion-free. This connection is defined by the Christoffel symbols $\Gamma^k_{ij}$ that already appeared in the geodesic equation — they encode, in coordinates, the "correction" needed when differentiating a vector field on a curved space. Defining the connection intrinsically, characterizing it axiomatically, and exploring what it means to "parallel transport" a vector around a loop — carrying it along while keeping it as constant as the manifold allows — is the subject of the next chapter. And it is the connection that will finally allow us to define curvature.

---

*The foundational reference for Riemannian geometry is do Carmo's* Riemannian Geometry *(1992), rigorous and concise. Lee's* Introduction to Riemannian Manifolds *(2nd ed., 2018) covers the same ground with more detail. The Hopf-Rinow theorem is proved in both. For the role of the exponential map in Lie groups and symmetric spaces, see Helgason's* Differential Geometry, Lie Groups, and Symmetric Spaces *(1978).*
