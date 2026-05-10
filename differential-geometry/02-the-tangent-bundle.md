---

title: "The Tangent Bundle"
subtitle: "Tangent vectors as derivations, pushforward and pullback, vector fields, and the flows they generate"
---



The notion of a tangent vector on a curve in $\mathbb{R}^n$ is elementary: the derivative $\gamma'(t)$ at a point. But on an abstract manifold — a space with no ambient Euclidean structure, no global coordinates, no way to subtract two points — the phrase "tangent vector at $p$" requires a definition. What exactly is a direction at a point on a sphere, when the sphere is not sitting inside $\mathbb{R}^3$ but is simply given by charts and transition maps?

There are three equivalent answers, each revealing something different about the structure. The most geometric: a tangent vector is an equivalence class of curves through $p$, two curves being equivalent if they have the same velocity in any chart. The most coordinate-heavy: a tangent vector is a collection of $n$ numbers, one per coordinate, that transform according to the Jacobian when the chart changes. The most algebraically clean, and the one we adopt: a tangent vector is a *derivation* — a linear operator on smooth functions satisfying the Leibniz rule. The derivation definition is the right one for abstract manifolds because it refers only to the smooth structure, not to any embedding or coordinate system.

## Tangent Vectors as Derivations

The key observation is that in $\mathbb{R}^n$, the directional derivative in direction $v$ at point $p$ acts on smooth functions by

$$D_v f = \sum_{i=1}^n v^i \frac{\partial f}{\partial x^i}\bigg|_p.$$

This operator is linear in $f$ and satisfies the product rule: $D_v(fg) = f(p) D_v g + g(p) D_v f$. The direction $v$ is entirely encoded in the operator $D_v$ — different vectors give different operators. We take this as the definition.

> [!definition] 2.1 — Tangent Vector and Tangent Space
> Let $M$ be a smooth manifold and $p \in M$. A **tangent vector at $p$** is a linear map $v: C^\infty(M) \to \mathbb{R}$ satisfying the **Leibniz rule**:
>
> $$v(fg) = f(p)\, v(g) + g(p)\, v(f) \quad \text{for all } f, g \in C^\infty(M).$$
>
> Such a map is called a **derivation at $p$**. The set of all derivations at $p$ is the **tangent space** $T_pM$, which is a vector space under the operations
>
> $$(v + w)(f) = v(f) + w(f), \qquad (cv)(f) = c\, v(f).$$


The definition is intrinsic: it depends only on the ring $C^\infty(M)$ of smooth functions, which is determined by the smooth structure. No chart, no embedding, no ambient space is needed.

Two consequences follow immediately from the Leibniz rule. First, if $f$ is constant, then $v(f) = 0$ for every tangent vector $v$ — derivations annihilate constants. (Proof: $v(1) = v(1 \cdot 1) = 1 \cdot v(1) + 1 \cdot v(1) = 2v(1)$, so $v(1) = 0$; linearity handles all constants.) Second, if $f(p) = g(p) = 0$, then $v(fg) = 0$ — the product of two functions vanishing at $p$ has zero derivative there.

To make the definition concrete, choose a chart $(U, \varphi)$ with $p \in U$ and coordinate functions $x^1, \ldots, x^n$. For each $i$, define the operator

$$\frac{\partial}{\partial x^i}\bigg|_p(f) = \frac{\partial (f \circ \varphi^{-1})}{\partial r^i}\bigg|_{\varphi(p)},$$

where $r^1, \ldots, r^n$ are the standard coordinates on $\mathbb{R}^n$. This is a derivation at $p$ — it satisfies linearity and the Leibniz rule because the ordinary partial derivative does. The operators $\partial/\partial x^i|_p$ are the **coordinate basis vectors** associated with the chart.

> [!theorem] 2.1 — Tangent Space is n-Dimensional
> Let $M$ be a smooth $n$-manifold and $p \in M$. For any chart $(U, \varphi)$ around $p$ with coordinate functions $x^1, \ldots, x^n$, the coordinate vectors $\partial/\partial x^1|_p, \ldots, \partial/\partial x^n|_p$ form a basis for $T_pM$. In particular, $\dim T_pM = n$.


The proof uses bump functions: one constructs, for each $v \in T_pM$, smooth functions $x^i$ that locally look like coordinate projections, and shows that $v = \sum_i v(x^i) \, \partial/\partial x^i|_p$ by checking both sides agree on every smooth function (using a Taylor expansion in the chart). The key step is that any smooth function $f$ can be written near $p$ as $f = f(p) + \sum_i (x^i - x^i(p)) g_i$ for smooth functions $g_i$ with $g_i(p) = \partial f/\partial x^i|_p$, and derivations annihilate constants and quadratic-order terms.

The components $v^i = v(x^i)$ are the coordinates of $v$ in the basis $\{\partial/\partial x^i|_p\}$. Under a change of chart from $(x^1, \ldots, x^n)$ to $(\tilde{x}^1, \ldots, \tilde{x}^n)$, the chain rule gives

$$\frac{\partial}{\partial x^i}\bigg|_p = \sum_j \frac{\partial \tilde{x}^j}{\partial x^i}\bigg|_p \frac{\partial}{\partial \tilde{x}^j}\bigg|_p.$$

The coordinates of the *same* tangent vector transform by the Jacobian matrix of the transition map — this is the "contravariant" transformation rule. The derivation definition makes this transformation automatic rather than axiomatic.

## The Differential of a Smooth Map

One of the central operations in calculus is the derivative of a map: the best linear approximation to the map near a point. For smooth maps between manifolds, this is the **differential** (also called the **pushforward**).

> [!definition] 2.2 — Differential (Pushforward)
> Let $F: M \to N$ be a smooth map and $p \in M$. The **differential of $F$ at $p$** is the linear map
>
> $$dF_p: T_pM \to T_{F(p)}N$$
>
> defined by
>
> $$(dF_p\, v)(f) = v(f \circ F) \quad \text{for all } f \in C^\infty(N),\, v \in T_pM.$$


In words: to push a tangent vector $v$ at $p$ forward to $T_{F(p)}N$, act with $v$ on functions pulled back through $F$. The chain rule is built into the definition: if $G: N \to P$ is another smooth map, then $d(G \circ F)_p = dG_{F(p)} \circ dF_p$.

In local coordinates, if $(x^i)$ are coordinates near $p$ and $(y^\alpha)$ are coordinates near $F(p)$, then $F$ is represented by $n$ functions $y^\alpha = F^\alpha(x^1, \ldots, x^n)$, and

$$dF_p\left(\frac{\partial}{\partial x^i}\bigg|_p\right) = \sum_\alpha \frac{\partial F^\alpha}{\partial x^i}\bigg|_p \frac{\partial}{\partial y^\alpha}\bigg|_{F(p)}.$$

The matrix of $dF_p$ in the coordinate bases is the Jacobian matrix $(\partial F^\alpha / \partial x^i)$. The differential is thus the coordinate-free version of the Jacobian: a linear map between tangent spaces that encodes the first-order behavior of $F$ near $p$.

The rank of $dF_p$ classifies smooth maps:

- $F$ is an **immersion at $p$** if $dF_p$ is injective ($\operatorname{rank} = \dim M$)
- $F$ is a **submersion at $p$** if $dF_p$ is surjective ($\operatorname{rank} = \dim N$)
- $F$ is a **local diffeomorphism at $p$** if $dF_p$ is an isomorphism ($\dim M = \dim N$, rank maximal)

The inverse function theorem from Chapter 1 can now be restated cleanly: $F$ is a local diffeomorphism near $p$ if and only if $dF_p$ is an isomorphism.

> [!remark]
> The **pullback** of a smooth function $f \in C^\infty(N)$ through $F$ is $F^*f = f \circ F \in C^\infty(M)$. The differential satisfies $(dF_p\, v)(f) = v(F^*f)$ — pushing a vector forward through $F$ is dual to pulling a function back. This duality between pushforward on tangent vectors and pullback on functions is the prototype for a recurring theme: tangent-type objects (vectors) push forward; cotangent-type objects (forms) pull back. We return to this in Chapter 3.


## The Tangent Bundle

The collection of all tangent spaces assembled into a single smooth manifold is the tangent bundle.

> [!definition] 2.3 — Tangent Bundle
> The **tangent bundle** of a smooth $n$-manifold $M$ is
>
> $$TM = \bigsqcup_{p \in M} T_pM = \{(p, v) : p \in M,\, v \in T_pM\}.$$
>
> It comes equipped with the **projection** $\pi: TM \to M$ defined by $\pi(p, v) = p$. The fiber $\pi^{-1}(p) = T_pM$ is the tangent space at $p$.


The tangent bundle is itself a smooth $2n$-manifold. Given a chart $(U, \varphi)$ on $M$ with coordinates $x^1, \ldots, x^n$, a point $(p, v) \in \pi^{-1}(U) \subseteq TM$ has the representation $v = \sum_i v^i\, \partial/\partial x^i|_p$, giving a chart on $TM$ by

$$\tilde{\varphi}(p, v) = (x^1(p), \ldots, x^n(p),\, v^1, \ldots, v^n) \in \mathbb{R}^{2n}.$$

The transition maps for these charts on $TM$ are smooth because they involve the Jacobian of the transition maps of $M$, which are smooth. The tangent bundle is thus a smooth manifold of dimension $2n$.

The tangent bundle is the simplest example of a **vector bundle**: a smooth manifold $E$ with a projection $\pi: E \to M$ such that each fiber $\pi^{-1}(p)$ is a vector space, varying smoothly from point to point. Much of differential geometry can be organized as the study of vector bundles over manifolds — the cotangent bundle, tensor bundles, the frame bundle. We will see these structures again in Chapters 3 and 8.

## Vector Fields

A smooth choice of tangent vector at each point of $M$ is a vector field.

> [!definition] 2.4 — Vector Field
> A **vector field** on $M$ is a smooth map $X: M \to TM$ such that $\pi \circ X = \mathrm{id}_M$ — that is, $X(p) \in T_pM$ for every $p$. Equivalently, $X$ is a smooth **section** of the tangent bundle.
>
> In a local chart $(U, x^i)$, a vector field $X$ on $U$ is written
>
> $$X = \sum_{i=1}^n X^i \frac{\partial}{\partial x^i},$$
>
> where the **component functions** $X^i: U \to \mathbb{R}$ are smooth.


The set of all smooth vector fields on $M$, denoted $\mathfrak{X}(M)$, is a module over $C^\infty(M)$: one can add vector fields and multiply them by smooth functions. The action of a vector field $X$ on a smooth function $f$ gives another smooth function:

$$(Xf)(p) = X_p(f) \in \mathbb{R}, \quad \text{so } Xf \in C^\infty(M).$$

This makes vector fields first-order differential operators on $C^\infty(M)$.

## Integral Curves and Flows

Every vector field generates a family of curves — its integral curves — obtained by following the direction the field points at each moment.

> [!definition] 2.5 — Integral Curve
> An **integral curve** of a vector field $X$ through a point $p \in M$ is a smooth curve $\gamma: (-\varepsilon, \varepsilon) \to M$ satisfying
>
> $$\gamma(0) = p, \qquad \gamma'(t) = X_{\gamma(t)} \quad \text{for all } t \in (-\varepsilon, \varepsilon).$$


In local coordinates, the condition $\gamma'(t) = X_{\gamma(t)}$ becomes the ODE system

$$\frac{d\gamma^i}{dt}(t) = X^i(\gamma^1(t), \ldots, \gamma^n(t)), \quad i = 1, \ldots, n,$$

with initial condition $\gamma^i(0) = x^i(p)$. The Picard–Lindelöf theorem guarantees existence and uniqueness of solutions for short times, given that $X$ is smooth (in particular, Lipschitz in coordinates). The solution may fail to exist for all time — think of a vector field on $\mathbb{R}$ given by $X = x^2 \, \partial/\partial x$, whose integral curve through $x_0 = 1$ blows up at $t = 1$. When integral curves exist for all time, $X$ is called **complete**.

Assembling the integral curves parametrized by their starting points gives the **flow** of the vector field.

> [!definition] 2.6 — Flow of a Vector Field
> The **flow** of a vector field $X$ is the smooth map
>
> $$\theta: \mathcal{D} \subseteq \mathbb{R} \times M \to M,$$
>
> where $\mathcal{D}$ is an open neighborhood of $\{0\} \times M$, defined by $\theta(t, p) = \gamma_p(t)$, the integral curve of $X$ through $p$ at time $t$.
>
> For fixed $t$, we write $\theta_t: M \to M$ for the map $p \mapsto \theta(t, p)$. Each $\theta_t$ (where it is defined) is a diffeomorphism. The family $\{\theta_t\}$ satisfies the **group law**:
>
> $$\theta_0 = \mathrm{id}_M, \qquad \theta_{s+t} = \theta_s \circ \theta_t.$$


The group law says that flowing for time $s$ and then time $t$ is the same as flowing for time $s + t$. This makes $\{\theta_t\}$ a one-parameter group of diffeomorphisms. Vector fields are infinitesimal generators of such groups — a theme that will become central when we study Lie groups in Chapter 7.

The flow has a compelling interpretation: given a vector field $X$ on $M$, think of $M$ as a fluid in motion, with $X_p$ giving the velocity of the fluid particle at $p$. Then $\theta_t(p)$ is where the particle starting at $p$ ends up after time $t$. The integral curves are the trajectories of individual particles; the flow maps $\theta_t$ describe the entire fluid at each moment.

## The Lie Bracket

Given two vector fields $X$ and $Y$ on $M$, one might hope their composition $X \circ Y$ — first apply $Y$, then $X$ — is again a vector field. It is not: $X(Yf)$ involves second derivatives of $f$, and the second-order terms do not satisfy the Leibniz rule. However, the **commutator** $XY - YX$ does:

> [!definition] 2.7 — Lie Bracket
> The **Lie bracket** of vector fields $X, Y \in \mathfrak{X}(M)$ is the vector field $[X, Y]$ defined by
>
> $$[X, Y](f) = X(Yf) - Y(Xf) \quad \text{for all } f \in C^\infty(M).$$


That $[X, Y]$ is indeed a vector field (a derivation, not a second-order operator) follows from a direct computation: the second-derivative terms in $X(Yf)$ and $Y(Xf)$ cancel, leaving a first-order operator. In local coordinates,

$$[X, Y] = \sum_{i,j} \left( X^j \frac{\partial Y^i}{\partial x^j} - Y^j \frac{\partial X^i}{\partial x^j} \right) \frac{\partial}{\partial x^i}.$$

The Lie bracket is the infinitesimal measure of how much the flows of $X$ and $Y$ fail to commute. To see this precisely, consider the composition of flows:

$$\theta^X_{\sqrt{t}} \circ \theta^Y_{\sqrt{t}} \circ \theta^X_{-\sqrt{t}} \circ \theta^Y_{-\sqrt{t}}(p).$$

This "commutator of flows" — go along $X$ for $\sqrt{t}$, then $Y$ for $\sqrt{t}$, then back along $X$, then back along $Y$ — measures how far you end up from where you started. A Taylor expansion shows this displacement is $t\, [X,Y]_p + O(t^{3/2})$: to first order in $t$, the failure of the flows to commute is exactly the Lie bracket.

> [!theorem] 2.2 — Properties of the Lie Bracket
> The Lie bracket $[\cdot, \cdot]: \mathfrak{X}(M) \times \mathfrak{X}(M) \to \mathfrak{X}(M)$ satisfies:
>
> 1. **Bilinearity**: $[aX + bY, Z] = a[X, Z] + b[Y, Z]$ for $a, b \in \mathbb{R}$
> 2. **Antisymmetry**: $[X, Y] = -[Y, X]$
> 3. **Jacobi identity**: $[X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0$
> 4. **Leibniz rule for $C^\infty(M)$**: $[fX, gY] = fg[X,Y] + f(Xg)Y - g(Yf)X$
>
> The first three properties make $\mathfrak{X}(M)$ a **Lie algebra** — an infinite-dimensional one.


The Jacobi identity is not a theorem to be proved so much as a constraint to be verified: it holds because both sides, when applied to $f$, reduce to the same expression by expansion. Its geometric meaning is subtle: it says the Lie bracket respects a certain three-body consistency, ensuring that the "failure to commute" of three flows is itself internally consistent.

The Lie bracket of the coordinate fields $\partial/\partial x^i$ vanishes: $[\partial/\partial x^i, \partial/\partial x^j] = 0$ for all $i, j$. This is the integrability condition behind the following important fact: $n$ vector fields $X_1, \ldots, X_n$ on an $n$-manifold can simultaneously be used as a coordinate basis (i.e., there exists a chart in which $X_i = \partial/\partial x^i$) if and only if their pairwise Lie brackets all vanish. Non-vanishing Lie brackets are thus a measure of the "non-flatness" of the frame — a precursor to the notion of curvature we develop in Chapter 6.

## Pushforward of Vector Fields

The differential $dF_p$ pushes a single tangent vector at $p$ to a tangent vector at $F(p)$. For a *vector field* on $M$ — a choice of tangent vector at every point — the pushforward requires more care: there is no guarantee that $F$ is a bijection, so the image of a vector field under $F$ need not be well-defined everywhere on $N$.

> [!definition] 2.8 — F-Related Vector Fields
> Let $F: M \to N$ be a smooth map. Vector fields $X \in \mathfrak{X}(M)$ and $Y \in \mathfrak{X}(N)$ are **$F$-related** if
>
> $$dF_p(X_p) = Y_{F(p)} \quad \text{for all } p \in M.$$
>
> When $F$ is a diffeomorphism, every vector field $X$ on $M$ has a unique **pushforward** $F_*X \in \mathfrak{X}(N)$ defined by $(F_*X)_q = dF_{F^{-1}(q)}(X_{F^{-1}(q)})$.


The Lie bracket is natural with respect to $F$-related fields: if $X_1$ is $F$-related to $Y_1$ and $X_2$ is $F$-related to $Y_2$, then $[X_1, X_2]$ is $F$-related to $[Y_1, Y_2]$. Equivalently, for diffeomorphisms: $F_*[X_1, X_2] = [F_*X_1, F_*X_2]$. The Lie bracket is a diffeomorphism-invariant operation.

## The Tangent Bundle as the Infinitesimal Picture

At this point it is worth stepping back to see the full picture the tangent bundle provides. A smooth manifold $M$ is a topological space that locally looks like $\mathbb{R}^n$. The tangent bundle $TM$ makes this local Euclidean structure precise at the infinitesimal level: it assigns to each point $p$ the $n$-dimensional vector space $T_pM$ of all "directions" available at $p$.

The differential of a smooth map $F: M \to N$ is a linear map $dF_p: T_pM \to T_{F(p)}N$ at each point — the best linear approximation to $F$ near $p$. This linearization is one of the most powerful tools in differential geometry: it reduces local questions about smooth maps to linear algebra.

Vector fields are the sections of this bundle: they pick out one tangent vector at each point, smoothly, defining a "flow" that moves every point of $M$ along an ODE. The Lie bracket of two vector fields measures how their flows interact, encoding the non-commutativity of the curved space in a purely algebraic object.

What is still missing is a way to *measure* tangent vectors — to assign a notion of length or angle to the arrows in each $T_pM$. This requires an inner product on each tangent space, varying smoothly across $M$: the Riemannian metric of Chapter 4. Before getting there, we need one more fundamental object: the cotangent bundle and differential forms, which live on the "dual side" of the tangent bundle and provide the right language for integration on manifolds.

---

*The derivation approach to tangent vectors is due to the modern axiomatic treatment of differential geometry; a clean account is in Lee's* Introduction to Smooth Manifolds *(Chapter 3). The flow of a vector field and its relation to ODEs is treated in detail in Spivak's* A Comprehensive Introduction to Differential Geometry *(Vol. 1). The Lie bracket as the infinitesimal commutator of flows is beautifully explained in Arnold's* Ordinary Differential Equations *(Chapter 8).*
