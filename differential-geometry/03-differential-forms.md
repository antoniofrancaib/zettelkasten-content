---

title: "Differential Forms and Integration"
subtitle: "The cotangent bundle, exterior algebra, the exterior derivative, and Stokes' theorem"
---



There are two kinds of objects that live naturally on a manifold. The first — tangent vectors — point *along* the manifold. The second — differential forms — measure *across* it. The distinction is not merely aesthetic. When you integrate over a region, the integrand is not a function but a form: it is the form that knows how to be measured against a patch of the manifold, producing a number regardless of which coordinate chart you use. Integration is, at its core, the pairing of forms against oriented pieces of manifold.

This chapter builds the theory of differential forms from the ground up. We begin with the algebraic structure — the exterior algebra of a vector space — then promote it to the smooth setting of the cotangent bundle. The exterior derivative gives forms a calculus of their own, and Stokes' theorem — the single identity that unifies the fundamental theorem of calculus, Green's theorem, and the divergence theorem — reveals the full power of the framework.

## The Cotangent Space

Chapter 2 defined the tangent space $T_pM$ as the space of derivations at $p$. Its **dual** is the cotangent space.

> [!definition] 3.1 — Cotangent Space and Covectors
> The **cotangent space** at $p \in M$ is the dual vector space
>
> $$T_p^*M = (T_pM)^* = \{\xi: T_pM \to \mathbb{R} \mid \xi \text{ linear}\}.$$
>
> Elements of $T_p^*M$ are called **covectors** or **1-forms at $p$**. The natural pairing between a covector $\xi \in T_p^*M$ and a tangent vector $v \in T_pM$ is written $\langle \xi, v \rangle$ or $\xi(v)$.


Since $T_pM$ is $n$-dimensional, so is $T_p^*M$. Given a chart $(U, x^i)$ around $p$, the coordinate basis $\{\partial/\partial x^i|_p\}$ of $T_pM$ has a dual basis $\{dx^i|_p\}$ of $T_p^*M$, characterized by

$$\left\langle dx^i\big|_p,\, \frac{\partial}{\partial x^j}\bigg|_p \right\rangle = \delta^i_j.$$

Any covector $\xi \in T_p^*M$ can be written $\xi = \sum_i \xi_i\, dx^i|_p$ where $\xi_i = \xi(\partial/\partial x^i|_p)$.

The most natural source of covectors is the differential of a smooth function. For $f \in C^\infty(M)$ and $p \in M$, define

$$df|_p : T_pM \to \mathbb{R}, \qquad df|_p(v) = v(f).$$

This is linear in $v$ by definition of $T_pM$, so $df|_p \in T_p^*M$. In coordinates, $df|_p = \sum_i \frac{\partial f}{\partial x^i}|_p\, dx^i|_p$ — the covector $df$ encodes all directional derivatives of $f$ at $p$. Note that $dx^i|_p = d(x^i)|_p$ in this notation, confirming the dual basis interpretation: the covectors $dx^i$ are the differentials of the coordinate functions.

Under a change of chart from $(x^i)$ to $(\tilde x^j)$, the chain rule gives

$$d\tilde x^j\big|_p = \sum_i \frac{\partial \tilde x^j}{\partial x^i}\bigg|_p\, dx^i\big|_p.$$

Covector components transform by the *transpose* Jacobian — the inverse of how tangent vector components transform. This is the "covariant" transformation rule, explaining the word "covector."

## The Cotangent Bundle and 1-Forms

Assembling the cotangent spaces gives the cotangent bundle.

> [!definition] 3.2 — Cotangent Bundle and Differential 1-Forms
> The **cotangent bundle** of $M$ is
>
> $$T^*M = \bigsqcup_{p \in M} T_p^*M,$$
>
> with the smooth structure defined analogously to $TM$: a chart $(U, x^i)$ on $M$ gives coordinates $(x^1, \ldots, x^n, \xi_1, \ldots, \xi_n)$ on $T^*M|_U$, where $\xi_i$ are the components of a covector in the basis $\{dx^i\}$.
>
> A **differential 1-form** (or just **1-form**) on $M$ is a smooth section of $T^*M$ — a smooth map $\omega: M \to T^*M$ with $\omega_p \in T_p^*M$ for every $p$. The set of all 1-forms is denoted $\Omega^1(M)$.


In local coordinates, a 1-form is written $\omega = \sum_i \omega_i(x)\, dx^i$ where the coefficient functions $\omega_i: U \to \mathbb{R}$ are smooth. The differential $df$ of any smooth function is a 1-form. Not every 1-form arises this way globally — those that do are called **exact** — but every 1-form is locally the differential of some function.

The pairing of a 1-form $\omega$ with a vector field $X$ gives a smooth function: $\omega(X)(p) = \omega_p(X_p) \in \mathbb{R}$. This is the natural duality between $\Omega^1(M)$ and $\mathfrak{X}(M)$.

**Pullback.** A smooth map $F: M \to N$ pulls 1-forms on $N$ back to 1-forms on $M$. For $\omega \in \Omega^1(N)$, define

$$(F^*\omega)_p(v) = \omega_{F(p)}(dF_p\, v) \quad \text{for } v \in T_pM.$$

Pullback is functorial: $(G \circ F)^* = F^* \circ G^*$. It is the natural operation on forms, just as pushforward is the natural operation on vectors. This asymmetry — vectors push forward, forms pull back — will remain throughout.

## The Exterior Algebra

To integrate over $k$-dimensional submanifolds, we need objects that can be evaluated against $k$ tangent vectors simultaneously — and that reverse sign when two of those vectors are swapped, encoding the orientation of the piece being integrated over.

> [!definition] 3.3 — Alternating Multilinear Forms and Exterior Algebra
> Let $V$ be an $n$-dimensional real vector space. A **$k$-covector** on $V$ is a multilinear map $\alpha: V^k \to \mathbb{R}$ that is **alternating**: $\alpha(\ldots, v, \ldots, w, \ldots) = -\alpha(\ldots, w, \ldots, v, \ldots)$ whenever two arguments are swapped. The space of all $k$-covectors on $V$ is denoted $\Lambda^k(V^*)$.
>
> The **wedge product** $\alpha \wedge \beta$ of $\alpha \in \Lambda^k(V^*)$ and $\beta \in \Lambda^\ell(V^*)$ is the $(k+\ell)$-covector
>
> $$(\alpha \wedge \beta)(v_1, \ldots, v_{k+\ell}) = \frac{1}{k!\,\ell!} \sum_{\sigma \in S_{k+\ell}} \mathrm{sgn}(\sigma)\, \alpha(v_{\sigma(1)}, \ldots, v_{\sigma(k)})\, \beta(v_{\sigma(k+1)}, \ldots, v_{\sigma(k+\ell)}).$$


The wedge product is bilinear, associative, and **graded-commutative**: $\alpha \wedge \beta = (-1)^{k\ell}\, \beta \wedge \alpha$. In particular, $\alpha \wedge \alpha = 0$ for any odd-degree form. The direct sum $\Lambda^*(V^*) = \bigoplus_{k=0}^n \Lambda^k(V^*)$ is the **exterior algebra** of $V^*$, with $\Lambda^0(V^*) = \mathbb{R}$ and $\Lambda^k(V^*) = 0$ for $k > n$.

Given a basis $\{e_1, \ldots, e_n\}$ for $V$ with dual basis $\{e^1, \ldots, e^n\}$ for $V^*$, the exterior algebra has basis

$$\{e^{i_1} \wedge \cdots \wedge e^{i_k} : 1 \leq i_1 < i_2 < \cdots < i_k \leq n\}$$

for $\Lambda^k(V^*)$, giving $\dim \Lambda^k(V^*) = \binom{n}{k}$. The top exterior power $\Lambda^n(V^*)$ is 1-dimensional, spanned by $e^1 \wedge \cdots \wedge e^n$. An element of $\Lambda^n(V^*)$ is a **volume form**: it assigns a signed volume to any $n$-tuple of vectors, with sign encoding orientation.

The geometric meaning of the wedge product: $\alpha \wedge \beta$ evaluated on vectors $(v_1, \ldots, v_{k+\ell})$ measures the "signed $(k+\ell)$-dimensional volume" spanned by the projections of those vectors onto the subspaces that $\alpha$ and $\beta$ are sensitive to. For $\xi, \eta \in V^*$, the 2-form $\xi \wedge \eta$ satisfies $(\xi \wedge \eta)(u, v) = \xi(u)\eta(v) - \xi(v)\eta(u)$ — the signed area of the parallelogram with sides $\xi(u), \xi(v)$ and $\eta(u), \eta(v)$, i.e., the $2\times2$ determinant.

## Differential $k$-Forms

Promoting the exterior algebra to the smooth setting:

> [!definition] 3.4 — Differential k-Forms
> A **differential $k$-form** on a smooth manifold $M$ is a smooth section of the bundle $\Lambda^k(T^*M)$ — that is, a smooth assignment $p \mapsto \omega_p \in \Lambda^k(T_p^*M)$.
>
> The space of all smooth $k$-forms on $M$ is denoted $\Omega^k(M)$, with the convention $\Omega^0(M) = C^\infty(M)$.
>
> In local coordinates $(x^1, \ldots, x^n)$, every $k$-form writes as
>
> $$\omega = \sum_{i_1 < \cdots < i_k} \omega_{i_1 \cdots i_k}(x)\, dx^{i_1} \wedge \cdots \wedge dx^{i_k},$$
>
> where the coefficient functions $\omega_{i_1 \cdots i_k}: U \to \mathbb{R}$ are smooth.


**Pullback of $k$-forms.** For $F: M \to N$ smooth and $\omega \in \Omega^k(N)$, the pullback $F^*\omega \in \Omega^k(M)$ is defined pointwise:

$$(F^*\omega)_p(v_1, \ldots, v_k) = \omega_{F(p)}(dF_p\, v_1, \ldots, dF_p\, v_k).$$

Pullback is an algebra homomorphism: $F^*(\omega \wedge \eta) = F^*\omega \wedge F^*\eta$. It commutes with all the operations we define below, making it the canonical way to move forms between manifolds.

## The Exterior Derivative

The exterior derivative is the fundamental differential operator on forms — the unique operator that extends the differential $d: C^\infty(M) \to \Omega^1(M)$ to all degrees while satisfying a graded Leibniz rule and squaring to zero.

> [!theorem] 3.1 — Exterior Derivative
> There exists a unique collection of $\mathbb{R}$-linear maps $d: \Omega^k(M) \to \Omega^{k+1}(M)$, for all $k \geq 0$, satisfying:
>
> 1. **Extension**: for $f \in \Omega^0(M) = C^\infty(M)$, $df$ is the usual differential.
> 2. **Graded Leibniz rule**: $d(\omega \wedge \eta) = d\omega \wedge \eta + (-1)^k \omega \wedge d\eta$ for $\omega \in \Omega^k(M)$.
> 3. **Nilpotency**: $d \circ d = 0$, i.e., $d^2\omega = 0$ for all $\omega$.
>
> In local coordinates, if $\omega = \sum_{I} \omega_I\, dx^I$ (using multi-index notation), then
>
> $$d\omega = \sum_I d\omega_I \wedge dx^I = \sum_I \sum_j \frac{\partial \omega_I}{\partial x^j}\, dx^j \wedge dx^I.$$


The formula in coordinates provides existence; uniqueness follows from the three axioms together with the fact that forms are locally generated by functions and their differentials.

In $\mathbb{R}^3$ with coordinates $(x, y, z)$, the exterior derivative recovers all of classical vector calculus:

- On 0-forms: $df = \frac{\partial f}{\partial x}dx + \frac{\partial f}{\partial y}dy + \frac{\partial f}{\partial z}dz$ — this is the **gradient**
- On 1-forms: $d(P\,dx + Q\,dy + R\,dz) = \left(\frac{\partial R}{\partial y} - \frac{\partial Q}{\partial z}\right)dy\wedge dz + \cdots$ — this is the **curl**
- On 2-forms: $d(P\,dy\wedge dz + Q\,dz\wedge dx + R\,dx\wedge dy) = \left(\frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y} + \frac{\partial R}{\partial z}\right)dx\wedge dy\wedge dz$ — this is the **divergence**

The identity $d^2 = 0$ encodes the classical identities $\mathrm{curl}(\nabla f) = 0$ and $\mathrm{div}(\mathrm{curl}\, F) = 0$ in a single coordinate-free statement.

**Closed and exact forms.** A form $\omega$ is **closed** if $d\omega = 0$, and **exact** if $\omega = d\eta$ for some form $\eta$. Since $d^2 = 0$, every exact form is closed. The converse — is every closed form exact? — is a global topological question answered by de Rham cohomology: $H^k_{\mathrm{dR}}(M) = \ker(d: \Omega^k \to \Omega^{k+1}) / \mathrm{im}(d: \Omega^{k-1} \to \Omega^k)$. The **Poincaré lemma** says closed forms are locally exact: if $d\omega = 0$ on a contractible open set $U$, then $\omega = d\eta$ on $U$. Globally, obstructions to exactness are measured by the topology of $M$.

The exterior derivative is natural with respect to pullback: $F^*(d\omega) = d(F^*\omega)$. This is a key computational identity — one can either differentiate and then pull back, or pull back and then differentiate, with the same result.

## Orientation and Integration

To integrate an $n$-form over an $n$-manifold, we first need the manifold to be oriented.

> [!definition] 3.5 — Orientation
> A smooth $n$-manifold $M$ is **orientable** if it admits a nowhere-vanishing $n$-form $\Omega \in \Omega^n(M)$. A choice of such $\Omega$ (up to multiplication by a positive function) is an **orientation** of $M$.
>
> Equivalently, $M$ is orientable if and only if it admits an atlas whose transition maps all have positive Jacobian determinant. The sphere $S^n$, tori, and Lie groups are all orientable. The Möbius band and $\mathbb{RP}^2$ are not.


On an oriented $n$-manifold $M$, the integral of a compactly supported $n$-form $\omega$ is defined as follows. In a positively oriented chart $(U, \varphi)$ with $\mathrm{supp}(\omega) \subseteq U$, write $\omega = f\, dx^1 \wedge \cdots \wedge dx^n$ and set

$$\int_M \omega = \int_{\varphi(U)} (f \circ \varphi^{-1})(x)\, d^n x,$$

where the right side is the ordinary Lebesgue integral over $\mathbb{R}^n$. For forms not supported in a single chart, use a partition of unity to decompose $\omega$ into pieces, each supported in a chart, and sum. The change-of-variables formula for the Lebesgue integral, applied to the transition maps, guarantees that the result is independent of the choice of charts and partition of unity — the sign-sensitive change-of-variables formula exactly matches the alternating transformation law of the $n$-form.

This is the key point: the alternating property of forms is not a technicality but precisely what makes integration coordinate-independent. An $n$-form automatically picks up the absolute value of the Jacobian determinant when the coordinates change — that is the content of the $n$-form transformation law — and the orientation determines the sign.

## Manifolds with Boundary

Stokes' theorem requires the notion of a manifold with boundary.

> [!definition] 3.6 — Manifold with Boundary
> A smooth **$n$-manifold with boundary** $M$ is defined like a manifold, but allowing charts to map into either $\mathbb{R}^n$ or the **half-space** $\mathbb{H}^n = \{(x^1, \ldots, x^n) : x^n \geq 0\}$.
>
> The **boundary** $\partial M$ is the set of points that map to $\{x^n = 0\}$ in some (equivalently, every) half-space chart. It is a smooth $(n-1)$-manifold without boundary: $\partial(\partial M) = \emptyset$.
>
> If $M$ is oriented, then $\partial M$ carries an induced orientation, called the **outward normal orientation**: at $p \in \partial M$, a basis $(v_1, \ldots, v_{n-1})$ for $T_p(\partial M)$ is positively oriented if and only if $(n, v_1, \ldots, v_{n-1})$ is positively oriented in $T_pM$, where $n$ is an outward-pointing normal.


Examples: a closed interval $[a, b]$ is a 1-manifold with boundary $\{a, b\}$. A closed disk $\overline{D}^2$ is a 2-manifold with boundary $S^1$. A closed ball $\overline{B}^3$ has boundary $S^2$. The sphere $S^n$ has empty boundary: $\partial S^n = \emptyset$.

## Stokes' Theorem

> [!theorem] 3.2 — Stokes' Theorem
> Let $M$ be a smooth oriented $n$-manifold with boundary, and let $\omega \in \Omega^{n-1}(M)$ be a compactly supported smooth $(n-1)$-form. Then
>
> $$\int_M d\omega = \int_{\partial M} \omega,$$
>
> where $\partial M$ carries the induced orientation and the right side is zero if $\partial M = \emptyset$.


The proof in local coordinates reduces to the one-dimensional statement $\int_a^b f'(x)\, dx = f(b) - f(a)$, applied coordinate by coordinate, with the alternating structure of the exterior derivative managing all the sign and orientation bookkeeping.

The breadth of what Stokes' theorem unifies is extraordinary:

- **$n = 1$, $M = [a, b]$**: $\int_a^b df = f(b) - f(a)$ — the fundamental theorem of calculus.
- **$n = 2$, $M \subseteq \mathbb{R}^2$**: with $\omega = P\,dx + Q\,dy$, one gets $\int_M (Q_x - P_y)\, dx\wedge dy = \oint_{\partial M} P\,dx + Q\,dy$ — **Green's theorem**.
- **$n = 3$, $M \subseteq \mathbb{R}^3$ a surface**: with $\omega$ a 1-form corresponding to a vector field $F$, one gets $\int_M (\nabla \times F) \cdot dS = \oint_{\partial M} F \cdot ds$ — the classical **Stokes' theorem**.
- **$n = 3$, $M \subseteq \mathbb{R}^3$ a solid region**: with $\omega$ a 2-form corresponding to a vector field $F$, one gets $\int_M \nabla \cdot F\, dV = \oint_{\partial M} F \cdot dS$ — the **divergence theorem (Gauss' theorem)**.

All four are instances of a single identity: the integral of $d\omega$ over a region equals the integral of $\omega$ over its boundary. The exterior derivative $d$ is the boundary operator in disguise — the identity $d^2 = 0$ is algebraically dual to the geometric fact that $\partial(\partial M) = \emptyset$.

> [!remark]
> Stokes' theorem has a clean algebraic summary. The exterior derivative $d: \Omega^{k-1}(M) \to \Omega^k(M)$ and the boundary operator $\partial$ on chains are **adjoint operators**: $\langle d\omega, C \rangle = \langle \omega, \partial C \rangle$, where $\langle \cdot, \cdot \rangle$ denotes integration of a form over a chain. The de Rham cohomology $H^k_{\mathrm{dR}}(M)$ and the singular homology $H_k(M; \mathbb{R})$ are dual via this pairing — this is the **de Rham theorem**, one of the deepest results in differential topology.


## The Interior Product and Cartan's Magic Formula

There is one more operation on forms that will be indispensable, especially in symplectic geometry (Chapter 9): the interior product, which contracts a form with a vector field.

> [!definition] 3.7 — Interior Product
> For a vector field $X \in \mathfrak{X}(M)$ and a $k$-form $\omega \in \Omega^k(M)$, the **interior product** (or **contraction**) $\iota_X \omega \in \Omega^{k-1}(M)$ is defined by
>
> $$(\iota_X \omega)(v_1, \ldots, v_{k-1}) = \omega(X, v_1, \ldots, v_{k-1}).$$
>
> It satisfies $\iota_X(\omega \wedge \eta) = (\iota_X \omega) \wedge \eta + (-1)^k \omega \wedge (\iota_X \eta)$ for $\omega \in \Omega^k$.


The interior product $\iota_X$ lowers degree by one, while the exterior derivative $d$ raises degree by one. Together they give the **Lie derivative** of a form along $X$:

> [!theorem] 3.3 — Cartan's Magic Formula
> For any vector field $X$ and differential form $\omega$,
>
> $$\mathcal{L}_X \omega = d(\iota_X \omega) + \iota_X(d\omega),$$
>
> where $\mathcal{L}_X \omega$ is the **Lie derivative** of $\omega$ along $X$, defined as the infinitesimal rate of change of $\omega$ under the flow of $X$:
>
> $$(\mathcal{L}_X \omega)_p = \lim_{t \to 0} \frac{(\theta_t^* \omega)_p - \omega_p}{t}.$$


Cartan's formula says that the Lie derivative — a global, flow-based concept — decomposes into two purely algebraic operations: $d$ and $\iota_X$. The proof follows from the naturality of $d$ under pullback: $\theta_t^*(d\omega) = d(\theta_t^*\omega)$, differentiated at $t = 0$.

The Lie derivative of a form measures how the form changes as you drag it along the flow of $X$. A form $\omega$ is **invariant under the flow of $X$** if and only if $\mathcal{L}_X \omega = 0$, which by Cartan's formula is equivalent to $d(\iota_X \omega) + \iota_X(d\omega) = 0$. When $\omega$ is closed ($d\omega = 0$), invariance reduces to $d(\iota_X \omega) = 0$ — the contraction $\iota_X \omega$ is itself closed. This is the starting point for the study of Hamiltonian mechanics, where the invariant closed 2-form is the symplectic form (Chapter 9).

## Forms as the Language of Geometry

The framework assembled in this chapter — cotangent bundle, exterior algebra, exterior derivative, integration, Stokes' theorem — is not an isolated corner of geometry. It is a language that permeates everything that follows.

The Riemannian metric (Chapter 4) is a smooth choice of inner product on each $T_pM$, which is the same thing as a symmetric element of $T_p^*M \otimes T_p^*M$ — a tensor built from covectors. Volume on a Riemannian manifold is computed by integrating the volume form $\sqrt{\det g}\, dx^1 \wedge \cdots \wedge dx^n$, a top form constructed from the metric.

Curvature (Chapter 6) is expressed through the curvature 2-form, a form with values in the Lie algebra of the structure group. The Gauss-Bonnet theorem — which equates the integral of curvature over a closed surface with a topological invariant — is, at its core, an application of Stokes' theorem.

Symplectic geometry (Chapter 9) is the study of a closed, non-degenerate 2-form $\omega$ on a manifold. The entire formalism of Hamiltonian mechanics — conservation laws, Poisson brackets, the geometry of phase space — flows from the properties of this single form, especially its closedness.

Information geometry (Chapter 10) defines a Riemannian metric — the Fisher-Rao metric — on a statistical manifold, whose Riemannian volume form controls the natural measure on the space of distributions.

In each case, the appropriate object is a differential form of some degree, and the appropriate calculus is the exterior calculus developed here. The graded structure $\Omega^0, \Omega^1, \ldots, \Omega^n$, the exterior derivative $d$, and Stokes' theorem constitute the lingua franca of modern differential geometry.

---

*The algebraic approach to differential forms is developed in Lee's* Introduction to Smooth Manifolds *(Chapters 11–16). The de Rham theorem connecting differential forms and singular cohomology is proved in Bott and Tu's* Differential Forms in Algebraic Topology *(1982), one of the most beautiful books in mathematics. For the physicist's perspective and Cartan's magic formula in the context of classical mechanics, see Abraham and Marsden's* Foundations of Mechanics *(2nd ed., 1978).*
