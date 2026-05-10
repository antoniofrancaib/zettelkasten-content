---

title: "Smooth Manifolds"
subtitle: "Topological spaces, charts, atlases, smooth maps, and what it means for a space to have local coordinates"
---



Geometry begins with a question about the nature of space itself. Euclidean space $\mathbb{R}^n$ is the prototype: flat, global coordinates, calculus works everywhere. But most spaces of interest — the sphere, the torus, the group of rotations, the space of probability distributions — are not Euclidean. They are curved, bounded, or otherwise non-flat in ways that prevent any single coordinate system from covering them consistently. The question is how to do calculus on such spaces.

The answer is the theory of smooth manifolds. A manifold is a space that is *locally* Euclidean — not globally flat, but flat enough in small neighborhoods that calculus makes sense. The sphere is not $\mathbb{R}^2$, but every small patch of it looks like a piece of $\mathbb{R}^2$. This local-to-global philosophy is the engine of all that follows: build the theory locally using familiar Euclidean tools, then assemble the local pieces into a coherent global picture.

## Topological Manifolds

Before we can do calculus, we need a notion of space with enough structure to define continuity. A **topological space** is a set $M$ together with a collection of "open sets" that encode which points are near which others, satisfying the axioms that the empty set and $M$ are open, arbitrary unions of open sets are open, and finite intersections of open sets are open. Continuity of maps $f: M \to N$ is then defined exactly as in calculus: the preimage of every open set is open.

The simplest interesting topological spaces are those that look locally like $\mathbb{R}^n$.

> [!definition] 1.1 — Topological Manifold
> A **topological manifold of dimension $n$** is a topological space $M$ that is:
>
> 1. **Hausdorff**: any two distinct points have disjoint open neighborhoods,
> 2. **Second-countable**: the topology has a countable base, and
> 3. **Locally Euclidean of dimension $n$**: every point $p \in M$ has an open neighborhood $U \ni p$ homeomorphic to an open subset of $\mathbb{R}^n$.
>
> The dimension $n$ is the same at every point (for a connected manifold) and is an intrinsic property of $M$.


The Hausdorff and second-countability conditions are technical requirements that rule out pathological examples and ensure the manifold behaves well for analysis — in particular, they guarantee the existence of partitions of unity, which are indispensable for assembling local constructions into global ones. For every manifold that arises in practice, these conditions hold automatically.

The key condition is local Euclideanness. A homeomorphism $\varphi: U \to \hat{U} \subseteq \mathbb{R}^n$ assigns coordinates to points in $U$: a point $p \in U$ gets coordinates $\varphi(p) = (x^1(p), \ldots, x^n(p)) \in \mathbb{R}^n$. This pair $(U, \varphi)$ is called a **coordinate chart** or simply a chart.

The canonical examples: $\mathbb{R}^n$ itself is an $n$-manifold (trivially). The circle $S^1$ is a 1-manifold. The sphere $S^2$ is a 2-manifold. The torus $\mathbb{T}^2 = S^1 \times S^1$ is a 2-manifold. The real projective plane $\mathbb{RP}^2$ is a 2-manifold. None of the last four can be given a single global coordinate chart.

## The Need for Smooth Structure

Topological manifolds are the right setting for continuous maps and topological invariants, but they are too weak for calculus. Continuity does not distinguish between a smooth curve and a kinked one. We need a notion of *differentiability* for maps between manifolds — and to define that, we need the coordinate transitions to be smooth.

Here is the subtlety. Suppose a point $p \in M$ lies in two overlapping charts $(U_\alpha, \varphi_\alpha)$ and $(U_\beta, \varphi_\beta)$. The same point gets coordinates $\varphi_\alpha(p)$ and $\varphi_\beta(p)$ in each chart. The **transition map**

$$\varphi_\beta \circ \varphi_\alpha^{-1} : \varphi_\alpha(U_\alpha \cap U_\beta) \to \varphi_\beta(U_\alpha \cap U_\beta)$$

is a map between open subsets of $\mathbb{R}^n$, so we know what it means for it to be smooth. If all transition maps are $C^\infty$ (infinitely differentiable), then whatever we compute in one chart and translate to another via the transition map will preserve differentiability. The smooth structure is precisely the choice of which charts are "compatible" in this sense.

> [!definition] 1.2 — Smooth Atlas and Smooth Manifold
> An **atlas** on a topological manifold $M$ is a collection of charts $\{(U_\alpha, \varphi_\alpha)\}$ whose domains cover $M$: $M = \bigcup_\alpha U_\alpha$.
>
> An atlas is **smooth** (or $C^\infty$) if every transition map $\varphi_\beta \circ \varphi_\alpha^{-1}$ is a $C^\infty$ diffeomorphism between open subsets of $\mathbb{R}^n$ wherever it is defined.
>
> Two smooth atlases are **compatible** if their union is also a smooth atlas. A **smooth structure** on $M$ is a maximal smooth atlas — one that contains every chart compatible with it. A **smooth manifold** is a topological manifold equipped with a smooth structure.


The maximality condition is a bookkeeping convenience: it ensures that any chart compatible with our structure is already in it, so we never have to worry about whether a chart is "allowed." In practice, one specifies a smooth manifold by giving any smooth atlas; the maximal atlas is the equivalence class it generates.

> [!remark]
> A given topological manifold may admit multiple non-diffeomorphic smooth structures. In dimensions 1, 2, and 3, the smooth structure is unique (up to diffeomorphism). In dimension 4, exotic smooth structures exist — Milnor's discovery in 1956 that $S^7$ admits 28 non-diffeomorphic smooth structures was one of the great shocks in twentieth-century mathematics. For the manifolds arising in physics, geometry, and machine learning, one smooth structure is natural and we work with it without comment.


## Examples

The sphere $S^n = \{x \in \mathbb{R}^{n+1} : |x| = 1\}$ is an $n$-manifold. No single chart covers it — any continuous bijection from $S^n$ to an open subset of $\mathbb{R}^n$ fails to be a homeomorphism at at least one point. The standard smooth atlas uses two charts: **stereographic projection** from the north pole $N = (0, \ldots, 0, 1)$ and from the south pole $S = (0, \ldots, 0, -1)$. The chart from the north pole maps $S^n \setminus \{N\}$ to $\mathbb{R}^n$ by projecting through $N$ onto the equatorial hyperplane; the chart from the south pole does the same through $S$. The transition map on the overlap $S^n \setminus \{N, S\}$ is the inversion $x \mapsto x / |x|^2$ in $\mathbb{R}^n \setminus \{0\}$, which is $C^\infty$.

The **general linear group** $\mathrm{GL}(n, \mathbb{R}) = \{A \in \mathbb{R}^{n \times n} : \det A \neq 0\}$ is an open subset of $\mathbb{R}^{n^2}$ (since $\det$ is continuous and $\{0\}$ is closed), so it is an $n^2$-dimensional manifold with the single chart being the identity map. The **special orthogonal group** $\mathrm{SO}(n) = \{A \in \mathbb{R}^{n \times n} : AA^T = I, \det A = 1\}$ is a manifold of dimension $n(n-1)/2$ — it is a submanifold of $\mathrm{GL}(n, \mathbb{R})$, and also a Lie group, a theme we will return to in Chapter 7.

The **$n$-torus** $\mathbb{T}^n = S^1 \times \cdots \times S^1$ ($n$ factors) is an $n$-manifold. Charts are products of the standard angle coordinates on $S^1$, and the transitions are smooth. The torus arises in mechanics as the configuration space of systems of pendula, and in number theory as the domain of Fourier series.

**Product manifolds** are straightforward: if $M$ is an $m$-manifold and $N$ is an $n$-manifold, then $M \times N$ with the product topology and the charts $\{(U_\alpha \times V_\beta, \varphi_\alpha \times \psi_\beta)\}$ is an $(m+n)$-manifold.

## Smooth Maps

With smooth manifolds defined, the natural morphisms are the maps that preserve the smooth structure.

> [!definition] 1.3 — Smooth Maps
> Let $M$ and $N$ be smooth manifolds of dimensions $m$ and $n$. A continuous map $F: M \to N$ is **smooth** if for every chart $(U, \varphi)$ on $M$ and $(V, \psi)$ on $N$ with $F(U) \subseteq V$, the coordinate representation
>
> $$\psi \circ F \circ \varphi^{-1} : \varphi(U) \subseteq \mathbb{R}^m \to \psi(V) \subseteq \mathbb{R}^n$$
>
> is a $C^\infty$ map between Euclidean spaces.
>
> A smooth map $F: M \to N$ that is bijective with smooth inverse is a **diffeomorphism**. Two manifolds are **diffeomorphic** if a diffeomorphism between them exists.


Smoothness does not depend on which compatible charts are used — this is exactly what the smooth structure guarantees. The definition says: represent $F$ in coordinates, and require the resulting Euclidean map to be $C^\infty$. If it is $C^\infty$ in one pair of charts, it is $C^\infty$ in all compatible pairs, because the transitions between charts are themselves $C^\infty$.

The composition of smooth maps is smooth. The identity on any smooth manifold is smooth. Smooth manifolds with smooth maps form a category, denoted $\mathbf{Man}^\infty$ or $\mathbf{Diff}$. Diffeomorphisms are the isomorphisms of this category: two diffeomorphic manifolds are, from the smooth-geometric point of view, the same.

Important classes of smooth maps:

- A smooth map $\gamma: (a, b) \to M$ is a **smooth curve** in $M$.
- A smooth map $f: M \to \mathbb{R}$ is a **smooth function** on $M$. The set of all smooth functions on $M$ is denoted $C^\infty(M)$ and is a commutative ring under pointwise operations.
- A smooth map $F: M \to N$ with $dF_p$ injective at every $p$ is an **immersion**; if it is also a homeomorphism onto its image, it is an **embedding**.

## Submanifolds

The most natural way manifolds arise in practice is as subsets of larger manifolds defined by equations. The sphere $S^n \subset \mathbb{R}^{n+1}$ is defined by the single equation $|x|^2 = 1$. The orthogonal group $\mathrm{O}(n) \subset \mathbb{R}^{n^2}$ is defined by the $n(n+1)/2$ equations $(AA^T)_{ij} = \delta_{ij}$. The question is when the solution set of a system of equations is itself a manifold.

The answer is given by the rank theorem and its corollaries.

> [!definition] 1.4 — Regular Value and Submanifold
> Let $F: M \to N$ be a smooth map. A point $q \in N$ is a **regular value** of $F$ if for every $p \in F^{-1}(q)$, the linear map $dF_p: T_pM \to T_qN$ (the differential, defined precisely in Chapter 2) is surjective.
>
> If $q$ is a regular value of $F: M \to N$ and $F^{-1}(q) \neq \emptyset$, then $F^{-1}(q)$ is a smooth submanifold of $M$ of dimension $\dim M - \dim N$.


The proof is a direct consequence of the implicit function theorem: if $dF_p$ is surjective at every point of the level set, then locally $F$ looks like a projection, and the level set looks like a flat slice of $\mathbb{R}^m$. For the sphere: $F(x) = |x|^2 - 1$ is a smooth map $\mathbb{R}^{n+1} \to \mathbb{R}$, and $dF_x = 2x^T$ is surjective for every $x \neq 0$. In particular it is surjective on $S^n = F^{-1}(0)$, so $S^n$ is a smooth $n$-manifold.

> [!theorem] 1.1 — Inverse Function Theorem
> Let $F: M \to N$ be a smooth map and $p \in M$. If $dF_p: T_pM \to T_{F(p)}N$ is an isomorphism (i.e., $\dim M = \dim N$ and $dF_p$ is invertible), then there exist open neighborhoods $U \ni p$ and $V \ni F(p)$ such that $F|_U: U \to V$ is a diffeomorphism.


The inverse function theorem is a local statement: it says nothing about global injectivity, only that $F$ is locally invertible near $p$ if its linearization is invertible there. Combined with the implicit function theorem (which it implies), it is the main computational tool for recognizing when a set defined by equations is a submanifold.

## Partitions of Unity

A recurrent technical need in differential geometry is to pass between local and global constructions: to define something chart by chart and assemble the pieces consistently. The tool that makes this possible is the partition of unity.

> [!definition] 1.5 — Partition of Unity
> Let $M$ be a smooth manifold and $\{U_\alpha\}$ an open cover of $M$. A **smooth partition of unity subordinate to** $\{U_\alpha\}$ is a collection of smooth functions $\{\rho_\alpha : M \to [0,1]\}$ such that:
>
> 1. $\mathrm{supp}(\rho_\alpha) \subseteq U_\alpha$ for each $\alpha$,
> 2. the collection $\{\mathrm{supp}(\rho_\alpha)\}$ is locally finite (every point has a neighborhood meeting only finitely many supports), and
> 3. $\sum_\alpha \rho_\alpha(p) = 1$ for all $p \in M$.


The existence of smooth partitions of unity on any smooth manifold follows from the Hausdorff and second-countability conditions. Partitions of unity are used throughout: to define integration on manifolds (Chapter 3), to construct Riemannian metrics (Chapter 4), and to glue together locally defined objects into global ones.

## The Category of Smooth Manifolds

It is worth pausing to appreciate what the smooth manifold framework achieves. We have a category whose objects are spaces that look locally like $\mathbb{R}^n$ and whose morphisms are maps that are smooth in local coordinates. This is the correct generalization of "doing calculus" to curved spaces:

- Every object locally looks like the familiar Euclidean setting.
- Every morphism is locally represented by a $C^\infty$ map between Euclidean spaces.
- The global topology can be arbitrarily complicated, but calculus is available everywhere.

The framework is also highly flexible. By varying the smoothness condition on transition maps, one gets different categories: $C^k$ manifolds (transition maps are $k$-times differentiable), real-analytic manifolds (transition maps are analytic), complex manifolds (transition maps are holomorphic). For our purposes, $C^\infty$ is the right choice: smooth enough for all of analysis, without the rigidity of analyticity.

What the framework does not yet provide is any way to measure lengths, angles, or distances — any way to distinguish a small circle from a large one, or a sphere from an ellipsoid. Topological manifolds have the same topology as their deformations; smooth manifolds have the same smooth structure as their diffeomorphisms. Geometry — in the sense of measurement — requires additional structure. That additional structure is the Riemannian metric, introduced in Chapter 4. But to even define what a metric is, we need the tangent space at each point — the infinitesimal Euclidean approximation to the manifold. That is the subject of the next chapter.

## What Smooth Manifolds Are Not

It is instructive to name what is missing. A smooth manifold has no preferred coordinates, no notion of distance, no way to compare vectors at different points, no canonical way to differentiate vector fields. All of these deficiencies are real: they distinguish manifolds from Euclidean spaces, and they are precisely what makes geometry on manifolds interesting and non-trivial.

The absence of preferred coordinates means that meaningful geometric objects must be defined coordinate-independently. A function $f: M \to \mathbb{R}$ is well-defined on $M$; its value at $p$ does not depend on which chart we use. But the *gradient* of $f$ — the direction of steepest ascent — requires a metric to define, because "steepest" depends on how we measure lengths of directions. The coordinate-independence requirement is not a burden but a feature: it forces us to identify which structures are truly geometric (intrinsic to the manifold) and which are artifacts of a coordinate choice.

The absence of a way to compare tangent vectors at different points means that "parallel" has no meaning yet. A vector field on a sphere — a smooth assignment of a tangent vector to each point — does not automatically tell us how to transport a vector from one point to another while keeping it "the same." That requires a connection (Chapter 5), and which connection we use is an additional geometric choice.

These gaps are not defects. They are the questions that the rest of the monograph answers.

---

*The standard references for this chapter are Lee's* Introduction to Smooth Manifolds *(2nd ed., 2013), which gives the most complete modern treatment, and do Carmo's* Riemannian Geometry *(1992) for the path toward metric structure. Milnor's original paper on exotic spheres (1956) remains worth reading for the shock of the result.*
