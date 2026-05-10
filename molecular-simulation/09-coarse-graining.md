---

title: "Coarse-Graining"
subtitle: "Effective degrees of freedom, systematic reduction, and the multiscale hierarchy"
---



A water molecule has three atoms, six degrees of freedom, and can be simulated with a timestep of 2 fs. A lipid bilayer patch of biological interest contains $\sim 10^5$ atoms and must be simulated for microseconds to observe a single flip-flop event. A protein undergoing a large conformational change involves $\sim 10^4$ atoms and dynamics on the millisecond timescale. The gap between what atomistic simulation can access and what biology requires is not a matter of faster computers — it is a matter of decades of Moore's law. The solution is to use a different description: one that discards the degrees of freedom irrelevant to the process of interest, retains only those that matter, and replaces the omitted physics with an effective potential that approximately restores their influence. This is coarse-graining.

## The Coarse-Graining Mapping

A coarse-grained (CG) model replaces $n$ atomistic degrees of freedom with $N < n$ collective degrees of freedom — the **CG sites** — linked to the atomistic coordinates by a **mapping operator**.

> [!definition] 9.1 — The Coarse-Graining Mapping
> Let $\mathbf{r} = (\mathbf{r}_1, \ldots, \mathbf{r}_n) \in \mathbb{R}^{3n}$ denote the atomistic coordinates. A **linear CG mapping** defines $N$ CG sites at positions $\mathbf{R} = (\mathbf{R}_1, \ldots, \mathbf{R}_N) \in \mathbb{R}^{3N}$ by:
>
> <Equation number="9.1">
> $$\mathbf{R}_I = \sum_{i=1}^n c_{Ii}\, \mathbf{r}_i, \quad I = 1, \ldots, N,$$
> </Equation>
>
> where $c_{Ii} \geq 0$ are mapping coefficients satisfying $\sum_i c_{Ii} = 1$ for each $I$. The most common choice is center-of-mass mapping: $c_{Ii} = m_i / M_I$ if atom $i$ belongs to CG site $I$ and 0 otherwise, where $M_I = \sum_i m_i$ is the total mass of the CG bead. Each CG bead typically represents a small group of atoms — a pair of heavy atoms, a sugar ring, a methylene group — that move approximately rigidly on the timescales of interest.


The mapping reduces $3n$ to $3N$ degrees of freedom — a compression ratio of $n/N$, which for typical biomolecular CG models (MARTINI-level, ~4 heavy atoms per bead) is roughly 4:1. This gives a $4\times$ reduction in degrees of freedom, but the speedup is far larger: (1) the CG model uses a timestep 10–100$\times$ larger than atomistic MD because fast bond vibrations are absent; (2) the smoother CG landscape means the system diffuses faster through configuration space; and (3) fewer force evaluations are needed per step. Typical CG simulations run $10^3$–$10^4\times$ faster than atomistic ones.

## The Many-Body Potential of Mean Force

The central object of coarse-graining theory is the **many-body potential of mean force** (MB-PMF) — the exact effective potential for the CG degrees of freedom.

> [!definition] 9.2 — The Many-Body Potential of Mean Force
> Given a mapping operator $\mathbf{R} = \mathbf{M}(\mathbf{r})$, the **many-body PMF** $U_{\mathrm{CG}}(\mathbf{R})$ is defined by:
>
> <Equation number="9.2">
> $$e^{-\beta U_{\mathrm{CG}}(\mathbf{R})} = \frac{1}{Z} \int e^{-\beta U(\mathbf{r})}\, \delta(\mathbf{M}(\mathbf{r}) - \mathbf{R})\, d\mathbf{r},$$
> </Equation>
>
> where $U(\mathbf{r})$ is the atomistic potential energy and the delta function restricts the integral to atomistic configurations that map to the CG configuration $\mathbf{R}$. The MB-PMF is the free energy of the CG configuration: it integrates out all atomistic degrees of freedom consistent with the CG coordinates, exactly capturing both the energetic and entropic contributions of the omitted atoms.
>
> The equilibrium distribution of CG configurations is $\pi_{\mathrm{CG}}(\mathbf{R}) \propto e^{-\beta U_{\mathrm{CG}}(\mathbf{R})}$, which exactly reproduces the marginal of the atomistic Boltzmann distribution projected onto the CG degrees of freedom.


The MB-PMF is the theoretically exact answer to the coarse-graining problem. It is also practically intractable: it depends on all $N$ CG coordinates simultaneously, making it a function in $3N$ dimensions that cannot be tabulated or parameterized without approximation. The goal of systematic coarse-graining is to find a tractable approximation $\hat{U}_{\mathrm{CG}}(\mathbf{R})$ — typically written as a sum of pairwise potentials — that reproduces selected properties of the exact MB-PMF as accurately as possible.

## Force Matching and Multiscale Coarse-Graining

The **force-matching** method (Ercolessi and Adams, 1994; extended to CG by Izvekov and Voth, 2005) finds the CG force field by minimizing the mean squared error between the CG forces and the atomistic forces projected onto the CG sites.

> [!definition] 9.3 — Force Matching (MS-CG)
> The **multiscale coarse-graining** (MS-CG) objective is:
>
> <Equation number="9.3">
> $$\chi^2[\hat{U}_{\mathrm{CG}}] = \frac{1}{3N} \left\langle \sum_{I=1}^N \left|\mathbf{G}_I(\mathbf{r}) + \frac{\partial \hat{U}_{\mathrm{CG}}(\mathbf{M}(\mathbf{r}))}{\partial \mathbf{R}_I}\right|^2 \right\rangle_{\mathbf{r}},$$
> </Equation>
>
> where $\mathbf{G}_I(\mathbf{r}) = -\sum_{i \in I} \nabla_{\mathbf{r}_i} U(\mathbf{r})$ is the total atomistic force on CG site $I$ (the sum of atomic forces within the bead), and the average is over atomistic configurations. The CG force field $\hat{U}_{\mathrm{CG}}$ is chosen to minimize $\chi^2$.


The key result of MS-CG theory (Noid et al., 2008) is that the minimizer of $\chi^2$ over all possible force fields is the MB-PMF itself: $\argmin \chi^2 = U_{\mathrm{CG}}$. Force matching is therefore a variational approach that approximates the MB-PMF within the chosen functional form of $\hat{U}_{\mathrm{CG}}$. In practice, $\hat{U}_{\mathrm{CG}}$ is parameterized as a sum of pair potentials $\sum_{I < J} v(R_{IJ})$, tabulated on a grid or expanded in a basis. The minimization of $\chi^2$ over these parameters reduces to a linear least-squares problem, solvable at low computational cost once the atomistic forces have been collected.

## Relative Entropy Minimization

An alternative variational approach, due to Shell (2008), minimizes the **relative entropy** (Kullback–Leibler divergence) between the CG distribution generated by $\hat{U}_{\mathrm{CG}}$ and the exact projected atomistic distribution:

> [!equation] 9.4
> $$S_{\mathrm{rel}}[\hat{U}_{\mathrm{CG}}] = \int \pi_{\mathrm{CG}}(\mathbf{R})\, \ln \frac{\pi_{\mathrm{CG}}(\mathbf{R})}{\hat{\pi}_{\mathrm{CG}}(\mathbf{R})}\, d\mathbf{R},$$


where $\pi_{\mathrm{CG}} \propto e^{-\beta U_{\mathrm{CG}}}$ is the exact marginal and $\hat{\pi}_{\mathrm{CG}} \propto e^{-\beta \hat{U}_{\mathrm{CG}}}$ is the distribution generated by the approximate CG potential. The relative entropy $S_{\mathrm{rel}} \geq 0$, with equality if and only if $\hat{\pi}_{\mathrm{CG}} = \pi_{\mathrm{CG}}$, i.e., when the CG model is exact.

Minimizing $S_{\mathrm{rel}}$ over the parameters of $\hat{U}_{\mathrm{CG}}$ is equivalent to maximum likelihood estimation: given samples of CG configurations $\{\mathbf{R}_k\}$ from the atomistic simulation, find the CG potential that makes those configurations most probable. The gradient of $S_{\mathrm{rel}}$ with respect to the CG parameters is:

$$\frac{\partial S_{\mathrm{rel}}}{\partial \theta_\alpha} = \beta \left[\left\langle \frac{\partial \hat{U}_{\mathrm{CG}}}{\partial \theta_\alpha} \right\rangle_{\hat{\pi}} - \left\langle \frac{\partial \hat{U}_{\mathrm{CG}}}{\partial \theta_\alpha} \right\rangle_{\pi}\right],$$

which requires ensemble averages under both the CG model (from CG simulations) and the atomistic model (from atomistic simulations). This is more expensive than force matching, which requires only atomistic data, but it directly optimizes the thermodynamic properties of interest rather than the forces.

## Iterative Boltzmann Inversion

**Iterative Boltzmann inversion** (IBI, Reith, Pütz, and Müller-Plathe, 2003) is an older, heuristic approach that adjusts CG pair potentials to match target structural correlation functions from atomistic simulation — typically the radial distribution function (RDF) $g(r)$.

The connection between the pair potential $v(r)$ and the RDF is exact only for an ideal gas: for interacting systems, the potential of mean force $w(r) = -k_BT \ln g(r)$ is not the pair potential but the effective pair interaction averaged over all other particles. IBI uses this as a starting guess and iterates:

$$v^{(k+1)}(r) = v^{(k)}(r) + k_BT \ln \frac{g^{(k)}(r)}{g^{\mathrm{target}}(r)},$$

where $g^{(k)}(r)$ is the RDF produced by the CG model at iteration $k$ and $g^{\mathrm{target}}(r)$ is the target RDF from atomistic simulation. IBI converges when $g^{(k)} = g^{\mathrm{target}}$ — the CG model exactly reproduces the atomistic RDF. It is simple to implement and works well for homogeneous liquids. It has no rigorous optimality property and can produce potentials that reproduce the RDF but fail to reproduce other structural or thermodynamic properties.

> [!remark] 9.1
> The **representability problem** is the central difficulty of pairwise CG models: a pair potential $v(r)$ that exactly reproduces one structural property (the RDF) of the MB-PMF does not in general reproduce others (the pressure, the compressibility, the three-body correlation functions). Different systematic methods — force matching, relative entropy minimization, IBI — minimize different measures of the error between $\hat{U}_{\mathrm{CG}}$ and the exact MB-PMF, and they generally produce different potentials that agree on some properties and disagree on others. This is not a failure of the methods but a fundamental consequence of approximating a many-body function by a sum of pair terms: the pair approximation is an underdetermined system, and different objectives select different solutions within it.


## Top-Down Parameterization: MARTINI

The methods above are **bottom-up**: CG parameters are derived from atomistic simulation data. An alternative philosophy is **top-down**: parameterize the CG force field directly against experimental observables — partition coefficients, densities, heat capacities, phase transition temperatures — without reference to atomistic trajectories.

The **MARTINI force field** (Marrink et al., 2004, 2007) is the dominant example. MARTINI maps approximately 4 heavy atoms to one CG bead, assigns each bead to one of ~20 interaction types based on its chemical character (polar, nonpolar, charged, ring), and parameterizes the Lennard-Jones interactions between bead types against experimental octanol/water partition coefficients and bulk densities. The result is a transferable force field: MARTINI parameters for a lipid tail bead are the same whether the tail is in a membrane, a micelle, or solution.

MARTINI's success comes from its transferability and its domain coverage — it has been parameterized for lipids, proteins, carbohydrates, DNA, polymers, and small molecules. Its limitations are the inherent approximations of top-down parameterization: MARTINI systematically overestimates the melting temperature of lipid bilayers, underestimates the free energy of partitioning for some solutes, and cannot reproduce properties that were not included in the training data. The recently released MARTINI 3 (Souza et al., 2021) substantially improved these limitations through a more systematic parameterization protocol.

## The Mori–Zwanzig Formalism

The bottom-up and top-down approaches share a fundamental approximation: the CG potential is a static function of instantaneous CG coordinates. But the exact dynamics of the CG degrees of freedom — as projected from the full atomistic dynamics — is not Newtonian. It contains **memory effects**: the force on a CG bead at time $t$ depends not only on the current CG configuration but on the history of CG configurations at all earlier times. This is the content of the **Mori–Zwanzig** (MZ) projection formalism.

The MZ formalism begins by defining a projection operator $\mathcal{P}$ that projects functions of the full phase space onto functions of the CG degrees of freedom. Applying the Dyson identity to the Liouville operator governing the atomistic dynamics gives, for any CG observable $A(\mathbf{R})$:

> [!equation] 9.5
> $$M_I \ddot{\mathbf{R}}_I(t) = -\nabla_{\mathbf{R}_I} U_{\mathrm{CG}}(\mathbf{R}(t)) - \int_0^t \mathbf{K}_I(t - t')\, \dot{\mathbf{R}}_I(t')\, dt' + \boldsymbol{\xi}_I(t).$$


This is the **generalized Langevin equation** (GLE): the CG bead $I$ experiences the mean force from the MB-PMF, a memory kernel $\mathbf{K}_I(t-t')$ (a time-dependent friction that depends on the history of velocities), and a fluctuating force $\boldsymbol{\xi}_I(t)$ with zero mean. The memory kernel and fluctuating force are related by the second fluctuation-dissipation theorem: $\langle \boldsymbol{\xi}_I(t) \cdot \boldsymbol{\xi}_I(0) \rangle = k_BT\, \mathbf{K}_I(t)$.

Standard CG MD treats the CG beads as Newtonian particles — replacing the full GLE by Newton's equations with the MB-PMF gradient. This approximation is exact only when the memory kernel decays instantaneously, i.e., when the CG dynamics are Markovian. For well-separated timescale hierarchies — when the omitted fast degrees of freedom equilibrate much faster than the CG motions — the Markovian approximation is good. When fast and slow timescales overlap, non-Markovian effects are significant and the standard CG MD produces incorrect dynamics even when the thermodynamics is correct.

> [!remark] 9.2
> The practical consequence of non-Markovian dynamics is that CG MD simulations, even with a perfect MB-PMF, may produce diffusion coefficients, relaxation times, and spectral densities that are systematically wrong. The CG model is thermodynamically correct (it samples the right equilibrium distribution) but dynamically incorrect (trajectories evolve too fast because the friction from the omitted degrees of freedom is absent). The Mori–Zwanzig GLE provides the formal correction: fitting the memory kernel $\mathbf{K}(t)$ from atomistic trajectory data and incorporating it into the CG dynamics gives a model that is both thermodynamically and dynamically consistent. Several practical approaches — including the generalized Langevin thermostat, colored noise methods, and data-driven memory kernels — implement this in practice, with varying degrees of fidelity and computational cost.


## The Multiscale Hierarchy

Coarse-graining is not a single step but a hierarchy of approximations, each valid at a different length and timescale:

**Quantum mechanics** → $\mathcal{O}(10^1\text{–}10^3)$ atoms, femtosecond–picosecond timescales; electrons explicitly treated.

**Atomistic force fields** → $\mathcal{O}(10^3\text{–}10^6)$ atoms, picosecond–microsecond; electrons averaged into fixed partial charges and parameterized LJ interactions.

**Coarse-grained models** (MARTINI, CGMD) → $\mathcal{O}(10^5\text{–}10^8)$ effective particles, nanosecond–millisecond; small groups of atoms represented as single beads.

**Mesoscale models** (dissipative particle dynamics, field-theoretic simulation) → $\mathcal{O}(10^8\text{–}10^{12})$ effective particles, microsecond–second; entire polymer chains or lipid headgroups as single objects.

**Continuum models** (finite element, fluid dynamics) → no particles; matter described by density fields and constitutive relations.

Each transition up the hierarchy integrates out degrees of freedom and introduces effective interactions. Information flows both ways: atomistic data parameterizes CG models (bottom-up); macroscopic experimental data parameterizes top-down models. Coupling models at different levels — running atomistic and CG simulations simultaneously, exchanging information at interfaces — is the domain of **adaptive resolution** and **multiscale** methods, an active area of development.

> [!remark] 9.3
> Machine learning has dramatically expanded the repertoire of coarse-graining methods. Neural network CG potentials can represent many-body interactions that pair potentials cannot, approaching the accuracy of the MB-PMF within the pair approximation's range of validity. Equivariant neural networks (Chapter 10) can enforce physical symmetries (rotational, translational, permutational invariance) exactly, rather than relying on the approximate symmetries of tabulated pair potentials. Variational autoencoders and graph neural networks have been used to learn CG mappings — the $c_{Ii}$ coefficients — jointly with the CG potential, allowing both the resolution and the interactions to be optimized simultaneously. These approaches are discussed in Chapter 10.


---

*The formal theory of coarse-graining and the potential of mean force originates with Kirkwood (1935). The force-matching method was introduced by Ercolessi and Adams (1994) for interatomic potentials and extended to systematic CG by Izvekov and Voth (2005); the variational MS-CG theory was established by Noid, Chu, Ayton, Krishna, Andricioaei, Voth, Field, Henderson, and Izvekov (2008). Relative entropy minimization was introduced by Shell (2008). Iterative Boltzmann inversion was developed by Reith, Pütz, and Müller-Plathe (2003). The MARTINI force field was introduced by Marrink, Risselada, Yefimov, Tieleman, and de Vries (2007) for lipids and extended to proteins by Monticelli et al. (2008); MARTINI 3 was released by Souza et al. (2021). The Mori–Zwanzig projection formalism was developed independently by Mori (1965) and Zwanzig (1960, 1961); its application to CG dynamics was clarified by Español and Warren (1995) and Hijon, Español, Vanden-Eijnden, and Delgado-Buscalioni (2010). Reviews by Noid (2013) and Kmiecik et al. (2016) provide comprehensive coverage of CG methodology and applications.*
