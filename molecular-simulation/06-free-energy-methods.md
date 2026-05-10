---

title: "Free Energy Methods"
subtitle: "Perturbation theory, thermodynamic integration, and alchemical transformations"
---



The free energy difference between two states is among the most important quantities in molecular science. It determines which polymorph of a drug crystal is thermodynamically stable, how tightly a ligand binds to its receptor, which reaction pathway is favored, and whether a protein folds. Chapter 4 showed that the free energy landscape governs both thermodynamics and kinetics. Chapter 5 developed methods for mapping free energy profiles along collective variables. This chapter addresses a different but related problem: computing the **free energy difference** $\Delta F = F_B - F_A$ between two thermodynamic states that may differ in their Hamiltonians, their boundary conditions, or even in whether certain atoms exist.

The fundamental difficulty is the same as in Chapter 1: the free energy $F = -k_BT \ln Z$ requires the partition function $Z$, which cannot be evaluated directly. What free energy methods do instead is compute differences — exploiting the fact that $\Delta F = F_B - F_A = -k_BT \ln (Z_B / Z_A)$ involves a ratio of partition functions in which most of the intractable normalization cancels. The art lies in constructing a pathway from $A$ to $B$ along which this ratio can be estimated efficiently from simulation.

## Free Energy Perturbation

The oldest and most transparent free energy method is **free energy perturbation** (FEP), derived by Zwanzig in 1954. Let state $A$ have Hamiltonian $H_A$ and state $B$ have Hamiltonian $H_B$. The exact free energy difference is:

> [!definition] 6.1 — Free Energy Perturbation
> The **Zwanzig formula** expresses the free energy difference between two states in terms of an ensemble average over the reference state $A$:
>
> <Equation number="6.1">
> $$\Delta F_{A \to B} = F_B - F_A = -k_BT \ln \left\langle e^{-\beta [H_B(\mathbf{q}) - H_A(\mathbf{q})]} \right\rangle_A,$$
> </Equation>
>
> where $\langle \cdot \rangle_A$ denotes an ensemble average over configurations drawn from the Boltzmann distribution of state $A$: $\pi_A(\mathbf{q}) \propto e^{-\beta H_A(\mathbf{q})}$.


The derivation is a single line:

$$e^{-\beta \Delta F} = \frac{Z_B}{Z_A} = \frac{\int e^{-\beta H_B}\, d\mathbf{q}}{\int e^{-\beta H_A}\, d\mathbf{q}} = \left\langle \frac{e^{-\beta H_B}}{e^{-\beta H_A}} \right\rangle_A = \left\langle e^{-\beta \Delta H} \right\rangle_A,$$

where $\Delta H(\mathbf{q}) = H_B(\mathbf{q}) - H_A(\mathbf{q})$. The formula is exact. The computational task is to estimate the exponential average on the right by sampling from $\pi_A$.

> [!remark] 6.1
> The exponential average in the Zwanzig formula is dominated by configurations where $\Delta H(\mathbf{q})$ is large and negative — configurations that are rare under $\pi_A$ but highly probable under $\pi_B$. If the two Hamiltonians are very different, such configurations are almost never sampled in a simulation of state $A$, and the estimator has enormous variance despite formally converging to the correct answer. This is the **overlap problem**: FEP requires substantial overlap between the configuration spaces explored by states $A$ and $B$. When overlap is poor, the variance of the estimator grows exponentially with $\beta \Delta F$, making direct FEP unusable for large perturbations.
>
> The practical remedy is **staging**: introduce a sequence of intermediate states $A = \lambda_0, \lambda_1, \ldots, \lambda_M = B$ and compute $\Delta F$ as a sum of small-perturbation steps $\sum_{m=0}^{M-1} \Delta F_{\lambda_m \to \lambda_{m+1}}$. Each step involves a small perturbation with good overlap; the total free energy difference is recovered by summing. Staging transforms a single exponentially hard problem into $M$ tractable ones.


## The Bennett Acceptance Ratio

The Zwanzig formula uses only the simulation of state $A$ to estimate $\Delta F$. If simulations of both $A$ and $B$ are available, a much better estimator is possible. **Bennett's acceptance ratio** (BAR, 1976) is the minimum-variance unbiased estimator of $\Delta F$ given samples from both endpoints:

> [!equation] 6.2
> $$\Delta F_{A \to B} = k_BT \ln \frac{\langle f(\Delta H - C) \rangle_B}{\langle f(-\Delta H + C) \rangle_A} + C,$$


where $f(x) = 1/(1 + e^{\beta x})$ is the Fermi function and $C$ is a constant determined self-consistently by requiring the two sides of (6.2) to be equal. The Fermi function weights configurations that appear in both ensembles — those in the overlap region — most heavily, and downweights configurations that are far from overlap. BAR achieves the minimum variance among all estimators that use only the energy differences $\Delta H$ evaluated at the sampled configurations.

The multistate generalization, MBAR (see Chapter 5), applies when more than two states are simulated — as is always the case in thermodynamic integration or staged FEP.

## Thermodynamic Integration

**Thermodynamic integration** (TI, Kirkwood 1935) avoids the exponential averaging problem entirely by differentiating the free energy with respect to a coupling parameter rather than exponentiating an energy difference.

> [!definition] 6.2 — Thermodynamic Integration
> Introduce a coupling parameter $\lambda \in [0, 1]$ that interpolates between the Hamiltonians of states $A$ and $B$:
>
> $$H(\mathbf{q}; \lambda) = (1 - \lambda)\, H_A(\mathbf{q}) + \lambda\, H_B(\mathbf{q}).$$
>
> The free energy at coupling $\lambda$ is $F(\lambda) = -k_BT \ln Z(\lambda)$. Differentiating with respect to $\lambda$:
>
> <Equation number="6.3">
> $$\frac{dF}{d\lambda} = \left\langle \frac{\partial H}{\partial \lambda} \right\rangle_\lambda = \left\langle H_B - H_A \right\rangle_\lambda,$$
> </Equation>
>
> where $\langle \cdot \rangle_\lambda$ is the ensemble average at coupling $\lambda$. The free energy difference is then recovered by integration:
>
> $$\Delta F_{A \to B} = \int_0^1 \left\langle \frac{\partial H}{\partial \lambda} \right\rangle_\lambda d\lambda.$$


TI converts the problem of computing a ratio of partition functions into computing a definite integral of an ensemble average. At each value of $\lambda$, one runs an independent simulation (or a single long simulation at multiple $\lambda$ values) and records $\langle \partial H / \partial \lambda \rangle_\lambda$. The integral over $\lambda$ is then evaluated numerically using quadrature — Gaussian quadrature is particularly efficient because the integrand is smooth for well-chosen coupling schedules.

The key advantage of TI over FEP is that $\langle \partial H / \partial \lambda \rangle_\lambda$ is an ordinary ensemble average, not an exponential average: it has well-controlled variance that does not blow up when states are far apart. The key requirement is a smooth coupling $H(\mathbf{q}; \lambda)$ such that the integrand is well-behaved along the entire path.

> [!remark] 6.2
> When state $B$ contains an atom that is absent in state $A$ (or vice versa), the linear interpolation $H = (1-\lambda)H_A + \lambda H_B$ has a **singularity at the endpoints**: as $\lambda \to 0$, a Lennard-Jones or Coulomb interaction involving the appearing atom diverges for configurations where the new atom overlaps with existing atoms. These near-singularities cause the integrand $\langle \partial H / \partial \lambda \rangle_\lambda$ to diverge as $\lambda \to 0$ or $1$, making numerical integration inaccurate.
>
> The standard cure is **soft-core potentials** (Beutler et al. 1994; Zacharias et al. 1994): the Lennard-Jones interaction involving the alchemical atom is modified so that the repulsive core shrinks to zero at $\lambda = 0$, preventing divergence. The modified potential has the form
>
> $$V_{\mathrm{sc}}(r; \lambda) = 4\varepsilon\, \lambda^n \left[\frac{1}{\left(\alpha(1-\lambda)^m + (r/\sigma)^6\right)^2} - \frac{1}{\alpha(1-\lambda)^m + (r/\sigma)^6}\right],$$
>
> with parameters $\alpha$, $m$, $n$ controlling the softening. Soft-core interactions allow atoms to appear and disappear smoothly without coordinate singularities, at the cost of introducing a more complex $\lambda$-dependent potential.


## Alchemical Transformations

The coupling parameter $\lambda$ need not interpolate between two physically realizable states. **Alchemical transformations** connect states that are related by non-physical changes — turning one atom type into another, annihilating an atom entirely, switching off all interactions of a molecule — along a path that exists only in simulation. The name is deliberate: like medieval alchemy, one atom is transmuted into another, but the purpose is not to create gold from lead but to compute a free energy difference.

> [!definition] 6.3 — Alchemical Transformation
> An **alchemical transformation** is a path in parameter space along which the Hamiltonian changes from $H_A$ to $H_B$ by continuously modifying interaction parameters. Common examples include:
>
> - **Mutation**: changing the identity of a residue in a protein (e.g., Ala → Val) by interpolating between the force field parameters of the two residues.
> - **Annihilation**: removing a molecule from the system by scaling its interactions to zero, $H(\lambda) = H_{\mathrm{protein+solvent}} + \lambda\, H_{\mathrm{ligand-interactions}}$.
> - **Decoupling**: turning off interactions between a ligand and its environment (but not within the ligand), used to compute absolute binding free energies via a thermodynamic cycle.
>
> In each case, the path from $\lambda = 0$ to $\lambda = 1$ is unphysical — no experiment could observe the intermediate states — but the endpoints are physical, and $\Delta F$ between them is a well-defined thermodynamic quantity.


The power of alchemical methods emerges when combined with **thermodynamic cycles**. Consider the binding free energy of a ligand $L$ to a protein $P$:

$$\Delta F_{\mathrm{bind}} = F_{PL} - F_P - F_L,$$

where $F_{PL}$, $F_P$, $F_L$ are the free energies of the protein-ligand complex, the apo protein, and the free ligand. Direct simulation of the binding event is difficult — it requires sampling from the unbound state to the bound state across a barrier, on a timescale that may be microseconds to milliseconds.

The alchemical shortcut uses a thermodynamic cycle. Because free energy is a state function:

> [!equation] 6.4
> $$\Delta F_{\mathrm{bind}}(L_2) - \Delta F_{\mathrm{bind}}(L_1) = \Delta F_{\mathrm{alch}}^{\mathrm{complex}} - \Delta F_{\mathrm{alch}}^{\mathrm{solvent}},$$


where $\Delta F_{\mathrm{alch}}^{\mathrm{complex}}$ is the alchemical free energy of mutating $L_1 \to L_2$ in the protein-ligand complex, and $\Delta F_{\mathrm{alch}}^{\mathrm{solvent}}$ is the same mutation in solvent. Both alchemical legs are far more tractable than simulating the binding event directly: the ligand is already bound (or already solvated), and the perturbation from $L_1$ to $L_2$ is a small chemical change. This is the basis of **relative binding free energy** (RBFE) calculations, which have become a standard tool in computational drug discovery.

## Absolute Binding Free Energies

Equation (6.4) gives only the **relative** binding affinity of $L_2$ versus $L_1$. To compute the **absolute** binding free energy of a single ligand, one must evaluate $\Delta F_{\mathrm{bind}}(L)$ directly, using a double-annihilation or double-decoupling scheme.

The double-decoupling approach (Gilson et al. 1997) runs two alchemical transformations:

1. **In the complex**: decouple the ligand from the protein and solvent, computing $\Delta F_{\mathrm{decouple}}^{\mathrm{complex}}$.
2. **In solution**: decouple the ligand from the solvent alone, computing $\Delta F_{\mathrm{decouple}}^{\mathrm{solvent}}$.

The absolute binding free energy is:

$$\Delta F_{\mathrm{bind}} = \Delta F_{\mathrm{decouple}}^{\mathrm{complex}} - \Delta F_{\mathrm{decouple}}^{\mathrm{solvent}} + \Delta F_{\mathrm{restraint}},$$

where $\Delta F_{\mathrm{restraint}}$ is an analytical correction for the restraints used to keep the decoupled ligand in the binding site (preventing it from wandering through the simulation box once its interactions are turned off). Absolute free energy calculations are more expensive than relative ones — each leg requires a full alchemical pathway — but provide $K_d$ predictions directly comparable to experimental affinities.

> [!remark] 6.3
> Absolute and relative free energy calculations in drug discovery have achieved remarkable accuracy over the past decade. For congeneric series of ligands — small molecules sharing a common scaffold with varying substituents — RBFE calculations with modern force fields and staging protocols routinely achieve mean unsigned errors of 0.5–1.0 kcal/mol compared to experimental $K_d$ values. This accuracy is sufficient to rank-order analogs and prioritize synthesis: a computational cost of hours to days per compound versus weeks and significant material cost for synthesis and assay. The success depends on accurate force fields, sufficient sampling (especially for ligand conformational flexibility and water displacement from the binding site), and careful staging. End-state artifacts — poor convergence at $\lambda = 0$ or $\lambda = 1$ due to soft-core implementation details — remain a practical source of error.


## Staging and Convergence

The practical implementation of TI or staged FEP requires choosing:

**The $\lambda$ schedule**: how many $\lambda$ values to simulate and where to place them. For smooth integrands, Gaussian quadrature points (e.g., 5 or 9 Gauss-Legendre nodes in $[0,1]$) are more efficient than equally spaced points. Near the endpoints, where soft-core potentials are active and the integrand may change rapidly, denser spacing is needed.

**The simulation length at each $\lambda$**: convergence of $\langle \partial H / \partial \lambda \rangle_\lambda$ requires the system to be equilibrated at each $\lambda$ value. Slow conformational relaxation — protein side-chains reorganizing around the alchemically modified ligand — can prevent convergence even with long simulations. Replica exchange between $\lambda$ values (lambda-REMD) can accelerate this by allowing configurations to diffuse across the $\lambda$ schedule.

**Hysteresis detection**: running the $\lambda$ schedule in both directions ($0 \to 1$ and $1 \to 0$) and comparing the resulting $\Delta F$ values. Persistent hysteresis after sufficient sampling indicates that the system is trapped in different conformational states at intermediate $\lambda$ values — a sign that enhanced sampling (metadynamics, REST2, or additional collective variable restraints) is needed along the alchemical path.

The combination of staged FEP or TI with MBAR analysis and enhanced sampling — sometimes called **FEP+** or **ABFE+** in the industry — represents the current state of the art for binding free energy calculations. Automated pipelines now run thousands of such calculations in parallel for drug discovery campaigns, integrating seamlessly with structure-based design workflows.

---

*Free energy perturbation was derived by Zwanzig (1954). Thermodynamic integration and the potential of mean force were introduced by Kirkwood (1935). The Bennett acceptance ratio is due to Bennett (1976); its multistate generalization MBAR to Shirts and Chodera (2008). Alchemical free energy methods for binding were pioneered by Tembe and McCammon (1984), who first computed a relative binding free energy by simulation, and by Jorgensen and Ravimohan (1985), who introduced the free energy perturbation approach for mutations. Soft-core potentials to eliminate end-point singularities were introduced by Beutler et al. (1994). The double-decoupling method for absolute binding free energies was formalized by Gilson et al. (1997). Reviews by Chipot and Pohorille (2007) and Cournia, Allen, and Sherman (2017) cover the methodology and its application to drug discovery in depth. The accuracy benchmark for RBFE in lead optimization series was established by Wang et al. (2015) using the FEP+ platform.*
