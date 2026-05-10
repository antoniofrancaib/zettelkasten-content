---

title: "Rare Events"
subtitle: "Transition state theory, the committor, transition path sampling, and the statistics of infrequent transitions"
---



The methods of Chapters 5 and 6 compute free energy differences. They do not compute rates. Knowing that state $B$ is 5 kcal/mol more stable than state $A$ tells you nothing about how quickly the system converts between them — whether the transition happens on the nanosecond timescale or the geological. Rate constants are independent thermodynamic quantities, determined by the height and shape of the barrier separating the states, not just by the free energy difference between them. This chapter develops the theoretical and computational machinery for computing rates and characterizing the rare reactive events that carry the system from one metastable state to another.

## The Rate Problem

A molecular system with two metastable states $A$ and $B$ reaches a steady state in which the net flux from $A$ to $B$ equals the net flux from $B$ to $A$: detailed balance. In the long-time limit, the fraction of time spent in each state is determined by thermodynamics alone — by the free energy difference $\Delta F$. But the timescale on which this equilibrium is approached is kinetic: it is set by the rate constants $k_{A \to B}$ and $k_{B \to A}$, related by detailed balance to $\Delta F$ via

$$\frac{k_{A \to B}}{k_{B \to A}} = e^{-\beta \Delta F_{A \to B}}.$$

Measuring $\Delta F$ without measuring the rates, or vice versa, gives only half the picture. A ligand may bind tightly ($\Delta F \ll 0$) but dissociate rapidly ($k_{\mathrm{off}}$ large), giving poor in-vivo efficacy despite high thermodynamic affinity. Conversely, a weakly binding compound may have a long residence time — slow $k_{\mathrm{off}}$ — making it pharmacologically more effective than its $K_d$ suggests. Rates and free energies are complementary, not redundant.

## Transition State Theory

The earliest quantitative theory of reaction rates is **transition state theory** (TST), developed by Eyring and by Evans and Polanyi in 1935. TST assumes that any trajectory passing through a dividing surface $\Sigma^{\ddagger}$ separating $A$ from $B$ — the transition state — commits to $B$ without recrossing $\Sigma^{\ddagger}$.

> [!definition] 7.1 — Transition State Theory
> Let $\Sigma^{\ddagger}$ be a hypersurface in configuration space separating states $A$ and $B$, chosen to pass through the saddle point of the free energy. The **TST rate** is:
>
> <Equation number="7.1">
> $$k_{A \to B}^{\mathrm{TST}} = \frac{1}{2} \langle |v_{\perp}| \rangle_{\Sigma^{\ddagger}} \cdot \frac{e^{-\beta F(\Sigma^{\ddagger})}}{Z_A / Z},$$
> </Equation>
>
> where $\langle |v_{\perp}| \rangle_{\Sigma^{\ddagger}}$ is the mean speed perpendicular to $\Sigma^{\ddagger}$ at the transition state, $e^{-\beta F(\Sigma^{\ddagger})} / (Z_A/Z)$ is the equilibrium probability of being on $\Sigma^{\ddagger}$ relative to being in $A$, and the factor $1/2$ selects trajectories moving from $A$ to $B$. For a one-dimensional barrier of height $\Delta F^{\ddagger}$ and harmonic wells:
>
> $$k_{A \to B}^{\mathrm{TST}} = \frac{\omega_A}{2\pi}\, e^{-\beta \Delta F^{\ddagger}},$$
>
> where $\omega_A$ is the angular frequency of oscillation in state $A$. This is the Arrhenius formula with a TST prefactor.


TST is an upper bound on the true rate: any recrossing of $\Sigma^{\ddagger}$ (trajectories that cross and return without committing to $B$) reduces the actual rate below the TST estimate. The exact rate is related to the TST rate by the **transmission coefficient** $\kappa \in (0, 1]$:

$$k_{A \to B} = \kappa\, k_{A \to B}^{\mathrm{TST}}.$$

The transmission coefficient accounts for recrossing, non-equilibrium effects at the barrier, and the difference between the chosen dividing surface and the true dynamical bottleneck. For reactions in solution, $\kappa$ can be as low as $10^{-2}$–$10^{-3}$ due to solvent friction — the bath re-randomizes momenta so frequently that many trajectories recross the barrier multiple times before committing. Computing $\kappa$ requires knowledge of the actual dynamics near the transition state and is handled by the reactive flux formalism.

> [!remark] 7.1
> The choice of dividing surface $\Sigma^{\ddagger}$ affects the TST rate and the transmission coefficient but not the product $\kappa \cdot k^{\mathrm{TST}}$, which is the true rate. An optimal dividing surface minimizes recrossing (maximizes $\kappa$), which means it passes through the **transition state ensemble** — the set of configurations for which the committor $q_+(\mathbf{q}) = 1/2$. Identifying this surface without prior knowledge of the committor is the central difficulty of rate calculation and motivates the transition path methods below.


## The Committor and Transition Path Theory

The theoretical foundation for rare event methods is the **committor function**, first emphasized in the context of transition path theory by Weinan E and Vanden-Eijnden (2006).

> [!definition] 7.2 — The Committor
> The **forward committor** $q_+(\mathbf{q})$ is the probability that a trajectory initiated from configuration $\mathbf{q}$ with momenta drawn from the Maxwell–Boltzmann distribution reaches state $B$ before state $A$:
>
> $$q_+(\mathbf{q}) = \mathbb{P}\!\left[\text{reaches } B \text{ before } A \mid \mathbf{q}(0) = \mathbf{q},\; \mathbf{p}(0) \sim \pi_T\right].$$
>
> The committor satisfies boundary conditions $q_+(\mathbf{q}) = 0$ for $\mathbf{q} \in A$ and $q_+(\mathbf{q}) = 1$ for $\mathbf{q} \in B$. In the overdamped (Brownian) limit, $q_+$ satisfies the backward Kolmogorov equation:
>
> <Equation number="7.2">
> $$-\nabla U \cdot \nabla q_+ + k_BT\, \nabla^2 q_+ = 0, \quad \mathbf{q} \notin A \cup B,$$
> </Equation>
>
> with the given boundary conditions. The **transition state ensemble** is the iso-surface $\{q_+ = 1/2\}$ — configurations equally likely to reach $B$ or $A$ next.


**Transition path theory** (TPT) uses the committor to give exact expressions for all kinetic quantities: the rate, the reactive current, and the distribution of transition paths. The reactive current $\mathbf{J}(\mathbf{q})$ — the net flux of reactive trajectories passing through configuration $\mathbf{q}$ per unit time — is:

> [!equation] 7.3
> $$\mathbf{J}(\mathbf{q}) = k_BT\, \pi(\mathbf{q})\, \nabla q_+(\mathbf{q}),$$


where $\pi(\mathbf{q}) \propto e^{-\beta U(\mathbf{q})}$ is the equilibrium distribution. The rate is the total flux through any surface separating $A$ from $B$:

$$k_{A \to B} = \frac{1}{Z_A} \int_\Sigma \mathbf{J}(\mathbf{q}) \cdot \hat{n}\, d\sigma,$$

where $\hat{n}$ is the outward normal and $Z_A = \int_A e^{-\beta U} d\mathbf{q}$. TPT is exact — it makes no assumption about the shape of the barrier, the choice of dividing surface, or the degree of recrossing. The difficulty is that computing $\mathbf{J}$ requires knowing $q_+(\mathbf{q})$, which is as hard as solving the full $3N$-dimensional backward Kolmogorov equation.

## Transition Path Sampling

Rather than computing the committor analytically, **transition path sampling** (TPS, Dellago, Bolhuis, Csajka, and Chandler, 1998) generates an ensemble of reactive trajectories — trajectories that start in $A$ and end in $B$ — using a Monte Carlo walk in trajectory space.

A reactive trajectory is a sequence of phase space points $\{(\mathbf{q}(t), \mathbf{p}(t)) : 0 \leq t \leq T\}$ satisfying the equations of motion and the boundary conditions that the trajectory begins in $A$ and ends in $B$. The **path ensemble** is the set of all such trajectories weighted by the path action. TPS samples from this ensemble by proposing new trajectories through **shooting moves**: from a configuration on an existing reactive trajectory, a new trajectory is initiated with slightly perturbed momenta, then run forward and backward in time. If the perturbed trajectory still starts in $A$ and ends in $B$, it is accepted; otherwise it is rejected. The procedure is exactly a Metropolis Monte Carlo in path space.

**Aimless shooting** (Peters and Trout, 2006) refines this: the shooting point is selected at random from the existing path, and momenta are fully randomized (not perturbed), removing the dependence on the previous trajectory's momenta entirely. This produces a more efficient exploration of the transition state ensemble and allows the committor to be estimated by running short trajectories from the shooting points and asking how often they reach $B$ versus $A$.

TPS offers a unique window into mechanism: it reveals which degrees of freedom distinguish reactive from non-reactive trajectories, often uncovering reaction coordinates that differ from the obvious geometric choice. Its main limitation is that the shooting point must be near the transition state — if the initial trajectory was obtained by brute force (which is rarely feasible) or by a biased path generation method — and it provides rates only via additional analysis.

## Forward Flux Sampling

**Forward flux sampling** (FFS, Allen, Frenkel, and ten Wolde, 2006) computes the rate directly without requiring a reactive trajectory to seed the calculation. The method defines a sequence of interfaces $\lambda_0, \lambda_1, \ldots, \lambda_n$ in order parameter space, with $\lambda_0$ at the boundary of $A$ and $\lambda_n$ at the boundary of $B$.

> [!definition] 7.3 — Forward Flux Sampling
> FFS computes the rate as a product of conditional probabilities:
>
> <Equation number="7.4">
> $$k_{A \to B} = \Phi_{A,0} \cdot \prod_{i=0}^{n-1} P(\lambda_{i+1} \mid \lambda_i),$$
> </Equation>
>
> where $\Phi_{A,0}$ is the flux of trajectories crossing the first interface $\lambda_0$ from $A$ (measured by a brute-force simulation in $A$, which is feasible since the system never leaves $A$), and $P(\lambda_{i+1} \mid \lambda_i)$ is the conditional probability that a trajectory crossing $\lambda_i$ subsequently crosses $\lambda_{i+1}$ before returning to $A$. Each conditional probability is estimated by launching an ensemble of short trial trajectories from configurations stored on interface $\lambda_i$ and recording the fraction that reach $\lambda_{i+1}$.


The power of FFS is that each factor in the product (7.4) can be estimated with modest computational effort, because the probability of crossing one interface given that the previous interface was crossed is typically not exponentially small — the exponential rarity of the overall event has been distributed across $n$ tractable calculations. If each $P(\lambda_{i+1} \mid \lambda_i) \approx 0.1$ and there are 20 interfaces, the overall rate is suppressed by $10^{-20}$ — an event that would take forever by brute force — but each factor is estimated from a few hundred short trajectories.

The factored form (7.4) requires that the order parameter $\lambda$ is a reasonable approximation to the committor: if the interfaces are not ordered by their $q_+$ values (if some configurations at $\lambda_i$ are more likely to return to $A$ than configurations at $\lambda_{i-1}$), the product overestimates the crossing probabilities and the rate is wrong. FFS inherits the CV problem from all order-parameter-based methods.

> [!remark] 7.2
> **Milestoning** (Faradjian and Elber, 2004) is a closely related method that partitions configuration space by hypersurfaces called **milestones** and computes the rate and mean first-passage time by estimating the statistics of transitions between adjacent milestones. Unlike FFS, milestoning can handle arbitrary milestone geometries (not just contours of a single order parameter) and integrates naturally with the local equilibrium approximation at each milestone to give not just the rate but the full first-passage time distribution. In the limit of many finely spaced milestones, milestoning approaches the exact rate regardless of the quality of the reaction coordinate.


## Finding the Transition Path: String and NEB Methods

The methods above sample or count reactive trajectories. A complementary goal is to **find** the most probable transition path — the minimum free energy path (MFEP) connecting $A$ to $B$ — without running many dynamics simulations.

The **nudged elastic band** (NEB) method represents a path as a chain of images connected by fictitious springs. Each image is evolved under the combined influence of the true force (perpendicular to the path) and the spring force (along the path), until the chain relaxes to the minimum energy path on the potential energy surface. NEB finds the MEP of $U(\mathbf{q})$ — the path of steepest descent from the saddle point — but not the free energy path, which requires integrating over the orthogonal degrees of freedom.

The **string method** (Weinan E, Ren, and Vanden-Eijnden, 2002) evolves a curve in configuration space under the free energy gradient, finding the minimum free energy path. In the zero-temperature limit, the string method finds the MEP of $U$; at finite temperature, it finds the MFEP of $F$, accounting for entropy. The **finite-temperature string method** alternates between evolving images under the mean force $\langle -\nabla U \rangle_{\mathbf{q}_k}$ (estimated by running short constrained simulations at each image) and reparametrizing the string to maintain equal arc-length spacing. The result is a one-dimensional free energy profile $F(s)$ along the reaction path and a set of configurations representative of each stage of the transition.

> [!remark] 7.3
> The committor can be estimated computationally by running many short trajectories from a trial configuration and computing the fraction that reach $B$ before $A$ — the **committor test** or **shooting analysis**. If a proposed reaction coordinate $\xi$ is the true committor, then configurations sampled uniformly from the surface $\{\xi = 1/2\}$ should all have committor value close to 1/2. If the distribution of committor values at $\xi = 1/2$ is broad or bimodal, $\xi$ is not the committor — important slow degrees of freedom are missing. Machine learning approaches (neural committor networks) now train a neural network $q_\theta(\mathbf{q})$ to minimize a loss function derived from the backward Kolmogorov equation, directly approximating the committor on simulation data. These methods are discussed further in Chapter 10.


## A Unified Picture

The rare event methods of this chapter — TST, TPS, FFS, milestoning, and string methods — differ in what they compute and what they assume, but they share a common theoretical substrate: the committor. TST approximates the rate by assuming that $\Sigma^{\ddagger}$ is the committor iso-surface $\{q_+ = 1/2\}$ and that there is no recrossing. TPS samples the distribution of reactive trajectories, which is concentrated near the committor iso-surface. FFS approximates the committor by a sequence of order-parameter interfaces. Milestoning makes the committor's role explicit through the local equilibrium assumption at each milestone. The string method finds the MFEP, which is the ridge of the free energy surface that the committor iso-surface must cross.

Accurate rate calculations thus require, in one form or another, a good approximation to the committor — the single function that encodes the entire kinetics of the $A \to B$ transition. Finding it efficiently, without solving the $3N$-dimensional backward Kolmogorov equation, remains one of the central challenges of computational molecular science and motivates the machine-learning approaches of Chapter 10.

---

*Transition state theory was developed simultaneously and independently by Eyring (1935) and by Evans and Polanyi (1935). The reactive flux formalism and the transmission coefficient were formalized by Chandler (1978). Transition path theory in its modern form, with the committor and reactive current as central objects, was developed by Weinan E and Vanden-Eijnden (2006); an earlier related framework was given by Hänggi and Borkovec (1990). Transition path sampling was introduced by Dellago, Bolhuis, Csajka, and Chandler (1998); the aimless shooting variant by Peters and Trout (2006). Forward flux sampling was introduced by Allen, Frenkel, and ten Wolde (2006). Milestoning was introduced by Faradjian and Elber (2004) and placed on a rigorous theoretical foundation by Vanden-Eijnden and Venturoli (2009). The nudged elastic band method was developed by Jónsson, Mills, and Jacobsen (1998). The string method was introduced by Weinan E, Ren, and Vanden-Eijnden (2002) and extended to finite temperature by the same group (2007). Reviews by Bolhuis and Dellago (2010) and Escobedo, Borrero, and Araque (2009) cover path sampling and interface methods comprehensively.*
