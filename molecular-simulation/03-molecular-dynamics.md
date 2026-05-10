---

title: "Molecular Dynamics"
subtitle: "Hamilton's equations, symplectic integrators, thermostats, and the ergodic hypothesis"
---



Monte Carlo generates configurations by accepting or rejecting random proposals — a stochastic process that makes no reference to how a molecular system actually moves in time. Molecular dynamics takes the opposite approach: integrate the equations of motion and let the system evolve. The ergodic hypothesis then connects the resulting trajectory to statistical mechanics — if the simulation runs long enough and the dynamics are sufficiently mixing, the time average of any observable along the trajectory converges to its ensemble average. The result is a simulation that is simultaneously a movie of molecular motion and a sampler of the equilibrium distribution.

## Hamiltonian Mechanics

The dynamical foundation is Hamilton's formulation of classical mechanics. For a system of $N$ atoms, let $\mathbf{q} = (\mathbf{r}_1, \ldots, \mathbf{r}_N) \in \mathbb{R}^{3N}$ denote the positions and $\mathbf{p} = (\mathbf{p}_1, \ldots, \mathbf{p}_N) \in \mathbb{R}^{3N}$ the momenta. The **Hamiltonian** is the total energy:

> [!equation] 3.1
> $$H(\mathbf{q}, \mathbf{p}) = K(\mathbf{p}) + U(\mathbf{q}) = \sum_{i=1}^N \frac{|\mathbf{p}_i|^2}{2m_i} + U(\mathbf{r}_1, \ldots, \mathbf{r}_N),$$


where $K(\mathbf{p})$ is the kinetic energy and $U(\mathbf{q})$ is the potential energy function from Chapter 1. Hamilton's equations of motion are:

> [!equation] 3.2
> $$\dot{\mathbf{q}}_i = \frac{\partial H}{\partial \mathbf{p}_i} = \frac{\mathbf{p}_i}{m_i}, \qquad \dot{\mathbf{p}}_i = -\frac{\partial H}{\partial \mathbf{q}_i} = -\frac{\partial U}{\partial \mathbf{r}_i} \equiv \mathbf{F}_i(\mathbf{q}),$$


where $\mathbf{F}_i(\mathbf{q})$ is the force on atom $i$. The first equation identifies momentum as mass times velocity. The second is Newton's second law. Together, they define a flow on phase space $\Gamma = \mathbb{R}^{6N}$: given the initial state $(\mathbf{q}(0), \mathbf{p}(0))$, the equations determine the trajectory $(\mathbf{q}(t), \mathbf{p}(t))$ for all $t > 0$.

Two properties of Hamiltonian flow are central to everything that follows. First, the Hamiltonian is conserved: $dH/dt = 0$ along any trajectory. This follows directly from the equations by the chain rule, and it means that pure Hamiltonian MD evolves at constant energy — it samples the **microcanonical ensemble** (fixed $N$, $V$, $E$). Second, the flow preserves volume in phase space: **Liouville's theorem** states that $\partial \dot{q}_i / \partial q_i + \partial \dot{p}_i / \partial p_i = 0$ for all $i$, so the phase space volume element $d\mathbf{q}\, d\mathbf{p}$ is invariant under the Hamiltonian flow. This is not a numerical accident — it is a fundamental property of the underlying physics, and any good MD integrator must preserve it.

## The Ergodic Hypothesis

If MD samples the microcanonical ensemble — trajectories confined to a constant-energy hypersurface — how does it relate to the canonical ensemble of Chapter 1? The connection is the **ergodic hypothesis**.

> [!definition] 3.1 — The Ergodic Hypothesis
> A Hamiltonian system is **ergodic** with respect to the microcanonical measure $\mu_E$ (uniform measure on the energy surface $H = E$) if, for $\mu_E$-almost every initial condition, the time average of any integrable observable $A$ equals its microcanonical ensemble average:
>
> <Equation number="3.3">
> $$\lim_{T \to \infty} \frac{1}{T} \int_0^T A(\mathbf{q}(t), \mathbf{p}(t))\, dt = \langle A \rangle_{\mu_E}.$$
> </Equation>
>
> In practice, a system is ergodic when the trajectory visits all regions of the energy surface in proportion to their measure.


The ergodic hypothesis cannot be proven for realistic molecular systems — it is an assumption. Its justification is empirical: for many systems of physical interest, simulations initialized from different starting configurations converge to the same ensemble averages, which agrees with thermodynamic measurements. The assumption fails precisely in the situations described in Chapter 4: when energy barriers prevent the trajectory from crossing between metastable basins within the simulation time.

For canonical ensemble sampling — the context most relevant to chemistry and biology — the connection is more direct. Under weak ergodicity assumptions and for large $N$, the canonical distribution can be recovered from microcanonical dynamics via the equivalence of ensembles, or more practically by modifying the equations of motion to couple the system to a heat bath. The modified dynamics destroy strict Hamiltonian structure but produce a trajectory whose time averages converge to canonical ensemble averages.

## Numerical Integration: The Verlet Family

Hamilton's equations are a system of $6N$ ordinary differential equations. Except for toy models, they cannot be solved analytically. The trajectory must be discretized in time with a small step size $\Delta t$, replacing the continuous flow with a numerical map $(\mathbf{q}^n, \mathbf{p}^n) \to (\mathbf{q}^{n+1}, \mathbf{p}^{n+1})$ that approximates the exact solution over one timestep.

The simplest approach — forward Euler — is numerically disastrous for oscillatory Hamiltonian systems: it systematically injects energy, causing trajectories to spiral outward and eventually diverge. The correct integrators are **symplectic**: they preserve the geometric structure of Hamiltonian mechanics and, in consequence, the phase space volume.

The foundational symplectic integrator for MD is the **Verlet algorithm**.

> [!definition] 3.2 — The Verlet Algorithm
> The **Verlet algorithm** (Verlet, 1967) advances positions using the Taylor expansion of $\mathbf{r}_i(t \pm \Delta t)$ about $t$:
>
> $$\mathbf{r}_i(t + \Delta t) = 2\mathbf{r}_i(t) - \mathbf{r}_i(t - \Delta t) + \frac{\Delta t^2}{m_i}\, \mathbf{F}_i(\mathbf{q}(t)) + \mathcal{O}(\Delta t^4).$$
>
> This requires only positions at times $t$ and $t - \Delta t$, plus the forces at time $t$. The truncation error in positions is $\mathcal{O}(\Delta t^4)$ per step. Velocities are not explicitly required by the algorithm but can be estimated as $\mathbf{v}_i(t) \approx [\mathbf{r}_i(t+\Delta t) - \mathbf{r}_i(t-\Delta t)] / (2\Delta t)$ with error $\mathcal{O}(\Delta t^2)$.


The modern workhorse of MD is the **velocity Verlet algorithm**, which is algebraically equivalent to the Verlet algorithm but propagates positions and velocities together, making it more convenient for computing kinetic energy and applying thermostats:

> [!equation] 3.4
> $$\mathbf{r}_i(t + \Delta t) = \mathbf{r}_i(t) + \Delta t\, \mathbf{v}_i(t) + \frac{\Delta t^2}{2m_i}\, \mathbf{F}_i(t),$$


> [!equation] 3.5
> $$\mathbf{v}_i(t + \Delta t) = \mathbf{v}_i(t) + \frac{\Delta t}{2m_i}\left[\mathbf{F}_i(t) + \mathbf{F}_i(t + \Delta t)\right].$$


The velocity update requires the forces at both $t$ and $t + \Delta t$: in practice, the position step (3.4) is performed first, the new forces $\mathbf{F}_i(t + \Delta t)$ are computed, and then the velocity step (3.5) is completed. One force evaluation per step is required, which is the dominant computational cost for most MD simulations.

## Symplecticity and Long-Time Stability

The distinction between symplectic and non-symplectic integrators is decisive for molecular simulation. A **symplectic map** preserves the symplectic 2-form $\omega = \sum_i d\mathbf{q}_i \wedge d\mathbf{p}_i$ — the mathematical object that underpins Liouville's theorem. Practically, this means the integrator maps phase space volume-preservingly, exactly as the true Hamiltonian flow does.

The consequences for long simulations are profound. A non-symplectic integrator (like a standard Runge-Kutta method) typically either gains or loses energy at every step; over millions of steps, this secular drift accumulates and the simulation explores a systematically wrong region of phase space. A symplectic integrator like velocity Verlet does not conserve the true Hamiltonian $H$ exactly — it conserves a **shadow Hamiltonian** $\tilde{H}$ that differs from $H$ by a term of order $\Delta t^2$ — but the deviation $|H - \tilde{H}|$ remains bounded for all time, rather than growing without limit. In practice, this means that a properly tuned velocity Verlet simulation shows energy fluctuations around a stable mean with no systematic drift, even for microsecond-scale trajectories.

> [!remark] 3.1
> The leapfrog integrator — a common variant where velocities are updated at half-integer timesteps, $\mathbf{v}_i(t + \Delta t/2) = \mathbf{v}_i(t - \Delta t/2) + (\Delta t / m_i)\mathbf{F}_i(t)$, followed by $\mathbf{r}_i(t + \Delta t) = \mathbf{r}_i(t) + \Delta t\, \mathbf{v}_i(t + \Delta t/2)$ — is algebraically equivalent to velocity Verlet and equally symplectic. It is the default integrator in GROMACS. The two algorithms produce identical trajectories to within floating-point precision.


## Choosing the Timestep

The timestep $\Delta t$ must be small enough that the fastest motions in the system are well resolved. The fundamental constraint is the **Nyquist criterion**: to accurately integrate a vibration with period $\tau_{\min}$, one needs $\Delta t \lesssim \tau_{\min} / 10$.

In biomolecular simulations, the fastest motions are bond stretching vibrations (O–H, N–H bonds at ~3000 cm$^{-1}$), with periods of roughly 10 fs. This dictates $\Delta t \leq 1$ fs for a fully flexible model. Bending motions (period ~30 fs) and dihedral rotations (period ~100–1000 fs) are slower and pose no additional constraint once bond stretching is resolved.

In practice:

- **$\Delta t = 1$ fs**: Standard for all-atom simulations without constraints.
- **$\Delta t = 2$ fs**: Achievable with constraint algorithms (SHAKE, LINCS) that freeze bond lengths involving hydrogen, effectively removing the fastest degrees of freedom. Most production biomolecular simulations run at this timestep.
- **$\Delta t = 4$ fs**: Possible with heavy hydrogen mass repartitioning, where the mass of hydrogen atoms is transferred to their bonded heavy atom, slowing down H vibrations.
- **$\Delta t \sim 10$–$100$ fs**: Typical for coarse-grained models where fast atomistic degrees of freedom have been integrated out.

Too large a timestep causes integration instability: the trajectory diverges, positions blow up, and the simulation crashes. Too small a timestep is computationally wasteful. The optimal choice is the largest $\Delta t$ that keeps the energy fluctuations acceptably small.

## Controlling Temperature: Thermostats

Pure velocity Verlet integrates the microcanonical ensemble. To simulate at constant temperature — the canonical ensemble — the equations of motion must be modified to couple the system to a heat bath. The mechanism is a **thermostat**.

The instantaneous temperature in an MD simulation is defined through the equipartition theorem:

> [!equation] 3.6
> $$T_{\mathrm{inst}} = \frac{2 K(\mathbf{p})}{N_f k_B} = \frac{1}{N_f k_B} \sum_{i=1}^N \frac{|\mathbf{p}_i|^2}{m_i},$$


where $N_f = 3N - N_c$ is the number of degrees of freedom after removing $N_c$ constraints (typically 3 for fixed center-of-mass momentum). At thermal equilibrium, $\langle T_{\mathrm{inst}} \rangle = T$.

**Velocity rescaling** is the simplest thermostat: after each step, multiply all velocities by the factor $\lambda = \sqrt{T / T_{\mathrm{inst}}}$. This forces $T_{\mathrm{inst}} = T$ at every step. It is fast and stable, but it generates an incorrect distribution of kinetic energies — the kinetic energy does not fluctuate as it should in the canonical ensemble. It is useful for equilibration but not for production runs where the correct ensemble is required.

> [!definition] 3.3 — The Nosé–Hoover Thermostat
> The **Nosé–Hoover thermostat** (Nosé 1984, Hoover 1985) extends the system by a fictitious degree of freedom $\xi$ with associated "mass" $Q$ (units of energy $\times$ time$^2$), representing the heat bath. The extended equations of motion are:
>
> <Equation number="3.7">
> $$\dot{\mathbf{r}}_i = \frac{\mathbf{p}_i}{m_i}, \quad \dot{\mathbf{p}}_i = \mathbf{F}_i - \xi\, \mathbf{p}_i, \quad \dot{\xi} = \frac{1}{Q}\left(\sum_{i=1}^N \frac{|\mathbf{p}_i|^2}{m_i} - N_f k_B T\right).$$
> </Equation>
>
> The thermostat coordinate $\xi$ acts as a velocity-dependent friction: when the system is too hot ($\sum |\mathbf{p}_i|^2 / m_i > N_f k_B T$), $\dot{\xi} > 0$ and the friction term decelerates the particles; when too cold, $\dot{\xi} < 0$ and particles are accelerated. These equations of motion have the canonical distribution as their stationary distribution, generating the correct $NVT$ ensemble.


The "mass" $Q$ of the thermostat controls the coupling between the bath and the system. A small $Q$ produces rapid temperature fluctuations and fast equilibration but can cause ringing (oscillatory behavior in the temperature). A large $Q$ gives slow, gentle coupling. The physically motivated choice is $Q = N_f k_B T / \omega^2$, where $\omega$ is the characteristic frequency of the thermostat oscillation. In practice, $Q$ is tuned so that the temperature autocorrelation time is comparable to the timescale of the slowest motions of interest.

A practically simpler and increasingly popular alternative is **Langevin dynamics**, which models the heat bath as a random force and a velocity-dependent friction:

> [!equation] 3.8
> $$m_i \ddot{\mathbf{r}}_i = \mathbf{F}_i(\mathbf{q}) - \gamma m_i \dot{\mathbf{r}}_i + \boldsymbol{\eta}_i(t),$$


where $\gamma$ is the friction coefficient (units of inverse time) and $\boldsymbol{\eta}_i(t)$ is a Gaussian white noise with zero mean and covariance $\langle \boldsymbol{\eta}_i(t) \cdot \boldsymbol{\eta}_j(t') \rangle = 2\gamma m_i k_B T\, \delta_{ij}\, \delta(t - t')$. The relationship between the friction amplitude $\gamma m_i$ and the noise amplitude $\sqrt{2\gamma m_i k_B T}$ is the **fluctuation-dissipation theorem**: the noise that heats the system must be matched by friction that cools it, at exactly the ratio that gives the target temperature.

> [!remark] 3.2
> Langevin dynamics has a double role. As a thermostat, it generates the canonical ensemble correctly for any $\gamma > 0$. As a sampling method, the stochastic kicks randomize particle velocities and can help the system escape from local kinetic traps that would persist in purely deterministic dynamics. At high friction ($\gamma$ large), Langevin dynamics approximates overdamped Brownian motion — the velocity equilibrates nearly instantaneously and the effective dynamics is a gradient flow on $U(\mathbf{q})$ perturbed by noise. Many enhanced sampling methods (Chapter 5) are built on top of Langevin dynamics.


## Controlling Pressure: Barostats

For NPT simulations, the pressure must be controlled in addition to the temperature. A **barostat** achieves this by allowing the simulation box dimensions to fluctuate.

The instantaneous pressure in an MD simulation is related to the **virial**:

$$P_{\mathrm{inst}} = \frac{1}{3V}\left[N_f k_B T + \mathbf{W}\right], \quad \mathbf{W} = \sum_{i<j} \mathbf{r}_{ij} \cdot \mathbf{F}_{ij},$$

where $\mathbf{W}$ is the virial, summed over all interacting pairs. The Berendsen barostat rescales the box (and coordinates) every step to drive $P_{\mathrm{inst}}$ toward the target pressure $P_0$, with a time constant $\tau_P$. Like the Berendsen thermostat, it is stable and fast but does not generate the correct NPT ensemble.

The **Parrinello–Rahman barostat** (Parrinello and Rahman, 1980) treats the box vectors as dynamic variables with their own equations of motion, coupled to the instantaneous pressure tensor. It generates the correct isothermal-isobaric ensemble and also allows the shape of the simulation cell to fluctuate, enabling simulation of systems under anisotropic stress — essential for studying solid-state phase transitions and membranes.

## MD versus Monte Carlo

Molecular dynamics and Monte Carlo are complementary tools, each with distinct strengths and limitations. The choice between them depends on the property of interest and the system under study.

**What MD offers that MC cannot**: MD generates a physically meaningful trajectory — positions and velocities as functions of time. This makes MD the only method for computing **dynamical properties**: diffusion coefficients, viscosity, reaction rates, relaxation times, spectroscopic lineshapes, and any other quantity that depends on correlations across time. The velocity autocorrelation function, from which the diffusion coefficient is extracted via the Green-Kubo relation, requires a time-ordered trajectory. MC generates an unordered set of configurations and cannot access such quantities.

**What MC offers that MD cannot**: MC proposes moves that have no physical analogue. The grand canonical ensemble — where particles are inserted and deleted — is natural for MC but requires modifying the equations of motion in a non-physical way for MD. Non-local moves (cluster moves, configurational-bias growth of chains, pivot moves for polymers) can drastically accelerate sampling of specific degrees of freedom in ways that have no MD counterpart. For purely equilibrium properties of complex topologies — ring polymers, branched molecules, lattice models — MC is often far more efficient.

> [!remark] 3.3
> The choice is not always either/or. **Hybrid Monte Carlo** (Duane et al., 1987, now more commonly called Hamiltonian Monte Carlo or HMC) uses a short MD trajectory as a trial move in a Metropolis–Hastings scheme. The MD trajectory proposes a new configuration that is far from the current one and correlated in a physically meaningful way; the Metropolis step then accepts or rejects the entire trajectory based on the energy difference, correcting for the discrete-time integration error. HMC is exact (no timestep bias), highly efficient for smooth, high-dimensional distributions, and is the basis of many modern machine learning samplers. It illustrates that the boundary between MD and MC is not fundamental — they are both strategies for navigating the Boltzmann distribution.


## The Force Calculation Problem

The dominant computational cost in MD is evaluating the forces $\mathbf{F}_i(\mathbf{q}) = -\partial U / \partial \mathbf{r}_i$ at each timestep. For a system of $N$ atoms with pairwise interactions, the naive cost is $\mathcal{O}(N^2)$ per step — one force evaluation for each pair. Several algorithmic strategies reduce this:

**Cutoffs and neighbor lists**: Short-range interactions (van der Waals, short-range part of electrostatics) are truncated at a cutoff radius $r_c \sim 10$–$12$ Å. A **Verlet neighbor list** precomputes all pairs within $r_c + r_{\mathrm{skin}}$ and is rebuilt only every $\sim 20$ steps, reducing the cost per step to $\mathcal{O}(N)$.

**Particle mesh Ewald** (PME): Long-range electrostatic interactions cannot simply be truncated — doing so introduces artifacts of the same order as the true long-range contribution. PME splits the Coulomb sum into a short-range part handled in real space and a long-range part evaluated efficiently on a Fourier grid, giving $\mathcal{O}(N \log N)$ scaling.

**Multiple time stepping**: Fast motions (bond stretching, angle bending) are updated every step; slow motions (long-range electrostatics, nonbonded interactions beyond a short cutoff) are updated every $k$ steps. This can give a factor of $k$ speedup at the cost of introducing a small integration error.

These algorithmic developments — developed over the 1980s and 1990s and now standard in all major MD packages — enable simulations of $N \sim 10^5$–$10^6$ atoms for microsecond timescales on modern hardware.

---

*Molecular dynamics was introduced by Alder and Wainwright (1957, 1959) for hard-sphere fluids, where the equations of motion reduce to event-driven billiard collisions. The continuous-force MD algorithm with the Verlet integrator was introduced for a Lennard-Jones fluid by Verlet (1967). The velocity Verlet form was derived by Swope, Andersen, Berens, and Wilson (1982). The Nosé thermostat was derived from an extended Lagrangian by Nosé (1984) and reformulated in a convenient non-Hamiltonian form by Hoover (1985); the resulting equations are now universally called Nosé–Hoover dynamics. The Parrinello–Rahman barostat is due to Parrinello and Rahman (1980, 1981). Hybrid Monte Carlo (Hamiltonian Monte Carlo) was introduced by Duane, Kennedy, Pendleton, and Roweth (1987) in lattice field theory and later imported into computational statistics by Neal (2011). Particle mesh Ewald was introduced by Darden, York, and Pedersen (1993). The textbooks by Frenkel and Smit (2002) and Allen and Tildesley (1987, 2017) provide comprehensive treatments of MD algorithms and their practical implementation.*
