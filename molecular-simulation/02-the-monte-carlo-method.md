---

title: "The Monte Carlo Method"
subtitle: "Metropolis–Hastings, detailed balance, Markov chains, and the art of sampling by random walk"
---



Chapter 1 established that ensemble averages are intractable integrals and that the path forward is importance sampling: generate configurations distributed according to the Boltzmann distribution $\pi$, then estimate averages as arithmetic means. This chapter delivers the algorithm that generates those configurations. The method — introduced by Metropolis and colleagues in 1953 — is simple enough to state in five lines of pseudocode. Understanding *why* it works requires thinking carefully about convergence of random processes, the geometry of configuration space, and a condition called detailed balance that sits at the foundation of equilibrium statistical mechanics. Both the simplicity and the depth matter for using the method well.

## The Failure of Naive Sampling

The original Monte Carlo idea — draw $M$ independent samples uniformly at random from the integration domain, then average the integrand — fails completely for high-dimensional Boltzmann distributions.

The problem is concentration. The Boltzmann weight $e^{-\beta U(\mathbf{r})}$ is negligible for the overwhelming majority of configurations in $\mathbb{R}^{3N}$. A configuration drawn uniformly from the space will almost certainly contain atomic overlaps — pairs of atoms closer than their van der Waals radii — because close contacts are geometrically unavoidable in high-dimensional random packings. For a hard-core potential, such a configuration has $U = +\infty$ and Boltzmann weight exactly zero. For more realistic potentials, the weight is exponentially suppressed by the large repulsive energy. The integrand $A(\mathbf{r}) e^{-\beta U(\mathbf{r})}$ is effectively zero almost everywhere, and essentially all computational effort is wasted evaluating the integrand in regions that contribute nothing to the integral.

This is not a problem that more samples can fix. The fraction of configuration space with non-negligible Boltzmann weight does not grow with the number of samples; it remains vanishingly small regardless of $M$. What is needed is a method that concentrates its evaluations where the Boltzmann weight is large — a method that navigates toward the typical configurations rather than searching for them blindly.

## Markov Chains

The solution is to construct a *dependent* sequence of configurations, each obtained by making a small random modification to the previous one, with the modification designed so that the sequence explores configuration space in proportion to the Boltzmann weight.

> [!definition] 2.1 — Markov Chain
> A **Markov chain** is a sequence of random variables $\mathbf{R}^{(0)}, \mathbf{R}^{(1)}, \mathbf{R}^{(2)}, \ldots$ taking values in configuration space $\Omega$, satisfying the **Markov property**: the conditional distribution of $\mathbf{R}^{(t+1)}$ depends only on $\mathbf{R}^{(t)}$, not on the earlier history:
>
> $$P(\mathbf{R}^{(t+1)} \in A \mid \mathbf{R}^{(0)}, \mathbf{R}^{(1)}, \ldots, \mathbf{R}^{(t)}) = P(\mathbf{R}^{(t+1)} \in A \mid \mathbf{R}^{(t)}).$$
>
> The dynamics are entirely encoded in the **transition kernel** $T(\mathbf{r}' \mid \mathbf{r})$: the probability density of moving to $\mathbf{r}'$ when currently at $\mathbf{r}$.


A Markov chain is useful as a sampling algorithm only if its distribution at time $t$ converges to $\pi$ as $t \to \infty$, regardless of where it started. A distribution $\pi$ with this property is called the **stationary distribution** of the chain. The convergence condition is:

> [!equation] 2.1
> $$\pi(\mathbf{r}') = \int_\Omega T(\mathbf{r}' \mid \mathbf{r})\, \pi(\mathbf{r})\, d\mathbf{r}.$$


This says that if the chain is started in $\pi$, it remains in $\pi$ after one step — $\pi$ is preserved. The challenge is to *design* a transition kernel $T$ for which $\pi$ (the Boltzmann distribution) is stationary.

## Detailed Balance

A sufficient condition for stationarity — and the one used in almost all molecular simulation algorithms — is **detailed balance**.

> [!definition] 2.2 — Detailed Balance
> A transition kernel $T$ satisfies **detailed balance** with respect to $\pi$ if, for all pairs of configurations $\mathbf{r}, \mathbf{r}' \in \Omega$:
>
> <Equation number="2.2">
> $$\pi(\mathbf{r})\, T(\mathbf{r}' \mid \mathbf{r}) = \pi(\mathbf{r}')\, T(\mathbf{r} \mid \mathbf{r}').$$
> </Equation>
>
> This condition asserts that the probability flux from $\mathbf{r}$ to $\mathbf{r}'$ — the rate at which the chain transitions from $\mathbf{r}$ to $\mathbf{r}'$ in stationarity — exactly equals the reverse flux from $\mathbf{r}'$ to $\mathbf{r}$. The chain is in *microscopic reversibility*: every transition is balanced by its time-reverse.


To verify that detailed balance implies stationarity, integrate both sides of (2.2) over $\mathbf{r}$:

$$\int_\Omega \pi(\mathbf{r})\, T(\mathbf{r}' \mid \mathbf{r})\, d\mathbf{r} = \pi(\mathbf{r}') \int_\Omega T(\mathbf{r} \mid \mathbf{r}')\, d\mathbf{r} = \pi(\mathbf{r}'),$$

where the last step uses the normalization $\int T(\mathbf{r} \mid \mathbf{r}')\, d\mathbf{r} = 1$ (the chain must go somewhere). This is exactly the stationarity condition (2.1).

The physical intuition behind detailed balance is direct: it is the condition of microscopic reversibility at equilibrium. In an equilibrium molecular system, the rate at which the system transitions from configuration $\mathbf{r}$ to $\mathbf{r}'$ equals the rate of the reverse transition. Designing an algorithm that satisfies this condition is therefore equivalent to designing an algorithm that respects the symmetry of equilibrium.

> [!remark] 2.1
> Detailed balance is sufficient but not necessary for stationarity. There exist valid sampling algorithms — notably those based on the **skew-detailed balance** condition — that do not satisfy (2.2) but still have $\pi$ as their stationary distribution. These algorithms can sometimes mix faster by introducing a net probability flux that circulates through configuration space, analogous to the flow of an incompressible fluid. However, the Metropolis–Hastings algorithm satisfies the stronger detailed balance condition, and we will work within that framework throughout this chapter.


## The Metropolis–Hastings Algorithm

Detailed balance provides the design criterion for a sampling kernel. The Metropolis–Hastings algorithm implements it by decomposing each step into a *proposal* followed by an *acceptance decision*.

Choose any convenient **proposal distribution** $Q(\mathbf{r}' \mid \mathbf{r})$ — a distribution over new configurations $\mathbf{r}'$ given the current configuration $\mathbf{r}$. Then accept or reject the proposal with a probability chosen specifically to enforce detailed balance:

> [!equation] 2.3
> $$\alpha(\mathbf{r} \to \mathbf{r}') = \min\!\left(1,\, \frac{\pi(\mathbf{r}')\, Q(\mathbf{r} \mid \mathbf{r}')}{\pi(\mathbf{r})\, Q(\mathbf{r}' \mid \mathbf{r})}\right).$$


This is the **Metropolis–Hastings acceptance probability**. For the Boltzmann distribution $\pi(\mathbf{r}) \propto e^{-\beta U(\mathbf{r})}$, the normalization constant $Z$ cancels in the ratio $\pi(\mathbf{r}') / \pi(\mathbf{r})$:

$$\frac{\pi(\mathbf{r}')}{\pi(\mathbf{r})} = \exp\!\big(-\beta\,[U(\mathbf{r}') - U(\mathbf{r})]\big) = e^{-\beta\,\Delta U},$$

where $\Delta U = U(\mathbf{r}') - U(\mathbf{r})$ is the energy change of the proposed move. For a **symmetric proposal** — one satisfying $Q(\mathbf{r}' \mid \mathbf{r}) = Q(\mathbf{r} \mid \mathbf{r}')$ — the proposal ratio cancels and the acceptance probability reduces to:

> [!equation] 2.4
> $$\alpha(\mathbf{r} \to \mathbf{r}') = \min\!\left(1,\, e^{-\beta\, \Delta U}\right).$$


This is the original **Metropolis criterion**. It has an immediate physical interpretation: moves that lower the energy ($\Delta U < 0$) are always accepted. Moves that raise the energy ($\Delta U > 0$) are accepted with probability $e^{-\beta \Delta U}$, which decreases exponentially with the magnitude of the energy increase. At low temperature ($\beta$ large), only energetically favorable moves are accepted. At high temperature ($\beta$ small), uphill moves are accepted more readily, and the system explores more of configuration space at the cost of less discrimination between high- and low-energy regions.

> [!definition] 2.3 — The Metropolis–Hastings Algorithm
> Given a symmetric proposal distribution $Q$ and current configuration $\mathbf{r}^{(t)}$:
>
> 1. **Propose**: Draw a candidate $\mathbf{r}' \sim Q(\,\cdot \mid \mathbf{r}^{(t)})$.
> 2. **Evaluate**: Compute $\Delta U = U(\mathbf{r}') - U(\mathbf{r}^{(t)})$.
> 3. **Accept**: Draw $u \sim \mathrm{Uniform}[0,1]$. If $u < \min(1, e^{-\beta\, \Delta U})$, set $\mathbf{r}^{(t+1)} = \mathbf{r}'$; else set $\mathbf{r}^{(t+1)} = \mathbf{r}^{(t)}$.
> 4. **Collect**: Record $A(\mathbf{r}^{(t+1)})$ for any observable $A$ of interest.
> 5. **Repeat** from step 1.
>
> After an equilibration period, the time average $\frac{1}{M}\sum_{m=1}^M A(\mathbf{r}^{(m)})$ converges to the ensemble average $\langle A \rangle_\pi$.


The correctness of the algorithm — the fact that it produces samples from $\pi$ — follows directly from detailed balance. The effective transition kernel of the Metropolis chain for $\mathbf{r}' \neq \mathbf{r}$ is $T(\mathbf{r}' \mid \mathbf{r}) = Q(\mathbf{r}' \mid \mathbf{r})\, \alpha(\mathbf{r} \to \mathbf{r}')$. Checking detailed balance for the case $\Delta U = U(\mathbf{r}') - U(\mathbf{r}) \geq 0$ (the uphill direction):

$$\pi(\mathbf{r})\, T(\mathbf{r}' \mid \mathbf{r}) = \frac{e^{-\beta U(\mathbf{r})}}{Z} \cdot Q(\mathbf{r}' \mid \mathbf{r}) \cdot e^{-\beta\Delta U} = \frac{e^{-\beta U(\mathbf{r}')}}{Z} \cdot Q(\mathbf{r}' \mid \mathbf{r}).$$

In the reverse direction, the move from $\mathbf{r}'$ to $\mathbf{r}$ is downhill and accepted with probability 1:

$$\pi(\mathbf{r}')\, T(\mathbf{r} \mid \mathbf{r}') = \frac{e^{-\beta U(\mathbf{r}')}}{Z} \cdot Q(\mathbf{r} \mid \mathbf{r}') \cdot 1 = \frac{e^{-\beta U(\mathbf{r}')}}{Z} \cdot Q(\mathbf{r}' \mid \mathbf{r}),$$

using the symmetry of $Q$. The two sides are equal. Detailed balance is satisfied, and $\pi$ is stationary.

## Trial Moves in Practice

The proposal distribution $Q$ determines the *efficiency* of sampling, while the Metropolis criterion guarantees *correctness* for any $Q$. The art of Monte Carlo in molecular simulation lies in choosing proposals that explore configuration space rapidly.

The canonical trial move is the **random displacement**: select one particle $i$ uniformly at random, then propose

> [!equation] 2.5
> $$\mathbf{r}_i' = \mathbf{r}_i + \boldsymbol{\xi},$$


where $\boldsymbol{\xi}$ is drawn uniformly from the cube $[-d_{\max}, d_{\max}]^3$. All other particle positions are unchanged. This proposal is symmetric by construction.

The maximum displacement $d_{\max}$ controls a fundamental tradeoff. If $d_{\max}$ is too small, proposed moves are nearly always accepted (since $\Delta U \approx 0$ for tiny displacements), but each accepted move barely changes the configuration — successive samples are highly correlated and configuration space is explored slowly. If $d_{\max}$ is too large, proposed moves generate large positive energy changes and are almost always rejected — the chain stagnates at the current configuration. The optimal choice balances these: a commonly cited target for simple liquids is an acceptance rate of roughly 30–50%. In practice, $d_{\max}$ is tuned during equilibration by monitoring the running acceptance fraction and adjusting $d_{\max}$ upward if the rate is too high, downward if too low.

> [!remark] 2.2
> The acceptance rate is not, in itself, a measure of sampling efficiency. A chain with 5% acceptance rate might be exploring configuration space efficiently if successful moves are large and geometrically uncorrelated. A chain with 95% acceptance rate might be making negligibly small, highly correlated steps. The meaningful quantity is the **integrated autocorrelation time** $\tau_A$ — the effective number of steps between statistically independent samples of an observable $A$ — which depends on both acceptance rate and step size in a coupled way. For a given system and observable, the optimal $d_{\max}$ minimizes $\tau_A$, not maximizes the acceptance rate.


For molecular systems — where particles are not single atoms but rigid or flexible molecules — single-atom displacement moves are inefficient because molecular geometry constraints make independent atom moves highly correlated. More effective proposals include:

- **Rigid-body moves**: translate and rotate an entire molecule simultaneously, preserving its internal geometry.
- **Volume moves**: for NPT simulations, propose a random change $V \to V' = V + \Delta V$, rescale all positions by $(V'/V)^{1/3}$, and apply a modified acceptance criterion that includes the pressure-volume work $P\Delta V$ and the entropic term $N k_B T \ln(V'/V)$.
- **Configurational-bias moves**: for chain molecules, build the proposed conformation one monomer at a time, sampling each monomer position from the local Boltzmann distribution of the growing chain. The acceptance criterion is corrected using Rosenbluth weights to account for the asymmetric proposal.

## Ergodicity and Convergence

Detailed balance guarantees that $\pi$ is *a* stationary distribution of the Metropolis chain. Two additional properties are needed to guarantee that the chain *converges* to $\pi$ and that the ergodic theorem applies.

> [!definition] 2.4 — Ergodicity
> A Markov chain is **irreducible** with respect to $\pi$ if, for every pair of configurations with $\pi(\mathbf{r}), \pi(\mathbf{r}') > 0$, there exists a finite number of steps $t$ such that the chain can reach $\mathbf{r}'$ from $\mathbf{r}$ with positive probability. The chain is **aperiodic** if it does not return to states with a fixed period. An irreducible, aperiodic Markov chain with stationary distribution $\pi$ is **ergodic**: for $\pi$-almost every starting configuration, the marginal distribution of $\mathbf{R}^{(t)}$ converges to $\pi$ as $t \to \infty$.


The ergodic theorem for Markov chains then guarantees:

> [!equation] 2.6
> $$\frac{1}{M} \sum_{m=1}^{M} A\!\left(\mathbf{r}^{(m)}\right) \xrightarrow{M\to\infty} \langle A \rangle_\pi \quad \text{almost surely,}$$


for any observable $A$ with finite variance under $\pi$. The time average along the chain converges to the ensemble average, regardless of the starting configuration, provided the chain is run long enough.

For the Metropolis algorithm with random displacement moves, irreducibility holds as long as $d_{\max} > 0$ and the potential energy $U$ is finite on a connected subset of configuration space with positive measure under $\pi$. Aperiodicity follows from the positive probability of rejection: the chain can stay at $\mathbf{r}^{(t)}$ for any number of steps, so it cannot be periodic.

There is, however, a practical failure mode that satisfies these formal conditions yet defeats convergence in any finite simulation: **metastability**. If the configuration space consists of multiple regions separated by barriers of height $\Delta F \gg k_B T$, the chain is irreducible in principle (it can always cross the barrier given enough steps) but in practice never does so within any accessible simulation time. The formal ergodicity guarantee says nothing about *how long* convergence takes, only that it eventually occurs. A chain trapped in a single metastable basin will produce biased estimates of any observable that differs between the basins. This is the sampling problem of Chapter 4.

## Equilibration and Statistical Analysis

Every Monte Carlo simulation has two phases: an **equilibration phase**, during which the chain is run without collecting data until it has lost memory of its initial configuration, and a **production phase**, during which configurations are collected and observables averaged.

There is no general rule for the equilibration length, but two diagnostics are standard. First, monitor the potential energy $U(\mathbf{r}^{(t)})$ as a function of step number: equilibration is complete when this quantity has stopped drifting and fluctuates around a stable mean. Second, if computationally feasible, launch multiple independent chains from different initial configurations and verify that their observable averages have converged to the same value.

Even in the production phase, successive configurations are not independent. The **integrated autocorrelation time**,

> [!equation] 2.7
> $$\tau_A = \frac{1}{2} \sum_{k=-\infty}^{\infty} \frac{C_A(k)}{C_A(0)}, \qquad C_A(k) = \langle A^{(t)} A^{(t+k)} \rangle - \langle A \rangle^2,$$


measures the effective number of steps between independent samples of $A$. A production run of $M$ steps yields an effective sample size of approximately $M_{\text{eff}} = M / (2\tau_A + 1)$, and the standard error of $\langle A \rangle$ is $\sqrt{\mathrm{Var}(A) / M_{\text{eff}}}$ rather than $\sqrt{\mathrm{Var}(A) / M}$.

A practical approach to error estimation is **block averaging**: divide the production run into $B$ non-overlapping blocks of length $L$, compute the block mean $\bar{A}_b$ for each block, and estimate the variance of $\langle A \rangle$ as $\mathrm{Var}(\bar{A}_b) / B$. When the block length $L \gg \tau_A$, the block means are approximately independent, and this estimator is consistent.

## The Partition Function Is Never Needed

One feature of the Metropolis algorithm deserves emphasis, because it has both a practical consequence and a fundamental limitation. The partition function $Z$ never appears in the acceptance criterion. The ratio $\pi(\mathbf{r}') / \pi(\mathbf{r}) = e^{-\beta \Delta U}$ requires only the *difference* in potential energies, and $Z$ cancels. This is the computational miracle that makes Boltzmann sampling tractable: we can generate samples from a distribution without knowing its normalization.

The limitation is that **absolute free energies are inaccessible** from a standard Metropolis run. The ensemble average $\langle A \rangle$ is estimated cheaply from the sample mean. But the Helmholtz free energy $F = -k_B T \ln Z$ requires knowing $Z$ itself — information that the Metropolis algorithm systematically discards in every acceptance step. Computing free energy differences $\Delta F = F_1 - F_0 = -k_B T \ln(Z_1 / Z_0)$ between two thermodynamic states requires a different strategy entirely — one that connects the two states and carefully tracks the probability weight along the connection. These strategies are the subject of Chapter 6.

> [!remark] 2.3
> The practical consequence is sharp: Metropolis Monte Carlo computes thermal averages of configurational observables efficiently, but cannot compute the free energy of a single state. To obtain the free energy difference between a folded and unfolded protein, or the binding free energy of a drug, one cannot simply collect configurations in each state and compare their energies — the entropic contribution is invisible to that comparison. One must compute the free energy difference directly, using methods that are covered in Chapter 6.


## Brief Extensions

The Metropolis algorithm is the foundation of a much larger family of Monte Carlo methods. Several extensions are important enough to mention here, even if their full development belongs to later chapters.

**Gibbs sampling** is an alternative update scheme for systems where the conditional distribution of one variable given all others is easy to sample. Rather than proposing a random displacement and accepting or rejecting, one updates each coordinate (or block of coordinates) by drawing exactly from its conditional distribution. No acceptance step is needed; the algorithm always accepts. Gibbs sampling is particularly useful in lattice models and in Bayesian inference problems embedded in molecular simulation.

**Parallel tempering** (replica exchange Monte Carlo) runs multiple copies of the system simultaneously at different temperatures, periodically proposing swaps of configurations between replicas with a Metropolis criterion applied to the swapping. High-temperature replicas can cross energy barriers freely, passing their configurations to lower-temperature replicas that then equilibrate locally. This dramatically improves sampling of rugged energy landscapes. It is the canonical enhanced sampling method and is treated in Chapter 5.

**Wang–Landau sampling** estimates the density of states — the number of configurations at each energy level — rather than sampling from the Boltzmann distribution directly. By iteratively flattening the energy histogram, it produces an estimate of $\ln Z(E)$ across the full energy range, from which free energies at any temperature can be obtained. It is a rare example of an algorithm that makes the partition function computationally accessible.

---

*The Monte Carlo method was developed at Los Alamos in the late 1940s by John von Neumann, Stanislaw Ulam, and Nicholas Metropolis for nuclear calculations, with the name coined to evoke the randomness of the casino at Monte Carlo. Its application to statistical mechanics is due to Metropolis, Arianna Rosenbluth, Marshall Rosenbluth, Augusta Teller, and Edward Teller, whose 1953 paper "Equation of State Calculations by Fast Computing Machines" introduced the acceptance criterion and demonstrated Boltzmann sampling for a hard-disk fluid. The asymmetric generalization — allowing non-symmetric proposals with the correction factor $Q(\mathbf{r} \mid \mathbf{r}') / Q(\mathbf{r}' \mid \mathbf{r})$ — is due to W.K. Hastings (1970). The theoretical foundations of Markov chain convergence were developed by Markov himself and formalized for continuous state spaces by Meyn and Tweedie (1993). The configurational-bias Monte Carlo method, extending the algorithm to chain molecules, is due to Siepmann and Frenkel (1992). Block averaging for autocorrelated time series is discussed in Flyvbjerg and Petersen (1989). The treatment of Monte Carlo methods in Frenkel and Smit (2002), Chapters 3–4, remains the definitive reference for practical implementation in molecular simulation.*
