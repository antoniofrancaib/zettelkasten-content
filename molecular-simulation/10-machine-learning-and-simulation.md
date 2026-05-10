---

title: "Machine Learning and Simulation"
subtitle: "Neural network potentials, learned collective variables, Boltzmann generators, and the convergence with generative flows"
---



The preceding nine chapters have told a coherent story: statistical mechanics defines the target distribution, simulation methods sample from it, enhanced sampling and free energy methods overcome its barriers, and coarse-graining reduces its dimensionality. Throughout, the potential energy function $U(\mathbf{q})$ has been treated as a given — expensive to evaluate, but exogenous to the sampling methodology. Machine learning changes this. It changes what $U$ can be, how collective variables are found, and how distributions are sampled. Most profoundly, it reveals that molecular simulation and generative modeling are the same problem viewed from different disciplines: both are concerned with learning and sampling from complex, high-dimensional probability distributions.

## Neural Network Potentials

Classical force fields (Chapter 1) parameterize $U(\mathbf{q})$ as a sum of fixed functional forms — harmonic bonds, cosine dihedrals, Lennard-Jones pairs, Coulomb interactions. The parameters are fitted to reproduce small-molecule quantum chemistry calculations and experimental observables. This approach is fast and transferable, but its functional form is rigid: it cannot represent bond breaking and forming, polarization effects, or the subtle many-body interactions that govern chemical reactivity. Quantum chemistry (density functional theory, coupled cluster) captures these effects but is computationally prohibitive for systems larger than a few hundred atoms.

**Neural network potentials** (NNPs) bridge this gap: a neural network is trained on quantum chemistry data to reproduce the potential energy and forces, then used as a drop-in replacement for both classical force fields and quantum methods in MD simulations.

> [!definition] 10.1 — Neural Network Potentials
> A **neural network potential** represents the total energy as a sum of atomic contributions:
>
> <Equation number="10.1">
> $$U(\mathbf{r}_1, \ldots, \mathbf{r}_N) = \sum_{i=1}^N \varepsilon_i(\mathbf{r}_i, \{\mathbf{r}_j : j \in \mathcal{N}(i)\}),$$
> </Equation>
>
> where $\varepsilon_i$ is the energy contribution of atom $i$, computed by a neural network from the local chemical environment: the positions of atom $i$ and all atoms within a cutoff radius $r_c$. The network takes as input a representation of the local environment that is invariant under global translations and rotations and equivariant (or invariant) under permutation of atoms of the same element. The total energy is the sum of atomic energies; the forces are computed by automatic differentiation: $\mathbf{F}_i = -\nabla_{\mathbf{r}_i} U$.


The foundational architecture is the **Behler–Parrinello network** (2007): the local environment is encoded as a set of hand-crafted symmetry functions (radial and angular descriptors that are invariant under rotation and permutation), which are passed through a fully connected network. Each element type has its own network; the total energy is the sum over all atoms. The descriptors are fixed and non-adaptive but effectively capture the local structure for most systems.

Modern NNPs replace hand-crafted descriptors with learned representations, constrained to be equivariant under the symmetry group of physical space. **NequIP** (Batzner et al., 2022) and **MACE** (Batatia et al., 2022) use equivariant message-passing neural networks — each layer propagates features that transform as irreducible representations of SO(3) (the rotation group), ensuring that the learned representation respects physical symmetry exactly. These equivariant architectures achieve near-DFT accuracy with dramatically less training data than non-equivariant alternatives, because they do not waste model capacity learning invariances that could be enforced by architecture design.

> [!remark] 10.1
> NNPs face a fundamental challenge: accuracy requires a training set that densely covers the relevant regions of configuration space, but the relevant regions — transition states, strained geometries, unusual bonding environments — are precisely the ones that are rare in equilibrium simulations. **Active learning** addresses this by coupling the NNP to a simulation that explores configuration space, using a committee of NNPs or an uncertainty estimate to identify configurations where the model is uncertain, querying the quantum chemistry oracle there, and retraining. The **Deep Potential Molecular Dynamics** (DeePMD) workflow (Wang et al., 2018) automates this cycle. Convergence of active learning is not guaranteed — the simulation may find unexpected configurations that the current committee does not recognize as uncertain — and remains an active area of research. Approaches based on conformal prediction and Bayesian neural networks are beginning to provide rigorous uncertainty bounds.


## Learned Collective Variables

Chapter 4 established that the quality of collective variables (CVs) determines the quality of any enhanced sampling calculation. Chapter 7 showed that the optimal CV is the committor — a function of $3N$ coordinates that cannot be known a priori. Machine learning now provides systematic methods for approximating the committor and other useful low-dimensional representations from simulation data.

**Variational Approach to Markov Processes** (VAMP, Wu and Noé, 2020; Mardt et al., 2018) provides a variational framework for learning the slow degrees of freedom of a molecular system. For a stochastic process described by a transfer operator $\mathcal{T}$ (the operator that propagates distributions forward by a lag time $\tau$), the VAMP score measures how well a set of functions $\{\chi_k(\mathbf{q})\}$ represents the leading eigenfunctions of $\mathcal{T}$:

> [!equation] 10.2
> $$\text{VAMP}_r = \left\| \mathbf{C}_{00}^{-1/2}\, \mathbf{C}_{0\tau}\, \mathbf{C}_{\tau\tau}^{-1/2} \right\|_r^2,$$


where $\mathbf{C}_{00}$, $\mathbf{C}_{\tau\tau}$, and $\mathbf{C}_{0\tau}$ are the instantaneous and time-lagged covariance matrices of the functions $\{\chi_k\}$, and $\|\cdot\|_r$ is the Schatten $r$-norm. Maximizing the VAMP-2 score ($r = 2$) over the parameterization of $\chi_k$ is equivalent to finding the functions that maximize the autocorrelation at lag time $\tau$ — i.e., the slowest modes of the dynamics. **VAMPnets** (Mardt et al., 2018) implement this by parameterizing $\chi_k$ as a neural network and maximizing the VAMP score by gradient ascent on trajectory data. The resulting functions are the learned slow collective variables: they capture the conformational transitions that are hardest to sample.

**Neural committor networks** directly approximate the committor $q_+(\mathbf{q})$ from simulation data. The committor satisfies the backward Kolmogorov equation (7.2); equivalently, for overdamped dynamics, it minimizes the Dirichlet energy:

$$\mathcal{L}[q_+] = \int \pi(\mathbf{q})\, |\nabla q_+(\mathbf{q})|^2\, d\mathbf{q}, \quad q_+\big|_A = 0, \quad q_+\big|_B = 1.$$

Parameterizing $q_+$ as a neural network and minimizing $\mathcal{L}$ over a training set of configurations sampled from the transition region — obtained, for example, from transition path sampling or from a biased simulation — gives an approximation to the true committor. This approximation can then be used as the CV in umbrella sampling, metadynamics, or forward flux sampling, dramatically improving convergence.

## Boltzmann Generators

The sampling methods of Chapters 2–5 all generate equilibrium configurations by running a Markov chain or molecular dynamics trajectory that eventually explores the Boltzmann distribution. **Boltzmann generators** (Noé et al., 2019) take a fundamentally different approach: train a generative model to directly sample from $\pi(\mathbf{q}) \propto e^{-\beta U(\mathbf{q})}$, bypassing the need for time-consuming trajectory simulation altogether.

> [!definition] 10.2 — Boltzmann Generators
> A **Boltzmann generator** is a normalizing flow $f_\theta: \mathbb{R}^{3N} \to \mathbb{R}^{3N}$ that maps a simple base distribution $p_z(\mathbf{z})$ (e.g., a Gaussian) to an approximation of the Boltzmann distribution $\pi(\mathbf{q})$. Samples are generated by drawing $\mathbf{z} \sim p_z$ and computing $\mathbf{q} = f_\theta(\mathbf{z})$. The model is trained to minimize the reverse KL divergence:
>
> <Equation number="10.3">
> $$\mathcal{L}_{\mathrm{KL}}(\theta) = \mathbb{E}_{\mathbf{z} \sim p_z}\!\left[\beta U(f_\theta(\mathbf{z})) - \log \left|\det \frac{\partial f_\theta}{\partial \mathbf{z}}\right|\right] + \mathrm{const},$$
> </Equation>
>
> where the first term penalizes high-energy configurations and the second term, the log absolute Jacobian determinant of $f_\theta$, ensures that the flow assigns the correct probability mass. Because $f_\theta$ must be invertible with a tractable Jacobian, normalizing flows (RealNVP, Glow, continuous normalizing flows) are the natural architecture.


The key advantage of Boltzmann generators is **one-shot sampling**: once trained, generating independent samples requires only a single forward pass through the network — no autocorrelation, no waiting for the chain to mix. The samples can be reweighted to the exact Boltzmann distribution using importance weights $w(\mathbf{q}) \propto e^{-\beta U(\mathbf{q})} / p_\theta(\mathbf{q})$, where $p_\theta$ is the density of the trained flow. **Transferable** Boltzmann generators (Köhler et al., 2023) condition the flow on molecular graph structure, allowing a single trained model to generalize across molecular conformations and chemical modifications.

> [!remark] 10.2
> The reverse KL loss $\mathcal{L}_{\mathrm{KL}}$ suffers from mode-seeking behavior: the trained flow concentrates on the dominant metastable states of $\pi$ while ignoring low-probability (but physically important) states such as transition states and rare conformations. **Forward KL training** — minimizing $\mathrm{KL}(\pi \| p_\theta)$ — requires samples from $\pi$ and is mode-covering, but this circular dependency is addressed by iterative refinement: generate samples with the current flow, reweight by importance sampling to approximate $\pi$-weighted expectations, and retrain. The combination of reverse and forward KL, combined with tempering, remains an active area of algorithmic development.


## The Convergence of Simulation and Generative Modeling

The developments above reveal a deep structural correspondence: molecular simulation and modern generative modeling are solving the same mathematical problem. Both seek to sample from a target distribution $\pi \propto e^{-\beta U}$ defined on a high-dimensional space $\mathbb{R}^{3N}$, and both face the fundamental challenge of multimodality — the distribution concentrates on exponentially many metastable configurations separated by high free-energy barriers.

This convergence has produced concrete algorithmic cross-pollination:
- **Score-based diffusion models** applied to molecular generation learn a denoising score $\nabla_\mathbf{q} \log p_t(\mathbf{q})$ at each noise level $t$, then run reverse diffusion to generate samples. For Boltzmann targets, the score at $t = 0$ is $-\beta \nabla_\mathbf{q} U$, the force field. Diffusion-based protein structure generation (RFdiffusion, Chroma) uses precisely this connection.
- **Equivariant flows** for molecular systems enforce the $SE(3)$ symmetry of physical space — translations, rotations, and (for identical particles) permutations — by construction, preventing the model from wasting capacity learning symmetries that the physics guarantees.
- **Flow matching** (Lipman et al., 2022) provides a simulation-free training objective for continuous normalizing flows: instead of running the ODE to compute the log-likelihood during training, one regresses directly on the vector field that interpolates between the base and target distributions. This is substantially more efficient for high-dimensional molecular targets.

The **MCMC–flow hybrid** approach combines the exactness of Markov chains with the efficiency of learned proposals: use a trained Boltzmann generator to propose large moves in configuration space, then accept or reject by Metropolis–Hastings. This is exact (satisfies detailed balance regardless of flow accuracy), overcomes mode-seeking bias, and can dramatically reduce the number of expensive force evaluations required to explore the energy landscape.

The story told by these ten chapters ends, appropriately, at an open frontier. The tools developed across statistical mechanics, enhanced sampling, free energy methods, and machine learning are converging toward a unified computational framework: one in which learned representations, physical symmetries, and stochastic dynamics collaborate to make the Boltzmann distribution tractable. The mathematical language underlying all of it — probability theory, differential geometry, stochastic calculus, and the theory of flows on manifolds — is the subject of the companion volumes in this series.

---

*Neural network potentials are reviewed in Behler (2021) and Batatia et al. (2023). VAMPnets are introduced in Mardt et al. (2018); neural committor methods in Khoo et al. (2019) and Li et al. (2022). Boltzmann generators are introduced in Noé et al.* Science *(2019). The convergence of diffusion models and molecular simulation is surveyed in Arts et al. (2023). The deep connection between normalizing flows and Jarzynski-weighted free energy methods is developed in Rizzi et al. (2021) and Wirnsberger et al. (2022).*
