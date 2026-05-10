---

title: "Boltzmann Generators and the Physics of Generation"
subtitle: "From statistical mechanics to transferable generative models of molecular conformation"
---



The preceding chapters developed generative flows as an abstract mathematical framework — vector fields on probability spaces, interpolants between noise and data, transport on manifolds. In this chapter the abstraction meets concrete physics. The molecules of life are among the most complex high-dimensional objects that science asks us to model: a single protein may have thousands of degrees of freedom, yet its function is dictated by the precise distribution over conformational states it populates at physiological temperature. Sampling this distribution faithfully is a problem that has occupied computational chemists for half a century and remains unsolved at scale.

Flow matching and its relatives offer a new approach, grounded in the same mathematical language we have developed throughout this monograph but now interfacing with the well-established formalism of statistical mechanics. The central object is the **Boltzmann Generator**: a generative model trained to approximate the Boltzmann distribution of a molecular system, with the critical property that its samples can be exactly reweighted to the true target. This chapter develops the framework from its statistical-mechanical foundations, traces the progression from normalizing flows through continuous flows to autoregressive approaches, and identifies the open problems that will define the field in the coming years.

## 8.1 Statistical Mechanics and the Sampling Problem

A molecular system consisting of $N$ atoms in classical mechanics is described by positions $q \in \mathbb{R}^{3N}$ and momenta $p \in \mathbb{R}^{3N}$. The dynamics are governed by the Hamiltonian

$$H(q, p) = K(p) + U(q) = \sum_{i=1}^N \frac{\|p_i\|^2}{2m_i} + U(q),$$

where $K(p)$ is kinetic energy, $U(q)$ is the potential energy surface (PES) encoding all bonded and non-bonded interactions, and Hamilton's equations give deterministic Newtonian trajectories:

$$\dot{q} = \frac{p}{m}, \qquad \dot{p} = -\nabla_q U(q) =: F(q).$$

In practice, molecular systems are not isolated: they are embedded in a thermal bath (solvent, membrane, cellular environment) at temperature $T$. The bath exchanges energy stochastically with the solute, driving the system toward thermal equilibrium. This is captured by **Langevin dynamics**, which augments Newton's equations with a friction term and Gaussian white noise:

> [!equation] 8.1
> $$\dot{p} = -\nabla_q U(q) - \gamma p + \sqrt{2 \gamma m k_B T}\, \xi(t), \qquad \dot{q} = \frac{p}{m},$$


where $\gamma > 0$ is the friction coefficient, $k_B$ is Boltzmann's constant, and $\xi(t) \sim \mathcal{N}(0, I)$ is standard white noise. The fluctuation–dissipation relation, encoded in the coefficient $\sqrt{2\gamma m k_B T}$, ensures that the system thermalizes to the **canonical (Boltzmann) distribution**:

> [!definition] 8.1 — Boltzmann Distribution
> For a system with potential energy $U : \mathbb{R}^{3N} \to \mathbb{R}$ at inverse temperature $\beta = 1/(k_B T)$, the canonical (Boltzmann) distribution over configurations is
>
> $$\rho(q) = \frac{1}{Z}\, e^{-\beta U(q)}, \qquad Z = \int_{\mathbb{R}^{3N}} e^{-\beta U(q)}\, dq,$$
>
> where $Z$ is the **partition function**. Thermodynamic observables — free energies, heat capacities, binding affinities — are ensemble averages $\langle f \rangle_\rho = \int f(q)\, \rho(q)\, dq$.


The partition function $Z$ is intractable: it is a $3N$-dimensional integral of a highly non-convex integrand. This is the fundamental computational challenge of statistical mechanics.

**Molecular Dynamics as a sampling algorithm.** The classical approach is to simulate Langevin dynamics (8.1) directly — discretizing it with the BAOAB (Langevin Middle) integrator, which interleaves half-steps of deterministic drift (B = force update, A = position update) with an Ornstein–Uhlenbeck step (O) on the momenta:

$$p' = p + \tfrac{\Delta t}{2} F(q), \quad q' = q + \tfrac{\Delta t}{2m} p', \quad p'' = e^{-\gamma \Delta t} p' + \sqrt{m k_B T (1 - e^{-2\gamma \Delta t})}\, \eta,$$
$$q^* = q' + \tfrac{\Delta t}{2m} p'', \quad p^* = p'' + \tfrac{\Delta t}{2} F(q^*),$$

where $\eta \sim \mathcal{N}(0, I)$. By the ergodic theorem, time averages along a sufficiently long trajectory converge to ensemble averages under $\rho$.

**The sampling problem.** The fatal flaw of direct MD simulation is the **separation of timescales** endemic to complex molecular systems. The fastest motions (bond vibrations, $\sim 10^{-15}$ s) constrain the integration step $\Delta t \lesssim 2$ fs, while the slowest biologically relevant conformational changes (helix–coil transitions, folding, binding) occur on timescales of microseconds to seconds — a gap of $10^9$–$10^{15}$ timesteps. Even with specialized hardware (Anton supercomputers, millisecond-scale simulations), most of configuration space is never explored. Generative models offer an alternative path: **learn the Boltzmann distribution directly**, bypassing the dynamical bottleneck entirely.

## 8.2 Boltzmann Generators: The Three-Step Framework

The Boltzmann Generator framework (Noé et al., 2019) proposes a clean decomposition of the sampling problem:

1. **Model:** learn a generative model $q_\theta \approx \rho$ using equilibrium data and/or the energy function $U$.
2. **Sample:** draw i.i.d. samples $x^{(1)}, \ldots, x^{(N)} \sim q_\theta$, computing exact log-likelihoods $\log q_\theta(x^{(i)})$ for each.
3. **Reweight:** correct for the approximation error $q_\theta \neq \rho$ using importance sampling, obtaining unbiased estimators of thermodynamic observables under the true Boltzmann distribution.

This framework is remarkable because step 3 provides **exact thermodynamic estimates** even when the generative model is imperfect — as long as the log-likelihoods are computed exactly. The key observation is that the unnormalized Boltzmann density $\tilde{\rho}(q) = e^{-\beta U(q)}$ is accessible without knowing $Z$: one needs only the energy function $U(q)$, which is provided by the molecular mechanics force field.

**Molecular representation.** A critical design choice is the coordinate system. Cartesian coordinates $q \in \mathbb{R}^{3N}$ are natural but violate the physical symmetries of the problem: two configurations related by a global rotation or translation have identical energy $U(q)$ but different coordinates. Working in Cartesian space requires the model to implicitly learn SE(3) invariance, wasting capacity.

The standard alternative is **internal (torsional) coordinates**: bond lengths, bond angles, and dihedral (torsion) angles. A molecule with $N$ atoms has $3N - 6$ internal degrees of freedom (subtracting 3 translational and 3 rotational). Dihedral angles $\phi \in [-\pi, \pi]^{n_\tau}$, where $n_\tau$ is the number of rotatable bonds, capture the dominant conformational flexibility while being SE(3)-invariant by construction and directly compatible with molecular mechanics energy functions. The Jacobian of the transformation from Cartesian to internal coordinates (the so-called $G$-matrix of Wilson) can be computed analytically and is required for exact density transformation.

## 8.3 Importance Sampling and the Effective Sample Size

With samples $\{x^{(i)}\}_{i=1}^N \sim q_\theta$ and exact log-likelihoods, one can form **importance weights** that correct the distribution from $q_\theta$ to the target $\rho$:

$$w(x^{(i)}) = \frac{\rho(x^{(i)})}{q_\theta(x^{(i)})} = \frac{e^{-\beta U(x^{(i)})} / Z}{q_\theta(x^{(i)})}.$$

Since $Z$ is unknown, one uses **self-normalized importance sampling (SNIS)**. Define unnormalized weights $\tilde{w}(x^{(i)}) = \tilde{\rho}(x^{(i)}) / q_\theta(x^{(i)}) = e^{-\beta U(x^{(i)})} / q_\theta(x^{(i)})$, and normalize:

$$\bar{w}^{(i)} = \frac{\tilde{w}(x^{(i)})}{\sum_{j=1}^N \tilde{w}(x^{(j)})}.$$

The SNIS estimator for an observable $f$ is then:

> [!equation] 8.2
> $$\langle f \rangle_\rho \approx \sum_{i=1}^N \bar{w}^{(i)} f(x^{(i)}).$$


The log-weight admits an elegant decomposition:

$$\log \tilde{w}(x^{(i)}) = -\beta U(x^{(i)}) - \log q_\theta(x^{(i)}),$$

which requires exactly one energy evaluation and one likelihood evaluation per sample — a total of $N$ calls to each, embarrassingly parallelizable. Note that **no simulation is required**: samples are generated i.i.d. from $q_\theta$, and the energy $U$ is a fast deterministic function.

> [!definition] 8.2 — Effective Sample Size (ESS)
> For importance weights $\{w^{(i)}\}_{i=1}^N$, Kish's **effective sample size** is
>
> $$\mathrm{ESS} = \frac{\left(\sum_{i=1}^N w^{(i)}\right)^2}{\sum_{i=1}^N (w^{(i)})^2} \in [1, N].$$
>
> The **relative ESS** is $\mathrm{rESS} = \mathrm{ESS}/N \in [N^{-1}, 1]$. An $\mathrm{rESS}$ near 1 indicates that $q_\theta \approx \rho$ uniformly; $\mathrm{rESS} \approx N^{-1}$ indicates near-complete weight collapse, where a single sample dominates.


The ESS is the primary evaluation metric for Boltzmann Generators: it captures both **precision** (whether $q_\theta$ assigns mass to regions where $\rho$ is small — causing high-weight outliers) and **recall** (whether $q_\theta$ has missed modes of $\rho$ — causing near-zero weights there). The variance of the SNIS estimator satisfies $\mathrm{Var}[\hat{f}_{\mathrm{SNIS}}] \propto \mathrm{ESS}^{-1}$, making ESS directly interpretable as an equivalent i.i.d. sample count.

> [!remark] 8.1
> A practical subtlety noted by Tan et al. (2025): implementations often clip the largest 0.2% of importance weights to prevent single outliers from dominating the estimator. While this reduces variance, it introduces a systematic bias by selectively suppressing high-weight (high-probability) samples. A principled alternative is **bidirectional clipping** — truncating both extremes of the weight distribution — which controls variance while preserving unbiasedness in a wider sense. The optimal clipping strategy remains an open question.


Beyond ESS, a complementary evaluation metric is the **Wasserstein distance** between the reweighted sample distribution and ground-truth equilibrium data. When equilibrium MD data is available, one can directly compare histograms of dihedral angles or pairwise distances; when only the energy $U$ is available, ESS and free energy differences serve as proxies. The two metrics capture different failure modes: ESS measures distributional overlap, while Wasserstein distance measures geometric accuracy.

## 8.4 Continuous Normalizing Flows and Exact Likelihoods

The Boltzmann Generator framework requires exact log-likelihood evaluation $\log q_\theta(x)$. Early implementations used **normalizing flows** — invertible architectures (RealNVP, Glow) admitting closed-form log-determinant computation. These offer exact likelihoods but restrict the expressivity of the transformation: coupling layers, by design, leave half the coordinates unchanged at each step.

**Continuous normalizing flows** (Chapter 1, Theorem 1.1) remove this architectural restriction. A CNF parameterizes $q_\theta$ as the time-1 pushforward of a base distribution $p_0 = \mathcal{N}(0,I)$ under a vector field $v_\theta$, with exact log-likelihood given by:

$$\log q_\theta(x_1) = \log p_0(x_0) - \int_0^1 \nabla \cdot v_\theta(x_t, t)\, dt, \qquad x_0 = \phi_1^{-1}(x_1).$$

The cost is computational: each likelihood evaluation requires a full ODE solve (forward for sampling, backward for likelihood), with $O(d)$ per step for the vector field and $O(d)$ for the Hutchinson trace estimator of the divergence.

**FALCON.** The recent FALCON framework (Rehmann et al., 2025) addresses the cost of likelihood computation in the few-step regime. A *flow map* $F_\theta : \mathbb{R}^d \to \mathbb{R}^d$ directly maps noise to samples in a single evaluation but is not guaranteed to be invertible — a requirement for exact likelihood estimation. FALCON augments the standard flow matching loss with a **soft invertibility penalty**:

$$\mathcal{L}_{\mathrm{FALCON}}(\theta) = \mathcal{L}_{\mathrm{CFM}}(\theta) + \lambda\, \mathbb{E}_{x_0}\left[\|F_\theta(F_\theta^{-\mathrm{approx}}(F_\theta(x_0))) - F_\theta(x_0)\|^2\right],$$

where $F_\theta^{-\mathrm{approx}}$ is an approximate inverse obtained by a fixed-point iteration. This encourages the flow map to remain approximately invertible while retaining the few-step efficiency of distilled models. The resulting models achieve accurate likelihood estimates at 1–4 NFE, enabling Boltzmann Generator pipelines with minimal simulation overhead.

## 8.5 Training Objectives: Data versus Energy Supervision

The choice of training signal for a Boltzmann Generator involves a fundamental trade-off between two KL divergences, each with distinct geometric meaning.

**Data supervision (forward KL).** If equilibrium data $\{x^{(i)}\}_{i=1}^M \sim \rho$ is available, maximum likelihood training minimizes:

$$D_{\mathrm{KL}}^{\mathrm{fwd}}(\rho \| q_\theta) = \mathbb{E}_{x \sim \rho}[\log \rho(x) - \log q_\theta(x)].$$

This is a **mass-covering** objective: the gradient penalizes any region where $\rho$ has mass but $q_\theta$ does not, driving $q_\theta$ to spread its mass broadly. The result is a model that covers all modes but may assign probability to low-probability regions between modes.

**Energy supervision (reverse KL).** When equilibrium data is scarce or absent but the energy $U$ is available, one minimizes:

$$D_{\mathrm{KL}}^{\mathrm{rev}}(q_\theta \| \rho) = \mathbb{E}_{x \sim q_\theta}[\log q_\theta(x) - \log \rho(x)] = \mathbb{E}_{x \sim q_\theta}[\log q_\theta(x) + \beta U(x)] + \log Z.$$

This is a **mode-seeking** objective: the gradient is zero wherever $q_\theta(x) = 0$, so the model learns to concentrate on high-probability regions of $\rho$. The gradient of $D_{\mathrm{KL}}^{\mathrm{rev}}$ with respect to $\theta$ is:

$$\nabla_\theta D_{\mathrm{KL}}^{\mathrm{rev}} = \mathbb{E}_{x \sim q_\theta}\left[\nabla_\theta \log q_\theta(x) \cdot (\log q_\theta(x) + \beta U(x))\right],$$

which can be estimated using the REINFORCE estimator or, for CNFs, via the reparameterization trick through the ODE solver. The log-importance-weight $\log \tilde{w}(x) = -\beta U(x) - \log q_\theta(x)$ thus serves double duty: as the quantity to be reweighted in step 3, and as the reward signal in energy-supervised training.

> [!remark] 8.2
> In practice, Boltzmann Generators combine both signals. Pre-training on available MD data (forward KL) initializes $q_\theta$ near the target, providing a warm start that avoids the mode collapse common to pure reverse-KL training. Fine-tuning with energy supervision then corrects the remaining mismatch. The ratio of data to energy gradient steps is a key hyperparameter: too much energy supervision from a poor initialization drives the model into degenerate modes.


## 8.6 Temperature Annealing and Sequential Monte Carlo

A fundamental difficulty of energy-supervised training is that the Boltzmann distribution at physical temperature $T$ may be sharply multimodal — high free-energy barriers separate conformational states, making direct optimization of $D_{\mathrm{KL}}^{\mathrm{rev}}$ unstable. The **temperature annealing** approach (Schopmans & Friedrich, 2025; Akhound-Sadegh, Lee et al., 2025) exploits the fact that at elevated temperature $T' \gg T$, the Boltzmann distribution $\rho_{T'} \propto e^{-U(q)/k_B T'}$ is more uniform — barriers are suppressed — and hence easier to learn. One trains a family of models $\{q_\theta^{(\beta_k)}\}$ at decreasing inverse temperatures $\beta_1 < \beta_2 < \cdots < \beta_K = \beta_{\mathrm{phys}}$, using each $q_\theta^{(\beta_k)}$ as a warm-start or proposal for training $q_\theta^{(\beta_{k+1})}$.

**Sequential Monte Carlo (SMC).** A more principled version of this idea is SMC sampling through a sequence of intermediate Boltzmann distributions (Tan et al., 2024). Define $\rho_\tau \propto e^{-\tau \beta U(q)}$ for a tempering schedule $0 = \tau_0 < \tau_1 < \cdots < \tau_K = 1$. At each temperature level $\tau_k$:

1. **Propose:** run the current flow model $q_\theta^{(\tau_k)}$ to generate $N$ samples.
2. **Reweight:** compute importance weights $w^{(i)} \propto \rho_{\tau_{k+1}}(x^{(i)}) / \rho_{\tau_k}(x^{(i)}) = e^{-(\tau_{k+1} - \tau_k)\beta U(x^{(i)})}$.
3. **Resample:** draw $N$ new samples with replacement proportional to $\{w^{(i)}\}$.
4. **Refresh:** run short Langevin MD trajectories (BAOAB steps) to diversify the particle population.

SMC provides a theoretically justified way to bridge from the high-temperature prior to the physical Boltzmann distribution, with the flow model at each level acting as a learned proposal that dramatically reduces the number of MCMC steps needed for mixing.

## 8.7 Transferable Boltzmann Generators

Early Boltzmann Generators were trained for a single molecular system — one force field, one set of atoms, one topology. This non-transferability is a fundamental limitation: drug discovery requires sampling hundreds of thousands of candidate molecules, and training a separate model per compound is computationally prohibitive.

**Transferable Boltzmann Generators** (Klein & Noé, 2024) address this by conditioning the flow model on a molecular graph representation. The generative model becomes $q_\theta(x | G)$, where $G$ is the chemical graph (atom types, bond connectivity, stereochemistry). A single model is trained on a diverse dataset of molecular conformers $\{(x^{(i)}, G^{(i)})\}$, learning a transferable mapping from graph to conformation distribution.

For importance sampling under a target molecule $G^*$, one must filter out contributions from structurally incompatible training molecules. Define effective importance weights:

$$w_i^{\mathrm{eff}} = g_i \cdot w_i, \qquad g_i = \mathbf{1}[\text{conformer } i \text{ has chemical graph } G^*].$$

Only conformers from the target chemical topology receive non-zero weight. The ESS of the resulting estimator measures how well the transferred model represents the target molecule's Boltzmann distribution.

As the diversity of training data grows toward arbitrary small molecules, the chemical graph filter becomes increasingly selective. This motivates architectures that can extrapolate to new chemical environments: equivariant graph neural networks conditioned on atom types and partial charges provide the inductive bias needed for such transferability. The scaling behavior of transferable Boltzmann Generators — how ESS improves as a function of training data and model size — remains an open empirical question analogous to the scaling laws for language models.

## 8.8 Autoregressive Boltzmann Generators

A structurally distinct alternative to continuous flow-based models is the **autoregressive Boltzmann Generator** (Rehmann et al., 2025), which generates conformations by inserting atoms sequentially. The joint density is factored as:

> [!equation] 8.3
> $$q_\theta(x) = \prod_{j=1}^d q_\theta(x_j \mid x_{<j}),$$


where $d$ is the number of atoms (or spatial voxels) and $x_{<j} = (x_1, \ldots, x_{j-1})$ is the context. Each conditional $q_\theta(x_j | x_{<j})$ is a distribution over the position of atom $j$, typically represented as a mixture over spatial voxels. The log-likelihood $\log q_\theta(x) = \sum_j \log q_\theta(x_j | x_{<j})$ is **trivially exact** — no ODE solve or Jacobian computation is required. This is in stark contrast to CNFs, where exact likelihoods require solving a trace-augmented ODE.

The key trade-off is **conformational resolution**: each atom's position is discretized to a finite voxel grid, introducing quantization error. This error scales inversely with voxel resolution — finer grids yield higher accuracy but exponentially more computational cost. At typical resolutions used in practice, autoregressive models achieve accuracy competitive with CNF-based approaches while being significantly faster to evaluate.

**DUEL and discrete diffusion.** An interesting connection arises with discrete diffusion models. DUEL (Turok et al., 2026) establishes that fixing an unmasking order in a discrete diffusion model over atom positions recovers exact likelihood estimation via the factorization (8.3). This connects the autoregressive framework to the Chapter 2 material on discrete flow matching: the atom-insertion order defines a directed path through discrete state space, and the likelihood is the product of transition probabilities along that path. The choice of ordering — canonical, random, or learned — profoundly affects both sample quality and inference speed.

**Sampling speed and speculative decoding.** Autoregressive generation is inherently sequential: atom $j+1$ cannot be placed without knowing atom $j$. This limits throughput for large molecules. An analogy to speculative decoding in language models suggests a path forward: a small "draft" model proposes the next $k$ atoms simultaneously, and a larger "verifier" model accepts or rejects the proposal. The speed-quality tradeoff can be modulated by adjusting $k$ at inference time. Whether this analogy is formally grounded — and whether the verification step can be made exact in the sense of preserving the target distribution — are active research questions (see MDVerse, Tieman et al., 2024).

## 8.9 Equivariant Architectures and Molecular Representations

Physical symmetry is not an optional add-on for molecular generative models — it is a hard constraint. Two configurations $q$ and $Rq + b$ (related by a rotation $R \in \mathrm{SO}(3)$ and translation $b \in \mathbb{R}^3$) have identical energy $U(Rq + b) = U(q)$ and therefore identical probability $\rho(Rq + b) = \rho(q)$. A generative model that does not respect this symmetry wastes representational capacity on physically equivalent configurations and may fail to sample the correct ensemble.

> [!definition] 8.3 — SE(3)-Equivariant Model
> A generative model $q_\theta(x | G)$ is **SE(3)-invariant** if $q_\theta(Rx + b | G) = q_\theta(x | G)$ for all $R \in \mathrm{SO}(3)$, $b \in \mathbb{R}^3$. A vector field $v_\theta(x, t)$ is **SE(3)-equivariant** if $v_\theta(Rx + b, t) = R\, v_\theta(x, t)$ for all such $(R, b)$.


Two approaches to SE(3) symmetry dominate the literature:

**Internal coordinates.** By working in torsion-angle space $\phi \in [-\pi, \pi]^{n_\tau}$, one obtains SE(3) invariance by construction. The representation is compact ($n_\tau \ll 3N$), directly interpretable, and compatible with standard force-field energy functions. The limitation is that bond lengths and angles must either be treated as fixed (rigid-body approximation) or modeled separately, losing some flexibility.

**Equivariant Cartesian networks.** Architectures such as SE(3)-Transformers, SEGNN, MACE, and NequIP process 3D point clouds using tensor products of irreducible representations of SO(3). These networks are SE(3)-equivariant by construction, meaning the velocity field output rotates correctly with the input. Flow matching in Cartesian space with an equivariant network naturally produces an SE(3)-equivariant flow: if the initial noise is SE(3)-invariant and the velocity field is SE(3)-equivariant, the marginal distribution at every time $t$ is SE(3)-invariant.

A key architectural choice in Cartesian-space flow matching for molecules is how to represent the frame of reference for each residue or fragment. Framing-based approaches (FrameDiff, FrameFlow) represent each amino acid as a local coordinate frame (position + orientation) and apply flow matching on the product manifold $\mathbb{R}^3 \times \mathrm{SO}(3)^N$ — connecting directly to the Riemannian flow matching framework of Chapter 3.

## 8.10 Structure-Based Drug Design and Protein Generation

Boltzmann Generators address the **conformational sampling** problem — given a known molecular topology, sample the ensemble of structures it adopts. A related but distinct problem is **de novo generation**: design molecules with desired properties from scratch. The past three years have seen flow-based models achieve breakthrough results on both tasks.

**Protein backbone generation.** The protein backbone is determined by the positions of $C^\alpha$, $C$, $N$, and $O$ atoms, or equivalently by the sequence of backbone dihedral angles $(\phi_i, \psi_i, \omega_i)_{i=1}^L$ for a chain of $L$ residues. The space of physically realizable backbones is a submanifold of $(\mathbb{S}^1)^{3L}$ constrained by Ramachandran preferences and steric exclusion.

- **RFdiffusion** (Watson et al., 2023) applies diffusion directly in the space of residue frames, using a pretrained ProteinMPNN/RoseTTAFold backbone for score estimation. It demonstrates de novo protein design, motif scaffolding, and binder design at unprecedented quality.
- **FrameDiff / FrameFlow** (Yim et al., 2023, 2024) applies SE(3) diffusion and flow matching, respectively, on the product manifold of residue frames $SE(3)^L$. FrameFlow achieves state-of-the-art backbone generation with significantly fewer sampling steps than FrameDiff, illustrating the inference-efficiency advantage of flow matching over diffusion (Chapter 1, Section 1.7).
- **FoldFlow** (Bose et al., 2023) extends the stochastic interpolant framework to $SE(3)^L$ with a choice of geometric interpolant, showing that the OT interpolant on $SE(3)$ produces straighter geodesic flows and reduces the required NFE.

**Structure-based drug design (SBDD).** SBDD asks: given the 3D structure of a protein binding pocket, generate small molecules that bind with high affinity and selectivity. DiffSBDD (Schneuing et al., 2022), TargetDiff, and FlowSite (Stark et al., 2024) combine equivariant networks over the protein–ligand complex with diffusion or flow matching over ligand atom positions and types. These models jointly generate coordinates and atom identities, operating on the product of continuous (position) and discrete (atom type, bond order) spaces — the setting of Chapter 5 on product-space flow matching.

**Allosteric modulation and cryptic pockets.** An emerging challenge is designing ligands for binding sites that only appear in a small fraction of the protein's conformational ensemble — so-called cryptic pockets. Addressing this requires coupling Boltzmann Generator-style conformational sampling with structure-based design: first sample the conformation ensemble with a Boltzmann Generator, identify low-population conformers with druggable pockets, then design ligands for those conformers. The two problems are thus deeply intertwined, and models that solve both simultaneously are an active frontier.

## 8.11 Supervision with Physical Observables

The ultimate validation of a Boltzmann Generator is not ESS or Wasserstein distance against MD data — it is agreement with experiment. Experimental data provides a qualitatively different kind of supervision that complements both energy and data signals.

**Stationary observables.** Macroscopic thermodynamic quantities — solvation free energies, binding free energies, heat capacities, radial distribution functions — are ensemble averages under $\rho$. If such quantities are measured experimentally, they can be incorporated as constraints on the generative model. Lewis, Clementi & Noé (Science, 2024) demonstrate that incorporating experimental radial distribution functions as loss terms substantially improves the accuracy of generative models trained on short MD trajectories, particularly for properties that are poorly sampled by simulation.

Formally, if $\{O_k\}_{k=1}^K$ are observed experimental values of observables $\{f_k\}_{k=1}^K$, one seeks a model $q_\theta$ satisfying:

$$\mathbb{E}_{q_\theta}[f_k(x)] = O_k, \quad k = 1, \ldots, K.$$

By the maximum entropy principle, the distribution satisfying these constraints while being closest (in relative entropy) to a prior $q_0$ is $q^* \propto q_0(x) \exp(\sum_k \lambda_k f_k(x))$, where $\lambda_k$ are Lagrange multipliers. In practice, $q_\theta$ is trained to match these constraints by differentiating through the SNIS estimator of $\mathbb{E}_{q_\theta}[f_k]$.

**Dynamic observables.** Kinetic properties — association and dissociation rate constants $k_{\mathrm{on}}, k_{\mathrm{off}}$, diffusion coefficients, NMR relaxation times — depend not only on the stationary distribution $\rho$ but on the dynamics of transitions between states. Two drugs binding with similar $K_d = k_{\mathrm{off}}/k_{\mathrm{on}}$ may have vastly different clinical outcomes if their residence times $\tau = 1/k_{\mathrm{off}}$ differ by orders of magnitude.

Connecting generative models to kinetic observables is fundamentally harder than connecting them to thermodynamic ones: it requires modeling the *process* by which the system moves between states, not just its equilibrium distribution. Recent work on **transfer emulators** (Diez & Olsson, 2025) uses flow matching to learn maps between molecular dynamics trajectories at different temperatures or Hamiltonians, enabling prediction of kinetic properties at conditions where direct simulation is infeasible. The theoretical underpinning connects to non-equilibrium statistical mechanics and the Jarzynski equality, relating work distributions along driven processes to free energy differences.

**DNA-PAINT and single-molecule data.** Super-resolution microscopy techniques such as DNA-PAINT provide single-molecule tracking data that directly encodes kinetic rates ($k_{\mathrm{on}}, k_{\mathrm{off}}$) for DNA hybridization. Incorporating such data into Boltzmann Generator training is an emerging direction that would connect generative models to a rich experimental literature on molecular kinetics.

## 8.12 Open Problems and the Scaling Era

The notes from which this chapter draws close with a question: are we entering a **scaling era for Boltzmann Generators**? The analogy with language models is suggestive but not guaranteed. We close with a precise statement of the six open problems that will determine the answer.

**1. Sampling speed.** Current autoregressive and CNF-based models generate conformations orders of magnitude faster than MD but still require $O(N_{\mathrm{atoms}})$ sequential steps for autoregressive models or $O(N_{\mathrm{steps}})$ ODE evaluations for CNFs. Achieving sub-second generation for proteins with $10^4$ atoms requires either speculative decoding (autoregressive), flow distillation (CNF), or architecturally novel approaches.

**2. Larger system sizes.** Existing Boltzmann Generators handle peptides ($\lesssim 50$ residues) and small molecules ($\lesssim 50$ heavy atoms) reliably. Extension to full protein-ligand complexes ($10^3$–$10^4$ atoms), membrane-embedded proteins, or nucleic acid complexes requires models that scale sub-quadratically with system size. MDVerse (Tieman et al., 2024) demonstrates partial decoding — generating only $C^\alpha$ positions — as a scalability strategy, at the cost of lower structural resolution.

**3. Free energy estimation.** Computing binding free energies $\Delta G_{\mathrm{bind}}$ with chemical accuracy ($\pm 0.5$ kcal/mol) is the primary computational chemistry goal in drug discovery. Boltzmann Generators could in principle compute $\Delta G$ via the identity $Z = \mathbb{E}_{q_\theta}[\tilde{w}(x)]$, but the variance of this estimator grows exponentially with system size. Incorporating free energy perturbation (FEP) theory — which computes $\Delta G$ between structurally similar molecules via a perturbation expansion — into the Boltzmann Generator pipeline is an active direction.

**4–5. Binder design and motif scaffolding with observable supervision.** Designing proteins that bind to a prescribed epitope, or that display a specific secondary-structure motif in a novel context, are tasks that RFdiffusion has demonstrated at the level of experimental validation. Incorporating stationary and dynamic observable constraints into the design objective — designing not just for structural specificity but for kinetic stability, on-target selectivity, and solubility — requires generalizing the energy-supervision framework to multi-objective settings.

**6. Transfer and non-equilibrium processes.** Current models are trained for systems near equilibrium at a single thermodynamic state point. Real biological processes — protein folding, ligand binding, membrane permeation — are non-equilibrium events driven by concentration gradients, electric fields, or mechanical forces. Extending Boltzmann Generators to non-equilibrium steady states requires replacing the equilibrium Boltzmann distribution with the non-equilibrium stationary distribution of the Fokker–Planck equation — a harder target that lacks the Gibbs structure that makes importance sampling tractable.

> [!remark] 8.3
> The table below summarizes the design space of current Boltzmann Generator architectures along the dimensions that matter for practical deployment:
>
> | Approach | Modeling | Sampling | Representation | Domain |
> |---|---|---|---|---|
> | NF-based BG | Normalizing flows | SNIS | Internal coordinates | Non-transferable |
> | CNF-based BG | Continuous NF | SNIS / SMC | Global | Transferable |
> | Autoregressive BG | Autoregression | SNIS | Global voxels | Transferable |
> | FALCON | Few-step CNF | SNIS | Global | Transferable |
>
> The frontier is occupied by approaches that combine high-expressivity continuous models (for accurate likelihoods), transferability across chemical space (for drug discovery scale), and sequential Monte Carlo (for exact thermodynamics). No single current approach satisfies all three simultaneously — and this gap is where the next generation of methods will emerge.


---

The physics of molecular systems is both a demanding application domain and a profound source of mathematical structure: conservation laws, symmetry groups, metastability, detailed balance. Flow matching frameworks that are ignorant of this structure learn it the hard way, from data and gradient descent. Frameworks that respect it — through equivariant architectures, physically motivated interpolants, and principled importance-sampling corrections — achieve an order-of-magnitude better sample efficiency. The lesson generalizes: the best generative models for structured scientific domains are those built at the intersection of the mathematical theory developed in the preceding chapters and the domain-specific physics that constrains the problem. That intersection is where science happens.
