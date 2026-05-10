---

title: "What It Does in the World"
subtitle: "Proteins, molecules, language, audio, weather, and the emerging science of generative modeling — ongoing"
---



Theory earns its keep by what it enables. The nine preceding chapters have constructed an elaborate mathematical scaffold — from normalizing flows through the generator matching framework, from Euclidean spaces to product spaces and manifolds, from slow-but-exact ODE integration to one-step consistency generation. The scaffold is elegant and it is internally coherent. But its ultimate justification is practical: what can be built with it?

The answer, as of 2025, is a great deal. Flow matching and its generative cousins have become the de facto standard methodology across structural biology, drug discovery, speech synthesis, climate science, and several branches of physics. The speed of adoption has been unusual: a framework with theoretical origins in 2022 is running in production at major pharmaceutical companies, providing ensemble weather forecasts at global scale, and generating proteins that have been validated in laboratory experiments. This chapter maps the application landscape by domain, identifying not just what has been accomplished but why — which feature of the framework made it the right tool in each case.

## 10.1 What Flow Matching Offers Scientific Applications

Before surveying the domains, it is worth being explicit about what the framework contributes to each.

**Simulation-free training with exact gradients.** The central computational advantage is that flow matching requires no ODE simulation during training — one forward pass per gradient step, regardless of the dimensionality of the data. For scientific domains where data is scarce and model training is expensive, this matters enormously. A protein structure model that required 1000-step simulation at every gradient step would be impractical to train at scale; one that requires a single forward pass can be.

**Arbitrary source distributions and conditional generation.** The framework does not require Gaussian noise as the source distribution. For scientific applications, meaningful source distributions are often available: crystal-structure libraries for small molecules, coil conformations for proteins, uniform reference states for discrete sequences. Conditional generation — generating samples from $p(x | \text{context})$ — is a first-class citizen of the framework rather than a post-hoc add-on.

**Geometry-respecting generation on natural state spaces.** Proteins live in $SE(3)^N$, molecular dihedral angles live on the torus, amino acid sequences live in a discrete alphabet. The Riemannian and discrete extensions of Chapter 5 make the framework geometry-native: it generates on the right space from the start, without coordinate singularities, normalization ambiguities, or boundary violations.

**Interpretable and steerable trajectories.** Unlike diffusion models, whose stochastic trajectories are hard to interpret, flow matching produces deterministic ODE trajectories from noise to data. These can be visualized, analyzed for bottlenecks, and steered — by guidance or by changing the source distribution — without breaking the mathematical structure.

## 10.2 Structural Biology: Protein Backbone Generation

The protein structure generation problem — given a desired length $N$, sample a backbone geometry $(R_i, t_i)_{i=1}^N \in SE(3)^N$ that folds into a designable, physically plausible protein — became the proving ground for flow matching in structural biology.

The diffusion approach, developed by Yim, Trippe, De Bortoli, Mathieu, Doucet, Barzilay, and Jaakkola (**FrameDiff**, ICML 2023), placed a diffusion process directly on the space of $SE(3)$ frames, treating the rotation and translation components with separate Gaussian diffusion schedules. The resulting model generates protein backbones with high designability — the fraction of generated backbones for which ProteinMPNN can find a designable sequence, as measured by structure prediction self-consistency — and was the first model to demonstrate that $SE(3)$-equivariant diffusion could match or exceed the quality of hand-crafted heuristics for backbone generation.

**FoldFlow** (Bose, Akhound-Sadegh, Huguet, Wolf, Rekesh, Su, Zhang, Bengio, Bhatt, Lee, Tong, Hamilton, and Bronstein, NeurIPS 2023) replaced the diffusion process with the Riemannian flow matching framework of Chapter 5: geodesic interpolation on $SE(3)^N$, simulation-free training, IPA architecture, 2× fewer inference steps. **FoldFlow-2** (Huguet, Vuckovic, Fatras, Tong, Sun, Wolf, Bronstein, Tazi, and Bengio, 2024) scaled this with mini-batch OT conditioning on $SE(3)$ — matching noise frames to data frames by a joint Riemannian OT cost — producing straighter trajectories and competitive designability in 10 steps.

The field-defining result was **AlphaFold 3** (Abramson, Adler, Dunger, Evans, Green, Pritzel, Ronneberger, Willmore, Ballard, Bambrick, Bodenstein, Evans, Hung, O'Neill, Reiman, Sternke, Beattie, Jumper, and Hassabis, Nature 2024). AlphaFold 3 predicts the full-atom structure of proteins, nucleic acids, small molecules, and their complexes, using a diffusion module that generates all-atom coordinates from a noise distribution. The diffusion module operates on flat coordinates (not $SE(3)$ frames), conditioned on the output of a structure module derived from AlphaFold 2's Evoformer. AlphaFold 3 achieves state-of-the-art performance on protein-ligand structure prediction, protein-nucleic acid complexes, and protein-protein interfaces — the breadth of biological structural prediction, resolved in a single model. Though the diffusion module uses flat-coordinate diffusion rather than Riemannian flow matching, the architecture establishes that generative models — rather than discriminative predictors — are the right inductive bias for structural biology.

> [!remark] 10.1
> The competition between flow matching and diffusion in structural biology is a competition not between different mathematical frameworks — both can be viewed as instances of the generator matching framework of Chapter 6 — but between different inductive biases. Riemannian flow matching explicitly respects the $SE(3)$ geometry of protein frames, producing rotation-equivariant trajectories by construction. AlphaFold 3's flat-coordinate diffusion achieves equivariance through the architecture (equivariant conditioning from the Evoformer) rather than the generative process. At current scales, both approaches produce excellent structures; the theoretical advantages of geometry-native generation are clearest at long sequence lengths where flat-coordinate models begin to show geometric artifacts.


## 10.3 Structure-Based Drug Design

Given the three-dimensional structure of a protein's binding pocket, **structure-based drug design** (SBDD) asks for a small molecule that fits the pocket — satisfying geometric complementarity, chemical compatibility, and energetic affinity. Framed as a generative problem, SBDD is conditional generation:

> [!equation] 10.1
> $$p(\text{molecule} \mid \text{pocket}) = \int p(\text{molecule} \mid \text{pose}, \text{pocket})\, p(\text{pose} \mid \text{pocket})\, d(\text{pose}),$$


where the "pose" is the binding geometry of the molecule relative to the pocket. The product-space structure — atom types (discrete) and coordinates (continuous) conditioned on a fixed geometric context — makes this a natural product-space flow matching problem.

**DiffSBDD** (Schneuing, Du, Harris, Didi, Jamasb, Igashov, Du, Blundell, Lio, Giri, Cryst, and Correia, 2022) established diffusion as a competitive approach: diffuse atom types and coordinates jointly, conditioned on the pocket point cloud. Flow matching variants followed in 2024, applying the FlowMol framework to the conditional setting, achieving improved binding affinity distributions and geometry validity scores on the CrossDocked benchmark.

The challenge in SBDD — and why purely generative approaches remain an active research area — is that binding affinity is not a simple function of geometry: it requires accounting for protein flexibility, solvation, and entropic effects that are not visible in the crystal structure. The best current models generate geometrically valid molecules with correct binding poses but do not yet reliably optimize for affinity. The integration of physics-based scoring (docking software, force field calculations) with learned generative models — either through guidance (Chapter 9) or through joint training — is the central open problem.

## 10.4 Protein-Ligand Docking

A closely related problem is **docking**: given both a protein pocket and a small molecule, predict the binding pose — the position and orientation of the molecule within the pocket. This is a geometric prediction problem, not a generative design problem, but it can be framed identically:

$$p(\text{pose} \mid \text{molecule}, \text{pocket}).$$

**DiffDock** (Corso, Stärk, Jere, Barzilay, and Jaakkola, ICLR 2023) framed docking as diffusion over the ligand's degrees of freedom — translations in $\mathbb{R}^3$, rotations in $SO(3)$, and torsion angles on the torus $\mathbb{T}^k$ — given a fixed protein pocket. Treating each degree of freedom with its appropriate geometry and noise schedule, DiffDock substantially outperformed previous docking algorithms on standard benchmarks (PoseBusters, PDBBind), particularly on flexible ligands where classical docking algorithms fail. The success of DiffDock demonstrated that the Riemannian geometry tools of Chapter 5 are not just theoretically appealing — they produce measurable accuracy improvements on hard instances.

**FlowDock** (Jing, Stark, Barzilay, and Jaakkola, 2024) replaced the diffusion process with Riemannian flow matching on the same product space $\mathbb{R}^3 \times SO(3) \times \mathbb{T}^k$, achieving competitive accuracy with substantially fewer inference steps. The flow matching formulation also enables confidence estimation: the variance of predicted poses across multiple runs provides an estimate of binding uncertainty, useful for prioritizing molecules in virtual screening campaigns.

## 10.5 Single-Cell Biology: Modeling Cell State Transitions

Mammalian biology operates through continuous state transitions: stem cells differentiate into specialized types, immune cells activate upon encountering pathogens, neurons fire and recover. Single-cell RNA sequencing (scRNA-seq) captures snapshots of cell states — gene expression profiles in $\mathbb{R}^{20000+}$ — but these snapshots are cross-sectional: any given cell is measured once and then destroyed, making it impossible to directly observe trajectories.

Flow matching is an unusually natural framework for this problem: cells can be modeled as particles evolving through a high-dimensional gene expression space, their trajectories governed by the biology of differentiation and activation. If one can learn the flow from early-stage snapshots to late-stage snapshots, the model captures the underlying dynamical process.

> [!equation] 10.2
> $$\mathcal{L}_\mathrm{scFM}(\theta) = \mathbb{E}_{t,\; c_0 \sim p_0^{\text{cell}},\; c_1 \sim p_1^{\text{cell}}}\!\left[\left\|v_\theta\!\left(c_t, t\right) - \left(c_1 - c_0\right)\right\|^2\right], \qquad c_t = (1-t)c_0 + tc_1,$$


where $c_0$ and $c_1$ are gene expression vectors measured at early and late time points. The coupling between $c_0$ and $c_1$ can be random (linear interpolant with independent pairing), OT-coupled (minimizing gene expression transport cost), or biology-informed (pairing cells by known lineage markers).

**CellOT** (Bunne, Stark, Gut, del Castillo, Levesque, Lehmann, Pelkmans, Rätsch, and Krause, ICML 2022) applied optimal transport to model perturbation responses in single-cell data — predicting how a cell population shifts when treated with a drug, without observing the same cells before and after treatment. The OT coupling provides a principled way to match pre-treatment and post-treatment cell populations, and the resulting transport map is the flow. **TrajectoryNet** (Tong, Huang, Wolf, Van Dijk, and Krishnaswamy, ICML 2020) pioneered the continuous normalizing flow approach to single-cell trajectories; flow matching extensions followed naturally, reducing training cost from ODE simulation at every gradient step to a single forward pass.

The fundamental challenge in single-cell flow modeling is **unbalanced transport**: cells proliferate, die, and differentiate at different rates, so the marginal distributions $p_0$ and $p_1$ do not have the same total mass. Standard OT and flow matching assume mass conservation; unbalanced formulations that account for cell proliferation and death are an active research direction.

## 10.6 Speech Synthesis and Audio Generation

**Text-to-speech** (TTS) synthesis — generating natural-sounding speech audio given a text input — is a high-dimensional continuous generation problem. The target audio is typically represented as a mel-spectrogram (a time-frequency representation) in $\mathbb{R}^{T \times F}$, where $T$ is the number of time frames and $F$ the number of frequency bins, with $T \times F$ on the order of $10^4$ to $10^5$ for a few seconds of speech.

**Voicebox** (Le, Vyas, Shi, Karrer, Saade, Ramahi, Kawas, Rossié, Ellis, Bengio, and Dupoux, NeurIPS 2023) was the first large-scale flow matching model for speech. Voicebox conditions on a text transcript and speaker identity to generate mel-spectrograms via a conditional flow matching objective on the mel-spectrogram space, using a full-sequence transformer architecture that processes the entire spectrogram jointly. Trained on 60,000 hours of English speech, Voicebox substantially outperformed previous TTS systems on naturalness metrics (word error rate after automatic transcription, speaker similarity) while enabling zero-shot voice synthesis — generating speech in any voice given only a short reference clip.

**Matcha-TTS** (Mehta, Gongal, Shivakumar, Dawalatabad, Gupta, and Srivastava, ICASSP 2024) demonstrated that competitive TTS could be achieved with a much smaller model: a UNet-based architecture with 18M parameters, trained on LJSpeech, achieving natural-sounding synthesis with 2–4 inference steps via ODE integration. **E2 TTS** (Eskimez, Wang, Yang, Yang, Chen, Chen, Shi, Isaman, Li, Li, and Liu, 2024) further simplified the approach to a pure flow matching objective on mel-spectrograms with minimal text conditioning, showing that very simple architectures suffice for high-quality synthesis when flow matching provides the training signal.

> [!remark] 10.2
> The success of flow matching in audio synthesis reflects a general principle: whenever the target domain is a high-dimensional real-valued space with no strong global structure — mel-spectrograms do not have Riemannian geometry in the relevant sense — the Euclidean flow matching framework applies directly, with no domain-specific modifications required. The domain knowledge enters entirely through the architecture (time-frequency UNets, transformers with attention over time) and the training data, not through the generative framework. This separation of concerns — framework provides the training principle, architecture provides the inductive bias — is one of the practical strengths of flow matching.


## 10.7 Language Generation with Discrete Flows

The most intriguing and contested application of flow matching to language is **discrete text generation** — generating natural language token by token, or mask by mask, from a language flow model rather than an autoregressive model.

As established in Chapter 5, discrete flow matching with the masking interpolant is equivalent to masked language model training: given a partially masked sequence, predict the true tokens. This connection transforms the entire masked language model literature — BERT (Devlin, Chang, Lee, and Toutanova, 2019), RoBERTa, DeBERTa — into instances of discrete flow matching, trained to estimate the conditional generator $R_t^\theta$ of the masking CTMC.

**Generative** use of masked models — actually sampling text rather than re-ranking or scoring it — requires running the CTMC backwards: start from a fully masked sequence and progressively unmask, using the learned rate model to determine which positions to reveal and what tokens to place there. **MDLM** (Masked Diffusion Language Model; Sahoo, Arriola, Shi, Gokaslan, Maretic, Ju, Salamanca, and Rush, ICML 2024) showed that careful schedule design and training stabilizations enable competitive generative performance: perplexity scores on text generation benchmarks that approach, within a factor of two, the performance of autoregressive models of comparable size.

**SEDD** (Score Entropy Discrete Diffusion; Lou, Meng, and Ermon, ICML 2024) took a different approach, deriving a score-matching-like objective for discrete diffusion by working with the ratio $p_t(y)/p_t(x)$ — the discrete analogue of the score function — and showing that this ratio can be estimated from data without computing the partition function. On language modeling benchmarks, SEDD achieved perplexity scores competitive with GPT-2 at the same scale.

**Plaid** (Gulrajani and Hashimoto, 2024) argued that the right comparison for discrete diffusion language models is not against autoregressive models (which have very different inductive biases) but against the theoretical optimum — the model that exactly captures the data distribution. Measuring the gap between learned models and this optimum reveals that both autoregressive and discrete diffusion models have comparable shortcomings, concentrated in long-range dependencies.

The question of whether discrete flow matching can close the gap with autoregressive language models remains open. The key structural difference: autoregressive models generate left-to-right in a fixed order, which is computationally efficient but biologically unnatural; discrete flow models generate in arbitrary order, which is more flexible but requires solving a more complex inference problem. For scientific sequences — proteins, DNA — the arbitrary-order property is arguably more natural and may confer real advantages.

## 10.8 Weather and Climate Forecasting

Global weather forecasting is a canonical example of a high-stakes, high-dimensional scientific prediction problem with a known (but expensive) simulator. A global weather state — temperature, wind, pressure, humidity at each point on a grid — lives in a space of dimension $\sim 10^6$; the next day's state is determined by the Navier-Stokes equations, which are solved by numerical weather prediction (NWP) models at enormous computational cost. Operational forecasting centers (ECMWF, NOAA) run NWP models at supercomputer scale.

The case for generative models in weather forecasting is not to replace NWP but to replace the ensemble component: instead of running 50 slightly perturbed NWP simulations to estimate forecast uncertainty, run one NWP simulation and use a generative model to sample from the distribution of plausible futures conditioned on the current analysis state. This is conditional generation:

> [!equation] 10.3
> $$p\!\left(\text{state}_{t+\tau} \;\Big|\; \text{state}_t\right) \approx p_\theta\!\left(x \;\Big|\; c_t\right),$$


where $c_t$ is a summary of the current atmospheric state and $x$ is a sample from the predicted distribution at lead time $\tau$. The generative model must capture the full uncertainty of the forecast — not just the mean, but the spread of plausible outcomes — which is exactly what a well-calibrated probabilistic generative model provides.

**GenCast** (Price, Sanchez-Gonzalez, Amos, Kossen, El-Kadi, Masters, Dueben, Chantry, Garg, Hall, Mohamed, and Battaglia, Nature 2025) demonstrated that a diffusion model trained on ERA5 reanalysis data can generate ensemble forecasts that match or exceed the calibration of ECMWF's operational 50-member ensemble at a fraction of the computational cost. GenCast generates each ensemble member in $\sim 8$ minutes on a single GPU, compared to hours for an NWP ensemble run, enabling on-demand ensemble generation for applications that previously required supercomputing access. The model captures extreme event distributions — the tails of the forecast distribution that matter most for disaster preparedness — with accuracy comparable to the operational ensemble.

**ClimODE** (Verma, Heinonen, and Garg, 2024) applied continuous normalizing flows to global weather modeling, explicitly connecting the flow matching framework to the underlying atmospheric dynamics: the learned velocity field is interpretable as a transport operator on the space of atmospheric states. The connection to the physics — rather than treating the atmosphere as an unstructured high-dimensional space — provides additional structure that improves long-range forecast skill.

## 10.9 Physical Simulation and Inverse Problems

The flow matching framework is well-suited to two related problems in computational physics: **surrogate modeling** (learning an approximation to an expensive simulator) and **inverse problems** (inferring the cause from the observed effect).

**PDE surrogates.** For partial differential equations — fluid dynamics, heat transfer, quantum mechanics — the computational cost of a single simulation may be hours. Flow matching can learn the mapping from initial conditions (or boundary conditions) to solution fields:

> [!equation] 10.4
> $$p(\text{solution field} \mid \text{initial condition}) \approx p_\theta(x \mid c),$$


generating approximate solutions at near-zero marginal cost after training. The advantage over standard neural operators (FNO, DeepONet) is that a generative model captures the full distribution over solutions — relevant for chaotic systems where small perturbations lead to large output variation — rather than predicting only the mean. **PDEDM** and related work (2023–2024) demonstrated flow matching surrogates for turbulent flow simulations, achieving ten to hundredfold speedups over classical solvers with acceptable accuracy for ensemble statistics.

**Inverse problems.** In many branches of physics and imaging — cryo-EM structure determination, astronomical deconvolution, medical tomography — the observation $y$ is a lossy function of the true signal $x$: $y = A(x) + \epsilon$ where $A$ is a known (or learnable) forward model and $\epsilon$ is noise. Bayesian inference seeks the posterior $p(x | y)$. Flow matching provides a natural architecture: train a conditional flow model $v_\theta(x, t | y)$ that transports from a prior $p_0$ to the posterior $p(x | y)$ by conditioning the velocity field on the observation $y$.

**CryoFM** and related cryogenic electron microscopy (cryo-EM) flow models (2024) learn the posterior distribution over protein conformations given a noisy micrograph stack — a problem where both the observation model (Fourier projection with CTF blurring) and the prior over protein structures are well-characterized. By training a flow matching model conditioned on the micrograph, these approaches generate structurally diverse conformational ensembles from single-particle cryo-EM data, resolving the continuous conformational heterogeneity that static structure determination methods cannot capture.

## 10.10 Materials Science and Crystal Structure Generation

The generative modeling of crystalline materials — inorganic solids, pharmaceutically relevant polymorphs, metal-organic frameworks — presents a distinct set of challenges. Crystal structures live in a space that combines continuous coordinates (fractional atomic positions within the unit cell), discrete atom types, and a continuous structural variable (the lattice parameters, defining the unit cell geometry). The periodicity of crystals introduces translational and rotational symmetry constraints that must be respected exactly.

**DiffCSP** (Jiao, Lin, Wood, and Schütt, ICLR 2024) applied diffusion to crystal structure prediction, using separate processes for lattice parameters and fractional coordinates. **FlowMM** (Miller, Smidt, Mandt, and Sriram, 2024) extended this to flow matching, treating the lattice parameters with an unconstrained continuous flow and the fractional coordinates with a periodic flow on the torus — a natural product-space application of the Chapter 7 framework. The discrete atom types are handled with a CTMC masking process.

The practical relevance is significant: crystal structure prediction has been a grand-challenge problem in computational chemistry since Linus Pauling's era, with implications for pharmaceutical polymorph screening, materials discovery, and battery electrode design. The success of flow matching on this problem — achieving competitive performance with classical methods at a fraction of the computational cost — reflects the generality of the product-space framework.

> [!remark] 10.3
> A recurring theme across these applications is that the domain-specific challenges — crystal periodicity, protein frame equivariance, cell mass imbalance, atmospheric chaos — enter the framework through specific choices in the design space of Chapter 6: state space, interpolant, generator class, Bregman divergence. The generator matching framework's value as an engineering tool is precisely that it separates the domain-specific choices (what space, what noise) from the domain-agnostic machinery (training is simulation-free, Theorem 6.1 applies). New applications do not require new theory — they require new design choices within the existing theory.


## 10.11 Evaluation in Scientific Domains

Evaluating generative models in scientific applications is fundamentally different from evaluating them on images or text. The standard metrics of image generation — FID, Inception Score, human preference ratings — measure distributional similarity to training data and perceptual quality. In scientific applications, what matters is not resemblance to training data but **physical validity** and **functional utility**.

The metrics differ by domain:

**Proteins.** The primary metric is **designability**: for each generated backbone, use ProteinMPNN to design a sequence, predict the structure of that sequence with ESMFold or AlphaFold 2, and measure the backbone RMSD between the generated structure and the predicted folded structure. A backbone is "designable" if this self-consistency RMSD is below 2Å. Designability measures whether the generated backbone has a sequence that would actually fold into it — a physically necessary condition for a useful protein design. Additional metrics include **novelty** (TM-score to the nearest known protein) and **diversity** (spread of generated structures).

**Small molecules.** The standard metrics include **validity** (whether the generated graph represents a chemically valid molecule according to valence rules), **uniqueness** (fraction of generated molecules that are distinct), and **novelty** (fraction not in the training set). For SBDD, the binding affinity (Vina score) against the target pocket and the geometry validity (no steric clashes, correct bond angles) are primary metrics. The **PoseBusters** benchmark (Buttenschoen, Morris, and Deane, 2024) introduced a comprehensive set of physical validity checks that exposed significant failure modes in early docking models.

**Weather.** The primary metric is **calibration**: the spread of the generated ensemble should match the observed spread of outcomes. A well-calibrated 10-member ensemble should bracket the true outcome 10% more often than a poorly calibrated 10-member ensemble. Secondary metrics include **RMSE** of the ensemble mean at various lead times and **CRPS** (Continuous Ranked Probability Score), which measures both accuracy and calibration jointly.

> [!remark] 10.4
> A persistent challenge in scientific generative modeling is the **evaluation gap**: a model may achieve excellent benchmark scores while failing on the actual task of interest. In protein design, designability measures self-consistency (does ProteinMPNN-ESMFold agree with itself?), not experimental success (does the protein actually fold and function?). In drug design, Vina binding scores are computational approximations that are known to over-predict the activity of flat aromatic compounds and under-predict the activity of fragments. The gap between benchmark performance and experimental success is a known difficulty, addressed partly by more rigorous benchmarks (PoseBusters, Wetlab validation campaigns) and partly by integrating experimental feedback into the generative model via reinforcement learning or active learning.


## 10.12 Historical Notes and Emerging Directions

The applications surveyed in this chapter span roughly three years of intense activity — 2022 to 2025 — a period in which flow matching was proposed, theoretically developed, and deployed at scale across multiple scientific domains simultaneously. The speed reflects both the generality of the framework and the maturity of the upstream ML infrastructure (large-scale compute, equivariant architectures, efficient ODE solvers) on which it builds.

Several themes emerge from the history of these applications.

**Scientific problems led theoretical development.** The Riemannian extension (Chapter 5) was motivated by protein structure generation, not by abstract geometry. The discrete flow matching framework (Chapter 5) was motivated by sequence co-design, not by information theory. The product-space framework (Chapter 7) was motivated by joint protein generation, not by product topology. Theory followed application, clarifying what the applications were actually doing.

**Benchmarks drove progress.** The sequence of CASP competitions (protein structure prediction), PoseBusters (molecular docking), and WeatherBench (weather forecasting) provided the forcing function for methodological improvement. Generative models without clear evaluation criteria tend to produce impressive samples that don't withstand scrutiny; benchmarks impose the discipline of measurable progress.

**Experimental validation remains the bottleneck.** The most important open question in scientific generative modeling is not computational but experimental: can generated proteins be synthesized, folded, and made functional? Can generated drug candidates be synthesized and shown to bind? The computational metrics (designability, Vina score, RMSD) are proxies, and the proxies have known failure modes. The integration of experimental feedback — using wet-lab results to update generative models through active learning — is the frontier at which the computational and experimental sciences are currently converging.

**The emerging domains.** Beyond the applications covered here, flow matching is actively being explored for RNA structure prediction and design (extending AlphaFold 3's all-atom framework), antibody design (generating complementarity-determining regions with specific binding properties), enzyme engineering (generating active-site geometries with catalytic function), and materials property optimization (generating crystal structures conditioned on target band gaps or mechanical properties). In each case, the framework is the same; the domain knowledge enters through the design space.

The monograph ends here, at the edge of the known. The theory is complete enough to be useful; the applications are advanced enough to be impactful; the open questions are sharp enough to be productive. Flow matching is not the end of generative modeling for science — it is, more likely, the foundation on which the next decade of scientific machine learning will be built.
