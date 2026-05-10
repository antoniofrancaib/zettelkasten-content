---

title: "The Heterogeneous World"
subtitle: "Co-generating continuous and discrete data on product spaces — 2024 to 2025"
---



The preceding six chapters have treated data as living in a single space: real-valued vectors in $\mathbb{R}^d$, tokens in a finite alphabet $\mathcal{A}^L$, frames in $SE(3)^N$, probability vectors on $\Delta^{d-1}$. In each case, the entire data object is of one type, the generator is of one structural class, and the training objective is a single Bregman divergence applied to a single generator output. Real scientific data is rarely so homogeneous.

A protein is not a structure. It is simultaneously a structure — a sequence of rigid frames in $SE(3)^N$ — and a sequence — a string of amino acids in $\{A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y\}^N$. The two representations are not independent: the sequence determines which protein folds, and the structure determines which sites are functional, which sequence positions tolerate mutations, and which drugs will bind. Generating a structure without a sequence is generating a backbone without chemistry; generating a sequence without a structure is generating chemistry without a shape. Neither is the object of biological interest. What is wanted is joint generation: a model that samples coherent (structure, sequence) pairs simultaneously, from a learned joint distribution over the product space $SE(3)^N \times \mathcal{A}^N$.

This chapter develops the theory and practice of generative models on product spaces — Cartesian products of factors that may be continuous or discrete, Euclidean or curved, each with its own generator class and Bregman divergence. The framework of Chapter 6 was designed with product spaces in mind; this chapter works out what the design requires in detail.

## 7.1 The Architecture of Scientific Data

The product-space structure of scientific data is not an edge case — it is the rule.

**Proteins.** A protein of $N$ residues is described by a backbone structure $(R_i, t_i)_{i=1}^N \in SE(3)^N$ (rotations and translations for each residue frame) and an amino acid sequence $(a_i)_{i=1}^N \in \mathcal{A}^N$ where $|\mathcal{A}| = 20$. Full structural prediction requires also the side-chain conformation — dihedral angles $(\phi_i, \psi_i, \chi_i^{(1)}, \ldots) \in \mathbb{T}^{k_i}$ for each residue, where $k_i$ depends on the side-chain size. A complete protein model thus lives in $SE(3)^N \times \mathcal{A}^N \times \mathbb{T}^K$ — a product of three different types of spaces.

**Small molecules.** A molecule with $N$ atoms is described by atom types $(z_i)_{i=1}^N \in \mathcal{Z}^N$ (periodic table symbols, or a coarser vocabulary), three-dimensional coordinates $(r_i)_{i=1}^N \in \mathbb{R}^{3N}$, and optionally a bond graph $\mathcal{G}$ over the atoms (encoding single, double, triple, and aromatic bonds as discrete edge labels). The product space is $\mathcal{Z}^N \times \mathbb{R}^{3N} \times \mathcal{B}^{E}$, where $E$ counts bonds.

**Nucleic acids.** DNA or RNA with $L$ bases lives in $\{A,C,G,T\}^L$ (sequence) $\times$ $(S^1)^{L-1}$ (backbone dihedral angles) $\times$ $\mathbb{R}^{3L}$ (3D coordinates). The sequence determines the chemistry; the dihedrals determine the secondary structure; the coordinates give the tertiary fold.

**Language grounded in structure.** In molecular language modeling — generating SMILES strings, IUPAC names, or natural-language descriptions alongside 3D structures — the product space includes a discrete token sequence and a continuous geometric representation, with semantic alignment requirements between the two.

In every case, the object of scientific interest is the *joint* distribution over the full product space. Marginal models — generating only the sequence, or only the structure — are useful surrogates, but the biological problem is posed jointly.

## 7.2 Product-Space Generators

Let $\mathcal{X} = \mathcal{X}_c \times \mathcal{X}_d$ be a product of a continuous space $\mathcal{X}_c$ (a Euclidean space, Riemannian manifold, or product thereof) and a discrete space $\mathcal{X}_d$ (a finite alphabet, or a product of finite alphabets). A Markov process on $\mathcal{X}$ has a generator $\mathcal{L}_t$ acting on functions $f : \mathcal{X}_c \times \mathcal{X}_d \to \mathbb{R}$.

> [!definition] 7.1 — Product-Space Markov Process
> A **product-space Markov process** on $\mathcal{X}_c \times \mathcal{X}_d$ is one in which, conditionally on the joint state $(x_c, x_d) \in \mathcal{X}_c \times \mathcal{X}_d$, the continuous and discrete components evolve independently. Its generator decomposes as:
>
> $$(\mathcal{L}_t f)(x_c, x_d) = \bigl(\mathcal{L}_t^c\, f(\,\cdot\,, x_d)\bigr)(x_c) + \bigl(\mathcal{L}_t^d\, f(x_c, \,\cdot\,)\bigr)(x_d),$$
>
> where $\mathcal{L}_t^c$ is a generator on $\mathcal{X}_c$ (an ODE generator, an SDE generator, or a Riemannian flow generator) and $\mathcal{L}_t^d$ is a CTMC generator on $\mathcal{X}_d$.


The generator decomposition in Definition 7.1 can be written compactly as:

> [!equation] 7.1
> $$\mathcal{L}_t = \mathcal{L}_t^c \otimes I_d \;+\; I_c \otimes \mathcal{L}_t^d,$$


where $I_c$ and $I_d$ are identity operators on functions of $x_c$ and $x_d$ respectively. This is the operator-algebraic statement that the two generators act on orthogonal factors of the product space.

The adjoint $\mathcal{L}_t^* = (\mathcal{L}_t^c)^* \otimes I_d + I_c \otimes (\mathcal{L}_t^d)^*$ acts on the joint density $p_t(x_c, x_d)$. The Kolmogorov forward equation (6.2) on the product space is:

> [!equation] 7.2
> $$\frac{\partial}{\partial t} p_t(x_c, x_d) = (\mathcal{L}_t^{c*}\, p_t)(x_c, x_d) + (\mathcal{L}_t^{d*}\, p_t)(x_c, x_d).$$


The first term is the continuity equation (or Fokker-Planck equation) for the continuous component, marginalizing over $x_d$; the second is the Kolmogorov forward equation for the discrete component, marginalizing over $x_c$. Equation (7.2) says that probability mass flows through the product space simultaneously via two mechanisms: continuous transport and discrete jumping.

## 7.3 Joint Interpolants and Independent Coupling

To apply the conditional generator matching framework (Theorem 6.1) on $\mathcal{X}_c \times \mathcal{X}_d$, we need a family of conditional paths $p_t(\cdot|x_1)$ where $x_1 = (x_1^c, x_1^d) \in \mathcal{X}_c \times \mathcal{X}_d$ is a data point.

The simplest choice is **independent coupling**: define the joint conditional path as the product of the marginal conditional paths,

$$p_t(x_c, x_d \mid x_1^c, x_1^d) \;=\; p_t^c(x_c \mid x_1^c)\;\cdot\; p_t^d(x_d \mid x_1^d).$$

Under independent coupling, the conditional generator factorizes as well. For each data point $(x_1^c, x_1^d)$ and each joint state $(x_t^c, x_t^d)$, the conditional generator statistic is the pair:

$$\ell_t\!\left((x_t^c, x_t^d) \mid (x_1^c, x_1^d)\right) = \bigl(\ell_t^c(x_t^c \mid x_1^c),\; \ell_t^d(x_t^d \mid x_1^d)\bigr).$$

The continuous target $\ell_t^c$ is the velocity or score for the continuous interpolant (e.g., $(x_1^c - x_t^c)/(1-t)$ for the linear interpolant); the discrete target $\ell_t^d$ is the rate row for the masking interpolant (e.g., $R_t(\text{[M]}, x_1^d\,|\,x_1^d) = 1/(1-t)$). Each is computed from the known conditional path without access to the marginal distribution.

> [!remark] 7.1
> Under independent coupling, the conditional targets for the two components are entirely decoupled — the continuous target depends only on $(x_t^c, x_1^c)$ and the discrete target depends only on $(x_t^d, x_1^d)$. The *model*, however, is not constrained to be factorized: a neural network that predicts both outputs can condition each on the joint state $(x_t^c, x_t^d)$. The training targets factorize; the model does not. This asymmetry is the key to joint generation under independent coupling: the model learns cross-modal dependencies from data, even though the training signal is computed modality by modality.


## 7.4 The Joint Conditional Generator Matching Objective

The conditional generator matching objective (6.5) extends verbatim to product spaces. With a continuous ODE generator and squared-norm Bregman divergence for $\mathcal{X}_c$, and a CTMC generator and KL Bregman divergence for $\mathcal{X}_d$, the joint objective is:

> [!equation] 7.3
> $$\mathcal{L}_\mathrm{joint}(\theta) = \mathbb{E}_{t,\; x_1,\; x_t | x_1}\!\left[\lambda_c \cdot \left\|\ell_t^{c\theta}(x_t) - \ell_t^c(x_t^c \mid x_1^c)\right\|^2 + \lambda_d \cdot D_\mathrm{KL}\!\left(\ell_t^d(x_t^d \mid x_1^d)\;\Big\|\;\ell_t^{d\theta}(x_t)\right)\right],$$


where $\lambda_c, \lambda_d > 0$ are weighting hyperparameters, the model $\theta$ processes the joint state $x_t = (x_t^c, x_t^d)$ to predict both the continuous generator output $\ell_t^{c\theta}$ and the discrete generator output $\ell_t^{d\theta}$, and the training samples $(x_1, x_t)$ are drawn from the joint data distribution and joint conditional path respectively.

Theorem 6.1 applies directly: the gradient of (7.3) with respect to $\theta$ equals the gradient of the corresponding marginal joint generator matching objective. The proof is unchanged — the Bregman bias-variance decomposition (6.3) holds component-by-component, and the two terms in (7.3) both satisfy it independently.

Training with (7.3) requires nothing beyond: sample a data point $(x_1^c, x_1^d)$ from the dataset, sample a time $t \in [0,1]$, sample $(x_t^c, x_t^d)$ from the known conditional path (one Euclidean or geodesic interpolation step plus one masking operation), evaluate the conditional targets, and take a gradient step. No simulation, no marginalization, one forward pass per gradient step — just as in all the preceding chapters.

## 7.5 The Coordination Problem

Independent coupling gives a tractable training procedure, but it does not guarantee *semantic coherence* between the two modalities at intermediate times. Under independent coupling, the state $(x_t^c, x_t^d)$ at time $t \in (0,1)$ is drawn from $p_t^c(\cdot|x_1^c) \otimes p_t^d(\cdot|x_1^d)$: the continuous component is an interpolant between reference noise and the target structure, and the discrete component is an independently masked version of the target sequence. There is no constraint ensuring that $x_t^c$ (a partially noised backbone) and $x_t^d$ (a partially masked sequence) are "compatible" with each other.

The model's response to this incoherence is to learn, from data, what structure-sequence pairs are physically meaningful — relying on the rich statistics of the training distribution rather than on any structural constraint in the interpolant. In practice, this works: the shared-architecture model develops strong cross-modal conditioning, predicting the next token given the current partial structure and the current partial sequence simultaneously. But the quality of this coordination depends heavily on the training data, the architecture, and the balance between the two losses.

Two alternative approaches address the coherence problem more directly.

**Correlated joint interpolants.** One can define a joint conditional path $p_t(x_c, x_d | x_1^c, x_1^d)$ that maintains explicit correlations at intermediate $t$ — for instance, by coupling the unmasking schedule of the sequence to the denoising schedule of the structure, so that as the structure becomes clearer, the sequence becomes less masked at the same rate. Designing such correlated interpolants is a non-trivial problem in general, because ensuring that the resulting path is a valid Markov process on $\mathcal{X}_c \times \mathcal{X}_d$ requires careful construction.

**Sequential conditioning.** Alternatively, one can train separate models: first a structure model that generates $x_1^c \sim p^c$, then a sequence model that generates $x_1^d | x_1^c \sim p^d(\cdot | x_1^c)$. This was the dominant approach before joint generation models appeared — ProteinMPNN and LigandMPNN generate sequences conditioned on fixed structures generated by separate diffusion models. Sequential approaches are modular and interpretable but cannot model the joint uncertainty: they cannot represent structures that are plausible only given a particular sequence, or sequences that are good only for a particular structural context.

The advantage of joint generation is precisely this: it models the full joint distribution $p(x^c, x^d)$, capturing the correlations between modalities during generation rather than conditioning one on a fixed sample of the other.

## 7.6 Protein Co-Design: Sequence and Structure Together

The problem that most forcefully motivated product-space flow matching is **protein co-design** — the simultaneous generation of a protein's backbone structure and its amino acid sequence, such that the sequence is designable for the generated backbone and the backbone is realizable by the generated sequence.

The two tasks are deeply coupled. In protein design, a common failure mode of sequential approaches is **hallucination**: the structure generator produces a backbone that looks geometrically plausible but for which no amino acid sequence exists that would fold into it. Joint generation avoids this by forcing the model to learn, during training, which (backbone, sequence) pairs are consistent — i.e., which pairs appear in the PDB.

The state space is:

$$\mathcal{X} = \underbrace{SE(3)^N}_{\text{backbone frames}} \;\times\; \underbrace{\mathcal{A}^N}_{\text{amino acids}},$$

where $N$ is the number of residues and $|\mathcal{A}| = 20$. The reference distribution is a uniform distribution over noise frames on $SE(3)^N$ and an all-masked sequence $[\text{M}]^N$.

The conditional path under independent coupling is:
- **Structure component**: geodesic interpolant on $SE(3)^N$, separately for translation ($\mathbb{R}^3$) and rotation ($SO(3)$). Conditional velocity $(v_t^{(i)})_{i=1}^N$ where each $v_t^{(i)} = (\log_{R_t^{(i)}}(R_1^{(i)})/(1-t),\; (t_1^{(i)} - t_t^{(i)})/(1-t))$.
- **Sequence component**: masking interpolant. Each position $i$ independently: $p_t(a^{(i)}|a_1^{(i)}) = (1-t)\delta_{[\text{M}]} + t\,\delta_{a_1^{(i)}}$.

The model is a transformer that processes the joint state $(x_t^{\text{struct}}, x_t^{\text{seq}})$ — both partially noised structures and partially masked sequences — and predicts both the per-residue structure velocity and the per-position token probabilities.

## 7.7 MultiFlow: Architecture and Training

**MultiFlow** (Campbell, Yim, Barzilay, Jaakkola, and Gat, NeurIPS 2024) is the first model to realize this program at scale. It takes as input the joint state $(x_t^{\text{struct}}, x_t^{\text{seq}})$ — noisy backbone frames and masked sequences — and outputs both a velocity field over $SE(3)^N$ and a per-position token distribution over $\mathcal{A}$.

The training objective is equation (7.3) instantiated for proteins:

$$\mathcal{L}_\mathrm{MultiFlow}(\theta) = \mathbb{E}\!\left[\lambda_c \sum_{i=1}^N \left\|v_\theta^{(i)}(x_t, t) - \frac{\log_{x_t^{(i)}}(x_1^{(i)})}{1-t}\right\|_{SE(3)}^2 + \lambda_d \sum_{i=1}^N \mathcal{H}\!\left(a_1^{(i)},\; \hat{p}_\theta^{(i)}(x_t, t)\right)\right],$$

where $\mathcal{H}(a_1^{(i)}, \hat{p}^{(i)})$ is the cross-entropy of the model's predicted amino acid distribution against the true residue, and $\|\cdot\|_{SE(3)}^2$ is the squared $SE(3)$ norm combining translational and rotational components.

The architecture uses **Invariant Point Attention** (IPA) — the geometric attention mechanism from AlphaFold2 that processes protein residue features while respecting $SE(3)$ equivariance — extended with cross-attention over the joint (structure, sequence) representation. Masked residues contribute a learned mask token embedding; unmasked residues contribute their one-hot amino acid embedding alongside the frame representation.

> [!remark] 7.2
> The architecture choice is not incidental. For joint generation to capture structure-sequence correlations, the model must be able to condition the velocity of a residue's frame on the identity of surrounding residues, and to condition the predicted token at position $i$ on the current backbone geometry at nearby positions. IPA provides exactly this: it computes attention weights that are sensitive to both sequence distance and spatial distance, allowing the model to integrate local structural context into its sequence predictions and vice versa. The architecture implements the coordination that the independent-coupling training objective does not enforce.


## 7.8 Joint Inference: Interleaving ODE and CTMC Dynamics

Sampling from a trained joint model requires simulating the product-space Markov process: a continuous ODE for the structure component and a CTMC for the sequence component, running simultaneously from $t = 0$ to $t = 1$.

The standard approach is **$\tau$-leaping with Euler steps**: discretize $[0,1]$ into $T$ steps of size $\Delta t = 1/T$, and at each step:

> [!equation] 7.4
> $$x_{t+\Delta t}^{(i),\, \mathrm{struct}} = \mathrm{Exp}_{x_t^{(i),\, \mathrm{struct}}}\!\left(\Delta t \cdot v_\theta^{(i)}(x_t, t)\right), \quad i = 1, \ldots, N,$$


for the structure component (an exponential map step on $SE(3)$, as in Chapter 5), and for each masked position $j$:

$$a_{t+\Delta t}^{(j)} = \begin{cases} a^* & \text{with probability } R_t^\theta\!\left(\text{[M]}, a^*\right) \cdot \Delta t,\\ \text{[M]} & \text{otherwise,}\end{cases}$$

where $a^* = \argmax_a \hat{p}_\theta^{(j)}(x_t, t)$ is the predicted amino acid (or a sample from $\hat{p}_\theta^{(j)}$ for stochasticity).

The joint state after each step is passed back into the model for the next step. Structure and sequence co-evolve: as the backbone becomes more defined (closer to a real protein conformation), the sequence predictions become more confident, and the masking rate $1/(1-t)$ ensures that all positions are unmasked by $t = 1$.

> [!remark] 7.3
> The $\tau$-leaping approximation treats the CTMC rates as constant over each interval $[t, t+\Delta t)$, which is exact in the limit $\Delta t \to 0$ and is accurate in practice for $\Delta t \leq 0.01$. The number of function evaluations required — typically 100–500 for protein-scale generation — is the same for both modalities, since the ODE and CTMC advance together. This contrasts with purely diffusion-based models for joint generation, which may require different numbers of steps for the two components due to different noise-to-signal schedules.


## 7.9 Small Molecule Generation on Graph-Geometry Product Spaces

Proteins are the most mature application domain, but product-space flow matching has been applied across the full range of molecular structures.

**Small molecule generation.** A drug-like molecule with $N$ heavy atoms lives in $\mathcal{Z}^N \times \mathbb{R}^{3N}$ — a vector of atom types (from a vocabulary of, say, 15 heavy-element types) and a matrix of Cartesian coordinates. The two components are coupled by the physics: two carbon atoms with a bond must be approximately 1.5Å apart; an aromatic ring must be planar. Generating atom types and coordinates independently and composing them post-hoc produces geometrically nonsensical molecules; joint generation is essential.

**EDM and flow-matching extensions.** The equivariant diffusion model (Hoogeboom, Satorras, Vignac, and Welling, ICML 2022) introduced joint atom-type-plus-coordinate generation using a product of a multinomial diffusion (for types) and a coordinate diffusion (for positions) with an equivariant architecture. The flow matching extension — using a CTMC for atom types and a continuous flow for coordinates — is a direct product-space application of the framework developed here. Models in this family include FlowMol (Dunn and Koes, 2024), which uses independent coupling with a linear interpolant for coordinates and a masking interpolant for atom types, trained jointly via (7.3), achieving state-of-the-art performance on drug-like molecule generation benchmarks.

**Bond types.** Adding bond graph generation introduces a third factor: the edge labels $\mathcal{B}^E$ over all possible atom pairs. This is a higher-order product space with interactions between atom types, coordinates, and bond types. The standard approach is to treat bond type prediction as a function of atom type and coordinate predictions at inference time (using RDKit or a separate classifier), avoiding the need to generate bond types during the flow. Full end-to-end joint generation of atom types, coordinates, and bonds remains an active research direction.

## 7.10 OT Coupling on Product Spaces

The mini-batch optimal transport coupling of Chapter 4 (Section 4.5) extends to product spaces, though with additional complexity. In the single-modality case, OT conditioning finds a coupling $\pi(x_0, x_1)$ that minimizes the expected transport cost $\mathbb{E}[c(x_0, x_1)]$; straighter trajectories reduce the number of function evaluations needed at inference.

On a product space $\mathcal{X}_c \times \mathcal{X}_d$, a product-space OT coupling requires a cost function that accounts for both modalities. A natural choice is:

$$c\!\left((x_0^c, x_0^d),\, (x_1^c, x_1^d)\right) = \alpha\,d_{c}(x_0^c, x_1^c)^2 + \beta\,d_{d}(x_0^d, x_1^d)^2,$$

where $d_c$ is the geometric distance on $\mathcal{X}_c$ (e.g., the $SE(3)$ metric for protein frames) and $d_d$ is a distance on $\mathcal{X}_d$ (e.g., the Hamming distance between amino acid sequences, or the sequence edit distance). Minimizing this cost jointly assigns reference noise frames to target structures *and* reference mask tokens to target sequences simultaneously, in a way that minimizes the total cost.

In practice, joint OT coupling on product spaces is computationally expensive because the transport problem lives in a much larger space than in the single-modality case. FoldFlow-2 applied OT conditioning to the structure component only, with sequence conditioning handled through the architecture. Full joint OT coupling — simultaneously matching sequence and structure noise to data — is a technically challenging open problem, requiring efficient approximations to the joint OT problem on the combined $SE(3)^N \times \mathcal{A}^N$ space.

## 7.11 What Product Spaces Reveal

Working with product-space models has clarified several aspects of the framework that are less visible in the single-modality case.

**The generator decomposition is an idealization.** Equation (7.1) describes the generator of a process in which the two components evolve conditionally independently — they do not interact dynamically. Real protein co-evolution is not like this: the sequence and structure co-evolve in a coupled way, with each residue's chemical identity influencing the geometry of its neighbors and vice versa. The independent-coupling training target reflects this idealization; the neural network's cross-modal attention compensates by learning the coupling empirically. A natural research direction is to develop joint interpolants that are not product-form — that maintain explicit correlations between modalities at each intermediate time.

**The weighting $\lambda_c / \lambda_d$ matters.** In the joint objective (7.3), the relative weight of the structure loss and the sequence loss is a hyperparameter with significant practical consequences. If $\lambda_c \gg \lambda_d$, the model prioritizes structural accuracy at the cost of sequence diversity; if $\lambda_d \gg \lambda_c$, the reverse. Optimal weighting depends on the downstream application (design versus classification, de novo versus motif-scaffolding) and is typically tuned by validation on designability metrics. The Bregman framework provides a principled perspective: the two terms correspond to different divergences (squared norm vs. KL), and their natural scales are determined by the geometry of each factor.

**Architecture and objective interact.** In all single-modality models, the architecture and objective are largely orthogonal: any architecture that outputs the right type of vector can be trained with the flow matching objective. In joint models, the architecture must actively facilitate cross-modal information flow — structure predictions must condition on sequence state and vice versa — otherwise the model degenerates to two independent single-modality models with shared parameters but no coupling. The interaction between architecture design and joint-objective training is a distinctive feature of the product-space setting.

> [!remark] 7.4
> The product-space framework also clarifies what "multimodal generation" means at the level of the Markov process. Two modalities are genuinely jointly generated when the Markov process on the product space has a non-product-form generator — when the velocity of the continuous component depends on the discrete state, and vice versa. Under independent coupling with a cross-modal architecture, this non-product-form structure enters through the learned generator $\mathcal{L}_t^\theta$, even though the training target (the conditional generator statistic) is product-form. Conditional generator matching thus trains a non-factorized generator using a factorized target — a subtle but important distinction.


## 7.12 Historical Notes

**Protein co-design as a benchmark.** The joint structure-sequence generation problem has been a benchmark for protein generative models since the introduction of ProteinMPNN (**Dauparas, Anishchenko, Bennett, Beem, Ragotte, Milles, Wicky, Courbet, de Haas, Bethel, Leung, Huddy, Pellock, Tischer, Chan, Koepnick, Nguyen, Kang, Schlichthaering, Dalton, Dimaio, Trevizán, Brunette, and Baker, Science 2022**), a sequence design model that conditions on a fixed backbone structure. ProteinMPNN is a *sequential* model — it generates sequences given structures — not a joint model. The move to joint generation was driven by the observation that sequential models, by fixing the structure before generating the sequence, cannot explore the joint distribution: they cannot generate sequences that only make sense for structures they haven't seen, or backbones that require specific chemical interactions to fold correctly.

**MultiFlow.** The first joint sequence-structure flow matching model is **MultiFlow** (**Campbell, Yim, Barzilay, Jaakkola, and Gat, NeurIPS 2024**). It combines a continuous flow on $SE(3)^N$ for backbone frames (geodesic interpolant, IPA architecture) with a discrete CTMC on $\mathcal{A}^N$ for amino acids (masking interpolant, cross-entropy loss). Training uses the product-space objective (7.3); inference uses the $\tau$-leaping algorithm (7.4). MultiFlow generates proteins with higher designability scores than structure-only models, demonstrating that joint generation genuinely improves over sequential approaches.

**Concurrent work.** **Gat, Remez, Shaul, Kreuk, Chen, Synnaeve, Adi, and Lipman (NeurIPS 2024)** developed discrete flow matching simultaneously with an application to protein sequence and structure co-design. Their framework is equivalent to MultiFlow in the discrete component, with differences in the architecture and evaluation methodology.

**Small molecules.** The equivariant diffusion model of **Hoogeboom, Satorras, Vignac, and Welling (ICML 2022)** established joint atom-type-plus-coordinate generation as a tractable problem. The flow matching extension was developed in **Dunn and Koes (2024)** (FlowMol) and in parallel work on equivariant flow matching for molecular generation. **Jing, Stark, Barzilay, and Jaakkola (2024)** (FlowDock) extended the product-space framework to molecular docking — jointly generating the binding pose of a ligand given a protein pocket structure.

**The co-design problem in biology.** The scientific importance of joint generation stems from the sequence-structure duality of proteins, formalized in Anfinsen's dogma (1973): the amino acid sequence of a protein uniquely determines its three-dimensional structure under physiological conditions. This means that the *design* problem — specifying a protein to perform a desired function — is inherently a joint problem: one must simultaneously determine a structure with the right geometry and a sequence that folds into it. Generative models that capture the joint distribution over $(structure, sequence)$ pairs observed in nature are, in principle, the right tool for this problem. Chapter 10, on applications, will examine how well current models succeed in practice.
