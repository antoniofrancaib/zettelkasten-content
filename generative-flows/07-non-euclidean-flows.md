---

title: "Non-Euclidean"
subtitle: "Discrete flows via CTMCs and Riemannian flow matching — 2021 to 2025"
---



The generative flow framework — as developed through Chapter 6 — rests on two foundational assumptions: the data lives in $\mathbb{R}^d$, and the generative process is a differential equation on that space. Both assumptions break for the most scientifically important data types.

Protein sequences, DNA, natural language: drawn from finite alphabets where vectors and ODEs have no meaning. Protein structures, molecular conformations, point clouds on surfaces: continuous but curved, living on manifolds where straight lines exit the space and the Gaussian is the wrong reference measure.

This chapter develops flow matching for both non-Euclidean regimes. **Sections 7.1–7.11** address discrete state spaces, replacing ODEs with continuous-time Markov chains (CTMCs): the right language for jumping between tokens, residues, or graph states. **Sections 7.12–7.18** address curved continuous spaces, replacing Euclidean straight lines with geodesics: the right language for frames in $SE(3)$, dihedral angles on the torus, and probability vectors on the simplex. Both are special cases of the generator matching framework of Chapter 6, with the Markov generator and Bregman divergence chosen to match the geometry.

## 7.1 The Discrete Generative Modeling Problem

Let $\mathcal{X}$ be a finite state space of size $|\mathcal{X}| = S$. We write states as elements $x \in \mathcal{X}$ and identify $\mathcal{X}$ with $\{1, 2, \ldots, S\}$ when convenient. A probability distribution over $\mathcal{X}$ is a vector $p \in \Delta^{S-1}$, the $(S-1)$-simplex. The target distribution $p_{\text{data}}$ is an unknown distribution over $\mathcal{X}$, accessible only through samples.

For sequence modeling — the primary application — $\mathcal{X}$ is the space of all sequences of length $L$ over a vocabulary $\mathcal{V}$ of size $|\mathcal{V}|$, giving $S = |\mathcal{V}|^L$. For proteins with $L = 100$ residues over a 20-letter alphabet, $S \approx 10^{130}$, rendering any direct parameterization of $p_{\text{data}}$ as a vector in $\Delta^{S-1}$ hopelessly intractable. The key to tractability is **factorization**: modeling the sequence position-by-position, exploiting the product structure of $\mathcal{X} = \mathcal{V}^L$.

The transport perspective carries over: we seek a process that transforms a simple reference distribution $p_0$ (e.g., all-mask, all-uniform, or a prior over sequences) into $p_{\text{data}}$ over the time interval $[0,1]$. In the continuous case this was an ODE; in the discrete case it will be a CTMC. The core mathematical question is: given the analogy between vector fields and rate matrices, does the conditional flow matching trick — the key that unlocked simulation-free training — generalize to the discrete setting?

The answer, as we shall see, is yes — and the resulting objective reduces, in important special cases, to objectives already well-known in the language modeling literature, revealing that masked language models and discrete diffusion models are, in a precise sense, discrete flow matching models.

## 7.2 Continuous-Time Markov Chains

> [!definition] 7.1 — Rate Matrix and CTMC
> A **rate matrix** (or **generator**) is a matrix $Q \in \mathbb{R}^{S \times S}$ satisfying:
> - $Q(x, y) \geq 0$ for all $x \neq y$ (non-negative off-diagonal entries),
> - $Q(x, x) = -\sum_{y \neq x} Q(x, y)$ (rows sum to zero).
>
> A **continuous-time Markov chain** (CTMC) with time-dependent rate matrix $\{Q_t\}_{t \in [0,1]}$ is a continuous-time stochastic process $\{X_t\}_{t \in [0,1]}$ on $\mathcal{X}$ whose jump probabilities satisfy: $\mathbb{P}(X_{t+\epsilon} = y \mid X_t = x) = Q_t(x, y)\, \epsilon + o(\epsilon)$ for $x \neq y$.


The evolution of the marginal distribution $p_t(x) = \mathbb{P}(X_t = x)$ is governed by the **Kolmogorov forward equation** (also called the master equation):

> [!equation] 7.1
> $$\frac{d}{dt} p_t(x) = \sum_{y \in \mathcal{X}} p_t(y)\, Q_t(y, x) = (p_t Q_t)(x),$$


or in matrix notation, $\dot{p}_t = p_t Q_t$. This is the exact discrete-space analogue of the continuity equation $\partial_t p_t + \nabla \cdot (p_t v_t) = 0$ from Chapter 3. The formal solution is $p_t = p_0 \exp\!\left(\int_0^t Q_s\, ds\right)$, where the matrix exponential plays the role of the flow map $\phi_t$.

The condition $Q(x, y) \geq 0$ for $x \neq y$ ensures that probability cannot become negative — mass only jumps from $x$ to $y$ at rate $Q(x,y)$, never in the reverse direction due to the term $Q(x,y)$ alone. The zero-row-sum condition ensures total probability is conserved: $\sum_x \dot{p}_t(x) = \sum_x (p_t Q_t)(x) = 0$.

> [!remark] 7.1
> The analogy between vector fields and rate matrices is instructive but imperfect. A vector field specifies a direction at each point; a rate matrix specifies outgoing rates at each state to *all* neighbors simultaneously. In $\mathbb{R}^d$, one can reverse a flow by negating the vector field; in $\mathcal{X}$, reversing a CTMC requires computing the time-reversed rate matrix $\bar{Q}_t(x, y) = Q_{T-t}(y, x)\, p_{T-t}(y) / p_{T-t}(x)$, which requires knowing the marginal $p_t$ — analogous to the score function in continuous diffusion. This asymmetry has important consequences for training and sampling.


## 7.3 Discrete Flow Matching

The setup mirrors Chapter 3 precisely. For each data point $x_1 \sim p_{\text{data}}$, define:
- A **conditional probability path** $\{p_t(\cdot | x_1)\}_{t \in [0,1]}$: a family of distributions over $\mathcal{X}$ satisfying $p_0(\cdot | x_1) = p_0$ (reference) and $p_1(\cdot | x_1) = \delta_{x_1}$ (data point).
- A **conditional rate matrix** $Q_t^{x_1}$: a rate matrix generating the Kolmogorov equation for $p_t(\cdot | x_1)$, i.e., $\dot{p}_t(\cdot | x_1) = p_t(\cdot | x_1)\, Q_t^{x_1}$.

The marginal path $p_t(x) = \mathbb{E}_{x_1 \sim p_{\text{data}}}[p_t(x | x_1)]$ is generated by the **marginal rate matrix**:

> [!theorem] 7.1 — Marginal Rate Matrix Identity
> Suppose each conditional path $p_t(\cdot | x_1)$ satisfies the Kolmogorov equation with rate matrix $Q_t^{x_1}$. Then the marginal distribution $p_t(x) = \mathbb{E}_{x_1}[p_t(x|x_1)]$ satisfies $\dot{p}_t = p_t R_t$, where the marginal rate matrix is:
>
> $$R_t(x, y) = \mathbb{E}_{x_1 \sim p_{\text{data}}}\!\left[Q_t^{x_1}(x, y)\, \frac{p_t(x|x_1)}{p_t(x)}\right], \quad x \neq y,$$
>
> with diagonal entries $R_t(x, x) = -\sum_{y \neq x} R_t(x, y)$.


*Proof.* Differentiating $p_t(x) = \sum_{x_1} p_t(x|x_1)\, p_1(x_1)$ gives

$$\dot{p}_t(x) = \sum_{x_1} \dot{p}_t(x|x_1)\, p_1(x_1) = \sum_{x_1} \sum_y p_t(y|x_1)\, Q_t^{x_1}(y, x)\, p_1(x_1).$$

For $y \neq x$, the contribution to $(p_t R_t)(x)$ from the off-diagonal terms is

$$(p_t R_t)(x) = \sum_y p_t(y)\, R_t(y, x) = \sum_y \sum_{x_1} p_t(y|x_1)\, Q_t^{x_1}(y, x)\, p_1(x_1),$$

which matches $\dot{p}_t(x)$ upon substituting the definition of $R_t$. The diagonal ensures row sums vanish. $\square$

**The Discrete Flow Matching objective.** We wish to train a parameterized rate matrix $R_\theta(x, y, t)$ to match the marginal $R_t(x,y)$. The natural loss is a cross-entropy between the conditional rate matrix (the tractable target) and the model:

> [!equation] 7.2
> $$\mathcal{L}_{\text{DFM}}(\theta) = \mathbb{E}_{t,\, x_1 \sim p_{\text{data}},\, x \sim p_t(\cdot|x_1)} \sum_{y \neq x} Q_t^{x_1}(x, y) \Big[\log Q_t^{x_1}(x, y) - \log R_\theta(x, y, t)\Big].$$


Dropping terms independent of $\theta$, minimizing $\mathcal{L}_{\text{DFM}}$ is equivalent to minimizing $-\mathbb{E}[\sum_{y \neq x} Q_t^{x_1}(x,y) \log R_\theta(x, y, t)]$, a weighted cross-entropy. The analogue of Theorem 3.1 holds:

> [!theorem] 7.2 — DFM–FM Gradient Equivalence
> Define the marginal loss $\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, x \sim p_t} \sum_{y \neq x} R_t(x,y)[\log R_t(x,y) - \log R_\theta(x,y,t)]$. Then
>
> $$\nabla_\theta \mathcal{L}_{\text{DFM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{FM}}(\theta).$$
>
> Consequently, the global minimizer $R_\theta^* = R_t$ is achieved by training on the tractable conditional objective.


*Proof sketch.* Expanding the expectation in $\mathcal{L}_{\text{FM}}$ using $p_t(x) = \mathbb{E}_{x_1}[p_t(x|x_1)]$ and substituting the definition of $R_t(x,y)$ reduces the marginal cross-term to the conditional one by Fubini, exactly as in the continuous proof. $\square$

The DFM objective (7.2) is fully tractable: $p_t(\cdot|x_1)$ is a simple distribution we choose, $Q_t^{x_1}$ is the closed-form conditional rate matrix, and $R_\theta$ is evaluated at the sampled $(x, t)$ pair. No CTMC simulation is required during training.

## 7.4 Noise Processes and Conditional Paths

The theory above is agnostic to the choice of conditional path. The choice determines the inductive structure of the problem: which intermediate states $x \sim p_t(\cdot|x_1)$ the model sees during training, and what the conditional rate matrix $Q_t^{x_1}$ looks like. We describe the three most important choices.

### The Absorbing State (Masking) Process

Extend the vocabulary by one special token: $\mathcal{X}' = \mathcal{X} \cup \{m\}$, where $m$ is the **mask token**. The reference distribution is $p_0 = \delta_m$: everything starts masked. The conditional path linearly interpolates:

$$p_t(x | x_1) = \begin{cases} t & \text{if } x = x_1 \\ 1 - t & \text{if } x = m \\ 0 & \text{otherwise.} \end{cases}$$

Differentiating: $\dot{p}_t(x_1 | x_1) = 1$, $\dot{p}_t(m | x_1) = -1$. The unique rate matrix satisfying the Kolmogorov equation with this path is:

> [!equation] 7.3
> $$Q_t^{x_1}(m, x_1) = \frac{1}{1-t}, \qquad Q_t^{x_1}(m, y) = 0 \text{ for } y \neq x_1,$$
> $$Q_t^{x_1}(x, y) = 0 \text{ for all } x \neq m.$$


The mask token transitions to $x_1$ at rate $1/(1-t)$, and all non-mask states are absorbing. The DFM objective (7.2) then reduces to:

$$\mathcal{L}_{\text{mask}}(\theta) = -\mathbb{E}_{t,\, x_1,\, x \sim p_t(\cdot|x_1)} \mathbf{1}[x = m] \cdot \frac{1}{1-t} \log R_\theta(m, x_1, t),$$

a **time-weighted cross-entropy** at the mask position, predicting the original token $x_1$ from the masked state. At $t$ near 1, the weight $1/(1-t)$ diverges — the model must become increasingly confident near the end of generation — but this singularity is integrable and causes no numerical issues in practice with appropriate importance sampling of $t$.

**Connection to masked language models.** When $t$ is sampled uniformly, this objective matches the **masked diffusion language model** (MDLM) training objective of Sahoo et al. (2024) and Shi et al. (2024), which in turn reduces to a continuous-time limit of BERT-style masked language model pretraining. The mask probability at time $t$ is $1-t$, exactly the fraction of masked tokens in a standard BERT training run. Discrete flow matching thus provides a principled theoretical foundation for the ubiquitous masked language model: **BERT is a discrete flow matching model trained with the absorbing-state path and a uniform time distribution**.

### The Uniform Noise Process

Instead of masking, one can use the uniform distribution over $\mathcal{X}$ as the reference:

$$p_0 = \mathrm{Uniform}(\mathcal{X}).$$

The conditional path linearly interpolates between uniform and the data point:

$$p_t(x | x_1) = (1-t)\frac{1}{S} + t\, \delta_{x_1}(x).$$

The conditional rate matrix is:

$$Q_t^{x_1}(x, y) = \frac{1}{1-t}\left(\frac{1}{S} + (S-1) \cdot \mathbf{1}[y = x_1] \cdot p_t(x|x_1)^{-1}\right) - \delta_{xy},$$

which simplifies to a rate that preferentially transitions toward $x_1$ while maintaining the uniform marginal structure. This is the discrete analogue of the linear interpolant from Section 1.5.

> [!remark] 7.2
> The uniform process and the masking process are limiting cases of a broader family of **absorbing/non-absorbing mixtures** parameterized by a mixing coefficient $\kappa \in [0,1]$. At $\kappa = 1$, all probability mass passes through the mask token; at $\kappa = 0$, tokens transition directly between data-distribution states. The choice of $\kappa$ controls the "detour" taken through uninformative states, trading off training signal density (masking provides a clear token-to-predict) against transition diversity (uniform noise exposes the model to more varied context).


### Data-to-Data Couplings

Just as in the continuous setting (Section 1.6), one can couple the reference and target distributions through a transport plan $\pi(x_0, x_1)$ that is not the independent coupling. For discrete spaces, the **optimal transport coupling** solves the assignment problem over $\mathcal{X} \times \mathcal{X}$ with a cost matrix $c(x_0, x_1)$ — for example, the edit distance between sequences. The resulting coupled rate matrices produce flows with lower variance and straighter (in a discrete geodesic sense) interpolants.

## 7.5 Factored Models for High-Dimensional Sequences

In the sequence setting, $\mathcal{X} = \mathcal{V}^L$ and $S = |\mathcal{V}|^L$ is astronomically large. An unconstrained rate matrix $R_\theta \in \mathbb{R}^{S \times S}$ is intractable. The key structural assumption that makes discrete flow matching scalable is the **mean-field (factored) approximation**: each position in the sequence evolves independently, conditionally on the current sequence context.

Formally, the rate matrix is parameterized as:

$$R_\theta(x, y, t) = \sum_{\ell=1}^L R_\theta^\ell(x^\ell, y^\ell, x, t) \cdot \mathbf{1}[y^{\ell'} = x^{\ell'} \text{ for all } \ell' \neq \ell],$$

where $R_\theta^\ell(x^\ell, y^\ell, x, t)$ is the rate at which position $\ell$ transitions from token $x^\ell$ to $y^\ell$, conditioned on the full current sequence $x$ at time $t$. Only single-position transitions are permitted, and the rate at each position depends on the global context — this is parameterized by a **sequence model** (transformer) that processes the full current sequence and outputs position-wise logits.

Under the masking process with this factored rate matrix, the DFM objective becomes:

> [!equation] 7.4
> $$\mathcal{L}_{\text{DFM}}^{\text{seq}}(\theta) = -\mathbb{E}_{t,\, x_1,\, x \sim p_t(\cdot|x_1)} \sum_{\ell : x^\ell = m} \frac{1}{1-t} \log R_\theta^\ell(m, x_1^\ell, x, t),$$


a sum of cross-entropy losses at all masked positions, with the full context $x$ provided to the model. This is a **denoising objective**: given a partially masked sequence $x$, predict each masked token $x_1^\ell$ from the surrounding context. The $1/(1-t)$ weight upweights the contribution of nearly-complete sequences, where the model's uncertainty should be smallest.

**Connection to order-agnostic autoregressive models.** In the limit as the number of DFM sampling steps $N \to \infty$, discrete flow sampling approaches a **random-order autoregressive process**: tokens are unmasked one at a time, in a random order that is determined by the sequence of largest rates at each step. This connects discrete flow matching to order-agnostic autoregressive models (Uria et al., 2014; Yang et al., 2019), revealing them as deterministic limits of CTMC-based generative processes.

## 7.6 The CTMC Score Function and Discrete Diffusion

The connection between discrete flow matching and discrete diffusion models (D3PM, Austin et al. 2021; MDLM, Shi et al. 2024) runs deeper than a shared training objective. It mirrors exactly the continuous relationship between flow matching and score-based diffusion.

A discrete **diffusion process** in continuous time is a CTMC with a *forward* rate matrix $F_t$ that corrupts data toward the reference, and a *backward* (generative) rate matrix $B_t$ that reverses the process. The **discrete score function** — the quantity that parameterizes the backward rate — is defined as:

$$s_t(x, y) = \frac{p_t(y)}{p_t(x)}, \quad x \neq y,$$

and the backward rate matrix is $B_t(x, y) = F_t(y, x)\, s_t(x, y)$. The discrete score thus plays the same role as $\nabla \log p_t$ in the continuous setting: it encodes the relative density at neighboring states and determines the direction of the probability flow.

Training a discrete diffusion model is equivalent to learning the discrete score $s_t(x, y)$, typically via a ratio-estimation objective (Lou et al., 2023, SEDD). Under the masking process, the discrete score simplifies:

$$s_t(m, x_1) = \frac{p_t(x_1)}{p_t(m)} = \frac{t\, p_1(x_1)}{1-t} \cdot \frac{1}{p_0(m)} \approx \frac{t}{1-t}\, p_1(x_1),$$

and the backward rate $B_t(m, x_1) = F_t(x_1, m)\, s_t(m, x_1)$. Substituting $F_t(x_1, m) = 1$ (unit masking rate in the forward process) gives $B_t(m, x_1) = t\, p_1(x_1)/(1-t)$. The DFM conditional rate (7.3) for the masking process is $Q_t^{x_1}(m, x_1) = 1/(1-t)$, and the marginal rate $R_t(m, x_1) = p_1(x_1)/(1-t)$ matches exactly. **Discrete flow matching and masked discrete diffusion parameterize the same backward rate matrix**, just derived from different theoretical starting points.

## 7.7 Sampling from Discrete Flows

Given a trained rate matrix $R_\theta(x, y, t)$, generation proceeds by simulating a CTMC from $x_0 \sim p_0$ to $x_1 \sim p_{\text{data}}$. Two main approaches are available, mirroring the continuous Euler vs. ODE solver dichotomy.

**$\tau$-leaping (Euler method for CTMCs).** Discretize $[0,1]$ into $N$ steps of size $\Delta t = 1/N$. At each step:

1. Compute the transition probability matrix $P_t = I + \Delta t \cdot R_\theta(\cdot, \cdot, t)$.
2. Sample the next state: $X_{t+\Delta t} \sim \mathrm{Categorical}(P_t(X_t, \cdot))$.

Correctness requires $\Delta t \cdot R_\theta(x, y, t) \geq 0$ for all $x \neq y$, which holds if $\Delta t$ is small enough relative to the maximum rate. The local error is $O(\Delta t^2)$ in total variation.

**Gillespie's exact algorithm.** For exact simulation (zero discretization error), use Gillespie's stochastic simulation algorithm: at each state $x$, the time to the next jump is exponential with rate $\lambda(x,t) = -R_\theta(x, x, t)$, and the jump destination $y$ is drawn from the normalized rates $R_\theta(x, y, t) / \lambda(x,t)$. Exact simulation is computationally expensive for large vocabularies ($O(S)$ per step) but provides unbiased samples and is the gold standard for benchmarking.

**Predictor-corrector sampling.** A practical improvement combines $\tau$-leaping (the predictor) with a CTMC-version of Langevin correction (the corrector): after each $\tau$-leaping step, run a short sequence of forward–backward CTMC steps that leave $p_t$ invariant but improve mixing. This mirrors the predictor-corrector sampling introduced by Song et al. (2021) for continuous diffusion and substantially improves sample quality at a fixed NFE budget.

> [!remark] 7.3
> The NFE–quality trade-off in discrete flows has a qualitatively different character than in continuous flows. In continuous flow matching, straighter trajectories (achieved via OT couplings) directly reduce the required number of Euler steps. In discrete flows, the "straightness" of a trajectory is measured in terms of how few transitions are needed to go from $p_0$ to $p_{\text{data}}$. Under the masking process, each token is unmasked exactly once — the trajectory is already maximally "straight" — and the NFE is simply $L/N$, where $L$ is the sequence length and $N$ is the number of $\tau$-leaping steps. Reducing $N$ requires the model to unmask multiple tokens per step with high confidence.


## 7.8 Likelihood Computation

Unlike autoregressive models, which have trivially computable likelihoods via $\log p(x) = \sum_\ell \log p(x^\ell | x^{<\ell})$, CTMCs require integrating over all possible trajectories to compute the marginal log-likelihood $\log p_\theta(x_1)$. This is computationally intractable in general.

For the masking process, however, an efficient bound is available via the **evidence lower bound (ELBO)**:

$$\log p_\theta(x_1) \geq \mathcal{L}_{\text{ELBO}}(\theta, x_1) = \mathbb{E}_{x \sim p_t(\cdot|x_1)}\left[\int_0^1 \sum_y Q_t^{x_1}(x, y) \log R_\theta(x, y, t)\, dt\right].$$

Under the masking process, this reduces to a time-averaged cross-entropy at masked positions — the same objective as $\mathcal{L}_{\text{DFM}}^{\text{seq}}$ (7.4), confirming that the DFM training objective is simultaneously a variational lower bound on the data log-likelihood.

The key result of DUEL (Turok et al., 2026) is that **fixing the unmasking order** — processing positions in a deterministic sequence rather than randomly — recovers an exact likelihood factorization:

$$\log p_\theta(x_1) = \sum_{\ell=1}^L \log p_\theta(x_1^\ell \mid x_1^{\pi(1)}, \ldots, x_1^{\pi(\ell-1)}, m, \ldots, m),$$

where $\pi$ is the fixed unmasking order. This is precisely an autoregressive factorization with the unmasking order playing the role of the generation order. Thus a discrete diffusion model with a fixed unmasking order *is* an autoregressive model — the two are unified by the CTMC framework.

## 7.9 Flows on the Probability Simplex

A conceptually elegant bridge between the discrete and continuous frameworks is provided by **Dirichlet flow matching**, which lifts the discrete problem into a continuous one on the probability simplex $\Delta^{S-1}$.

Identify each token $x \in \mathcal{X}$ with its one-hot vector $e_x \in \{0,1\}^S$. A distribution $p \in \Delta^{S-1}$ is a soft assignment over tokens. Flow matching on $\Delta^{S-1}$ — a Riemannian manifold with the Fisher–Rao metric — defines vector fields tangent to the simplex, which correspond to flows between soft distributions. In the limit where all mass concentrates at vertices, such flows recover the discrete jumps of a CTMC.

Concretely, a path $p_t(\cdot | x_1) = (1-t)\, \alpha + t\, e_{x_1}$ (linear interpolation from a uniform Dirichlet prior $\alpha$ to a one-hot) defines a simplex vector field:

$$u_t(p | x_1) = \frac{e_{x_1} - p}{1-t},$$

which is the exact simplex analogue of the continuous OT field $u_t(x | x_1) = (x_1 - x)/(1-t)$ from Section 1.5. Training the vector field model $v_\theta(p, t)$ on the interior of $\Delta^{S-1}$ and discretizing its output at inference time (argmax, or sampling) recovers the discrete output.

The simplex perspective connects to the Riemannian flow matching of Section 7.13: $\Delta^{S-1}$ with the Fisher–Rao metric is a Riemannian manifold of non-constant curvature, and the geodesics under this metric are the natural interpolants. It also provides a way to handle **uncertainty at inference time**: instead of committing to a hard token at each step, the model maintains a distribution over tokens and refines it progressively.

## 7.10 Applications

The discrete flow matching framework has demonstrated strong empirical results across the domains where discrete structure is native.

**Protein sequence design.** Protein sequences are strings over a 20-letter amino acid alphabet, with lengths ranging from tens to thousands of residues. The fitness landscape of protein sequences — which sequences fold stably, bind targets, or catalyze reactions — is highly multi-modal and poorly understood. Discrete flow matching models trained on sequence databases (UniRef, UniProt) with structure-conditioned guidance (from predicted structure models like ESMFold) achieve competitive performance with autoregressive baselines on standard protein design benchmarks, with the additional benefit of bidirectional conditioning: any subset of positions can be fixed as context while the remaining positions are generated.

**DNA and RNA.** Regulatory DNA and RNA sequences have complex functional structure: promoter elements, splice sites, binding motifs for transcription factors. Masked diffusion and DFM models trained on genomic data can generate sequences with prescribed regulatory activity, conditioned on epigenetic context or cell type, making them powerful tools for synthetic biology.

**Language modeling.** The connection between DFM and masked language models (Section 7.5) implies that large-scale pretraining in the DFM framework can leverage the extensive infrastructure developed for transformers. MDLM (Sahoo et al., 2024) and similar models demonstrate that masked diffusion language models approach the perplexity of autoregressive transformers at comparable parameter counts, with additional generative flexibility: they can fill in arbitrary spans rather than being constrained to left-to-right generation.

**Graph and molecular generation.** Small molecules can be represented as graphs over atom and bond types — a product of discrete spaces for atom identity and continuous spaces for 3D coordinates. Discrete flow matching handles the atom type and bond order components, while continuous flow matching (or Riemannian flow matching over $SE(3)$, Section 7.13) handles the 3D geometry.

## 7.11 Historical Notes and Connections

The mathematical foundations of discrete generative modeling are older than the machine learning literature might suggest. The CTMC framework dates to Kolmogorov (1931) and Doeblin (1940). The idea of using CTMCs for generative modeling was first systematically explored by **Campbell, Benton, De Bortoli, Rainforth et al. (2022)** in "A Continuous Time Framework for Discrete Denoising Models," which established the score-based perspective on discrete diffusion and derived the backward rate matrix. **Austin, Johnson, Ho & Barber (2021)** introduced D3PM (Discrete Denoising Diffusion Probabilistic Models), a discrete-time analogue of DDPM for finite state spaces, establishing the connection to masked language models. The score-ratio parameterization was developed by **Lou, Meng & Ermon (2023)** in SEDD.

**Gat, Lipsitz, Ben-Hamu et al. (2024)** introduced **Discrete Flow Matching** proper — the CTMC analogue of the conditional flow matching trick, establishing Theorem 7.2 and the DFM objective (7.2). Concurrent and independent work by **Campbell, Harvey, Cautis et al. (2024)** in "Generative Flows on Discrete State-Spaces" developed essentially the same framework under different notation.

The **simplex / Dirichlet flow matching** perspective was developed by Stark, Jing et al. (2024) and Alcaide-Lopez et al. (2024), providing the geometric bridge to the Riemannian sections below.

**What remains open.** The central open questions in discrete flow matching concern:
- *Optimal noise schedules*: should one always use the linear schedule $p_t(x|x_1) = t\delta_{x_1} + (1-t)p_0$, or are there discrete analogues of VP/VE schedules that offer better training signal?
- *Multi-token transitions*: allowing simultaneous multi-position jumps (beyond the mean-field factorization) could dramatically reduce the required number of sampling steps but requires tractable parameterizations of high-order rate matrices.
- *Guidance*: how to condition a trained DFM model on auxiliary information (structure, function, property) at inference time without retraining — the discrete analogue of classifier-free guidance in continuous diffusion.
- *Theoretical guarantees*: convergence rates of $\tau$-leaping for non-homogeneous CTMCs, and conditions under which the trained rate matrix $R_\theta$ converges to the marginal $R_t$ in total variation.

---

The second extension replaces Euclidean space with a general Riemannian manifold $(\mathcal{M}, g)$. The core obstacle is that straight-line interpolation exits the manifold; the fix is to replace it with *geodesic* interpolation. The resulting training objective is simulation-free for any manifold with computable exponential and logarithm maps — a condition satisfied by all spaces relevant to structural biology and chemistry.

## 7.12 Riemannian Manifolds: A Working Primer

A **Riemannian manifold** $(\mathcal{M}, g)$ consists of a smooth manifold $\mathcal{M}$ and a Riemannian metric $g$: a family of positive definite inner products $g_x : T_x\mathcal{M} \times T_x\mathcal{M} \to \mathbb{R}$, one for each point $x \in \mathcal{M}$, varying smoothly with $x$. The tangent space $T_x\mathcal{M}$ is the space of all velocity vectors of smooth curves through $x$; it is a vector space of the same dimension as $\mathcal{M}$.

The two maps that make computation on Riemannian manifolds tractable are:

> [!definition] 7.2 — Exponential and Logarithm Maps
> The **exponential map** $\exp_x : T_x\mathcal{M} \to \mathcal{M}$ maps a tangent vector $v \in T_x\mathcal{M}$ to the point $\exp_x(v) \in \mathcal{M}$ reached by following the unique geodesic starting at $x$ with initial velocity $v$ for unit time.
>
> The **logarithm map** $\log_x : \mathcal{M} \to T_x\mathcal{M}$ (defined on a neighborhood of $x$) is the inverse: $\log_x(y)$ is the unique tangent vector at $x$ such that $\exp_x(\log_x(y)) = y$. Its magnitude $\|\log_x(y)\|_{g_x}$ equals the geodesic distance $d_\mathcal{M}(x, y)$.


**Geodesics** — the shortest paths on $\mathcal{M}$ — generalize straight lines. The geodesic from $x$ to $y$ is the curve $\gamma_t = \exp_x(t\, \log_x(y))$, $t \in [0,1]$, traversing the shortest path at constant speed. On the unit sphere $S^{d-1}$, geodesics are great circle arcs. On $SO(3)$, geodesics are rotations along a fixed axis. On the simplex with the Fisher-Rao metric, geodesics are arcs in the corresponding spherical orthant.

These maps are computable in closed form for several important manifolds:

- **Sphere** $S^{d-1}$: $\exp_x(v) = \cos(\|v\|)x + \sin(\|v\|)\hat{v}$, where $\hat{v} = v/\|v\|$.
- **$SO(3)$**: $\exp_R(V) = R \cdot \mathrm{expm}(R^\top V)$, where $\mathrm{expm}$ is the matrix exponential, and $V \in \mathfrak{so}(3)$ is a skew-symmetric matrix.
- **Symmetric positive definite matrices**: $\exp_X(V) = X^{1/2} \mathrm{expm}(X^{-1/2} V X^{-1/2}) X^{1/2}$.
- **Hyperbolic space** $\mathbb{H}^d$: explicit closed form via hyperbolic trigonometry.

## 7.13 Riemannian Flow Matching

Chen and Lipman's "Flow Matching on General Geometries" (ICLR 2024, outstanding paper) extended the conditional flow matching framework to arbitrary Riemannian manifolds, establishing a training procedure that is simulation-free, geometry-respecting, and computationally tractable for all manifolds with closed-form exponential and logarithm maps.

The conditional path on $\mathcal{M}$ is the **geodesic interpolant**: for source $x_0 \sim p_0$ (the reference distribution on $\mathcal{M}$, e.g. uniform or wrapped Gaussian) and target $x_1 \sim p_1 = p_{\text{data}}$:

> [!equation] 7.5
> $$x_t = \exp_{x_0}\!\left(t\, \log_{x_0}(x_1)\right), \qquad t \in [0, 1].$$


This is the geodesic from $x_0$ to $x_1$, evaluated at time $t$. It reduces to $x_t = x_0 + t(x_1 - x_0)$ in flat space, recovering the Euclidean linear interpolant. The conditional velocity field is the derivative of (7.5) with respect to $t$, which at time $t$ points along the geodesic toward $x_1$:

$$u_t(x_t \mid x_1) = \frac{\log_{x_t}(x_1)}{1 - t}.$$

This formula says: at any intermediate point $x_t$ on the geodesic, the target velocity points toward $x_1$ along the remaining geodesic arc, at a speed that increases as $t \to 1$ (the $1/(1-t)$ factor, matching the Euclidean formula). The conditional flow matching objective on $\mathcal{M}$ is then:

> [!equation] 7.6
> $$\mathcal{L}_{\mathrm{RFM}}(\theta) = \mathbb{E}_{t,\; x_1 \sim p_1,\; x_0 \sim p_0}\!\left[\left\|v_\theta(x_t, t) - \frac{\log_{x_t}(x_1)}{1-t}\right\|_{g(x_t)}^2\right],$$


where the squared norm is taken in the Riemannian metric at $x_t$, and $v_\theta(x_t, t) \in T_{x_t}\mathcal{M}$ is the learned velocity — a tangent vector at $x_t$.

> [!theorem] 7.3 — Riemannian Conditional Flow Matching
> On any Riemannian manifold $(\mathcal{M}, g)$ with computable exponential and logarithm maps, the conditional flow matching objective (7.6) is:
> 1. Simulation-free: each training step samples $(x_0, x_1, t)$, computes $x_t$ via (7.5) and the target via $\log_{x_t}(x_1)/(1-t)$, and takes one gradient step — no ODE solve required.
> 2. Equivalent to the marginal flow matching objective: the conditional and marginal objectives share the same minimizer.
> 3. Geometrically consistent: the learned flow stays on $\mathcal{M}$ by construction, since the velocity field $v_\theta$ is parametrized as a tangent-space-valued function.


The only computational requirement — computable $\exp$ and $\log$ maps — is satisfied in closed form for all manifolds arising in molecular and structural biology: spheres, tori, $SO(3)$, $SE(3)$, hyperbolic spaces, and the Stiefel manifold of orthonormal frames.

> [!remark] 7.4
> The formula $u_t(x_t|x_1) = \log_{x_t}(x_1)/(1-t)$ has a simple geometric interpretation: it is the "geodesic shooting" velocity that, if the ODE $\dot{x} = v_\theta(x,t)$ is solved exactly, would transport $x_t$ to $x_1$ along the remaining geodesic in the remaining time $1-t$. This is the Riemannian generalization of the Euclidean identity $u_t = (x_1 - x_t)/(1-t)$, which transports $x_t$ to $x_1$ along a straight line in time $1-t$. The Riemannian metric determines what "straight" means.


## 7.14 The Probability Simplex and Categorical Geometry

The probability simplex $\Delta^{d-1} = \{x \in \mathbb{R}^d_{\geq 0} : \sum_{i=1}^d x_i = 1\}$ deserves special treatment. It arises naturally in language modeling — where $x$ represents a distribution over a vocabulary — in population genetics, in mixture models, and in any setting where the object of interest is a probability vector over $d$ categories.

Euclidean flow matching is problematic on $\Delta^{d-1}$ for a fundamental reason: the linear interpolant $x_t = tx_1 + (1-t)x_0$ stays on the simplex (convex combinations of probability vectors are probability vectors), but the boundary of the simplex — where some $x_i = 0$ — is a fixed point of many dynamics, creating pathologies. More importantly, the Euclidean metric on $\Delta^{d-1}$ is not the natural one.

The natural metric on $\Delta^{d-1}$ is the **Fisher-Rao metric**: $g_{ij}^{\text{FR}}(x) = \delta_{ij}/x_i$. Under this metric, the simplex has a beautiful geometric structure: the map $\phi : \Delta^{d-1} \to S^{d-1}_+$ defined by $\phi(x)_i = \sqrt{x_i}$ is an isometry between $(\Delta^{d-1}, g^{\text{FR}})$ and the positive orthant of the unit sphere $S^{d-1}_+ = \{z \in S^{d-1} : z_i \geq 0\}$ with the round metric. Geodesics on the simplex correspond to great circle arcs on $S^{d-1}_+$ under this identification.

This isometry has immediate consequences for flow matching on the simplex:

**The canonical reference distribution** is the uniform distribution on $\Delta^{d-1}$ (symmetric Dirichlet $\mathrm{Dir}(\alpha = 1)$), which corresponds under $\phi$ to the uniform distribution on $S^{d-1}_+$.

**Geodesic interpolation** on the simplex is simply spherical interpolation of the square-root vectors: $\gamma_t = \phi^{-1}(\exp_{\phi(x_0)}(t\, \log_{\phi(x_0)}(\phi(x_1))))$. This avoids the boundary issues of Euclidean interpolation — great circle arcs on the positive orthant naturally stay in the interior when both endpoints are interior points.

**Dirichlet Flow Matching** (Stark, Jing, et al., 2024) exploits this structure for DNA sequence generation: represent each position in the sequence as a probability vector over $\{A, C, G, T\}$, use the Fisher-Rao geodesic as the interpolant, and train with the Riemannian flow matching objective (7.6) on the spherical parametrization. The trained flow generates DNA sequences by integrating the learned ODE from a symmetric Dirichlet sample to a one-hot vector, and the resulting sequences match the statistical properties of real genomic data better than models trained with Euclidean noise schedules.

## 7.15 SE(3): Flows for Proteins and Molecules

The application that most forcefully demonstrated the value of Riemannian flow matching was protein backbone generation — the task of generating the three-dimensional structure of a protein, residue by residue.

A protein backbone is a sequence of $N$ residues, each represented by a **frame** $(t_i, R_i) \in SE(3)$: a translation $t_i \in \mathbb{R}^3$ (the position of the $\alpha$-carbon) and a rotation $R_i \in SO(3)$ (the local orientation of the residue). The space of all protein backbones is therefore $SE(3)^N$ — a product of $N$ copies of the special Euclidean group.

$SE(3) = \mathbb{R}^3 \rtimes SO(3)$ is a Lie group (a manifold with a compatible group structure). Its geometry decomposes naturally: the translation component lives in Euclidean space ($\mathbb{R}^3$) and is handled by standard flow matching; the rotation component lives in $SO(3)$ and requires Riemannian methods.

**Geodesics on $SO(3)$.** The Lie algebra $\mathfrak{so}(3)$ consists of $3 \times 3$ skew-symmetric matrices, identifiable with $\mathbb{R}^3$ via the hat map: for $\omega \in \mathbb{R}^3$, the matrix $\hat\omega$ has $\hat\omega_{ij} = -\epsilon_{ijk}\omega_k$. The exponential map is:

$$\exp_I(\hat\omega) = I + \frac{\sin\|\omega\|}{\|\omega\|}\hat\omega + \frac{1-\cos\|\omega\|}{\|\omega\|^2}\hat\omega^2 \qquad \text{(Rodrigues' formula)},$$

and the logarithm map inverts this for $\|\omega\| < \pi$. Geodesics on $SO(3)$ are rotations about a fixed axis at constant angular velocity.

**FoldFlow** (Bose, Akhound-Sadegh, Huguet, Wolf, Rekesh, Su, Zhang, Bengio, Bhatt, Lee, Tong, Hamilton, and Bronstein, NeurIPS 2023) was the first model to apply flow matching on $SE(3)$ to protein backbone generation, using geodesic interpolation on each residue frame and a transformer architecture on the sequence of frames. The training is entirely simulation-free: sample a noise structure $\{(t_i^0, R_i^0)\}$ from a standard distribution on $SE(3)^N$, a target structure $\{(t_i^1, R_i^1)\}$ from the PDB, interpolate geodesically to get $\{(t_i^t, R_i^t)\}$, and regress the velocity field against $\{(\log_{t_i^t}(t_i^1)/(1-t), \log_{R_i^t}(R_i^1)/(1-t))\}$ using a single forward pass.

FoldFlow-2 (Huguet, Vuckovic, Fatras, Tong, Sun, Wolf, Bronstein, Tazi, and Bengio, 2024) scaled this architecture with OT conditioning on $SE(3)$: matching noise frames to data frames by a Riemannian OT cost on $SE(3)$, producing straighter geodesic paths and requiring fewer integration steps at inference. The resulting model generates protein backbones with designable, physically plausible structures at a quality competitive with — and in some metrics exceeding — diffusion-based predecessors like FrameDiff and RFdiffusion.

> [!remark] 7.5
> The separate treatment of translations (Euclidean) and rotations (Riemannian) in $SE(3)$ is justified by the product structure of the group, but it ignores the non-commutativity of rotations and translations in $SE(3)$. More principled approaches use the full Lie group geometry of $SE(3)$, where geodesics couple the translational and rotational components via the Maurer-Cartan form. For most practical protein generation tasks, the approximate product-space treatment produces excellent results; the theoretically exact Lie group treatment is relevant for applications requiring precise equivariance guarantees, such as docking and co-design.


## 7.16 Torus Flows and Dihedral Angles

One more manifold deserves mention: the torus $\mathbb{T}^k = (S^1)^k$ — the product of $k$ circles. It is the natural home of dihedral angles, the bond-angle degrees of freedom that determine molecular conformation.

A molecule with $k$ rotatable bonds has a conformation space that is a subset of $\mathbb{T}^k$: each dihedral angle $\phi_j \in (-\pi, \pi]$ parametrizes a $S^1$ factor. The metric on $\mathbb{T}^k$ is flat (no curvature), and geodesics are straight lines modulo $2\pi$ — angle interpolation. The exponential and logarithm maps are just angular addition and subtraction, with wrapping.

TorsionalDiffusion (Jing, Corso, Chang, Barzilay, Jaakkola, ICLR 2023) used torus diffusion for small-molecule conformation generation; FoldFlow's successor models extended the approach to protein side-chain dihedral angles in addition to the backbone frames. The combination of $SE(3)$ flow matching for backbones and $\mathbb{T}^k$ flow matching for side chains gives a complete conformational flow model with no Euclidean approximations.

## 7.17 The Emerging Picture

The extensions of this chapter share a common structure. In every setting — discrete tokens, spheres, $SO(3)$, the simplex, the torus — the framework is:

1. Choose a state space $\mathcal{X}$ appropriate to the data.
2. Choose a reference distribution $p_0$ on $\mathcal{X}$ (uniform on finite spaces, wrapped Gaussian or uniform on manifolds).
3. Define a conditional path $\{p_t(\cdot|x_1)\}$ from $p_0$ toward the data point $x_1$, using the natural structure of $\mathcal{X}$ (geodesics on manifolds, masking on discrete spaces).
4. Derive the conditional rate matrix or velocity field from the path.
5. Train by regression against this conditional target — one forward pass per step, no simulation.

The training cost is universal: one forward pass per gradient step, regardless of whether the space is Euclidean, curved, or discrete. The difference is entirely in the geometry of the path and the form of the regression target.

This universality reflects a single abstract principle: the Kolmogorov equation (7.1) and the continuity equation (3.1) are both special cases of the same object — the **Kolmogorov forward equation for a Markov generator** on a general state space. The vector field $u_t$ and the rate matrix $R_t$ are both instances of a Markov generator: the infinitesimal description of a Markov process. Chapter 6 made this precise: a single theorem that contains normalizing flows, diffusion models, flow matching, discrete flows, and Riemannian flows as special cases, stated at the level of a general Markov generator.

## 7.18 Historical Notes

**Discrete diffusion.** The systematic study of discrete diffusion models begins with **Austin, Johnson, Ho, Kingma, and Song (NeurIPS 2021)** ("Structured Denoising Diffusion Models in Discrete State-Spaces"), which introduced D3PM — a framework for discrete diffusion with absorbing, uniform, and token-proximity forward processes. **Hoogeboom, Nielsen, Jaini, Forré, and Welling (NeurIPS 2021)** ("Argmax Flows and Multinomial Diffusion") developed the multinomial diffusion model from the normalizing flow direction. **Lou, Meng, and Ermon (ICML 2024)** ("Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution") derived a score-matching analog for discrete spaces using ratio estimators.

**Discrete flow matching.** The CTMC-based flow matching framework was developed simultaneously by **Gat, Remez, Shaul, Kreuk, Chen, Synnaeve, Adi, and Lipman (NeurIPS 2024)** ("Discrete Flow Matching") and **Campbell, Yim, Barzilay, Jaakkola, and Gat (NeurIPS 2024)** ("Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design"). The connection between the masking interpolant and masked language models is established in both papers. A related concurrent framework is **Shi, Holderrieth, Haber, Schmidt, and Holderrieth (2024)** ("Simplified and Generalized Masked Diffusion for Discrete Data").

**Riemannian diffusion.** The Riemannian extension of score-based models was developed by **De Bortoli, Thornton, Heng, and Doucet (NeurIPS 2022)** ("Riemannian Score-Based Generative Modelling") and **Huang, Aghajohari, Bose, Panangaden, and Courville (NeurIPS 2022)** ("Riemannian Diffusion Models"), establishing that the score SDE framework extends to Riemannian manifolds with appropriate modifications to the Fokker-Planck equation.

**Riemannian flow matching.** The simulation-free extension is **Chen and Lipman (ICLR 2024, outstanding paper)** ("Flow Matching on General Geometries"), which introduced the geodesic conditional path (7.5), the formula $u_t = \log_{x_t}(x_1)/(1-t)$ (7.6), and the premetric approximation for manifolds without closed-form geodesics. A contemporaneous independent development is **Pooladian, Ben-Hamu, Domingo-Enrich, Amos, Lipman, and Chen (2023)**, which extended OT conditioning to Riemannian settings.

**Protein structure generation.** The SE(3) diffusion approach for protein backbones was introduced by **Yim, Trippe, De Bortoli, Mathieu, Doucet, Barzilay, and Jaakkola (ICML 2023)** ("SE(3) Diffusion Model with Application to Protein Backbone Generation"). The flow matching version — **FoldFlow** — is due to **Bose, Akhound-Sadegh, Huguet, Wolf, Rekesh, Su, Zhang, Bengio, Bhatt, Lee, Tong, Hamilton, and Bronstein (NeurIPS 2023)**, with the scaled-up version **FoldFlow-2 (Huguet et al., 2024)** incorporating OT conditioning on $SE(3)$. The AlphaFold-3 (Abramson et al., Nature 2024) architecture uses a diffusion process on atom coordinates without explicit $SE(3)$ structure, while FrameDiff (Yim et al.) and RFdiffusion (Watson, Juergens, et al., Nature 2023) use explicit frame representations.

**Simplex flows.** The Dirichlet flow matching framework for genomic sequences is developed in **Stark, Jing, Wang, Geiger, Barzilay, Jaakkola (ICML 2024)** ("Dirichlet Flow Matching with Applications to DNA Sequence Design"), which establishes the isometry between the simplex with Fisher-Rao metric and the spherical orthant and applies it to promoter sequence design. **Campbell, Benton, De Bortoli, Rainforth, Deligiannidis, and Doucet (NeurIPS 2022)** ("A Continuous Time Framework for Discrete Denoising Models") provided the CTMC theoretical foundation for these approaches.

## References

- [Austin, Johnson, Ho & Barber (NeurIPS 2021)](https://arxiv.org/abs/2107.03006) — Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM)
- [Lou, Meng & Ermon (ICML 2024)](https://arxiv.org/abs/2310.16834) — Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (SEDD)
- [Campbell, Benton, De Bortoli, Rainforth, Deligiannidis, and Doucet (NeurIPS 2022)](https://arxiv.org/abs/2205.14987) — A Continuous Time Framework for Discrete Denoising Models
- [Gat, Remez, Shaul, Kreuk, Chen, Synnaeve, Adi, and Lipman (NeurIPS 2024)](https://arxiv.org/abs/2407.15595) — Discrete Flow Matching
- [Campbell, Yim, Barzilay, Jaakkola, and Gat (NeurIPS 2024)](https://arxiv.org/abs/2406.04329) — Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design
- [Hoogeboom, Nielsen, Jaini, Forré, and Welling (NeurIPS 2021)](https://arxiv.org/abs/2102.05379) — Argmax Flows and Multinomial Diffusion
- [Sahoo et al. (2024) — MDLM](https://arxiv.org/abs/2406.07524) — Simple and Effective Masked Diffusion Language Models
- [Shi et al. (2024)](https://arxiv.org/abs/2405.02236) — Simplified and Generalized Masked Diffusion for Discrete Data
- [Uria et al. (2014)](https://arxiv.org/abs/1310.1757) — A Deep and Tractable Density Estimator
- [Yang et al. (2019) — XLNet](https://arxiv.org/abs/1906.08237) — XLNet: Generalized Autoregressive Pretraining for Language Understanding
- [Chen and Lipman's 'Flow Matching on General Geometries' (ICLR 2024, outstanding paper)](https://arxiv.org/abs/2302.03660) — Flow Matching on General Geometries
- [De Bortoli, Thornton, Heng, and Doucet (NeurIPS 2022)](https://arxiv.org/abs/2202.02763) — Riemannian Score-Based Generative Modelling
- [Huang, Aghajohari, Bose, Panangaden, and Courville (NeurIPS 2022)](https://arxiv.org/abs/2208.07949) — Riemannian Diffusion Models
- [Pooladian, Ben-Hamu, Domingo-Enrich, Amos, Lipman, and Chen (2023) — Riemannian OT](https://arxiv.org/abs/2304.14772) — Multisample Flow Matching: Straightening Flows with Minibatch Couplings
- [Yim, Trippe, De Bortoli, Mathieu, Doucet, Barzilay, and Jaakkola (ICML 2023)](https://arxiv.org/abs/2302.02277) — SE(3) Diffusion Model with Application to Protein Backbone Generation (FrameDiff)
- [Bose, Akhound-Sadegh, Huguet, Wolf, Rekesh, Su, Zhang, Bengio, Bhatt, Lee, Tong, Hamilton, and Bronstein (NeurIPS 2023) — FoldFlow](https://arxiv.org/abs/2310.02391) — SE(3)-Stochastic Flow Matching for Protein Backbone Generation
- [Huguet, Vuckovic, Fatras, Tong, Sun, Wolf, Bronstein, Tazi, and Bengio (2024) — FoldFlow-2](https://arxiv.org/abs/2406.03163) — FoldFlow-2: Improving Protein Backbone Generation with Flow Matching
- [Abramson et al. (Nature 2024) — AlphaFold-3](https://doi.org/10.1038/s41586-024-07487-w) — Accurate Structure Prediction of Biomolecular Interactions with AlphaFold 3
- [Watson, Juergens, et al. (Nature 2023) — RFdiffusion](https://doi.org/10.1038/s41586-023-06415-8) — De Novo Design of Protein Structure and Function with RFdiffusion
- [TorsionalDiffusion (Jing, Corso, Chang, Barzilay, Jaakkola, ICLR 2023)](https://arxiv.org/abs/2210.01776) — Torsional Diffusion for Molecular Conformations
- [Stark, Jing, Wang, Geiger, Barzilay, Jaakkola (ICML 2024)](https://arxiv.org/abs/2402.05841) — Dirichlet Flow Matching with Applications to DNA Sequence Design
- [Holderrieth, Yim, and Jaakkola (2024/2025) — Generator Matching](https://arxiv.org/abs/2409.11399) — Generator Matching: Generative Modeling with Arbitrary Markov Processes
