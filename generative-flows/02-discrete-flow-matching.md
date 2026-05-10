---

title: "Flows on Finite and Categorical Spaces"
subtitle: "Continuous-time Markov chains, rate matrix matching, and the geometry of masked generation"
---



The framework developed in Chapter 1 rests on a foundational assumption: the data lives in $\mathbb{R}^d$, and the generative process is an ordinary differential equation on that space. This assumption is natural for images, audio, and continuous molecular coordinates, but it immediately breaks down for the most pervasive objects in modern machine learning — discrete sequences. Protein sequences, DNA, small-molecule graphs, natural language: all are drawn from finite or countable state spaces where vectors and derivatives have no meaning.

One might hope that discretization is merely an engineering concern — embed the tokens, run continuous flow matching in the embedding space, read off the tokens at the end. But this approach forfeits the structure of the discrete space entirely. Crucially, it cannot exploit the fact that the target distribution $p_{\text{data}}$ assigns probability to a discrete set of points, nor can it provide meaningful likelihoods over sequences. A mathematically principled treatment requires a different starting point.

That starting point is the **continuous-time Markov chain** (CTMC): the discrete-space analogue of the ODE. Just as a vector field specifies how probability mass flows continuously through $\mathbb{R}^d$, a rate matrix specifies how probability mass jumps between states in a finite space. Discrete flow matching — introduced by Gat, Lipsitz et al. (2024) — builds on this analogy to construct a simulation-free training objective for CTMCs, with the same theoretical elegance as the CFM result of Chapter 1 and with direct applications to language, biology, and chemistry.

## 2.1 The Discrete Generative Modeling Problem

Let $\mathcal{X}$ be a finite state space of size $|\mathcal{X}| = S$. We write states as elements $x \in \mathcal{X}$ and identify $\mathcal{X}$ with $\{1, 2, \ldots, S\}$ when convenient. A probability distribution over $\mathcal{X}$ is a vector $p \in \Delta^{S-1}$, the $(S-1)$-simplex. The target distribution $p_{\text{data}}$ is an unknown distribution over $\mathcal{X}$, accessible only through samples.

For sequence modeling — the primary application — $\mathcal{X}$ is the space of all sequences of length $L$ over a vocabulary $\mathcal{V}$ of size $|\mathcal{V}|$, giving $S = |\mathcal{V}|^L$. For proteins with $L = 100$ residues over a 20-letter alphabet, $S \approx 10^{130}$, rendering any direct parameterization of $p_{\text{data}}$ as a vector in $\Delta^{S-1}$ hopelessly intractable. The key to tractability is **factorization**: modeling the sequence position-by-position, exploiting the product structure of $\mathcal{X} = \mathcal{V}^L$.

The transport perspective carries over: we seek a process that transforms a simple reference distribution $p_0$ (e.g., all-mask, all-uniform, or a prior over sequences) into $p_{\text{data}}$ over the time interval $[0,1]$. In the continuous case this was an ODE; in the discrete case it will be a CTMC. The core mathematical question is: given the analogy between vector fields and rate matrices, does the conditional flow matching trick — the key that unlocked simulation-free training — generalize to the discrete setting?

The answer, as we shall see, is yes — and the resulting objective reduces, in important special cases, to objectives already well-known in the language modeling literature, revealing that masked language models and discrete diffusion models are, in a precise sense, discrete flow matching models.

## 2.2 Continuous-Time Markov Chains

> [!definition] 2.1 — Rate Matrix and CTMC
> A **rate matrix** (or **generator**) is a matrix $Q \in \mathbb{R}^{S \times S}$ satisfying:
> - $Q(x, y) \geq 0$ for all $x \neq y$ (non-negative off-diagonal entries),
> - $Q(x, x) = -\sum_{y \neq x} Q(x, y)$ (rows sum to zero).
>
> A **continuous-time Markov chain** (CTMC) with time-dependent rate matrix $\{Q_t\}_{t \in [0,1]}$ is a continuous-time stochastic process $\{X_t\}_{t \in [0,1]}$ on $\mathcal{X}$ whose jump probabilities satisfy: $\mathbb{P}(X_{t+\epsilon} = y \mid X_t = x) = Q_t(x, y)\, \epsilon + o(\epsilon)$ for $x \neq y$.


The evolution of the marginal distribution $p_t(x) = \mathbb{P}(X_t = x)$ is governed by the **Kolmogorov forward equation** (also called the master equation):

> [!equation] 2.1
> $$\frac{d}{dt} p_t(x) = \sum_{y \in \mathcal{X}} p_t(y)\, Q_t(y, x) = (p_t Q_t)(x),$$


or in matrix notation, $\dot{p}_t = p_t Q_t$. This is the exact discrete-space analogue of the continuity equation $\partial_t p_t + \nabla \cdot (p_t v_t) = 0$ from Chapter 1. The formal solution is $p_t = p_0 \exp\!\left(\int_0^t Q_s\, ds\right)$, where the matrix exponential plays the role of the flow map $\phi_t$.

The condition $Q(x, y) \geq 0$ for $x \neq y$ ensures that probability cannot become negative — mass only jumps from $x$ to $y$ at rate $Q(x,y)$, never in the reverse direction due to the term $Q(x,y)$ alone. The zero-row-sum condition ensures total probability is conserved: $\sum_x \dot{p}_t(x) = \sum_x (p_t Q_t)(x) = 0$.

> [!remark] 2.1
> The analogy between vector fields and rate matrices is instructive but imperfect. A vector field specifies a direction at each point; a rate matrix specifies outgoing rates at each state to *all* neighbors simultaneously. In $\mathbb{R}^d$, one can reverse a flow by negating the vector field; in $\mathcal{X}$, reversing a CTMC requires computing the time-reversed rate matrix $\bar{Q}_t(x, y) = Q_{T-t}(y, x)\, p_{T-t}(y) / p_{T-t}(x)$, which requires knowing the marginal $p_t$ — analogous to the score function in continuous diffusion. This asymmetry has important consequences for training and sampling.


## 2.3 Discrete Flow Matching

The setup mirrors Chapter 1 precisely. For each data point $x_1 \sim p_{\text{data}}$, define:
- A **conditional probability path** $\{p_t(\cdot | x_1)\}_{t \in [0,1]}$: a family of distributions over $\mathcal{X}$ satisfying $p_0(\cdot | x_1) = p_0$ (reference) and $p_1(\cdot | x_1) = \delta_{x_1}$ (data point).
- A **conditional rate matrix** $Q_t^{x_1}$: a rate matrix generating the Kolmogorov equation for $p_t(\cdot | x_1)$, i.e., $\dot{p}_t(\cdot | x_1) = p_t(\cdot | x_1)\, Q_t^{x_1}$.

The marginal path $p_t(x) = \mathbb{E}_{x_1 \sim p_{\text{data}}}[p_t(x | x_1)]$ is generated by the **marginal rate matrix**:

> [!theorem] 2.1 — Marginal Rate Matrix Identity
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

This is the exact discrete analogue of Theorem 1.2 in Chapter 1. The marginal rate matrix is the posterior-weighted conditional rate matrix — the same reweighting by $p_t(x|x_1)/p_t(x)$ that appeared in the continuous case.

**The Discrete Flow Matching objective.** We wish to train a parameterized rate matrix $R_\theta(x, y, t)$ to match the marginal $R_t(x,y)$. The natural loss is a cross-entropy between the conditional rate matrix (the tractable target) and the model:

> [!equation] 2.2
> $$\mathcal{L}_{\text{DFM}}(\theta) = \mathbb{E}_{t,\, x_1 \sim p_{\text{data}},\, x \sim p_t(\cdot|x_1)} \sum_{y \neq x} Q_t^{x_1}(x, y) \Big[\log Q_t^{x_1}(x, y) - \log R_\theta(x, y, t)\Big].$$


Dropping terms independent of $\theta$, minimizing $\mathcal{L}_{\text{DFM}}$ is equivalent to minimizing $-\mathbb{E}[\sum_{y \neq x} Q_t^{x_1}(x,y) \log R_\theta(x, y, t)]$, a weighted cross-entropy. The analogue of Theorem 1.3 holds:

> [!theorem] 2.2 — DFM–FM Gradient Equivalence
> Define the marginal loss $\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, x \sim p_t} \sum_{y \neq x} R_t(x,y)[\log R_t(x,y) - \log R_\theta(x,y,t)]$. Then
>
> $$\nabla_\theta \mathcal{L}_{\text{DFM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{FM}}(\theta).$$
>
> Consequently, the global minimizer $R_\theta^* = R_t$ is achieved by training on the tractable conditional objective.


*Proof sketch.* Expanding the expectation in $\mathcal{L}_{\text{FM}}$ using $p_t(x) = \mathbb{E}_{x_1}[p_t(x|x_1)]$ and substituting the definition of $R_t(x,y)$ reduces the marginal cross-term to the conditional one by Fubini, exactly as in the continuous proof. $\square$

The DFM objective (2.2) is fully tractable: $p_t(\cdot|x_1)$ is a simple distribution we choose, $Q_t^{x_1}$ is the closed-form conditional rate matrix, and $R_\theta$ is evaluated at the sampled $(x, t)$ pair. No CTMC simulation is required during training.

## 2.4 Noise Processes and Conditional Paths

The theory above is agnostic to the choice of conditional path. The choice determines the inductive structure of the problem: which intermediate states $x \sim p_t(\cdot|x_1)$ the model sees during training, and what the conditional rate matrix $Q_t^{x_1}$ looks like. We describe the three most important choices.

### The Absorbing State (Masking) Process

Extend the vocabulary by one special token: $\mathcal{X}' = \mathcal{X} \cup \{m\}$, where $m$ is the **mask token**. The reference distribution is $p_0 = \delta_m$: everything starts masked. The conditional path linearly interpolates:

$$p_t(x | x_1) = \begin{cases} t & \text{if } x = x_1 \\ 1 - t & \text{if } x = m \\ 0 & \text{otherwise.} \end{cases}$$

Differentiating: $\dot{p}_t(x_1 | x_1) = 1$, $\dot{p}_t(m | x_1) = -1$. The unique rate matrix satisfying the Kolmogorov equation with this path is:

> [!equation] 2.3
> $$Q_t^{x_1}(m, x_1) = \frac{1}{1-t}, \qquad Q_t^{x_1}(m, y) = 0 \text{ for } y \neq x_1,$$
> $$Q_t^{x_1}(x, y) = 0 \text{ for all } x \neq m.$$


The mask token transitions to $x_1$ at rate $1/(1-t)$, and all non-mask states are absorbing. The DFM objective (2.2) then reduces to:

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

> [!remark] 2.2
> The uniform process and the masking process are limiting cases of a broader family of **absorbing/non-absorbing mixtures** parameterized by a mixing coefficient $\kappa \in [0,1]$. At $\kappa = 1$, all probability mass passes through the mask token; at $\kappa = 0$, tokens transition directly between data-distribution states. The choice of $\kappa$ controls the "detour" taken through uninformative states, trading off training signal density (masking provides a clear token-to-predict) against transition diversity (uniform noise exposes the model to more varied context).


### Data-to-Data Couplings

Just as in the continuous setting (Section 1.6), one can couple the reference and target distributions through a transport plan $\pi(x_0, x_1)$ that is not the independent coupling. For discrete spaces, the **optimal transport coupling** solves the assignment problem over $\mathcal{X} \times \mathcal{X}$ with a cost matrix $c(x_0, x_1)$ — for example, the edit distance between sequences. The resulting coupled rate matrices produce flows with lower variance and straighter (in a discrete geodesic sense) interpolants.

## 2.5 Factored Models for High-Dimensional Sequences

In the sequence setting, $\mathcal{X} = \mathcal{V}^L$ and $S = |\mathcal{V}|^L$ is astronomically large. An unconstrained rate matrix $R_\theta \in \mathbb{R}^{S \times S}$ is intractable. The key structural assumption that makes discrete flow matching scalable is the **mean-field (factored) approximation**: each position in the sequence evolves independently, conditionally on the current sequence context.

Formally, the rate matrix is parameterized as:

$$R_\theta(x, y, t) = \sum_{\ell=1}^L R_\theta^\ell(x^\ell, y^\ell, x, t) \cdot \mathbf{1}[y^{\ell'} = x^{\ell'} \text{ for all } \ell' \neq \ell],$$

where $R_\theta^\ell(x^\ell, y^\ell, x, t)$ is the rate at which position $\ell$ transitions from token $x^\ell$ to $y^\ell$, conditioned on the full current sequence $x$ at time $t$. Only single-position transitions are permitted, and the rate at each position depends on the global context — this is parameterized by a **sequence model** (transformer) that processes the full current sequence and outputs position-wise logits.

Under the masking process with this factored rate matrix, the DFM objective becomes:

> [!equation] 2.4
> $$\mathcal{L}_{\text{DFM}}^{\text{seq}}(\theta) = -\mathbb{E}_{t,\, x_1,\, x \sim p_t(\cdot|x_1)} \sum_{\ell : x^\ell = m} \frac{1}{1-t} \log R_\theta^\ell(m, x_1^\ell, x, t),$$


a sum of cross-entropy losses at all masked positions, with the full context $x$ provided to the model. This is a **denoising objective**: given a partially masked sequence $x$, predict each masked token $x_1^\ell$ from the surrounding context. The $1/(1-t)$ weight upweights the contribution of nearly-complete sequences, where the model's uncertainty should be smallest.

**Connection to order-agnostic autoregressive models.** In the limit as the number of DFM sampling steps $N \to \infty$, discrete flow sampling (Section 2.7) approaches a **random-order autoregressive process**: tokens are unmasked one at a time, in a random order that is determined by the sequence of largest rates at each step. This connects discrete flow matching to order-agnostic autoregressive models (Uria et al., 2014; Yang et al., 2019), revealing them as deterministic limits of CTMC-based generative processes.

## 2.6 The CTMC Score Function and Discrete Diffusion

The connection between discrete flow matching and discrete diffusion models (D3PM, Austin et al. 2021; MDLM, Shi et al. 2024) runs deeper than a shared training objective. It mirrors exactly the continuous relationship between flow matching and score-based diffusion.

A discrete **diffusion process** in continuous time is a CTMC with a *forward* rate matrix $F_t$ that corrupts data toward the reference, and a *backward* (generative) rate matrix $B_t$ that reverses the process. The **discrete score function** — the quantity that parameterizes the backward rate — is defined as:

$$s_t(x, y) = \frac{p_t(y)}{p_t(x)}, \quad x \neq y,$$

and the backward rate matrix is $B_t(x, y) = F_t(y, x)\, s_t(x, y)$. The discrete score thus plays the same role as $\nabla \log p_t$ in the continuous setting: it encodes the relative density at neighboring states and determines the direction of the probability flow.

Training a discrete diffusion model is equivalent to learning the discrete score $s_t(x, y)$, typically via a ratio-estimation objective (Lou et al., 2023, SEDD). Under the masking process, the discrete score simplifies:

$$s_t(m, x_1) = \frac{p_t(x_1)}{p_t(m)} = \frac{t\, p_1(x_1)}{1-t} \cdot \frac{1}{p_0(m)} \approx \frac{t}{1-t}\, p_1(x_1),$$

and the backward rate $B_t(m, x_1) = F_t(x_1, m)\, s_t(m, x_1)$. Substituting $F_t(x_1, m) = 1$ (unit masking rate in the forward process) gives $B_t(m, x_1) = t\, p_1(x_1)/(1-t)$. The DFM conditional rate (2.3) for the masking process is $Q_t^{x_1}(m, x_1) = 1/(1-t)$, and the marginal rate $R_t(m, x_1) = p_1(x_1)/(1-t)$ matches exactly. **Discrete flow matching and masked discrete diffusion parameterize the same backward rate matrix**, just derived from different theoretical starting points.

## 2.7 Sampling from Discrete Flows

Given a trained rate matrix $R_\theta(x, y, t)$, generation proceeds by simulating a CTMC from $x_0 \sim p_0$ to $x_1 \sim p_{\text{data}}$. Two main approaches are available, mirroring the continuous Euler vs. ODE solver dichotomy.

**$\tau$-leaping (Euler method for CTMCs).** Discretize $[0,1]$ into $N$ steps of size $\Delta t = 1/N$. At each step:

1. Compute the transition probability matrix $P_t = I + \Delta t \cdot R_\theta(\cdot, \cdot, t)$.
2. Sample the next state: $X_{t+\Delta t} \sim \mathrm{Categorical}(P_t(X_t, \cdot))$.

Correctness requires $\Delta t \cdot R_\theta(x, y, t) \geq 0$ for all $x \neq y$, which holds if $\Delta t$ is small enough relative to the maximum rate. The local error is $O(\Delta t^2)$ in total variation.

**Gillespie's exact algorithm.** For exact simulation (zero discretization error), use Gillespie's stochastic simulation algorithm: at each state $x$, the time to the next jump is exponential with rate $\lambda(x,t) = -R_\theta(x, x, t)$, and the jump destination $y$ is drawn from the normalized rates $R_\theta(x, y, t) / \lambda(x,t)$. Exact simulation is computationally expensive for large vocabularies ($O(S)$ per step) but provides unbiased samples and is the gold standard for benchmarking.

**Predictor-corrector sampling.** A practical improvement combines $\tau$-leaping (the predictor) with a CTMC-version of Langevin correction (the corrector): after each $\tau$-leaping step, run a short sequence of forward–backward CTMC steps that leave $p_t$ invariant but improve mixing. This mirrors the predictor-corrector sampling introduced by Song et al. (2021) for continuous diffusion and substantially improves sample quality at a fixed NFE budget.

> [!remark] 2.3
> The NFE–quality trade-off in discrete flows has a qualitatively different character than in continuous flows. In continuous flow matching, straighter trajectories (achieved via OT couplings) directly reduce the required number of Euler steps. In discrete flows, the "straightness" of a trajectory is measured in terms of how few transitions are needed to go from $p_0$ to $p_{\text{data}}$. Under the masking process, each token is unmasked exactly once — the trajectory is already maximally "straight" — and the NFE is simply $L/N$, where $L$ is the sequence length and $N$ is the number of $\tau$-leaping steps. Reducing $N$ requires the model to unmask multiple tokens per step with high confidence.


## 2.8 Likelihood Computation

Unlike autoregressive models, which have trivially computable likelihoods via $\log p(x) = \sum_\ell \log p(x^\ell | x^{<\ell})$, CTMCs require integrating over all possible trajectories to compute the marginal log-likelihood $\log p_\theta(x_1)$. This is computationally intractable in general.

For the masking process, however, an efficient bound is available via the **evidence lower bound (ELBO)**:

$$\log p_\theta(x_1) \geq \mathcal{L}_{\text{ELBO}}(\theta, x_1) = \mathbb{E}_{x \sim p_t(\cdot|x_1)}\left[\int_0^1 \sum_y Q_t^{x_1}(x, y) \log R_\theta(x, y, t)\, dt\right].$$

Under the masking process, this reduces to a time-averaged cross-entropy at masked positions — the same objective as $\mathcal{L}_{\text{DFM}}^{\text{seq}}$ (2.4), confirming that the DFM training objective is simultaneously a variational lower bound on the data log-likelihood.

The key result of DUEL (Turok et al., 2026) is that **fixing the unmasking order** — processing positions in a deterministic sequence rather than randomly — recovers an exact likelihood factorization:

$$\log p_\theta(x_1) = \sum_{\ell=1}^L \log p_\theta(x_1^\ell \mid x_1^{\pi(1)}, \ldots, x_1^{\pi(\ell-1)}, m, \ldots, m),$$

where $\pi$ is the fixed unmasking order. This is precisely an autoregressive factorization (8.3) of Chapter 8, with the unmasking order playing the role of the generation order. Thus, as noted in Chapter 8: a discrete diffusion model with a fixed unmasking order *is* an autoregressive model — the two are unified by the CTMC framework.

## 2.9 Flows on the Probability Simplex

A conceptually elegant bridge between the discrete and continuous frameworks is provided by **Dirichlet flow matching**, which lifts the discrete problem into a continuous one on the probability simplex $\Delta^{S-1}$.

Identify each token $x \in \mathcal{X}$ with its one-hot vector $e_x \in \{0,1\}^S$. A distribution $p \in \Delta^{S-1}$ is a soft assignment over tokens. Flow matching on $\Delta^{S-1}$ — a Riemannian manifold with the Fisher–Rao metric — defines vector fields tangent to the simplex, which correspond to flows between soft distributions. In the limit where all mass concentrates at vertices, such flows recover the discrete jumps of a CTMC.

Concretely, a path $p_t(\cdot | x_1) = (1-t)\, \alpha + t\, e_{x_1}$ (linear interpolation from a uniform Dirichlet prior $\alpha$ to a one-hot) defines a simplex vector field:

$$u_t(p | x_1) = \frac{e_{x_1} - p}{1-t},$$

which is the exact simplex analogue of the continuous OT field $u_t(x | x_1) = (x_1 - x)/(1-t)$ from Section 1.5. Training the vector field model $v_\theta(p, t)$ on the interior of $\Delta^{S-1}$ and discretizing its output at inference time (argmax, or sampling) recovers the discrete output.

The simplex perspective connects to the Riemannian flow matching of Chapter 3: $\Delta^{S-1}$ with the Fisher–Rao metric is a Riemannian manifold of non-constant curvature, and the geodesics under this metric are the natural interpolants. It also provides a way to handle **uncertainty at inference time**: instead of committing to a hard token at each step, the model maintains a distribution over tokens and refines it progressively.

## 2.10 Applications

The discrete flow matching framework has demonstrated strong empirical results across the domains where discrete structure is native.

**Protein sequence design.** Protein sequences are strings over a 20-letter amino acid alphabet, with lengths ranging from tens to thousands of residues. The fitness landscape of protein sequences — which sequences fold stably, bind targets, or catalyze reactions — is highly multi-modal and poorly understood. Discrete flow matching models trained on sequence databases (UniRef, UniProt) with structure-conditioned guidance (from predicted structure models like ESMFold) achieve competitive performance with autoregressive baselines on standard protein design benchmarks, with the additional benefit of bidirectional conditioning: any subset of positions can be fixed as context while the remaining positions are generated.

**DNA and RNA.** Regulatory DNA and RNA sequences have complex functional structure: promoter elements, splice sites, binding motifs for transcription factors. Masked diffusion and DFM models trained on genomic data can generate sequences with prescribed regulatory activity, conditioned on epigenetic context or cell type, making them powerful tools for synthetic biology.

**Language modeling.** The connection between DFM and masked language models (Section 2.5) implies that large-scale pretraining in the DFM framework can leverage the extensive infrastructure developed for transformers. MDLM (Sahoo et al., 2024) and similar models demonstrate that masked diffusion language models approach the perplexity of autoregressive transformers at comparable parameter counts, with additional generative flexibility: they can fill in arbitrary spans rather than being constrained to left-to-right generation.

**Graph and molecular generation.** Small molecules can be represented as graphs over atom and bond types — a product of discrete spaces for atom identity and continuous spaces for 3D coordinates. Discrete flow matching handles the atom type and bond order components, while continuous flow matching (or Riemannian flow matching over $SE(3)$, Chapter 3) handles the 3D geometry. The product-space framework for combining these two components is the subject of Chapter 5.

## 2.11 Historical Notes and Connections

The mathematical foundations of discrete generative modeling are older than the machine learning literature might suggest. The CTMC framework dates to Kolmogorov (1931) and Doeblin (1940). The idea of using CTMCs for generative modeling was first systematically explored by **Campbell, Benton, De Bortoli, Rainforth et al. (2022)** in "A Continuous Time Framework for Discrete Denoising Models," which established the score-based perspective on discrete diffusion and derived the backward rate matrix. **Austin, Johnson, Ho & Barber (2021)** introduced D3PM (Discrete Denoising Diffusion Probabilistic Models), a discrete-time analogue of DDPM for finite state spaces, establishing the connection to masked language models. The score-ratio parameterization was developed by **Lou, Meng & Ermon (2023)** in SEDD.

**Gat, Lipsitz, Ben-Hamu et al. (2024)** introduced **Discrete Flow Matching** proper — the CTMC analogue of the conditional flow matching trick, establishing Theorem 2.2 and the DFM objective (2.2). Concurrent and independent work by **Campbell, Harvey, Cautis et al. (2024)** in "Generative Flows on Discrete State-Spaces" developed essentially the same framework under different notation.

The **simplex / Dirichlet flow matching** perspective was developed by Stark, Jing et al. (2024) and Alcaide-Lopez et al. (2024), providing the geometric bridge to Chapter 3.

**What remains open.** The central open questions in discrete flow matching concern:
- *Optimal noise schedules*: should one always use the linear schedule $p_t(x|x_1) = t\delta_{x_1} + (1-t)p_0$, or are there discrete analogues of VP/VE schedules that offer better training signal?
- *Multi-token transitions*: allowing simultaneous multi-position jumps (beyond the mean-field factorization) could dramatically reduce the required number of sampling steps but requires tractable parameterizations of high-order rate matrices.
- *Guidance*: how to condition a trained DFM model on auxiliary information (structure, function, property) at inference time without retraining — the discrete analogue of classifier-free guidance in continuous diffusion.
- *Theoretical guarantees*: convergence rates of $\tau$-leaping for non-homogeneous CTMCs, and conditions under which the trained rate matrix $R_\theta$ converges to the marginal $R_t$ in total variation.

Chapter 3 takes the theory in a different direction: instead of making the state space discrete, it makes the state space curved — a Riemannian manifold. The mathematical bridge is the observation that the simplex $\Delta^{S-1}$, which appeared naturally at the end of this chapter, is itself a Riemannian manifold. The next chapter develops flow matching for the full generality of such spaces.

---

*A practitioner wishing to implement discrete flow matching for protein sequence design will find that the masking process with factored rate matrices (2.4), a standard transformer architecture, and $N = 100$ $\tau$-leaping steps at inference reproduces the qualitative behavior of MDLM. The theoretical machinery of this chapter — CTMC generators, Kolmogorov equations, the marginal rate identity — is what guarantees this simple procedure is doing something principled.*
