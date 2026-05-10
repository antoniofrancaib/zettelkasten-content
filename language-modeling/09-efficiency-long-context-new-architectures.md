---

title: "Efficiency, Long Context, and New Architectures"
subtitle: "MoE, FlashAttention, Mamba, and the question of what comes after the Transformer"
---


By 2022, the Transformer had won. It dominated natural language processing, was making inroads into vision and audio, and had become the default architecture for any problem involving sequences. The question of *what architecture to use* had effectively been settled. The question that replaced it was more practical and, in some ways, harder: *how do you use this architecture at the scales now required, without burning through compute and memory at rates that make deployment economically impossible?*

The problem was not obscure. Self-attention over a sequence of length $T$ costs $O(T^2)$ in both memory and compute. For the sequence lengths of the original Transformer — a few hundred tokens — this was manageable. For the sequence lengths that practical applications demanded — tens of thousands of tokens for long documents, hundreds of thousands for genomics, millions for entire codebases — it was catastrophic. A single forward pass over a 128,000-token sequence required storing an attention score matrix of size $T^2$ per head. That is not possible on any hardware that exists.

The inference problem was distinct from the training problem. During training, the full computation was batched across the sequence. During inference, the model had to maintain a **key-value cache** for all previous positions in the context, growing linearly with sequence length per token generated. For a 70-billion-parameter model running concurrent users with long contexts, the KV cache alone required hundreds of gigabytes of memory — more than the model weights themselves.

The field responded on several fronts simultaneously. Hardware-aware algorithms rewrote the attention kernel without changing its semantics. New positional encodings freed models from training-length limits. Sparse architectures decoupled parameter count from activation cost. And entirely new sequence models, drawing on classical signal processing and linear dynamical systems theory, proposed to compete with the Transformer by abandoning the attention mechanism entirely.

## FlashAttention

The key insight came from asking not whether attention was efficient in FLOP terms, but whether it was efficient given the specific hardware on which it ran. Attention is not compute-bound; it is **memory-bandwidth-bound**. A modern GPU has two levels of memory: a large high-bandwidth memory (HBM), typically 40–80 GB, with bandwidth around 2–3 TB/s; and a small, extremely fast on-chip SRAM, around 20 MB, with bandwidth roughly ten times higher. Standard attention reads the query, key, and value matrices from HBM, computes the full $T \times T$ attention score matrix, writes it to HBM, reads it back for the softmax, writes again, reads once more to multiply by values, and writes the output. The $T \times T$ matrix transits slow memory multiple times.

**FlashAttention** (Dao et al., 2022) restructured the attention computation to avoid materializing the full attention matrix in HBM. The technique is **tiling**: process queries, keys, and values in blocks that fit in SRAM. For each block of queries, iterate over all blocks of keys and values, computing partial attention contributions and accumulating the output without writing intermediate matrices to HBM. This requires maintaining a running softmax normalization across tiles, which is achievable via the log-sum-exp identity: the running maximum $m_i$ and normalization $\ell_i$ are updated as new key blocks are processed, and the output is corrected in-place at each step.

The result is **exact attention** — not an approximation — computed with asymptotically fewer HBM accesses. FlashAttention reduced memory usage from $O(T^2)$ to $O(T)$ and achieved 2–4× wall-clock speedup over standard implementations. FlashAttention-2 (Dao, 2023) extended this with improved work partitioning across GPU warps, reaching 50–70% of theoretical hardware FLOP utilization — exceptionally high for a memory-intensive operation.

The impact was immediate. Long contexts that had been computationally infeasible became routine. FlashAttention was integrated into nearly every production Transformer implementation within months of release, and subsequent efficient attention research largely abandoned approximate methods in favor of FlashAttention variants.

## Positional Encodings and Long Context

Making long contexts computationally tractable was only half the problem. The other half was representational: standard positional encodings broke beyond the training-length limit.

The sinusoidal positional encodings of the original Transformer were fixed functions of absolute position. In principle they could encode any position, but the model had only been trained with positions up to some maximum $T_{\max}$. Positions beyond $T_{\max}$ had never appeared during training. The model's behavior there was empirically poor — attention distributions became erratic, and performance degraded sharply.

**RoPE** (Rotary Position Embedding, Su et al., 2021) proposed a more principled approach. Instead of adding a fixed encoding to the token embedding, RoPE rotates the query and key vectors by an angle proportional to their position before computing dot products. The rotation is structured so that the inner product between a query at position $m$ and a key at position $n$ depends only on their *relative* displacement $m - n$:

$$\langle R_\Theta^m\, \mathbf{q},\; R_\Theta^n\, \mathbf{k} \rangle = \langle R_\Theta^{m-n}\, \mathbf{q},\; \mathbf{k} \rangle,$$

where $R_\Theta^m$ is a block-diagonal rotation matrix with angles $m\theta_i$ for $\theta_i = 10000^{-2i/d}$. The model never sees absolute positions, only relative displacements — which makes generalization to longer sequences more natural. RoPE was adopted in LLaMA and became the dominant positional encoding for decoder-only models.

**ALiBi** (Press et al., 2022) took a different approach: add a fixed linear bias to attention logits based on the distance between query and key positions, with no learned parameters. A query at position $t$ attending to position $j$ receives a bias $-m \cdot (t - j)$ for a head-specific slope $m$. This explicitly penalizes long-range attention — a useful inductive bias when nearby context is more relevant than distant context — and extrapolates naturally beyond the training length without any modification.

Context extension became a standard post-training procedure. **YaRN** (Peng et al., 2023) and related methods showed that RoPE-based models trained with 4K-token contexts could be extended to 128K or longer by rescaling the positional frequencies, with modest performance degradation. A two-week fine-tuning run could unlock a context window that would have required exponentially more pretraining compute to achieve directly.

But extending the context window raised a subtler question: did models actually *use* what they were given? Liu et al. (2023) documented a troubling pattern they called **lost in the middle**: language models systematically attended more to information at the beginning and end of long contexts, underweighting material in the interior. A model with a 128K-token context window did not use it like a random-access database. It used it like a document read in a hurry — with the middle skimmed. The context window was a necessary but not sufficient condition for genuine long-context reasoning.

## Mixture of Experts

The scaling laws had established that more parameters meant better performance. But more parameters also meant more compute per token — the two were coupled in a standard dense Transformer. **Mixture of Experts** architectures decoupled them.

In an MoE Transformer, the feedforward sublayer is replaced by $E$ expert networks, each a standard FFN. A learned **routing function** determines, for each token, which $k$ of the $E$ experts process it. Only the selected experts' weights are active for that token; the rest are skipped. Total parameter count scales with $E$, but compute per token scales with $k / E$ of total expert capacity — a fraction that can be made as small as desired by adding more experts.

The Switch Transformer (Fedus et al., 2021) used top-1 routing: each token was sent to exactly one expert. With 128 experts, a model could have 128 times the parameters of a dense model at comparable per-token compute cost. The bottleneck was engineering: expert parallelism required routing tokens across devices, with load balancing to prevent some experts from receiving most tokens while others idled. The auxiliary balance loss

$$\mathcal{L}_{\mathrm{balance}} = \alpha \cdot E \sum_{i=1}^{E} f_i \cdot P_i,$$

where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ is the average routing probability for expert $i$, penalized unequal utilization. Without it, routing collapsed: the model learned to send most tokens to a single expert that happened to produce lower loss early in training, wasting the rest of the capacity.

**Mixtral 8×7B** (Jiang et al., 2024) demonstrated MoE at practical scale. With 8 experts per layer and top-2 routing, Mixtral had 46.7 billion total parameters but only 12.9 billion active per token — comparable to a dense 13B model in per-token compute cost. On standard benchmarks, Mixtral matched or exceeded Llama 2 70B at roughly 5× lower inference cost. The caveat was memory: all expert weights had to be loaded, even if only two were active per token. An MoE model is memory-intensive but compute-efficient — a good tradeoff for the regime where inference throughput matters more than model footprint.

## State Space Models and Mamba

While the efficiency work around FlashAttention, RoPE, and MoE operated within the Transformer framework, a parallel research thread asked a more fundamental question: was the attention mechanism necessary at all?

**Structured State Space Models** (S4, Gu et al., 2021) drew on the theory of linear dynamical systems. The core idea is a continuous-time system

$$\mathbf{h}'(t) = A\,\mathbf{h}(t) + B\,x(t), \qquad y(t) = C\,\mathbf{h}(t),$$

with state $\mathbf{h}(t) \in \mathbb{R}^N$, discretized and unrolled to process discrete sequences. The choice of $A$ mattered enormously: S4 parameterized it as a **HiPPO matrix** — designed so that the state approximates the history of the input optimally under a particular measure — allowing the model to maintain informative summaries of arbitrarily long pasts without attention. In training, the SSM could be computed as a convolution in $O(T \log T)$ time. In inference, it ran as a recurrence in $O(1)$ per step, with $O(N)$ state. On the Long Range Arena benchmark, designed specifically to test long-range dependencies, S4 substantially outperformed Transformers while using less compute.

The limitation of S4 was its input independence. The transition matrix $A$ and input matrix $B$ were fixed across all tokens — the same dynamics regardless of what the sequence contained. This made it impossible for the model to selectively retain or discard information based on content, a capability that attention provided naturally and that LSTMs approximated via gating.

**Mamba** (Gu and Dao, 2023) addressed this by making the state transition parameters functions of the input. For each step, the matrices $B$ and $C$ and a discretization step $\Delta$ are computed from the current token:

$$(B_t,\; C_t,\; \Delta_t) = \mathrm{Linear}(\mathbf{x}_t).$$

This **selective scan** mechanism allowed the model to dynamically control what information entered and exited the state based on the current input — analogous to the gating of an LSTM, but implemented within the SSM framework and at scales where LSTMs had previously been untrainable. Because the transitions vary per step, the selective SSM cannot be computed as a single convolution; it requires a parallel scan, achievable in $O(T \log T)$ time. Mamba matched Transformer performance on language modeling benchmarks up to several billion parameters, while maintaining $O(1)$ per-step inference and $O(T)$ memory.

At larger scales, however, a limitation emerged. Tasks requiring **verbatim recall** — retrieving a specific string or fact stated earlier in the context — exposed a systematic weakness. The compressed state, however expressive, was a lossy summary. The Transformer's key-value cache was expensive in memory but was a random-access store; a query could attend directly to a specific earlier token regardless of what had happened in between. Mamba's state compressed the past continuously; by the time a query was posed, specific early tokens might have been overwritten by subsequent updates. On benchmarks designed to test in-context retrieval, Mamba underperformed Transformers at equivalent scale.

The practical response was **hybrid architectures**. Jamba (Lieber et al., 2024) and Zamba (Glorioso et al., 2024) interleaved Mamba layers with standard Transformer attention layers, achieving the inference efficiency of SSMs for the majority of compute while preserving attention's exact retrieval capability where needed. The fraction of attention layers became a design parameter, trading off memory and compute against recall. Early results suggested that a small fraction of attention layers — perhaps one in every four — could recover most of the retrieval capability at a fraction of the full attention cost.

**Mamba-2** (Dao and Gu, 2024) established a theoretical connection between selective SSMs and attention, showing that both could be viewed as instances of a broader class of **structured matrix multiplications** over sequences. The unification was more than aesthetic: it enabled more efficient hardware kernels and clarified which design choices were load-bearing versus incidental.

## Model Compression and the Deployment Frontier

In parallel with architectural work, a practically urgent thread focused on deploying existing large models within the memory and compute budgets of real hardware. Three techniques saw widespread adoption.

**Quantization** represents model weights in lower-precision formats. The difference between float16 and int4 quantization of a 70-billion-parameter model is the difference between roughly 140 GB and 35 GB in memory — between requiring eight high-end GPUs and two. **GPTQ** (Frantar et al., 2022) achieved post-training quantization to 4-bit precision with minimal accuracy loss, using second-order gradient information to compensate for quantization error layer by layer. **AWQ** (Lin et al., 2023) identified the weight channels most sensitive to quantization and protected them before applying low-bit compression to the remainder. Both methods made it possible to run billion-parameter models on hardware that would have been inadequate for their float16 counterparts, enabling a wave of local deployment on consumer GPUs and edge devices.

**Knowledge distillation** (Hinton et al., 2015, applied to language models in DistilBERT by Sanh et al., 2019) trains a small **student** model to match the output distribution of a large **teacher** rather than training from scratch on labels. The teacher's softmax probabilities over the full vocabulary — which reflect the relative plausibility of all tokens, not merely the argmax — convey richer supervision than one-hot targets. DistilBERT achieved 97% of BERT's benchmark performance at 40% fewer parameters. Applied to instruction-tuned LLaMA and Mistral derivatives, distillation produced models small enough to run on laptops while retaining a substantial fraction of the instruction-following capability of their larger teachers.

**Structured pruning** removed attention heads, feedforward neurons, or entire layers that contributed least to model quality. Michel et al. (2019) showed that attention heads had highly variable importance — some could be removed with negligible effect, while others were essential. The finding implied significant redundancy in standard Transformer training. Sheared LLaMA (Xia et al., 2023) demonstrated that structured pruning followed by continued pretraining could efficiently produce smaller models from large ones without the full compute cost of training from scratch, making it possible to obtain a 1.3B model from a 7B model at a fraction of the original pretraining budget.

## The Architecture Question, Reopened

By 2025, the question that had seemed settled — what architecture to use for language modeling — was genuinely open again, not because the Transformer had failed, but because its success had created conditions under which alternatives could be fairly evaluated.

The Transformer's dominance was partly a hardware-software cospecialization problem. Decades of GPU and TPU design had optimized hardware for the dense matrix multiplications that Transformers perform. Compilers, frameworks, and training infrastructure had been written for Transformers. Switching architectures was not a matter of demonstrating competitive benchmark numbers; it required rebuilding an ecosystem. A new architecture had to be not merely competitive but substantially superior to justify the switching cost.

This created an asymmetry that favored incremental improvements over architectural replacement. FlashAttention had made quadratic attention practical for most realistic sequence lengths. MoE had decoupled parameter count from per-token compute cost. Quantization had made large models deployable on modest hardware. The engineering infrastructure around the Transformer was growing faster than the theoretical case for replacing it. Each solved problem reduced the pressure to switch.

The Transformer had also accumulated the most important asset of any dominant technology: interpretability. Mechanistic understanding of how Transformers processed language — how attention heads specialized, how MLP layers stored and retrieved factual associations, how residual streams carried information across layers — was growing steadily. Building comparable interpretability for Mamba or hybrid architectures would require years of dedicated work. The scientific value of understanding what you have is not separable from the practical value of continuing to use it.

What the efficiency era established, regardless of how the architecture question is ultimately resolved, was that the field had moved beyond the assumption that the first architecture to succeed at scale was the one that should dominate forever. The Transformer had earned its position. It had not been anointed it. And the systematic investigation of its inefficiencies — the $O(T^2)$ attention cost, the KV cache, the fixed positional encodings, the coupled parameter and compute counts — had produced a clearer understanding of what a better architecture would need to provide.

The question of what came after the Transformer was not premature. It was the right question. The answer, as of the middle of the 2020s, remained open.

---

*FlashAttention is introduced in Dao et al. (2022), "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," extended in Dao (2023). RoPE is in Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary Position Embedding"; YaRN context extension in Peng et al. (2023). ALiBi is in Press et al. (2022), "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." The lost-in-the-middle phenomenon is documented in Liu et al. (2023). MoE scaling is introduced in Fedus et al. (2021), "Switch Transformers," and demonstrated at practical scale in Jiang et al. (2024), "Mixtral of Experts." S4 is in Gu et al. (2021), "Efficiently Modeling Long Sequences with Structured State Spaces"; Mamba in Gu and Dao (2023), "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"; Mamba-2 in Dao and Gu (2024). GPTQ is in Frantar et al. (2022); AWQ in Lin et al. (2023). DistilBERT is in Sanh et al. (2019). Attention head pruning is studied in Michel et al. (2019); Sheared LLaMA in Xia et al. (2023).*
