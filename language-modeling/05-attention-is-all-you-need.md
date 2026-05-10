---

title: "Attention Is All You Need"
subtitle: "Self-attention, positional encoding, layer normalization, and the Transformer"
---


On June 12, 2017, eight researchers at Google Brain and Google Research posted a preprint to arXiv. The title was a provocation: "Attention Is All You Need." The claim embedded in that title — that recurrence and convolution were unnecessary, that attention alone could handle sequence modeling — seemed bold to the point of recklessness. Recurrence had been the backbone of sequence processing for thirty years. LSTMs had just demonstrated state-of-the-art performance on machine translation, language modeling, and dozens of downstream tasks. The idea that you could throw them out entirely was not obvious.

It was, nevertheless, correct. The **Transformer** architecture introduced in that paper became the foundation of essentially every major language model developed since. GPT, BERT, T5, LLaMA, GPT-4, Claude — all are Transformers. Understanding why requires understanding what the Transformer actually is: not a single new idea, but a particular assembly of ideas that had been circulating for years, arranged in a way that achieved something qualitatively new.

## The Problem with Recurrence, Restated

To appreciate the Transformer's contribution, it helps to be precise about what was wrong with recurrent models. Chapter 3 discussed the vanishing gradient and the sequential bottleneck. These are two distinct problems.

The **vanishing gradient** is a training-time problem: gradients from distant positions shrink exponentially as they propagate backward through time. LSTMs mitigated this via gating, but even LSTMs struggled with very long-range dependencies.

The **sequential bottleneck** is a computational problem: the hidden state $\mathbf{h}_t$ cannot be computed until $\mathbf{h}_{t-1}$ is computed. This serializes the computation over the sequence length dimension, preventing parallelization. On modern GPU hardware, where compute is available in massive parallel batches, sequential computation is deeply inefficient.

Chapter 4 introduced self-attention as a mechanism that connects any two positions in a single operation. Self-attention over a sequence of length $T$ requires $O(T^2)$ operations total but only $O(1)$ sequential steps — every position can attend to every other position simultaneously. The path length between any two positions is 1, regardless of sequence length. This directly addresses both problems: gradients flow directly between any two positions without passing through intermediate states, and the entire attention computation can be parallelized.

The Transformer's contribution was to ask: what if the entire model were built from self-attention and nothing else?

## The Architecture

The Transformer is an **encoder-decoder** architecture for sequence-to-sequence tasks, though both halves are independently useful. The encoder maps an input sequence $x_1, \ldots, x_T$ to a sequence of continuous representations $\mathbf{z}_1, \ldots, \mathbf{z}_T$. The decoder generates the output sequence $y_1, \ldots, y_{T'}$ one token at a time, attending to both the encoder output and the previously generated tokens.

Each half consists of a stack of identical layers. The encoder has $N = 6$ layers, each containing two sublayers:

1. **Multi-head self-attention** over the encoder's own sequence
2. **Position-wise feedforward network** applied independently at each position

Each sublayer is wrapped in a **residual connection** and **layer normalization**:

$$\mathbf{x}^{(\ell+1)} = \mathrm{LayerNorm}\!\left(\mathbf{x}^{(\ell)} + \mathrm{Sublayer}(\mathbf{x}^{(\ell)})\right).$$

The decoder adds a third sublayer between the self-attention and feedforward: **cross-attention** over the encoder output, in which the decoder's representations serve as queries and the encoder's representations serve as keys and values.

This description is simple. What makes it work is the interplay among the components, and understanding each in turn.

## Positional Encoding

Self-attention is permutation-equivariant: if you shuffle the input tokens, the attention outputs shuffle in the same way, but the pairwise attention weights are unchanged (since they depend only on content, not position). This means a pure self-attention model has no notion of word order — "the cat ate the mouse" and "the mouse ate the cat" produce identical attention patterns up to permutation.

Recurrent models implicitly encode position through the sequential nature of their updates: $\mathbf{h}_t$ is computed from $\mathbf{h}_{t-1}$, so position is baked into the hidden state. Without recurrence, position must be injected explicitly.

Vaswani et al. add a **positional encoding** to the input embeddings before they enter the first layer. The encoding is a deterministic function of position:

$$\mathrm{PE}(t, 2k) = \sin\!\left(\frac{t}{10000^{2k/d_{\mathrm{model}}}}\right),$$
$$\mathrm{PE}(t, 2k+1) = \cos\!\left(\frac{t}{10000^{2k/d_{\mathrm{model}}}}\right),$$

where $t$ is the position index and $k$ indexes the dimension. The input to the first layer is the sum of the word embedding and the positional encoding:

$$\mathbf{x}_t = \mathbf{e}_{w_t} + \mathrm{PE}(t).$$

The sinusoidal encoding has a useful property: the encoding for position $t + \Delta$ can be expressed as a linear function of the encoding for position $t$, for any fixed offset $\Delta$. This means the model can potentially learn to attend to relative positions by attending to the difference in encodings, without needing the absolute position.

Why sinusoids rather than learned position embeddings? Vaswani et al. tested both and found similar performance. The sinusoidal version generalizes to sequence lengths not seen during training — an important property for deployment. Later work (most prominently RoPE, Rotary Position Embedding) found better ways to encode relative positions, but the sinusoidal encoding was adequate to establish the concept.

## Multi-Head Self-Attention

The core computation of each encoder layer is multi-head self-attention. Recall from Chapter 4 that attention is parameterized by projections of the input into query, key, and value spaces. In self-attention, all three come from the same sequence:

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V,$$

where $X \in \mathbb{R}^{T \times d_{\mathrm{model}}}$ is the matrix of input representations and $W^Q, W^K, W^V \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$ are learned projections.

The scaled dot-product attention is then:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.$$

For the encoder, this is a full attention: every position attends to every other position. The result at each position is a weighted combination of the value representations of all positions, where the weights reflect pairwise relevance.

Multi-head attention runs $H = 8$ such operations with separate learned projections:

$$\mathrm{head}_h = \mathrm{Attention}(X W_h^Q,\; X W_h^K,\; X W_h^V),$$
$$\mathrm{MultiHead}(X) = [\mathrm{head}_1;\; \ldots;\; \mathrm{head}_H]\, W^O.$$

With $d_k = d_v = d_{\mathrm{model}} / H = 64$ (for $d_{\mathrm{model}} = 512$, $H = 8$), the total parameter count is the same as single-head attention with $d_k = 512$, but the model can simultaneously attend to information from different representation subspaces. In practice, different heads reliably specialize in different linguistic relationships — syntactic dependencies, coreferential links, positional adjacency.

## The Feedforward Sublayer

After the multi-head attention, each position passes through a **position-wise feedforward network** (FFN): the same two-layer network applied independently at each position:

$$\mathrm{FFN}(\mathbf{x}) = \max(0,\; \mathbf{x} W_1 + \mathbf{b}_1)\, W_2 + \mathbf{b}_2,$$

where $W_1 \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{ff}}}$ and $W_2 \in \mathbb{R}^{d_{\mathrm{ff}} \times d_{\mathrm{model}}}$, with $d_{\mathrm{ff}} = 2048$ in the original paper (four times $d_{\mathrm{model}}$).

The FFN serves a different role than the attention: where attention mixes information *across* positions (every position sees a weighted combination of all others), the FFN processes each position *independently*, applying a nonlinear transformation that can represent complex functions of the local representation. In the mechanistic interpretability literature, the FFN layers of large Transformers have been characterized as storing and retrieving factual associations — a form of learned key-value memory. The attention layer *retrieves* contextual information; the FFN layer *processes and updates* the local representation.

The ratio $d_{\mathrm{ff}} / d_{\mathrm{model}} = 4$ (expanded to 8/3 in later models using SwiGLU activations) is a hyperparameter that controls the model's capacity for position-wise transformation. A larger $d_{\mathrm{ff}}$ means more parameters in the FFN and more expressive local transformations; it also means more compute per token.

## Layer Normalization and Residual Connections

Two components are present in every sublayer that tend to be described as engineering details but are, in fact, essential:

**Residual connections** (He et al., 2016) add the input of each sublayer to its output before normalization:

$$\mathbf{x}^{(\ell+1)} = \mathrm{LayerNorm}\!\left(\mathbf{x}^{(\ell)} + \mathrm{Sublayer}\!\left(\mathbf{x}^{(\ell)}\right)\right).$$

The residual shortcut creates a highway for gradients: the gradient of the loss with respect to $\mathbf{x}^{(\ell)}$ is the sum of the gradient through the sublayer and the gradient of the identity path (which is simply the gradient of the loss with respect to $\mathbf{x}^{(\ell+1)}$). Even if the sublayer produces near-zero gradients, the identity path preserves the signal. This is what makes it possible to train Transformers with 6, 12, 24, or more layers without vanishing gradients.

**Layer normalization** (Ba et al., 2016) normalizes the activations within each layer across the feature dimension:

$$\mathrm{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sigma} \odot \boldsymbol{\gamma} + \boldsymbol{\beta},$$

where $\mu$ and $\sigma$ are the mean and standard deviation computed over the feature dimension, and $\boldsymbol{\gamma}, \boldsymbol{\beta}$ are learned scale and shift parameters. This stabilizes the distribution of activations throughout the network, preventing the internal covariate shift that makes deep networks hard to train.

The original "Attention Is All You Need" paper places normalization *after* the residual addition (Post-LN). Most subsequent models switched to Pre-LN — normalizing before the sublayer — which was found to be more stable during training of very deep models, particularly at large scale.

## Masked Attention in the Decoder

The decoder's self-attention requires a modification. During training, the decoder receives the entire target sequence at once (teacher forcing). But during generation, the decoder must not see future tokens — it should condition only on what has been generated so far. This is enforced by **causal masking**: before the softmax, set all attention weights from position $t$ to positions $j > t$ to $-\infty$, which becomes 0 after the softmax:

$$e_{tj} = \begin{cases} \mathbf{q}_t \cdot \mathbf{k}_j / \sqrt{d_k} & j \leq t \\ -\infty & j > t. \end{cases}$$

The masking is applied to the attention score matrix before the softmax, ensuring that each position can only attend to itself and earlier positions. This makes the decoder autoregressive: the representation of position $t$ depends only on positions $1, \ldots, t$.

Causal masking is what makes decoder-only Transformers (the GPT family) work as language models: at every layer, each position attends only to the left context, and the representation at the final position predicts the next token.

## The Decoder's Cross-Attention

Between the masked self-attention and the feedforward sublayer, the decoder has a cross-attention layer that attends to the encoder's output. Here the queries come from the decoder's hidden states and the keys and values come from the encoder's output:

$$Q = \mathbf{s}^{(\ell)} W^Q, \quad K = \mathbf{z}\, W^K, \quad V = \mathbf{z}\, W^V,$$

where $\mathbf{s}^{(\ell)}$ is the decoder's current representation and $\mathbf{z}$ is the encoder output. This is precisely the attention mechanism Bahdanau introduced in 2015, but now applied inside a stack of layers rather than as a single-step addition to an RNN.

The cross-attention allows the decoder to dynamically route its attention over the source sequence as it generates each target token. Unlike the fixed context vector of the original seq2seq model, the attended source representation is different for every decoder layer and every decoder position — the model learns when to consult different parts of the source at different stages of generation.

## Results and Impact

On the standard WMT 2014 English-to-German and English-to-French translation benchmarks, the Transformer achieved BLEU scores of 28.4 and 41.0 respectively — surpassing every prior model, including ensembles of large LSTMs, and doing so with significantly less training time (12 hours on 8 GPUs, compared to weeks for the best LSTM systems).

The training speedup was the key practical contribution. Because the Transformer's computation at each layer is a matrix multiplication (with no sequential dependency), it could fully utilize GPU parallelism. Training a competitive translation model in 12 hours rather than 3 weeks was not merely a matter of convenience — it meant that experiments could be run faster, hyperparameters could be tuned more thoroughly, and the scale of models that could be explored was dramatically larger.

This scalability would turn out to be the defining property of the Transformer. Not its accuracy on any particular benchmark, but its ability to get better as you made it bigger — more layers, more heads, larger $d_{\mathrm{model}}$, more training data. The LSTM also got better with scale, but it ran into the sequential bottleneck before it reached truly large regimes. The Transformer did not.

## Why Attention Works

The question of *why* the Transformer works as well as it does is not fully settled, but several explanations have been advanced and partially validated.

**Arbitrary pairwise interactions.** An LSTM processes position $t$ by reading positions $1, \ldots, t-1$ through a chain. A Transformer computes an explicit interaction between every pair of positions simultaneously. In the worst case this is $O(T^2)$ computation, but it means no dependency is more than one hop away. For tasks that require resolving long-range dependencies — coreference, syntactic agreement across clauses, semantic entailment — this direct access matters.

**Parallel computation enables scale.** Scale, as the following chapters will show, is the central variable in modern language modeling. Models that can be trained faster can be trained larger, and larger models consistently outperform smaller ones given sufficient data. The Transformer's parallelism was not an incidental engineering benefit; it was what made the subsequent scaling revolution possible.

**Attention is interpretable.** The attention weights provide a window into the model's computation that the hidden state of an LSTM does not. While subsequent research has complicated the interpretation (attention weights are not the same as importance), the ability to inspect what each head is attending to has enabled mechanistic understanding of how the model processes language.

## The Transformer Family

In the years immediately following "Attention Is All You Need," the Transformer diversified into a family of architectures depending on which part of the encoder-decoder structure was retained.

**Encoder-only** models (BERT, RoBERTa, DeBERTa) use only the encoder stack with bidirectional attention — every position can attend to every other position. These are ideal for classification, NLU tasks, and representation learning, but cannot generate text autoregressively.

**Decoder-only** models (GPT, GPT-2, LLaMA) use only the decoder stack with causal masking. Every position attends only to previous positions. These are natural language models — they are trained to predict the next token and can generate text by sampling autoregressively.

**Encoder-decoder** models (T5, BART, the original Transformer) retain both stacks. The encoder processes the input bidirectionally; the decoder generates the output autoregressively, attending to both the encoded input and previous outputs. These are natural for seq2seq tasks: translation, summarization, question answering.

The choice between these architectures — and the training objective used with each — is the subject of Chapter 6.

## What the Transformer Left Unsolved

The Transformer was a decisive architectural advance, but it was not without limitations, some fundamental and some practical.

**Quadratic attention cost.** Self-attention over a sequence of length $T$ requires $O(T^2)$ memory and $O(T^2)$ computation. For the sequence lengths typical in 2017 (a few hundred tokens), this was manageable. For long documents — thousands or tens of thousands of tokens — it became prohibitive. A large literature on efficient attention (Longformer, BigBird, Linformer, FlashAttention) developed to address this, with varying success.

**No structural inductive bias.** An LSTM has a strong prior: information flows sequentially, and recent information is more accessible than distant information. This is often appropriate for language. The Transformer has no such prior — it must learn from data whether position matters and in which direction. For small datasets, this makes the Transformer harder to train than an LSTM. The LSTM's inductive bias is a weakness at scale (it prevents parallelism) but a strength at small data.

**Context window.** Positional encodings are defined for sequences up to a maximum length set at training time. Handling inputs longer than the training context requires either truncation or architectural modifications (such as RoPE's relative position encodings, which extend more gracefully).

These limitations were all addressed, incompletely or well, in subsequent work. But they do not diminish the central achievement of "Attention Is All You Need": it established that sequence modeling could be done entirely without recurrence, and that doing so was not merely possible but superior in almost every respect. The paper became one of the most cited in the history of machine learning, not because it solved a particular problem neatly, but because it opened a new regime — one in which scale, pretraining, and fine-tuning would transform what language models could do.

---

*Key references: Vaswani et al. (2017) on the Transformer, Ba et al. (2016) on layer normalization, He et al. (2016) on residual connections, Gehring et al. (2017) on convolutional seq2seq (the concurrent work that the Transformer outperformed), Press and Wolf (2017) on input-output weight tying, Devlin et al. (2019) on BERT, Radford et al. (2018) on GPT.*
