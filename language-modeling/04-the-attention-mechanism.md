---

title: "The Attention Mechanism"
subtitle: "Bahdanau attention, seq2seq, and the soft database lookup that changed everything"
---


The LSTM solved one problem — the vanishing gradient — and in doing so revealed another. A recurrent encoder reading a sentence of length $T$ must compress everything it has read into a single fixed-size vector $\mathbf{h}_T \in \mathbb{R}^d$. The decoder must then reconstruct the meaning of the entire source from this vector alone. For short sentences this works reasonably well. For long sentences it catastrophically fails: the network forgets the beginning of the source by the time it encodes the end, and the decoder must work from a lossy summary rather than the original signal.

The solution was attention. Not the grand architectural mechanism that would come later, but a small, focused idea: let the decoder, at each step, look back at the encoder's hidden states and decide which ones are most relevant right now. A single vector bottleneck becomes a weighted query over all positions. The information is not discarded; it is indexed, and the decoder retrieves exactly what it needs.

This is the story of how that idea was born, formalized, and generalized — until the generalization became more important than the original application.

## Sequence-to-Sequence Learning

The architecture that made attention necessary was **sequence-to-sequence learning** (Sutskever, Vinyals, and Le, 2014). The problem is machine translation: given a source sentence $x_1, \ldots, x_T$ in one language, produce a target sentence $y_1, \ldots, y_{T'}$ in another. Source and target lengths differ and are not known in advance.

The seq2seq model has two components. The **encoder** is an LSTM that reads the source sequence left to right, updating its hidden state at each step:

$$\overrightarrow{\mathbf{h}}_t = \mathrm{LSTM}_{\mathrm{enc}}(\overrightarrow{\mathbf{h}}_{t-1},\, \mathbf{e}_{x_t}).$$

After reading the entire source, the final hidden state $\mathbf{c} = \overrightarrow{\mathbf{h}}_T$ is passed to the **decoder** as its initial state. The decoder is a separate LSTM that generates the target sequence one token at a time, conditioning each step on the previous token and the current hidden state:

$$\mathbf{s}_t = \mathrm{LSTM}_{\mathrm{dec}}(\mathbf{s}_{t-1},\, \mathbf{e}_{y_{t-1}}),$$
$$P(y_t \mid y_{<t},\, x) = \mathrm{softmax}(W_o\, \mathbf{s}_t + \mathbf{b}_o).$$

The model is trained end-to-end by maximizing the log-probability of the target sequence given the source, and the gradient flows back through both the decoder and encoder via BPTT.

Sutskever et al. achieved state-of-the-art BLEU scores on English-to-French translation. The result was surprising — prior machine translation systems were elaborate pipelines of hand-engineered components: phrase tables, language models, reordering models, alignment models. A single neural network, end-to-end, beat them. One engineering insight turned out to be essential: reversing the source sentence before encoding. "The cat sat on the mat" was fed to the encoder as "mat the on sat cat The." This placed the beginning of the source close to the beginning of the target in the decoding stream, making it easier for the decoder to generate the first few target words correctly. It was a hack, but it worked.

## The Bottleneck

The elegance of seq2seq concealed a fundamental architectural problem. The entire source sentence — its syntax, semantics, entities, coreference, and pragmatics — must be encoded into a vector of 1,000 or 2,000 floating-point numbers. This is a compression problem, and for long sentences the compression is severe.

Cho et al. (2014b) documented this empirically: translation quality degraded sharply as source sentence length increased. For sentences of length 30 or fewer, the model performed well. For sentences of length 60 or more, performance dropped substantially below the phrase-based baseline. The LSTM could represent short sentences well; it could not hold long ones in memory.

The problem was not the decoder or the output vocabulary. It was the **fixed-length context vector** — a single $d$-dimensional vector that the decoder received and never modified. All decoder steps had access to exactly the same representation of the source. If the source had 80 words, the decoder generating the 75th target word received the same context as the decoder generating the 1st target word. The representation could not adapt to what part of the source was currently relevant.

What if it could?

## Bahdanau Attention

In 2015, Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio published "Neural Machine Translation by Jointly Learning to Align and Translate." The central idea was simple and, in retrospect, obvious: instead of giving the decoder a single context vector, give it access to all encoder hidden states and let it compute a weighted combination at each decoding step.

The encoder is changed to be bidirectional: two LSTMs run in opposite directions, and their hidden states are concatenated to form annotation vectors:

$$\mathbf{h}_j = [\overrightarrow{\mathbf{h}}_j;\; \overleftarrow{\mathbf{h}}_j] \in \mathbb{R}^{2d}.$$

Each $\mathbf{h}_j$ encodes information about word $j$ in the context of the entire source sentence, not just the prefix up to position $j$.

The decoder at step $t$, instead of receiving a fixed context, computes a **context vector** $\mathbf{c}_t$ as a weighted sum of all encoder annotations:

$$\mathbf{c}_t = \sum_{j=1}^{T} \alpha_{tj}\, \mathbf{h}_j.$$

The weights $\alpha_{tj}$ — which sum to 1 over $j$ — are the **attention distribution**: how much attention the decoder pays to source position $j$ when generating target position $t$. They are computed from the current decoder state $\mathbf{s}_{t-1}$ and each encoder annotation $\mathbf{h}_j$ through a small feedforward network called the **alignment model**:

$$e_{tj} = \mathbf{v}_a^\top \tanh(W_a\, \mathbf{s}_{t-1} + U_a\, \mathbf{h}_j),$$
$$\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^{T} \exp(e_{tk})}.$$

The unnormalized scores $e_{tj}$ measure how well the decoder state at step $t-1$ matches each encoder annotation. The softmax converts them into a probability distribution. The context vector $\mathbf{c}_t$ is then the expected encoder annotation under this distribution.

The decoder update uses both the previous hidden state, the previous target token, and the new context vector:

$$\mathbf{s}_t = \mathrm{LSTM}_{\mathrm{dec}}(\mathbf{s}_{t-1},\, [\mathbf{e}_{y_{t-1}};\, \mathbf{c}_t]).$$

The entire system — encoder, decoder, and alignment model — is trained jointly by gradient descent. The alignment model learns what to attend to without any explicit supervision on word alignments.

## The Alignment Matrix

The attention mechanism produces, as a byproduct, an alignment matrix: for each target position $t$ and source position $j$, the weight $\alpha_{tj}$ quantifies how strongly they are associated. Bahdanau et al. visualized these matrices for English-to-French translation and found something striking: the attention weights were largely diagonal when the languages had similar word order, and off-diagonal in predictable ways where French and English systematically diverge (adjective-noun ordering, for example).

The model had learned to align source and target words without ever being shown a word alignment. Statistical machine translation systems had spent years building alignment models from bilingual corpora using the EM algorithm (IBM models 1–5). The neural model discovered alignment as a side effect of end-to-end training for translation.

This visualization became one of the most reproduced figures in NLP. It was persuasive evidence that the attention mechanism was doing something interpretable — not just improving numbers, but learning a structure that corresponded to a linguistically meaningful concept. Attention weights could be read as a soft version of the hard alignments that phrase-based MT systems computed.

## Luong Attention and the General Framework

Shortly after Bahdanau et al., Thang Luong, Hieu Pham, and Christopher Manning (2015) published "Effective Approaches to Attention-based Neural Machine Translation," which systematically explored variants of the attention score function.

Bahdanau's score is **additive** (also called MLP attention or concatenation attention): it computes a score via a learned nonlinear function of the decoder state and encoder annotation. Luong et al. showed that a simpler **dot-product** score often worked as well or better:

$$e_{tj} = \mathbf{s}_t^\top \mathbf{h}_j,$$

and introduced a **general** (bilinear) variant:

$$e_{tj} = \mathbf{s}_t^\top W_a\, \mathbf{h}_j.$$

The dot-product score has no learned parameters (beyond the representations themselves), making it computationally cheaper and easier to scale to long sequences. The general variant adds a learned weight matrix but avoids the bottleneck of the tanh nonlinearity in Bahdanau's formulation. On translation benchmarks, the differences were small.

Luong et al. also introduced the distinction between **global attention** (attending over all source positions, as Bahdanau did) and **local attention** (attending over a fixed-size window around a predicted center position), and proposed computing the context vector *after* the LSTM step rather than before it, allowing the hidden state at step $t$ to already partially condition the attention.

## Attention as Soft Database Lookup

The abstract structure of the attention operation invites a reinterpretation that would prove enormously generative. Think of the encoder hidden states not as a sequence but as a **database**: a set of key-value pairs $\{(\mathbf{k}_j, \mathbf{v}_j)\}_{j=1}^T$, where the key $\mathbf{k}_j$ is used to determine relevance and the value $\mathbf{v}_j$ is the information retrieved.

The decoder issues a **query** $\mathbf{q}_t$ (its current state). Relevance scores are computed between the query and all keys:

$$e_{tj} = \mathrm{score}(\mathbf{q}_t, \mathbf{k}_j).$$

The scores are normalized to weights:

$$\alpha_{tj} = \mathrm{softmax}(e_{t1}, \ldots, e_{tT})_j.$$

The retrieved value is the weighted sum of the database values:

$$\mathbf{c}_t = \sum_{j=1}^T \alpha_{tj}\, \mathbf{v}_j.$$

In Bahdanau attention, the keys and values are both the encoder annotations $\mathbf{h}_j$ — the same vectors are used for matching and for retrieval. But they need not be. If the keys and values are separate, the model can learn representations optimized for matching (keys) and representations optimized for retrieval (values) independently. This generalization — **separate key, value, and query projections** — is the form that would appear in the Transformer.

The database metaphor is imperfect but illuminating. A hard database lookup retrieves the single entry whose key exactly matches the query. Attention performs a *soft* lookup: every entry is retrieved, but weighted by relevance. No entry is completely ignored; no entry is retrieved with exclusive focus. This softness is what makes attention differentiable, and differentiability is what makes it trainable.

## Intra-Attention and Self-Attention

Once attention had been abstracted as a query-key-value operation, researchers began applying it in new configurations. The most consequential was **self-attention**: applying the attention mechanism not between an encoder and a decoder, but within a single sequence.

In self-attention, the sequence attends to itself. For a sequence of representations $\mathbf{h}_1, \ldots, \mathbf{h}_T$, the updated representation at position $t$ is computed by attending over all positions $j$:

$$\tilde{\mathbf{h}}_t = \sum_{j=1}^{T} \alpha_{tj}\, \mathbf{h}_j, \quad \alpha_{tj} = \mathrm{softmax}_j\!\left(\mathrm{score}(\mathbf{h}_t, \mathbf{h}_j)\right).$$

The result is a new representation at each position that incorporates information from all other positions, weighted by learned relevance. Unlike an LSTM, which aggregates context through a chain of sequential updates, self-attention connects any two positions in a *single operation* — a direct path that does not decay with distance.

Self-attention was explored in several papers before the Transformer. Cheng et al. (2016) used it for reading comprehension. Parikh et al. (2016) applied it to natural language inference. Paulus et al. (2017) used self-attention within the decoder of a summarization model. In each case, the mechanism allowed the model to resolve long-range dependencies that LSTMs had difficulty with: coreference, semantic entailment, document-level coherence.

The question was not whether self-attention was useful — it clearly was — but whether it could replace recurrence entirely. Could you build a model with no LSTM, no recurrence at all, using only self-attention?

## Scaled Dot-Product Attention

As attention mechanisms were applied at scale, a numerical issue emerged with dot-product attention. If the query and key vectors have dimension $d_k$ and their components are roughly unit variance, then the dot product $\mathbf{q} \cdot \mathbf{k}$ has variance approximately $d_k$. For large $d_k$ — say, $d_k = 512$ — the dot products become very large in magnitude, pushing the softmax into regions where its gradient is near zero. Training becomes slow and unstable.

Vaswani et al. (2017) proposed a simple fix: divide by $\sqrt{d_k}$ before the softmax. The **scaled dot-product attention** is:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V,$$

where $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, and $V \in \mathbb{R}^{m \times d_v}$ are matrices of queries, keys, and values respectively, and the softmax is applied row-wise. This formulation processes all $n$ queries simultaneously as a matrix operation — a crucial property for GPU efficiency.

The scaling factor $1/\sqrt{d_k}$ restores unit variance to the pre-softmax scores, keeping gradients well-conditioned throughout training. It is one of those engineering choices that matters enormously in practice and is almost invisible in descriptions of the method.

## Multi-Head Attention

A single attention function computes a single weighted average of the values. But different aspects of a sequence may require attending to different positions for different reasons: syntactic agreement requires attending to the subject of a verb; coreference requires attending to the antecedent of a pronoun; pragmatic presupposition may require attending to earlier discourse context. A single attention distribution cannot simultaneously focus on multiple positions for multiple reasons.

**Multi-head attention** runs $H$ attention operations in parallel, each with its own learned projections for queries, keys, and values:

$$\mathrm{head}_h = \mathrm{Attention}(Q W_h^Q,\; K W_h^K,\; V W_h^V),$$
$$\mathrm{MultiHead}(Q, K, V) = [\mathrm{head}_1;\; \ldots;\; \mathrm{head}_H]\, W^O,$$

where $W_h^Q \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$, $W_h^K \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$, $W_h^V \in \mathbb{R}^{d_{\mathrm{model}} \times d_v}$ are learned projection matrices for head $h$, and $W^O \in \mathbb{R}^{Hd_v \times d_{\mathrm{model}}}$ projects the concatenated outputs back to the model dimension.

With $H = 8$ heads and $d_k = d_v = d_{\mathrm{model}} / H$, the total computational cost is similar to single-head attention with full $d_{\mathrm{model}}$ dimensions, but the model can jointly attend to information from different representation subspaces at different positions. In the Transformer, different heads have been found to specialize in different linguistic functions — some attending to syntactic dependencies, others to semantic relationships, others to positional proximity.

## What Attention Changed

The attention mechanism, in its original 2015 form, was a component of a larger system — an add-on to seq2seq that improved translation quality, especially for long sentences. Its contribution to machine translation was significant but not revolutionary: BLEU scores improved by a few points, qualitative translation of long sentences improved substantially.

What changed the field was the realization of what attention *was*, abstracted from its specific application. It was a differentiable lookup mechanism. It was a way of computing context-sensitive representations without sequential processing. It was a structure that allowed any two positions in a sequence to interact directly, in a single operation, with a strength determined by learned relevance.

These properties solved problems that LSTMs had not: the information bottleneck in seq2seq (attention replaced the fixed context vector with a dynamic one), the difficulty of long-range dependencies (attention provided a direct path between distant positions), the inability to parallelize training (attention over an entire sequence can be computed as a single matrix multiplication, with no sequential dependency).

The LSTM processed a sequence by passing information forward through time, one step at a time. Attention processed a sequence by allowing every position to look at every other position simultaneously. These are fundamentally different computational models, and the second turned out to be strictly more capable for the kinds of tasks that language modeling requires.

## The Path to the Transformer

By late 2016, the components were all in place: scaled dot-product attention, multi-head attention, self-attention for within-sequence processing, separate key and value projections, position-wise feedforward networks for combining attention outputs. What remained was to assemble them into an architecture with no recurrence at all — and to show that this architecture, trained at scale, could do everything the LSTM could do and more.

That paper appeared in June 2017. Its title was a provocation: "Attention Is All You Need." The claim was literal: no convolutions, no recurrence, no long short-term memory, no gating. Just attention, feedforward layers, residual connections, and layer normalization.

The Transformer did not invent any single component from scratch. Every piece had appeared before. What Vaswani et al. did was recognize that these pieces, assembled in the right way, could replace recurrence entirely — and that the resulting architecture was faster to train, easier to scale, and, crucially, better at capturing the long-range dependencies that make language hard.

The age of recurrence was ending. The age of attention was beginning.

---

*Key references: Sutskever et al. (2014) on seq2seq learning, Cho et al. (2014b) on the encoder-decoder architecture, Bahdanau et al. (2015) on additive attention, Luong et al. (2015) on multiplicative attention variants, Cheng et al. (2016) on intra-sentence attention, Parikh et al. (2016) on decomposable attention for NLI, Vaswani et al. (2017) on scaled dot-product and multi-head attention.*
