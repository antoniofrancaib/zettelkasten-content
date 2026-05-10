---

title: "Sequence Memory"
subtitle: "RNNs, LSTMs, the vanishing gradient, and the first contextual representations"
---


Word embeddings gave language a geometry. What they could not give it was time. A sentence is not a bag of vectors — it is a sequence, and order matters. "The dog bit the man" and "The man bit the dog" have identical word embeddings; they mean opposite things. The structure of language is sequential, and meaning is not in words alone but in their arrangement and mutual context. To process language well, a model must not merely represent words but *read* them — maintaining a running understanding that updates with each new token and accumulates over arbitrarily long spans.

This is the problem of sequence memory, and it was the central preoccupation of neural NLP from roughly 2010 to 2017. The architecture that attempted to solve it was the recurrent neural network, and the history of that attempt is the history of a beautiful idea encountering a deep mathematical obstruction, and the engineering solutions that, imperfectly, got around it.

## Recurrent Neural Networks

A **recurrent neural network** (RNN) processes a sequence $x_1, x_2, \ldots, x_T$ by maintaining a hidden state $\mathbf{h}_t \in \mathbb{R}^d$ that is updated at each step:

$$\mathbf{h}_t = \tanh(W_h\, \mathbf{h}_{t-1} + W_x\, \mathbf{x}_t + \mathbf{b}),$$

where $W_h \in \mathbb{R}^{d \times d}$, $W_x \in \mathbb{R}^{d \times m}$, and $\mathbf{b} \in \mathbb{R}^d$ are learned parameters, $\mathbf{x}_t \in \mathbb{R}^m$ is the embedding of the $t$-th input token, and $\mathbf{h}_0$ is initialized to zero. The same weights are reused at every time step — this is what makes the network *recurrent*. To produce a prediction at step $t$ (for language modeling, the distribution over the next word), the hidden state is projected through an output layer:

$$P(w_{t+1} \mid w_1, \ldots, w_t) = \mathrm{softmax}(W_o\, \mathbf{h}_t + \mathbf{b}_o).$$

The intuition is clear: $\mathbf{h}_t$ is a *summary* of the entire left context $w_1, \ldots, w_t$, compressed into a fixed-dimensional vector. Unlike the n-gram model, which only conditions on the last $n-1$ words, the RNN in principle conditions on the full history. The Markov assumption is gone. The memory is, in principle, unbounded.

In practice, it was not. And understanding why requires understanding what happens to gradients as they propagate backward through time.

## Backpropagation Through Time

Training an RNN requires computing the gradient of the loss with respect to every parameter. The loss at time $t$ depends on the hidden state $\mathbf{h}_t$, which depends on $\mathbf{h}_{t-1}$, which depends on $\mathbf{h}_{t-2}$, and so on back to $\mathbf{h}_0$. The chain rule must be applied across all $T$ time steps — a procedure called **backpropagation through time** (BPTT).

The gradient of the loss $\mathcal{L}$ at step $T$ with respect to $\mathbf{h}_t$ involves a product of Jacobians:

$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} = \prod_{k=t+1}^{T} \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \prod_{k=t+1}^{T} \mathrm{diag}\!\left(\tanh'\!\left(W_h \mathbf{h}_{k-1} + W_x \mathbf{x}_k + \mathbf{b}\right)\right) W_h.$$

This is a product of $T - t$ matrices. What happens to such a product as the chain grows long depends on the spectral properties of the matrices involved.

## The Vanishing and Exploding Gradient

Hochreiter (1991) — then a diploma student in Munich, writing in German, largely ignored for years — identified the central problem. Bengio, Simard, and Frasconi made it precise in their 1994 paper "Learning Long-Term Dependencies with Gradient Descent is Difficult."

The argument is stark. If the weight matrix $W_h$ has spectral radius $\rho < 1$ (all singular values less than 1), then the product of matrices in the Jacobian chain shrinks exponentially with sequence length. The gradient of $\mathcal{L}_T$ with respect to $\mathbf{h}_t$ decays as roughly $\rho^{T-t}$. For $T - t = 20$ and $\rho = 0.9$, the gradient has decayed by a factor of $0.9^{20} \approx 0.12$. For $T - t = 100$, by a factor of $0.9^{100} \approx 2.6 \times 10^{-5}$. The signal from distant past tokens reaches the current parameters vanishingly small — the network cannot learn to use long-range context because the gradients that would teach it to do so are effectively zero.

The converse problem is **exploding gradients**: if $\rho > 1$, the product grows exponentially, and gradients become numerically unstable — gradients of $10^{10}$ or larger, causing parameter updates that completely destabilize training. The fix for this problem is straightforward: **gradient clipping**, introduced by Pascanu et al. (2013). If the gradient norm exceeds a threshold $\tau$, rescale the gradient vector to have norm $\tau$:

$$\hat{\mathbf{g}} = \begin{cases} \mathbf{g} & \text{if } \|\mathbf{g}\| \leq \tau \\ \tau \cdot \mathbf{g} / \|\mathbf{g}\| & \text{otherwise.} \end{cases}$$

Gradient clipping is a hack — it does not fix the instability, it just prevents catastrophic divergence. And it has no analog for vanishing gradients: you cannot amplify a signal that is genuinely close to zero.

The conclusion of the 1994 analysis was pessimistic: gradient descent on standard RNNs fundamentally cannot learn dependencies spanning more than roughly 10–15 time steps. For language, where coreference, topic coherence, and syntactic agreement can span dozens or hundreds of words, this is a serious limitation. The model that solved it had already been invented, three years earlier, by the same Hochreiter who first identified the problem.

## Long Short-Term Memory

In 1997, Sepp Hochreiter and Jürgen Schmidhuber published "Long Short-Term Memory" — a paper that introduced an architecture specifically designed to preserve gradient signal across long sequences. The core idea was to replace the single hidden state with two separate memory systems: a **cell state** $\mathbf{c}_t$ that flows through time with minimal transformation, and a **hidden state** $\mathbf{h}_t$ that is read out and updated via learned gates.

An LSTM cell computes four quantities at each time step, given the previous hidden state $\mathbf{h}_{t-1}$ and current input $\mathbf{x}_t$:

$$
\begin{aligned}
\mathbf{i}_t &= \sigma(W_i[\mathbf{h}_{t-1};\, \mathbf{x}_t] + \mathbf{b}_i) && \text{(input gate)} \\
\mathbf{f}_t &= \sigma(W_f[\mathbf{h}_{t-1};\, \mathbf{x}_t] + \mathbf{b}_f) && \text{(forget gate)} \\
\mathbf{o}_t &= \sigma(W_o[\mathbf{h}_{t-1};\, \mathbf{x}_t] + \mathbf{b}_o) && \text{(output gate)} \\
\mathbf{g}_t &= \tanh(W_g[\mathbf{h}_{t-1};\, \mathbf{x}_t] + \mathbf{b}_g) && \text{(cell update)}
\end{aligned}
$$

where $\sigma$ is the sigmoid function and $[\mathbf{h}_{t-1}; \mathbf{x}_t]$ denotes concatenation. The cell state and hidden state are then updated:

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t,$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t),$$

where $\odot$ denotes elementwise multiplication.

The mechanism deserves careful reading. The **forget gate** $\mathbf{f}_t \in (0, 1)^d$ controls how much of the previous cell state is retained: a value near 1 in dimension $k$ means "remember this," a value near 0 means "forget this." The **input gate** $\mathbf{i}_t$ controls how much of the newly computed update $\mathbf{g}_t$ is written into the cell. The **output gate** $\mathbf{o}_t$ controls how much of the cell state is exposed as the hidden state $\mathbf{h}_t$.

Crucially, the cell state update $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t$ is a *linear* operation on $\mathbf{c}_{t-1}$ — multiplication by a diagonal matrix $\mathrm{diag}(\mathbf{f}_t)$. If the forget gate is near 1, the gradient flows back through the cell state with almost no attenuation: this is the **constant error carousel** that Hochreiter had envisioned in 1991. The vanishing gradient problem is not solved — it is circumvented, by providing a highway for gradients that avoids the squashing nonlinearities that caused the problem in the first place.

The forget gate itself was not in Hochreiter and Schmidhuber's original 1997 paper; it was added by Gers, Schmidhuber, and Cummins in 2000, and turned out to be crucial for performance. Without it, the cell state could only accumulate, never selectively erase.

## The Gated Recurrent Unit

In 2014, Cho et al. proposed the **Gated Recurrent Unit** (GRU) as a simpler alternative to the LSTM. The GRU merges the cell state and hidden state into a single vector and uses only two gates:

$$\mathbf{z}_t = \sigma(W_z[\mathbf{h}_{t-1};\, \mathbf{x}_t]) \quad \text{(update gate)},$$
$$\mathbf{r}_t = \sigma(W_r[\mathbf{h}_{t-1};\, \mathbf{x}_t]) \quad \text{(reset gate)},$$
$$\tilde{\mathbf{h}}_t = \tanh(W[\mathbf{r}_t \odot \mathbf{h}_{t-1};\, \mathbf{x}_t]),$$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t.$$

The update gate $\mathbf{z}_t$ interpolates between retaining the old hidden state and adopting the new candidate $\tilde{\mathbf{h}}_t$: when $\mathbf{z}_t \approx 0$, the model effectively skips the current input and propagates the old state; when $\mathbf{z}_t \approx 1$, it performs a full update. The reset gate $\mathbf{r}_t$ controls how much of the old hidden state influences the candidate update — low values allow the model to "reset" and compute the candidate independently of the past.

The GRU has fewer parameters than the LSTM and trains faster. On most benchmarks they perform comparably. The choice between them is largely empirical and task-dependent. Both appeared in essentially every NLP system of the 2014–2017 period.

## Language Modeling with LSTMs

The paper that established LSTM language models as a serious competitor to n-grams was Zaremba, Sutskever, and Vinyals (2014): "Recurrent Neural Network Regularization." The key contribution was systematic application of **dropout** to LSTM language models.

Dropout (Srivastava et al., 2014) randomly sets a fraction $p$ of activations to zero during training, forcing the network to develop redundant representations. In convolutional networks and feedforward networks it was already standard practice. Applying it to RNNs was non-trivial: naive dropout on the recurrent connections destroyed the gradient flow through the cell state. Zaremba et al. showed that applying dropout only to the non-recurrent connections (the input-to-hidden and hidden-to-output pathways) while leaving the recurrent connections intact preserved both regularization benefits and gradient flow.

The result was dramatic. On Penn Treebank — the standard language modeling benchmark of the era — a 2-layer LSTM with dropout achieved a test perplexity of 78.4, compared to Kneser-Ney's perplexity around 141. The gap was not close. Neural language models had definitively overtaken n-gram models on the task that n-gram models had been optimized for over fifty years.

Subsequent work stacked more layers, used larger hidden dimensions, and refined the regularization. A 3-layer LSTM with variational dropout (Gal and Ghahramani, 2016) reached perplexity of 65.8. AWD-LSTM (Merity et al., 2017), with weight tying and additional regularization techniques, pushed this to 57.3. Each improvement required careful engineering, but the trend was clear: more capacity and better regularization kept driving perplexity down.

## Character-Level Language Models

An important parallel development was the application of LSTMs to **character-level language modeling**: predicting the next character rather than the next word. Graves (2013) demonstrated that character-level RNNs could generate surprisingly coherent text. Karpathy, Johnson, and Fei-Fei (2015) popularized the idea with a now-famous blog post demonstrating models trained on Shakespeare, Linux source code, and LaTeX papers.

Character-level models have several advantages. They handle the out-of-vocabulary problem trivially — every text, no matter how unusual its vocabulary, can be represented as a sequence of characters from a fixed small alphabet. They can learn morphology, spelling patterns, and even syntactic structure directly from character sequences, without any tokenization assumptions. A character-level LSTM trained on Linux source code learns to open and close brackets, maintain consistent indentation, and produce syntactically valid C without any explicit grammar.

The disadvantage is sequence length. A word-level model processes one token per word; a character-level model processes four to six tokens per word on average, making sequences five times longer and making long-range dependencies proportionally harder to maintain. This tradeoff — richer representation at the cost of longer sequences — anticipated a fundamental design tension that would recur in subsequent architectures.

## Bidirectional RNNs and Deep Architectures

A language model conditions only on the left context — it predicts $w_{t+1}$ given $w_1, \ldots, w_t$. But many NLP tasks require understanding a word in its full context, both what came before and what comes after. **Bidirectional RNNs** (Schuster and Paliwal, 1997) process the sequence in both directions, maintaining a forward hidden state $\overrightarrow{\mathbf{h}}_t$ and a backward hidden state $\overleftarrow{\mathbf{h}}_t$, and concatenate them to form the contextual representation of position $t$:

$$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t;\; \overleftarrow{\mathbf{h}}_t].$$

This is possible for encoder models (which read an entire sequence before making predictions) but not for autoregressive language models (which must predict each token before seeing the next). Bidirectionality became standard for sequence-to-sequence encoders and classification models.

**Deep RNNs** stack multiple LSTM layers, feeding the hidden state sequence of one layer as the input sequence to the next:

$$\mathbf{h}_t^{(\ell)} = \mathrm{LSTM}\!\left(\mathbf{h}_t^{(\ell-1)},\, \mathbf{h}_{t-1}^{(\ell)}\right).$$

Deeper models consistently outperformed shallower ones on language modeling benchmarks. Zaremba et al. used 2 layers; most subsequent models used 3–4. The intuition is that different layers learn representations at different levels of abstraction — low layers capture lexical and surface patterns, higher layers capture more syntactic and semantic structure.

## ELMo: The First Contextual Representations

Word embeddings, as Chapter 2 established, are static: each word has one vector regardless of context. The LSTM era produced a solution to this. Peters et al. (2018) introduced **ELMo** (Embeddings from Language Models): representations extracted from the internal states of a deep bidirectional LSTM language model.

ELMo is trained as a standard bidirectional language model on a large corpus, maximizing the combined log-likelihood of the forward and backward directions:

$$\mathcal{L} = \sum_{t=1}^{T} \left[\log P(w_t \mid w_1, \ldots, w_{t-1}) + \log P(w_t \mid w_{t+1}, \ldots, w_T)\right].$$

After training, the representation of word $w_t$ in a new task is not a fixed embedding but a weighted combination of the hidden states from all layers of the pretrained model:

$$\mathbf{e}_t^{\mathrm{ELMo}} = \gamma \sum_{k=0}^{L} s_k\, \mathbf{h}_t^{(k)},$$

where $L$ is the number of layers, $s_k$ are task-specific softmax-normalized weights, $\gamma$ is a task-specific scalar, and $\mathbf{h}_t^{(k)}$ is the hidden state at layer $k$ and position $t$. The weights $s_k$ and $\gamma$ are learned for each downstream task while the LSTM parameters are frozen.

The results were transformative. On six NLP benchmarks — question answering, textual entailment, sentiment analysis, named entity recognition, coreference resolution, semantic role labeling — ELMo improved the state of the art by an average of 3–5 absolute points. For the first time, a single pretrained model provided contextual representations that benefited every task in the suite.

The key word is *contextual*. Because the LSTM processes the full sequence, the representation of "bank" in "she withdrew money from the bank" is different from its representation in "they picnicked on the bank of the river." The disambiguation happens not at embedding time but at encoding time, as the LSTM accumulates context. This was exactly what static word vectors could not do, and exactly what downstream tasks needed.

## What RNNs Got Right

The recurrent architecture established a template that still shapes how we think about sequence processing. The fundamental intuition — that processing a sequence should involve a state that evolves over time, accumulating information — is not wrong, just limited in its implementation.

More specifically, RNNs established that **language modeling and representation learning are the same problem**. Training a model to predict the next word forces it to develop internal representations that encode everything useful about the context: syntax, semantics, discourse structure, world knowledge. You do not need a labeled dataset for these representations; the supervision signal is the text itself, and the internet contains an essentially unlimited supply of text. This insight — that self-supervised prediction is a general-purpose representation learning objective — is the foundational principle of every large language model that followed.

RNNs also established the value of **depth in the temporal direction**: stacking layers allowed hierarchical abstraction over time. ELMo demonstrated that different layers of an LSTM capture qualitatively different aspects of linguistic structure — lower layers more morphological and syntactic, higher layers more semantic. This layered interpretation of linguistic representation would carry forward into Transformers.

## The Wall

By 2017, LSTM language models were state-of-the-art on all the standard benchmarks, and ELMo demonstrated their value as general-purpose representation learners. And yet the field was quietly aware of a fundamental problem that better engineering could not fix.

The problem was **sequential dependency**. In a recurrent network, the representation of position $t$ depends on the representation of position $t-1$, which depends on position $t-2$, and so on. This means the entire computation is sequential: you cannot compute $\mathbf{h}_{100}$ until you have computed $\mathbf{h}_{99}$. Training on a sequence of length $T$ requires $T$ sequential steps regardless of how much parallel hardware is available. On modern GPU hardware, where parallelism is everything, this is a severe bottleneck.

The consequences were practical. LSTM training on large corpora was slow — far slower than the same number of parameters in a convolutional or fully-connected architecture, because the computation could not be parallelized across the sequence length dimension. This limited the scale of models that could feasibly be trained. And scale, as the coming years would reveal, mattered more than almost anything else.

The second problem was **information bottleneck**. All information about the left context must pass through the fixed-dimensional vector $\mathbf{h}_t$. For a short sequence, this is fine. For a long document, the hidden state at position 500 must somehow encode everything relevant from positions 1 through 499. In practice, LSTMs forget. They perform well on short and medium-range dependencies but degrade on very long-range ones. The forget gate helps, but it is a learned heuristic, not a principled solution.

What was needed was a mechanism that could, in a single operation, connect any two positions in a sequence — regardless of how far apart they were — without routing information through every intermediate position. That mechanism was attention, and it was already known. The question was whether it could replace the recurrent architecture entirely.

The answer, delivered in June 2017 in a paper titled "Attention Is All You Need," was yes.

---

*Key references: Elman (1990) on simple RNNs, Hochreiter (1991) and Bengio et al. (1994) on the vanishing gradient problem, Hochreiter and Schmidhuber (1997) on LSTMs, Gers et al. (2000) on the forget gate, Cho et al. (2014) on GRUs, Zaremba et al. (2014) on LSTM regularization, Graves (2013) and Karpathy et al. (2015) on character-level RNNs, Pascanu et al. (2013) on gradient clipping, Peters et al. (2018) on ELMo.*
