---

title: "Distributed Representations"
subtitle: "From co-occurrences to learned embeddings — Word2Vec, GloVe, and the geometry of meaning"
---


The failure of n-gram models was not a failure of statistics. It was a failure of representation. A word in a Kneser-Ney model is an index — an arbitrary integer with no relationship to any other integer. "Cat" and "feline" are as unrelated as "cat" and "democracy." This makes statistical sharing across similar words impossible: every observation about "cat" is quarantined from "feline," and vice versa. The curse of dimensionality is not just a computational fact about the size of the space; it is a consequence of representing words as isolated atoms rather than as points in a structured geometry.

The solution had been gestured at since the 1950s. It would take fifty years to implement it in a form that actually worked.

## The Distributional Hypothesis

In 1957, the British linguist J.R. Firth wrote: *"You shall know a word by the company it keeps."* The idea, now called the **distributional hypothesis**, is that words with similar meanings tend to appear in similar contexts. "Cat" and "dog" both follow "the," both precede "barked," both appear near "pet" and "food." Their distributional profiles — the pattern of contexts they occur in — are similar, and that similarity tracks semantic similarity.

Zellig Harris had formalized a related idea in 1954: linguistic environments determine meaning. What distinguishes morphological variants of the same word, synonyms, or antonyms is precisely the distribution of contexts in which they appear. If this is true, then the statistical structure of a large corpus contains, implicitly, a geometry of meaning — and that geometry can, in principle, be extracted.

The question was how to extract it. The answer required thinking about words not as atoms but as vectors.

## Co-occurrence Matrices and LSA

The most direct implementation of the distributional hypothesis is the **co-occurrence matrix**. Fix a vocabulary of $|\mathcal{V}|$ words and a context window of size $k$. For every pair of words $(i, j)$, let $X_{ij}$ count how many times word $j$ appears within $k$ words of word $i$ in the corpus. The result is a matrix $X \in \mathbb{R}^{|\mathcal{V}| \times |\mathcal{V}|}$.

Each row $X_{i\cdot}$ is a vector representation of word $i$ — its distributional profile. Words with similar rows are distributionally similar, and by the distributional hypothesis, semantically related. The problem is scale: for $|\mathcal{V}| = 50{,}000$, the matrix has $2.5 \times 10^9$ entries, most of them zero, with no obvious way to compute similarity efficiently.

**Latent Semantic Analysis** (Deerwester et al., 1990) addresses this with truncated singular value decomposition. Decompose

$$X \approx U \Sigma V^\top,$$

where $U \in \mathbb{R}^{|\mathcal{V}| \times d}$ and $\Sigma \in \mathbb{R}^{d \times d}$ contain the top $d$ singular vectors and values. The rows of $U\Sigma^{1/2}$ serve as low-dimensional word vectors: dense, $d$-dimensional (typically $d \in [50, 300]$), and capturing the dominant axes of co-occurrence variation.

LSA worked remarkably well for its era. Synonym detection, document retrieval, latent topic modeling — on all of these it substantially outperformed raw co-occurrence. But it had a critical limitation: the SVD is computed on a raw count matrix, with no principled choice of what counts should represent. High-frequency words ("the," "of," "and") dominate the co-occurrence statistics and drown out the signal from content words. Numerous heuristics — log-transforming counts, using positive pointwise mutual information (PPMI), subsampling frequent words — could partially compensate, but the method had no learning dynamics to adapt to.

## The Neural Probabilistic Language Model

The decisive conceptual break came in 2003. Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin published "A Neural Probabilistic Language Model" — a paper that introduced two ideas that would define the next two decades of NLP.

The first idea was **word embeddings**: every word $w$ in the vocabulary is assigned a continuous vector $\mathbf{e}_w \in \mathbb{R}^d$, where $d$ is a hyperparameter (they used $d = 30$ to $100$). These vectors are parameters of the model, learned end-to-end from data.

The second idea was to condition the language model on the concatenation of these vectors. Given a trigram context $(w_{t-2}, w_{t-1})$, predict the distribution over the next word $w_t$ using

$$P(w_t \mid w_{t-2}, w_{t-1}) = \mathrm{softmax}(W \cdot \tanh(U \cdot [\mathbf{e}_{w_{t-2}}; \mathbf{e}_{w_{t-1}}] + \mathbf{b}_1) + \mathbf{b}_2),$$

where $W$, $U$, $\mathbf{b}_1$, $\mathbf{b}_2$ are learned weight matrices and biases, and $[\cdot ; \cdot]$ denotes concatenation. The entire model — embeddings and network weights — is trained jointly by gradient descent to minimize cross-entropy loss on the training corpus.

The key insight is what this model can do that a lookup table cannot: **generalize across similar words**. If "cat" and "dog" have learned similar embedding vectors (because they appear in similar contexts), then a sequence involving "dog" will produce almost the same hidden representation as the corresponding sequence involving "cat," and the model's output distribution will be nearly the same. One training example simultaneously updates the predictions for all words with similar embeddings. The curse of dimensionality is not eliminated, but it is dramatically softened.

Bengio et al. demonstrated competitive perplexity on a small text corpus. The model was slower to train than Kneser-Ney by an order of magnitude, and for a decade that computational cost kept it from displacing n-gram models in production. But the idea was planted.

## The Softmax Bottleneck

The model in Bengio et al. required computing the softmax over the entire vocabulary at every training step. With $|\mathcal{V}| = 50{,}000$ words, this means computing $50{,}000$ dot products per token per forward pass, and $50{,}000$ gradient contributions per backward pass. On 2003 hardware, this was prohibitive. On a billion-word corpus, it remained prohibitive until specialized hardware and engineering techniques intervened.

Two approaches emerged. **Hierarchical softmax** (Morin and Bengio, 2005) organizes the vocabulary into a binary tree, reducing the cost from $O(|\mathcal{V}|)$ to $O(\log |\mathcal{V}|)$ per step. **Noise-contrastive estimation** (Gutmann and Hyvärinen, 2010) reframes the problem as binary classification: instead of predicting the correct word among all $|\mathcal{V}|$ options, the model learns to distinguish the true next word from a small number of randomly sampled "noise" words.

Both techniques sacrifice exact maximum likelihood training in exchange for computational tractability. They proved decisive when, in 2013, Tomáš Mikolov and colleagues at Google published Word2Vec — a paper that discarded everything Bengio's model did *except* the embedding layer, and in doing so created the word vectors that would power NLP for the next five years.

## Word2Vec: The Radical Simplification

Mikolov et al.'s insight was that the hidden layer of the neural language model was not contributing much. The embedding layer was doing all the work. What if you removed the nonlinearity entirely?

Word2Vec trains embeddings using two architectures, both stripped to the minimum:

**CBOW (Continuous Bag of Words)**: given the surrounding context words $w_{t-k}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+k}$, predict the center word $w_t$. The model averages the context embeddings and predicts through a linear softmax.

**Skip-gram**: given the center word $w_t$, predict each surrounding context word $w_{t+j}$ for $j \in [-k, k] \setminus \{0\}$. The objective is to maximize

$$\mathcal{J} = \frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-k \leq j \leq k \\ j \neq 0}} \log P(w_{t+j} \mid w_t).$$

Each conditional probability is modeled with a separate output embedding $\tilde{\mathbf{e}}_{w}$ ("context embedding") distinct from the input embedding $\mathbf{e}_w$:

$$P(w_{t+j} \mid w_t) = \frac{\exp(\tilde{\mathbf{e}}_{w_{t+j}} \cdot \mathbf{e}_{w_t})}{\sum_{w \in \mathcal{V}} \exp(\tilde{\mathbf{e}}_{w} \cdot \mathbf{e}_{w_t})}.$$

The softmax denominator is still intractable at scale. Mikolov et al. trained it with **negative sampling**: instead of the true softmax, optimize a binary objective that pushes the dot product $\tilde{\mathbf{e}}_{w_c} \cdot \mathbf{e}_{w_t}$ up for true context pairs $(w_t, w_c)$ and down for $k$ randomly sampled noise pairs:

$$\mathcal{J}_{\mathrm{NS}} = \log \sigma(\tilde{\mathbf{e}}_{w_c} \cdot \mathbf{e}_{w_t}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n}\!\left[\log \sigma(-\tilde{\mathbf{e}}_{w_i} \cdot \mathbf{e}_{w_t})\right],$$

where $\sigma$ is the sigmoid function and $P_n(w) \propto f(w)^{3/4}$ is a smoothed unigram noise distribution. With $k = 5$ to $20$ noise samples per true pair, training on billions of words became practical on a single machine.

The result was embeddings that were qualitatively different from anything produced before. "Paris" was close to "London" and "Berlin." "King" was close to "queen" and "emperor." "Running" was close to "walking" and "jogging." For the first time, a model had recovered, from raw text statistics, something that looked like a representation of conceptual structure.

## The Geometry of Meaning

The finding that electrified the NLP community was not the embeddings themselves but their **linear structure**. Mikolov et al. discovered that vector arithmetic in embedding space corresponded to semantic relationships:

$$\mathbf{e}_{\text{king}} - \mathbf{e}_{\text{man}} + \mathbf{e}_{\text{woman}} \approx \mathbf{e}_{\text{queen}}.$$

This analogy structure — king is to man as queen is to woman — was not explicitly trained. It emerged from the geometry of the embedding space. Similarly:

$$\mathbf{e}_{\text{Paris}} - \mathbf{e}_{\text{France}} + \mathbf{e}_{\text{Germany}} \approx \mathbf{e}_{\text{Berlin}},$$
$$\mathbf{e}_{\text{walked}} - \mathbf{e}_{\text{walk}} + \mathbf{e}_{\text{run}} \approx \mathbf{e}_{\text{ran}}.$$

Relational knowledge — the fact that capitals stand in a consistent relationship to their countries, that past-tense forms stand in a consistent relationship to present-tense forms — was encoded as a consistent *direction* in embedding space. The gender direction, the capital-of direction, the tense direction: each was a vector that could be extracted and applied to new words.

This was, in retrospect, the first evidence that neural networks trained on raw text could recover structured knowledge without explicit supervision. The implications would take years to fully absorb.

## GloVe: Count-Based Meets Prediction-Based

The success of Word2Vec reinvigorated interest in count-based methods. Levy and Goldberg (2014) showed, through careful analysis, that the skip-gram with negative sampling objective was implicitly factorizing the shifted positive pointwise mutual information matrix:

$$\mathrm{PPMI}(w, c) = \max\!\left(0,\; \log \frac{P(w, c)}{P(w)\, P(c)} - \log k\right),$$

where $k$ is the number of negative samples. This meant Word2Vec and LSA were not as different as they appeared: both were recovering a low-rank factorization of a co-occurrence-derived matrix, just via different objectives and optimization procedures.

Jeffrey Pennington, Richard Socher, and Christopher Manning at Stanford pursued this insight to its logical conclusion with **GloVe** (Global Vectors for Word Representation, 2014). Rather than training on a sliding window over the corpus as in Word2Vec, GloVe first constructs the full co-occurrence matrix $X$ (where $X_{ij}$ counts how often word $j$ appears in the context of word $i$), then directly optimizes a factorization objective:

$$\mathcal{J} = \sum_{i,j=1}^{|\mathcal{V}|} f(X_{ij})\left(\mathbf{e}_i \cdot \tilde{\mathbf{e}}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2,$$

where $f : \mathbb{R}_{\geq 0} \to [0, 1]$ is a weighting function that downweights very frequent co-occurrences (those pairs so common they carry little discriminative information):

$$f(x) = \begin{cases} (x / x_{\max})^\alpha & \text{if } x < x_{\max} \\ 1 & \text{otherwise,} \end{cases}$$

with $x_{\max} = 100$ and $\alpha = 3/4$ in the original paper. The objective says: the dot product between the embedding of word $i$ and the context embedding of word $j$, plus bias terms, should equal the log co-occurrence count. The weighting function ensures that rare co-occurrences and ubiquitous ones are both deemphasized relative to moderately frequent pairs.

GloVe embeddings matched or exceeded Word2Vec on most benchmarks with faster and more stable training. The reason is partly statistical: by working with global co-occurrence counts rather than a sliding window, GloVe makes efficient use of all the data without the stochasticity of negative sampling. The method made explicit what Word2Vec had left implicit: that language modeling through embeddings is, fundamentally, a matrix factorization problem dressed up as sequence prediction.

## Similarities, Analogies, and What the Benchmarks Measured

The period 2013–2015 saw an explosion of evaluation benchmarks for word vectors. The word analogy task (Mikolov et al., 2013b) — "Athens is to Greece as Oslo is to __?" — became the standard test, with the Google Analogy Dataset containing 19,544 such questions spanning geographic, relational, and syntactic categories. Similarity datasets like WordSim-353 and SimLex-999 measured whether the cosine similarity between word vectors correlated with human judgments of semantic relatedness.

Word2Vec and GloVe scored well on all of these. But the benchmarks measured a very particular kind of language competence: static, decontextual, world-knowledge-flavored semantic similarity. They could not measure whether a model understood that "bank" meant a financial institution in one sentence and a riverbank in another, because word vectors are **static** — each word has exactly one vector, regardless of context. "Bank" in "she went to the bank to deposit money" and "bank" in "he fished on the bank of the river" are represented by the same point in the same space.

This was not a bug that could be engineered away. It was a fundamental limitation of the representation. A static vector averages over all the contexts in which a word appears; for polysemous words, this average is a blurred superposition that does not correspond to any of the actual senses.

## Subword Representations and the Out-of-Vocabulary Problem

A related limitation was the handling of rare and unseen words. A vocabulary of 50,000 words leaves out most proper nouns, technical terms, and morphological variants. A word not in the vocabulary — an out-of-vocabulary (OOV) word — must be mapped to a single generic "unknown" token, losing all information about its form.

The solution developed in this era was **subword tokenization**. Rather than representing words as indivisible atoms, decompose them into smaller units — character n-grams, morphemes, or byte-pair encoding (BPE) segments — that can compose to represent any surface form.

Bojanowski et al.'s fastText (2017) extended Word2Vec by representing each word as the sum of its character n-gram embeddings:

$$\mathbf{e}_w = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g,$$

where $\mathcal{G}_w$ is the set of character n-grams in word $w$ (including the word itself, framed with special boundary markers), and $\mathbf{z}_g \in \mathbb{R}^d$ is the embedding of n-gram $g$. OOV words can be embedded by summing their n-gram embeddings, even if the word itself was never seen. This gave substantial gains on morphologically rich languages (Finnish, Turkish, Arabic) where word forms are highly variable and word-level vocabularies are enormous.

Byte-pair encoding (Sennrich et al., 2016), originally developed for machine translation, took a different approach: iteratively merge the most frequent adjacent subword units until a vocabulary of a target size is reached. The result is a vocabulary where common words appear as single tokens, common prefixes and suffixes appear as shared subword units, and rare words are decomposed into constituent parts. BPE became the dominant tokenization strategy for large language models and remains so today.

## The Limitations of Static Embeddings

By 2015, the word embedding era had matured. The representations were genuinely useful: downstream NLP tasks — named entity recognition, sentiment analysis, parsing, machine translation — all improved substantially when initialized with pretrained Word2Vec or GloVe vectors. The embeddings were a kind of compressed world knowledge, a prior that allowed models to start with semantic structure rather than learn it from scratch.

But several problems remained that no amount of engineering could fix within the static embedding paradigm.

**Polysemy** was the most fundamental. Every word with multiple senses had a single vector — a blend. "Java" (the island, the coffee, the programming language) had one point in space. ELMo and BERT would later solve this by making representations **contextual**: the vector for a word would depend on the sentence it appeared in.

**Syntactic and long-range structure** was absent. Word vectors captured distributional co-occurrence but not grammatical relationships, logical entailment, or coreference. "The cat ate the mouse" and "The mouse ate the cat" had the same bag-of-word-embeddings representation. Sequential and structural information required a model that processed sequences, not just bags of tokens.

**Compositionality** was approximate at best. The additive analogy arithmetic was suggestive but limited. The meaning of a phrase is not simply the sum of its word vectors — context modifies meaning, syntax constrains it, and pragmatics shapes it in ways that a sum of static vectors cannot capture.

These were not criticisms of word embeddings per se. They were statements about what kind of thing a language model needed to be: not a lookup table, not a static vector space, but a function that maps sequences to sequences, with context-sensitive representations at every step. That function required memory — a model that could read from and write to an evolving representation of what had been said so far.

## The Bridge

The word embedding era is, in retrospect, a transitional period. It solved the representation problem that n-gram models could not even formulate: how to give words positions in a continuous space where distance tracks meaning. But it did not solve the modeling problem: how to use those representations to make context-sensitive predictions over arbitrarily long sequences.

The models that would solve that problem — recurrent neural networks, LSTMs, and eventually the Transformer — were already being developed in parallel, largely by the same researchers. The key technical insight that connects them to word embeddings is that the embedding layer is simply the input to a deeper model. Once you have a continuous representation of each word, you can process sequences of those representations with any architecture that can read sequences.

The counting era gave language modeling its objective — cross-entropy minimization — and its evaluation metric — perplexity. The embedding era gave it a representation capable of generalization. What came next was a model capable of memory.

---

*Key references: Firth (1957), Harris (1954), Deerwester et al. (1990) on LSA, Bengio et al. (2003) on the neural probabilistic language model, Mikolov et al. (2013a) on Word2Vec, Mikolov et al. (2013b) on word analogies, Pennington et al. (2014) on GloVe, Levy and Goldberg (2014) on the relationship between Word2Vec and matrix factorization, Bojanowski et al. (2017) on fastText, Sennrich et al. (2016) on byte-pair encoding.*
