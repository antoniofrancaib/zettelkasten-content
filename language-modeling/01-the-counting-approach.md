---

title: "The Counting Approach"
subtitle: "Shannon entropy, n-gram models, perplexity, and Kneser-Ney smoothing"
---


Language arrived at the machine not through understanding but through counting. Before anyone asked what a word *meant*, they asked how often it appeared next to another word — and whether that frequency, carefully normalized, could stand in for meaning. It could not, exactly. But it was enough to build on, and what was built on it turned out to be more durable than its critics expected. The entire modern edifice of language modeling — the Transformer, the large language model, the in-context learner — rests on a conceptual foundation laid in the era of n-grams, perplexity, and Kneser-Ney smoothing. To understand where we are, it helps to understand what the first workers were actually trying to solve.

## Shannon and the Mathematical Theory of Communication

In 1948, Claude Shannon published "A Mathematical Theory of Communication" in the Bell System Technical Journal. The paper is not primarily about language; it is about the general problem of transmitting information reliably over a noisy channel. But to make the problem precise, Shannon needed a model of the source producing the messages — and for English text, that meant modeling the statistical structure of language.

Shannon introduced two ideas that would dominate the field for decades. The first was **entropy**: a measure of the average uncertainty in a random variable. For a discrete source producing symbols from a vocabulary $\mathcal{V}$ with probability distribution $p$, the entropy is

$$H(p) = -\sum_{w \in \mathcal{V}} p(w) \log_2 p(w).$$

Entropy is measured in bits. A source that always produces the same symbol has zero entropy. A source that produces symbols uniformly at random over a vocabulary of size $|\mathcal{V}|$ has entropy $\log_2 |\mathcal{V}|$. English text falls somewhere in between — structured, but not deterministic.

The second idea was that a message source could be modeled as a **stochastic process**: a sequence of random variables $W_1, W_2, W_3, \ldots$ where each $W_i$ takes values in the vocabulary. The question of language modeling is then the question of specifying the joint distribution over such sequences. Shannon showed that for a stationary ergodic process, the entropy rate

$$H = \lim_{n \to \infty} \frac{1}{n} H(W_1, W_2, \ldots, W_n)$$

is well-defined and can, in principle, be estimated from data. He even performed a famous experiment: he had human subjects guess the next letter in a passage of English text, then used their success rate to estimate the entropy of English at roughly 1.0–1.5 bits per character.

Shannon's framing was profound because it separated two questions that had previously been conflated: *what does language mean*, and *what is the statistical structure of language*. The second question could be answered without the first, and answering it turned out to be extremely useful.

## The Chain Rule and the Fundamental Problem

Any probability distribution over sequences can be factored exactly using the **chain rule of probability**. For a sequence of words $w_1, w_2, \ldots, w_n$, we have

$$P(w_1, w_2, \ldots, w_n) = \prod_{k=1}^{n} P(w_k \mid w_1, w_2, \ldots, w_{k-1}).$$

This is an identity — it requires no assumptions about independence or Markov structure. The entire modeling problem reduces to specifying the conditional distributions $P(w_k \mid w_1, \ldots, w_{k-1})$: the probability of the next word given everything that came before it. If we could specify these distributions perfectly, we would have a perfect model of language.

The difficulty is that each of these conditionals is a function of an arbitrarily long context $w_1, \ldots, w_{k-1}$. In practice, if the vocabulary has size $|\mathcal{V}|$, then the number of distinct contexts of length $k-1$ is $|\mathcal{V}|^{k-1}$ — exponential in history length. For English, with $|\mathcal{V}| \approx 50{,}000$ words, even a three-word context admits $50{,}000^3 \approx 1.25 \times 10^{14}$ possible histories. No corpus is large enough to estimate all of these.

The counting approach resolves this with a drastic but practical simplification.

## The Markov Assumption and N-gram Models

The **Markov assumption** truncates the conditioning history. An n-gram model asserts that the next word depends only on the preceding $n-1$ words:

$$P(w_k \mid w_1, \ldots, w_{k-1}) \approx P(w_k \mid w_{k-n+1}, \ldots, w_{k-1}).$$

The most common choices are:

- **Unigram** ($n=1$): words are independent, $P(w_k)$
- **Bigram** ($n=2$): one word of context, $P(w_k \mid w_{k-1})$
- **Trigram** ($n=3$): two words of context, $P(w_k \mid w_{k-2}, w_{k-1})$

Under this assumption, the full joint probability becomes

$$P(w_1, \ldots, w_n) \approx \prod_{k=1}^{n} P(w_k \mid w_{k-n+1}, \ldots, w_{k-1}).$$

The Markov assumption is obviously wrong — the beginning of a sentence does constrain words near its end, context from previous sentences matters, and meaning is not local. But it transforms an intractable estimation problem into a tractable one, and with large enough data, trigram models produce surprisingly fluent text.

## Maximum Likelihood Estimation

Given the Markov assumption, the parameters of an n-gram model are just conditional probabilities over short contexts. These are estimated from a training corpus by **maximum likelihood estimation** (MLE): count how often the n-gram appears, divide by how often the context appears.

For bigrams:

$$\hat{P}(w_i \mid w_{i-1}) = \frac{C(w_{i-1},\, w_i)}{C(w_{i-1})},$$

where $C(w_{i-1}, w_i)$ is the count of the bigram $w_{i-1} w_i$ in the training corpus and $C(w_{i-1})$ is the count of the unigram $w_{i-1}$. For trigrams:

$$\hat{P}(w_i \mid w_{i-2}, w_{i-1}) = \frac{C(w_{i-2},\, w_{i-1},\, w_i)}{C(w_{i-2},\, w_{i-1})}.$$

The MLE is simple, interpretable, and consistent: as the corpus grows to infinity, it converges to the true conditional distribution under the Markov model. The problem is what happens at finite corpus sizes, which is to say: always.

## The Curse of Sparsity

Consider trying to estimate trigram probabilities for English. The vocabulary has perhaps 50,000 common words. The number of distinct trigrams is $50{,}000^3 \approx 1.25 \times 10^{14}$. Even the largest corpora assembled in the 1990s — tens of millions of words — contain only a tiny fraction of these. Most trigrams have never been observed. Their MLE probability is zero.

A model that assigns zero probability to unseen trigrams assigns zero probability to any sentence containing one. But almost every sentence a user might type contains at least one unseen trigram. The model is useless.

This is not a failure of the Markov assumption; it is a manifestation of a universal principle. The space of possible natural-language sequences is vastly larger than any finite corpus. Data sparsity is the defining problem of statistical language modeling, and every technique in this chapter is, fundamentally, a response to it.

## Perplexity: A Unified Metric

Before turning to solutions, it is worth establishing how to measure the problem. The standard evaluation metric for a language model is **perplexity**, defined as the reciprocal of the geometric mean per-word probability assigned to a held-out test corpus:

$$\mathrm{PP}(W) = P(w_1, w_2, \ldots, w_N)^{-1/N} = \left( \prod_{i=1}^{N} \frac{1}{P(w_i \mid w_1, \ldots, w_{i-1})} \right)^{1/N}.$$

Perplexity can be interpreted as the *effective branching factor* of the language model: how many equally likely next words would produce the same level of uncertainty. A perplexity of 100 means the model is, on average, as uncertain as if it had to choose uniformly among 100 options at each step.

Lower perplexity is better. A model that has memorized the test set exactly has perplexity 1. A model that assigns uniform probability over a 50,000-word vocabulary has perplexity 50,000. English text, under a good trigram model, has perplexity in the range of 50–150.

Perplexity relates directly to cross-entropy. If $H(p, q) = -\frac{1}{N} \sum_{i} \log_2 q(w_i \mid w_{i-1})$ is the cross-entropy of the model distribution $q$ against the empirical test distribution, then

$$\mathrm{PP}(W) = 2^{H(p,\, q)}.$$

This connection makes perplexity a proper information-theoretic quantity: reducing perplexity by a factor of two is equivalent to reducing cross-entropy by one bit per word. The goal of language modeling is to reduce this number as far as possible, ideally approaching the true entropy of English.

## Smoothing: Making the Zero-Probability Problem Go Away

The collective name for techniques that address data sparsity is **smoothing**: methods that redistribute probability mass from observed n-grams to unobserved ones. The intuition is Bayesian — even if we have never seen a trigram, we are not certain its probability is zero. We should hedge.

**Add-one (Laplace) smoothing** is the simplest approach: pretend you observed every possible bigram at least once by adding 1 to every count before normalizing.

$$P_L(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + |\mathcal{V}|}.$$

This guarantees that no bigram has zero probability. The problem is that it shifts far too much probability mass to unseen events. For a vocabulary of 50,000 words and a context that appeared 100 times, Laplace smoothing adds $50{,}000$ phantom counts — 500 times the real data. High-count n-grams are drastically discounted; low-count n-grams are wildly inflated. In practice, add-one smoothing degrades perplexity by a factor of two or more compared to unsmoothed models that never encounter unseen n-grams at test time.

**Good-Turing smoothing**, developed by Alan Turing and I.J. Good during World War II (in the context of breaking Enigma) and formalized by Good in 1953, offers a principled alternative. The key insight is that the probability of seeing an event for the first time is related to the frequency of events we have seen exactly once. Define $n_c$ as the number of distinct n-grams that appeared exactly $c$ times in the corpus. Good-Turing replaces each count $c$ with a smoothed count

$$c^* = (c+1) \, \frac{n_{c+1}}{n_c}.$$

The total mass redistributed to unseen events is $n_1 / N$ — equal to the fraction of the corpus occupied by singletons, which serves as an estimate of the probability of observing a genuinely novel event.

Good-Turing is elegant and well-founded. But it is unstable: when $n_{c+1}$ is zero or noisy, the estimate breaks down. And it does not use the identity of the words — all bigrams with count 1 are treated identically, regardless of whether they involve common or rare words. The breakthrough came from recognizing that the problem was not just about counts, but about **continuation structure**.

## Kneser-Ney: The Pinnacle of Counting

The most celebrated smoothing algorithm in the n-gram era is **Kneser-Ney smoothing**, introduced by Reinhard Kneser and Hermann Ney in 1995. It remained the dominant language model until neural approaches overtook it around 2011, and even today it is a competitive baseline for tasks where data is limited.

The insight behind Kneser-Ney is subtle. Consider estimating the probability of the word "Francisco" in the context "I flew to San ___." A simple backoff model, unable to estimate the trigram "San Francisco" directly, falls back to the unigram probability of "Francisco" — which is high, because the word is common. But "Francisco" is common precisely because it almost always follows "San." In novel contexts, it is very rare. The word has low **continuation probability**: it appears after very few distinct contexts.

Kneser-Ney estimates the backoff distribution not from raw unigram counts, but from **continuation counts** — the number of distinct left-contexts a word appears in:

$$P_{\mathrm{cont}}(w) = \frac{|\{v : C(v, w) > 0\}|}{|\{(u, v) : C(u, v) > 0\}|}.$$

"Francisco" appears in very few distinct left-contexts (essentially only "San"), so $P_{\mathrm{cont}}(\text{Francisco})$ is low. "the," which follows hundreds of different words, has high continuation probability. This is exactly the right prior for a backoff distribution.

The full Kneser-Ney model, for bigrams, is:

$$P_{\mathrm{KN}}(w_i \mid w_{i-1}) = \frac{\max(C(w_{i-1}, w_i) - d,\; 0)}{C(w_{i-1})} + \lambda(w_{i-1})\, P_{\mathrm{cont}}(w_i),$$

where $d \in (0, 1)$ is a fixed discount applied to all observed bigrams, and $\lambda(w_{i-1})$ is a normalizing weight ensuring the distribution sums to one:

$$\lambda(w_{i-1}) = \frac{d \cdot |\{w : C(w_{i-1}, w) > 0\}|}{C(w_{i-1})}.$$

The logic has three parts. First, discount every observed bigram slightly — take $d$ probability mass away from it. Second, accumulate the total discount for a given context $w_{i-1}$: proportional to the number of distinct words that followed it. Third, distribute the accumulated mass over the continuation distribution. The discount $d$ is typically estimated from held-out data, often around 0.75.

**Interpolated Kneser-Ney** (Chen and Goodman, 1998) extends this to higher-order models by recursively applying the same discounting strategy at each level. For a trigram, the higher-order counts $C(w_{i-2}, w_{i-1}, w_i)$ are discounted, and the backoff weight is spread over the smoothed bigram model, which itself is discounted and backed off to a smoothed unigram model:

$$P_{\mathrm{KN}}(w_i \mid w_{i-2}, w_{i-1}) = \frac{\max(C(w_{i-2}, w_{i-1}, w_i) - d,\; 0)}{C(w_{i-2}, w_{i-1})} + \lambda(w_{i-2}, w_{i-1})\, P_{\mathrm{KN}}(w_i \mid w_{i-1}).$$

The result is a model that naturally uses lower-order statistics to fill in for unseen higher-order contexts, with each level trained on counts that reflect continuation behavior rather than raw frequency. On standard benchmarks like Penn Treebank and the Wall Street Journal corpus, interpolated Kneser-Ney reduced perplexity by 30–40% relative to previous smoothing methods and was not substantially beaten until the neural language models of 2011–2013.

## Scaling and the Limits of Memory

The success of Kneser-Ney prompted an obvious question: what if the n-gram model were applied to a truly enormous corpus? The answer was explored through the 2000s, culminating in work at Google and elsewhere with models trained on hundreds of billions of words — effectively the entire crawlable web.

With enough data, even simple n-gram models become surprisingly powerful. Brants et al. (2007) showed that a 5-gram model trained on 2 trillion tokens outperformed neural models of the time on machine translation. The key was data scale, not algorithmic sophistication. The practical bottleneck was memory: a 5-gram model over a vocabulary of 1 million words requires storing tens of billions of parameters. Compressed representations such as Bloom filters and randomized data structures were developed specifically to make these models fit in RAM.

But the fundamental limitation remained. Every n-gram model is, at its core, a lookup table. It can only express what it has counted. It has no notion of synonymy — "feline" and "cat" are unrelated entries. It has no notion of syntax — "The dog that the cat chased fled" is no more natural than "fled chased cat the dog the that." It cannot generalize from "the red car" to "the blue car" without having seen both. Each unseen context is, from the model's perspective, exactly as unknown as every other unseen context.

The smoothing techniques of this chapter are all attempts to paper over this rigidity. Kneser-Ney is the most sophisticated such attempt — it exploits the distributional structure of the data in a principled way. But it remains a heuristic layered on top of a fundamentally local representation. The question of how to generalize across contexts — how to represent words in a way that makes similar words share statistical strength — was the question that the next era of language modeling would answer.

## What Counting Got Right

It would be a mistake to dismiss the counting approach as merely a prelude. The n-gram paradigm established several things that remain true and important.

The **evaluation framework** it established has proven remarkably durable. Perplexity, as a measure of how well a model predicts held-out data, is still the primary intrinsic metric for language model quality. The cross-entropy connection means that improvements in perplexity translate directly to reductions in bits-per-character, which translate directly to better compression — and better compression is, in a deep sense, the same as better understanding.

The **probabilistic framing** — a language model as a distribution over sequences, trained to maximize likelihood on text — has remained essentially unchanged from Shannon to GPT-4. The architecture, the parameterization, and the scale have all changed. The objective has not. Every modern language model is, at its core, minimizing the same cross-entropy loss that n-gram models were minimizing against trigram counts.

The **chain rule decomposition** is not an approximation but an exact identity, and it remains the foundational structure of all autoregressive language models. When a Transformer generates text one token at a time, conditioning each prediction on all previous tokens, it is exactly implementing the chain rule. The n-gram model was the first to make this structure explicit.

And the **Markov assumption**, though wrong, revealed something true: local context carries most of the information about what comes next. Most of the perplexity reduction from using a trigram instead of a unigram is obtained in the first two steps; extending to 4-grams and 5-grams yields diminishing returns. Language has long-range dependencies, but they are overlaid on a strong local structure. The models that came after would need to capture both.

## The Wall

By the late 2000s, the n-gram paradigm had reached its limits. Larger corpora helped, but with diminishing returns. More sophisticated smoothing helped, but again with diminishing returns. The fundamental architecture — count, discount, backoff — was fully developed.

The problem was not engineering but representation. To a Kneser-Ney model, words are atoms. "King" is not related to "queen"; "Paris" is not related to "London"; "walked" is not related to "running." Every word is a dimension in a sparse, high-dimensional space, and similarity between words is invisible to the counting machinery.

This was the problem that Bengio et al. would identify in their 2003 paper "A Neural Probabilistic Language Model" — the paper that opened the neural era. Their proposal: instead of looking up a count in a table, represent each word as a vector in a low-dimensional space, and let the network learn the representation jointly with the conditional distribution. Words with similar meanings would end up near each other in this space, and a single training example involving "cat" would generalize, through the representation, to "kitten," "feline," and "tabby."

That chapter comes next. But to appreciate what it contributed, one must appreciate what it replaced. The counting approach was not naive. It was the mature, optimized product of fifty years of careful statistical thinking. What the neural approach required was not less cleverness, but a different kind of cleverness — one that worked in continuous space rather than discrete tables, in similarity rather than identity, in learned representations rather than observed frequencies.

The counting era ended not because it was wrong, but because it had exhausted what could be done without learning to represent.

---

*The key references for this chapter are Shannon (1948), Good (1953), Jelinek and Mercer (1980) on linear interpolation, Chen and Goodman (1998) on empirical comparisons of smoothing methods, Kneser and Ney (1995) on modified Kneser-Ney, and Brants et al. (2007) on large-scale n-gram models for machine translation.*
