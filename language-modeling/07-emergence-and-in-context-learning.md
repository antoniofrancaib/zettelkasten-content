---

title: "Emergence and In-Context Learning"
subtitle: "Scaling laws, GPT-3, and capabilities that appear at scale"
---


In early 2020, a group of researchers at OpenAI published a paper with a deceptively modest title: "Scaling Laws for Neural Language Models." The paper was not about a new architecture. It was not about a new training technique. It was about mathematics — about the relationship between three numbers: the size of a model, the size of its training data, and the amount of compute used to train it. Its findings were not what most people expected.

The community had assumed that scaling was subject to diminishing returns, that progress would slow as models grew larger and data requirements ballooned. What the paper showed was that, within the range studied, scaling was not subject to diminishing returns in any meaningful sense. Model performance — measured in bits per character, or perplexity on held-out text — improved as a **power law** in each of the three quantities, with no sign of a ceiling. Double the model size and performance improved by a predictable amount. Double the data and performance improved by a predictable amount. Double the compute and performance improved by a predictable amount. The three scaling axes were, to a first approximation, independent.

More precisely, for a fixed compute budget $C$, there was an optimal allocation between model size $N$ and training tokens $D$. The paper's central finding — the **Chinchilla scaling laws**, which we'll return to — was that most models at the time were undertrained: researchers were training larger models than was optimal for their compute budget, without providing enough data. But the deeper insight was the power law structure itself. Performance did not plateau. It obeyed a formula.

## The Power Laws

For a language model of $N$ parameters trained on $D$ tokens with compute $C \approx 6ND$, the test loss $L$ follows:
$$L(N) \approx \left(\frac{N_0}{N}\right)^{\alpha_N}, \qquad L(D) \approx \left(\frac{D_0}{D}\right)^{\alpha_D}, \qquad L(C) \approx \left(\frac{C_0}{C}\right)^{\alpha_C},$$
where $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$, $\alpha_C \approx 0.050$ are empirical exponents. The constants $N_0$, $D_0$, $C_0$ are fitted to data.

These are small exponents. Doubling compute improves loss by a factor of $2^{0.05} \approx 1.035$, which is a 3.5% reduction in bits-per-character. That sounds modest. But loss is a compressed metric: in language modeling, small improvements in perplexity correspond to large changes in the quality of generated text and downstream task performance. And the power law holds over many orders of magnitude with no sign of breaking.

The implications were stark. If scaling continued to work — and there was no theoretical reason it wouldn't — then larger models trained on more data would continue to improve. The question was no longer whether to scale but how. And because compute was expensive and data was not unlimited, the question of optimal allocation became urgent.

The 2022 **Chinchilla paper** by Hoffmann et al. revisited these tradeoffs. Its conclusion: for a given compute budget, you should train a model roughly twice as large and on roughly twice as much data as the 2020 laws had implied. Specifically, the number of training tokens should be approximately 20 times the number of parameters. A 70-billion parameter model should be trained on roughly 1.4 trillion tokens. The Chinchilla-optimal regime became the standard for subsequent model development.

## GPT-3 and In-Context Learning

Six months after the scaling laws paper, OpenAI published "Language Models are Few-Shot Learners" — the GPT-3 paper. GPT-3 was a 175-billion parameter Transformer decoder, trained on a 570 GB text corpus derived from Common Crawl, WebText, books, and Wikipedia. At the time of its release, it was by an order of magnitude the largest language model ever trained. The compute cost was estimated at several million dollars.

The architecture was recognizable: a standard Transformer decoder with 96 layers, 96 attention heads, and an embedding dimension of 12,288. No significant architectural innovation over GPT-2. The difference was scale.

But GPT-3 was not evaluated primarily on fine-tuning benchmarks. Instead, the paper focused on a new capability the model appeared to have acquired: **in-context learning**.

The observation was this. When you formatted a task as a natural language prompt — with a few examples of inputs and outputs followed by a new input — GPT-3 could often solve the task without any gradient updates. Without any modification to its weights. Just by seeing the examples in the prompt.

The paper distinguished three regimes:
- **Zero-shot**: no examples, just a task description. "Translate English to French: 'Hello' =>"
- **One-shot**: a single example in the prompt.
- **Few-shot**: typically 10–100 examples in the prompt.

As the number of in-context examples increased, performance improved substantially. GPT-3 few-shot performance matched or exceeded fine-tuned smaller models on many tasks, including translation, arithmetic, reading comprehension, and question answering. On some tasks — particularly those requiring world knowledge — GPT-3 zero-shot substantially outperformed any previous approach.

This was genuinely unexpected. The model had not been trained to do few-shot learning. It had been trained to predict the next token. Yet it had, apparently, learned to read a few examples and infer the pattern. The capability had not been designed in; it had emerged.

## What Emergence Means

"Emergence" became one of the most loaded words in language modeling research circa 2020-2022. The term comes from complex systems theory: emergent properties are those that appear in the aggregate but are not predictable from the behavior of individual components. A useful working definition for language models was offered by Wei et al. in 2022: a capability is emergent if it is absent for smaller models and present — apparently discontinuously — at larger scales.

The classic example is **few-shot arithmetic**. For models up to a few billion parameters, accuracy on multi-digit addition in-context is near chance. At around 10 billion parameters, accuracy jumps to above 50%. At 100 billion parameters, it reaches near-human levels. There is no smooth power-law progression. There is a threshold.

Why does arithmetic emerge discontinuously when language modeling loss decreases smoothly? The probable answer is a measurement artifact rather than a genuine discontinuity in capability. Arithmetic is evaluated on discrete pass/fail metrics. A model that almost-but-not-quite gets the carry operation gets zero credit; a model that gets it right gets full credit. Smoothly improving representations can cross a threshold that looks discontinuous in evaluation metrics. But the underlying capabilities — the representations of numbers, the understanding of positional notation — may be improving continuously throughout.

This resolution makes emergence less mysterious but no less important. The practical implication is the same: certain capabilities only appear above certain scales, and cannot be obtained simply by training longer or adjusting hyperparameters. They require the model to be large enough.

The **BIG-bench** collaboration (Srivastava et al., 2022) assembled a benchmark of over 200 tasks specifically designed to probe capabilities that might emerge at scale. The results confirmed the pattern: many tasks showed smooth improvement with scale; a significant fraction showed apparent threshold behavior; a handful showed **inverse scaling** — performance on the task decreased as models grew larger. The inverse scaling phenomenon was particularly interesting: it suggested that scale could sometimes reinforce incorrect reasoning patterns.

## The Mechanism of In-Context Learning

The question of *how* in-context learning works became an active research problem. Several competing hypotheses emerged.

The **Bayesian hypothesis**: in-context learning is implicit Bayesian inference. The pre-training distribution defines a prior over tasks. Given in-context examples, the model performs approximate posterior inference over which task is being requested, then generates accordingly. Under this hypothesis, in-context learning is not a new capability learned at scale but a natural consequence of the pre-training objective applied to a wide enough distribution of text.

The **gradient-descent hypothesis**: in-context learning implements implicit gradient descent in the activations. Akyürek et al. (2022) and Dai et al. (2022) showed that Transformer attention layers can implement one step of gradient descent on a simple linear model, suggesting that the forward pass through a Transformer might perform something analogous to optimization given in-context examples. This is a striking connection between inference and learning.

The **task-retrieval hypothesis**: the model retrieves the closest task it has seen in training and adapts to the in-context examples. Under this view, in-context learning is primarily about pattern matching to training data, not genuine generalization to novel tasks. Evidence both supports and challenges this: models do better on tasks similar to their training distribution, but can also perform reasonably on genuinely novel tasks.

The true mechanism is probably a mixture of all three, varying by task, model size, and prompt formulation. The research question remains open.

## Prompt Engineering and the Surprising Sensitivity of In-Context Learning

GPT-3's in-context learning was powerful but unstable. The same model, given the same task, could perform dramatically differently depending on phrasing — on which examples were included, in which order, with which formatting conventions, and with which instruction wording.

This sensitivity was both a research problem and a practical one. If you wanted GPT-3 to solve arithmetic problems, you had better format the examples as `Q: What is 4 + 5? A: 9` rather than `4 + 5 = ?`. The first format matched training distribution patterns more closely. The second did not. The model had no way to ask for clarification; it could only try to predict what came next.

**Prompt engineering** — the craft of choosing prompts to elicit good behavior from a language model — emerged as a legitimate research area. Techniques that proved consistently useful included:

**Chain-of-thought prompting** (Wei et al., 2022): instead of providing input-output pairs as examples, provide worked solutions that show intermediate reasoning steps. "Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 balls. How many does he have? A: Roger started with 5. 2 cans × 3 balls = 6 balls. 5 + 6 = 11." When few-shot examples contain reasoning chains, the model generates reasoning chains and achieves substantially better accuracy on arithmetic, logic, and multi-step reasoning. Chain-of-thought prompting was not successful at small scales — it only helped models above approximately 100 billion parameters.

**Self-consistency** (Wang et al., 2022): sample multiple reasoning chains from the model and take the majority vote on the final answer. Errors in reasoning chains tend to be diverse; the correct answer is reached by multiple paths. Self-consistency improved accuracy on reasoning tasks substantially over greedy decoding.

**Zero-shot chain-of-thought** (Kojima et al., 2022): append the phrase "Let's think step by step" to the prompt. Without any few-shot examples, this simple addition caused GPT-3 class models to generate reasoning chains and perform better on arithmetic and logical reasoning. The instruction to think step by step unlocked a capability that was present but latent.

## Knowledge in Parameters

GPT-3's world knowledge was one of its most surprising properties. The model had read a large fraction of the text on the internet, and it retained substantial amounts of factual information in its parameters. It knew the capital of France, the year of the French Revolution, the boiling point of water, the name of the 42nd president. It could answer trivia questions, write biographical summaries, and explain scientific concepts — without any retrieval from an external database.

This raised new questions about the nature of knowledge in neural networks. Where was the information stored? How was it organized? Could it be updated?

**Knowledge neurons** (Dai et al., 2021) identified specific neurons in Transformer MLP layers that appeared to encode specific factual associations. Suppressing these neurons reduced the model's ability to produce specific facts; activating them could, in some cases, produce factual statements spontaneously. The MLP layers appeared to function as a distributed key-value memory: the attention mechanism retrieves which fact is relevant; the MLP layers store and return the fact's value.

This distributed storage had a troubling implication: **hallucination**. GPT-3 confidently stated false information. It would claim that a real person had won awards they hadn't won, that a real event had occurred in a different year, that a fictional fact was real. The model had no representation of uncertainty about its own parameters. It would generate fluent, confident text regardless of whether that text was accurate.

Hallucination was not a bug in the sense of an implementation error. It was a fundamental property of autoregressive language models trained on next-token prediction: the objective rewarded fluency and plausibility, not truth. A model that generates "The capital of Australia is Sydney" receives only weak correction from the training data (Canberra appears in capital-city contexts, but Sydney appears far more frequently in contexts that also mention Australia). The training signal was too diffuse to guarantee factual accuracy.

## The Scale Debate

By 2021, the debate over the right path forward in language modeling had crystallized into a question of scaling versus architecture.

The **scaling camp** held that the path to better language models was clear: more parameters, more data, more compute. The power laws predicted continued improvement. GPT-3 had demonstrated in-context learning without any architectural innovation. The argument was empirical: scaling had worked consistently; there was no reason to expect it to stop.

The **architecture camp** held that scaling could not be the whole story. Certain failure modes — hallucination, arithmetic, logical consistency — seemed structural rather than scale-dependent. A model that failed to multiply three-digit numbers at 100 billion parameters would not be fixed by 1 trillion parameters; the failure was architectural. What was needed were better inductive biases: retrieval augmentation, modular reasoning components, explicit memory.

Both camps were partially right. Scaling continued to produce dramatic improvements in fluency, world knowledge, and in-context reasoning. It did not fix hallucination. It did not, by itself, produce reliable arithmetic without chain-of-thought prompting. And it did not produce models that could follow complex instructions reliably — that would require a different intervention entirely.

The next chapter's development — alignment and instruction following — did not emerge from scaling alone. It required a new training signal: human feedback. The models produced by scaling were extraordinarily capable but also unpredictable, potentially harmful, and difficult to use. The question was not only what they could do but what they should do.

GPT-3 was the demonstration that scale changed the nature of what language models were capable of. The capabilities that emerged — in-context learning, world knowledge, few-shot task performance — were not present in GPT-2, were not predictable from GPT-2's behavior, and were not achieved by any architectural change. They came from scale alone. The lesson was absorbed by every major AI lab: the path to powerful language models ran through large datasets, large models, and the compute to train them. And the next question was what to do with models that powerful once you had them.

---

*Scaling laws for language models are established in Kaplan et al. (2020),* "Scaling Laws for Neural Language Models," *and revised by Hoffmann et al. (2022),* "Training Compute-Optimal Large Language Models" *(the Chinchilla paper). GPT-3 is introduced in Brown et al. (2020),* "Language Models are Few-Shot Learners." *Emergent capabilities are surveyed in Wei et al. (2022),* "Emergent Abilities of Large Language Models," *and the BIG-bench paper (Srivastava et al., 2022). Chain-of-thought prompting is introduced in Wei et al. (2022); zero-shot chain-of-thought in Kojima et al. (2022). The gradient-descent interpretation of in-context learning is in Akyürek et al. (2022) and Dai et al. (2022). Knowledge neurons are identified in Dai et al. (2021). Hallucination is surveyed in Ji et al. (2023).*
