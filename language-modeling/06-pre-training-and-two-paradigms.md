---

title: "Pre-training and the Two Paradigms"
subtitle: "GPT, BERT, and the transfer learning revolution"
---


The Transformer had been published. The architecture was clean, its performance on translation was decisive, and the research community had absorbed its core ideas within months. But something else was happening at the same time, something that would prove to be the more consequential development. The community was beginning to ask a different question — not *how to build a better sequence model* but *how to use unlabeled text*. The answer it found would change the field permanently.

The question had been lurking for years. The internet contained hundreds of billions of words of text. Wikipedia, books, news archives, web crawls — an enormous corpus of human-generated language, almost none of it labeled for any particular task. Labeled datasets, by contrast, were expensive to build and small: a few thousand examples for sentiment analysis, tens of thousands for natural language inference, a hundred thousand for reading comprehension. This asymmetry felt wrong. It seemed like an enormous waste to ignore the raw text and rely only on the small labeled slices.

The solution, it turned out, was already understood in computer vision. ImageNet pre-training — initializing a model on large-scale image classification, then fine-tuning on a smaller task — had been the dominant paradigm in vision since 2012. Language had its own version, word embeddings (Chapters 2 and 3), but these were shallow: you transferred the representations of individual words, not the deep contextual understanding built by processing entire sentences. The question was whether you could do the same thing with deep models. Could you pre-train a Transformer on raw text and transfer the entire network?

Two groups answered yes in 2018, and they did it in opposite ways.

## The First Paradigm: GPT

OpenAI published "Improving Language Understanding by Generative Pre-training" in June 2018. The paper introduced **GPT** (Generative Pre-trained Transformer), and its logic was almost disarmingly simple.

Take the decoder side of the original Transformer — the part that generates output tokens autoregressively. Train it on a language modeling objective: predict the next token given all preceding tokens. Do this on a large corpus of books (BookCorpus, roughly 800 million words). Then fine-tune the pre-trained model on downstream tasks by appending a task-specific linear layer and training on labeled examples.

The language modeling objective had a compelling motivation. Every sentence in the corpus is its own labeled example: the input is the context, the label is the next word. A model that becomes excellent at next-word prediction must, implicitly, have learned something deep about syntax, semantics, and world knowledge — because the only way to predict language well is to understand it. This is **self-supervised learning**: the labels come from the data itself, with no human annotation required.

The architecture was a 12-layer Transformer decoder with 117 million parameters. The training corpus was BookCorpus, chosen for its long, coherent passages — better for learning long-range dependencies than shorter web text. The fine-tuning procedure adapted the input formatting to each task: for text classification, a start and end token wrapped the input; for natural language inference, a separator token divided premise and hypothesis. The model learned to use these formatting conventions from the small labeled data, while the pre-trained weights carried the language understanding forward.

The results were striking. GPT set new state-of-the-art performance on 9 of the 12 tasks evaluated, including natural language inference, question answering, and semantic similarity. The gains were particularly large on tasks with small training sets, exactly where pre-training would be expected to help most. Something powerful was being transferred.

But there was a limitation baked into the design. GPT used a **causal** (left-to-right, autoregressive) language model: each token could only attend to the tokens preceding it. This made sense for generation — you can't use future tokens when predicting the next word — but for understanding tasks like classification, it meant the model saw each token only in its left context. The word "bank" in "the river bank" and "the bank account" differs primarily in its right context. A left-to-right model would process both contexts separately, but never simultaneously. The representation at each position was built from one direction only.

## The Second Paradigm: BERT

Google published "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" in October 2018. BERT took the opposite architectural choice and went bidirectional.

**BERT** — Bidirectional Encoder Representations from Transformers — used the encoder side of the original Transformer. In the encoder, every token attends to every other token in the sequence. The representation of each token is built from full context: everything to the left and everything to the right simultaneously. This bidirectionality was BERT's central innovation, and it required solving a non-trivial training problem.

The standard language modeling objective — predict the next token — cannot be applied to a bidirectional encoder without trivially revealing the answer. If every token attends to every other token, and the task is to predict token $t$, then token $t$ can simply attend to itself. The objective collapses. You need a training task that forces genuine understanding of context without leaking the target.

The solution was **Masked Language Modeling (MLM)**. Before processing a sequence, randomly mask 15% of the tokens. Replace them with a special `[MASK]` token (with some probability), or a random word, or leave them unchanged (to prevent the model from learning to ignore masked positions). Train the model to predict the original tokens from context. Because the target positions are masked, the model cannot look at them directly — it must infer them from the surrounding words.

MLM has a beautiful property: it forces the model to learn rich bidirectional representations, because predicting a masked word from context requires understanding the meaning of the surrounding sentence from both sides. The word "bank" masked from "the river ___ was eroded by floods" requires knowing that river banks erode; the same word masked from "the ___ charged overdraft fees" requires knowing that financial institutions charge fees. The same architectural mechanism, applied to two different contexts, must produce two different representations.

BERT also used a second pre-training task: **Next Sentence Prediction (NSP)**. Given two sentences, predict whether the second sentence follows the first in the original document. Half the training pairs were consecutive sentences; half were random pairs. This was intended to teach BERT about inter-sentence relationships, useful for tasks like question answering (does the passage answer the question?) and natural language inference (does the hypothesis follow from the premise?). In retrospect, NSP proved less important than MLM — later work showed that removing it often helped rather than hurt performance — but it was a reasonable hypothesis at the time.

BERT was trained on BookCorpus and the entirety of English Wikipedia (2.5 billion words combined), substantially more data than GPT's BookCorpus-only pre-training. It was also larger: two variants were released, **BERT-Base** (110 million parameters, 12 layers) and **BERT-Large** (340 million parameters, 24 layers). Both were substantially more expensive to train than GPT but the compute was available at Google scale.

The results were not merely incremental. BERT improved the state of the art on eleven NLP benchmarks by substantial margins — in many cases, by more than all prior work combined. On **GLUE** (General Language Understanding Evaluation, a suite of diverse NLP tasks), BERT-Large improved over the previous best by over 7 absolute points. On **SQuAD** (Stanford Question Answering Dataset, a reading comprehension benchmark), BERT surpassed human performance. On natural language inference, semantic similarity, and question answering, the improvements were similar in character: large, consistent, and clearly attributable to the pre-training.

The NLP community had a phrase for this kind of result: *it crushed it*. Papers that had taken years of careful engineering to reach a particular benchmark performance were overtaken in weeks by teams that simply fine-tuned BERT. The landscape of the field rearranged overnight.

## Fine-tuning: The Transfer Learning Recipe

What BERT and GPT shared was more important than what they differed on: the general framework of **pre-train, then fine-tune**.

The recipe had three steps. First, pre-train a large Transformer on a self-supervised objective (language modeling for GPT, MLM for BERT) using abundant unlabeled text. Second, for a downstream task, take the pre-trained model and add a small task-specific head — typically a linear classifier on top of the model's output. Third, fine-tune all parameters jointly on the small labeled dataset, using a learning rate small enough not to destroy the pre-trained representations.

This framework changed what it meant to do NLP. Previously, building a competitive model for a new task meant careful feature engineering, task-specific architecture design, and substantial labeled data. Now it meant: download BERT, fine-tune for a few hours, submit. The barrier to entry dropped dramatically. The focus of research shifted from task-specific modeling to pre-training methodology.

Fine-tuning raised new technical questions. How should the input be formatted for different task types? BERT's solution was input templates with special tokens: `[CLS]` at the start, `[SEP]` between sentences and at the end. The `[CLS]` token's output representation became the aggregate sequence representation fed to the classification head. For sentence pairs (premise-hypothesis, question-passage), both sentences were concatenated with a separator. A learned **segment embedding** distinguished the two parts. These conventions became standard practice.

How many layers should be fine-tuned? Empirically, fine-tuning all layers consistently outperformed fine-tuning only the top layers, suggesting that pre-trained representations at every level were useful and needed task-specific adaptation. The lower layers, encoding syntactic features, transferred most readily; the higher layers, encoding more semantic and task-specific features, required more updating. **Catastrophic forgetting** — the model losing pre-trained knowledge during fine-tuning — was a real risk with large learning rates, managed by warming up the learning rate and using smaller rates for lower layers.

## Representations as Understanding

One of the most illuminating aspects of the BERT era was the cottage industry of **probing** studies that tried to understand what pre-trained models had actually learned.

The method was simple: take a pre-trained model, freeze its weights, train a linear classifier on its representations to predict some linguistic property (part-of-speech tag, syntactic dependency, semantic role), and measure accuracy. High accuracy means the property is encoded in the representation; low accuracy means it isn't. By probing layer by layer, you could map out what the model learned where.

The results were consistent and surprising. BERT's lower layers encoded surface features (word morphology, POS tags), middle layers encoded syntactic structure (parse trees could be recovered), and upper layers encoded semantic information and task-relevant features. The model had learned a rough hierarchy of linguistic abstraction, not by design but as a consequence of predicting masked words. Representations built for next-word prediction turned out to contain syntactic parse trees. Representations built for masked word prediction turned out to understand coreference, semantic roles, and logical relationships.

This suggested something deeper than engineering: that language understanding and language prediction are, at some level, the same task. A model that can predict what word goes in a gap in any sentence must understand the sentence. The probing studies provided empirical evidence for what had been a theoretical intuition.

## The Divergence of Two Approaches

GPT and BERT represented not just two architectures but two different theories about what language models were for.

BERT was an **encoder**: it built representations of text. It was not designed to generate new text. Its masked objective and bidirectional attention made it excellent at understanding — at tasks that required reading input and producing a label or score. It was the dominant approach for discriminative NLP from 2018 onward.

GPT was a **decoder**: it generated text. Its causal objective made it excellent at completion tasks — given a prompt, continue it. For downstream fine-tuning on discriminative tasks, it performed slightly worse than BERT, because its left-to-right representations were less rich than BERT's bidirectional ones. But GPT's generative ability pointed toward something BERT couldn't do: produce language.

The question that GPT raised, but that GPT-1 answered only modestly, was: *what happens if you scale up?* GPT-1 had been trained on BookCorpus. What if you trained a much larger decoder on a much larger corpus? Could you get, not just better fine-tuned performance, but better *zero-shot* performance — good behavior with no fine-tuning at all?

OpenAI asked this question and answered it with **GPT-2**, published in February 2019. GPT-2 was 1.5 billion parameters — 12 times larger than GPT-1 — and trained on **WebText**, a 40 GB dataset of Reddit-linked web pages (filtered for quality by the upvote count as a proxy for human interest). The paper's title captured the hypothesis: "Language Models are Unsupervised Multitask Learners."

The results were striking in a different way from BERT. GPT-2 was not evaluated on fine-tuned benchmarks. Instead, it was tested *zero-shot*: in what tasks could a language model trained purely on next-word prediction already perform reasonably, without any task-specific training? The answer was: many tasks. Given a reading comprehension dataset formatted as a document and a question, GPT-2 generated reasonable answers without being trained on reading comprehension. Given a French sentence, it translated to English — without translation training. Given a summarization prompt, it produced reasonable summaries.

The zero-shot performance was not state-of-the-art. It was clearly worse than fine-tuned BERT on GLUE tasks. But the conceptual implication was profound: a language model, trained on nothing but raw text, was already learning to perform tasks that researchers had spent years building specialized models for. Language modeling was not merely a training signal for representation learning — it was, in some sense, *task learning*.

OpenAI declined to release the full GPT-2 model at launch, citing concerns about misuse for generating disinformation. This was controversial — many researchers criticized the decision as performative given the model's modest capabilities — but it marked a new kind of awareness in the field: that the capabilities of language models could have consequences beyond research benchmarks.

## The Encoder-Decoder Middle Ground

Between pure encoders (BERT) and pure decoders (GPT) lay a third option: the full encoder-decoder Transformer of the original "Attention Is All You Need" paper, trained with a unified pre-training objective.

Google's **T5** (Text-to-Text Transfer Transformer, Raffel et al., 2019) explored this space systematically. T5's central insight was format unification: cast every NLP task as a text-to-text problem. Classification? Output the label as text. Translation? Output the target sentence as text. Summarization? Output the summary as text. Question answering? Output the answer as text. With every task in the same format, you could train on mixtures of tasks and datasets, fine-tune with the same objective (language modeling on the target), and compare apples-to-apples across tasks.

T5 also ran a systematic ablation study of pre-training choices — dataset, objective, model size, training compute — that produced the most careful empirical analysis of pre-training design to date. Its findings: span corruption (masking contiguous spans rather than individual tokens) slightly outperformed BERT-style individual token masking; more data consistently helped; larger models consistently helped; pre-training on cleaner data (the **C4 dataset**, Colossal Clean Crawled Corpus, 750 GB) outperformed pre-training on noisy web crawls.

The T5 11-billion parameter model achieved state-of-the-art on GLUE, SuperGLUE, and a range of other benchmarks. But its more lasting contribution was methodological: the text-to-text framework unified NLP and demonstrated that task format mattered less than scale and data quality.

## What Pre-training Actually Teaches

By 2020, the pre-training era had produced a set of empirical regularities that were difficult to explain from first principles.

Pre-trained models were not merely better initializations. They learned qualitatively different representations — representations that encoded linguistic structure, semantic relationships, and world knowledge in ways that could not be learned from small labeled datasets alone, regardless of model architecture. Fine-tuning a pre-trained model on ten examples of sentiment analysis worked far better than training a smaller model from scratch on ten thousand.

The mystery was why. The self-supervised objectives — next-word prediction, masked word prediction — seemed too simple to teach what the models appeared to know. The answer, which the community began to articulate around 2020, was that language is not simple. The set of all next-word predictions over a large text corpus *is* an implicit encoding of world knowledge, causal relationships, social norms, factual associations, and logical structure. A model that predicts language well must have internalized all of this, not because it was told to, but because the data required it.

This observation set the stage for the next chapter's central question: what happens when the model becomes large enough to not merely predict language well, but to *use language capabilities* on novel tasks it was never trained to perform? What happens when scale changes the nature of what a language model can do?

The answer — emergence — was still a year away. But its prerequisites were all in place: the architecture (Transformer), the training paradigm (pre-training), the data (web-scale text), and the growing conviction that scale was the key variable. GPT-2 had been a hint. GPT-3 would be the demonstration.

---

*GPT was introduced in Radford et al. (2018),* "Improving Language Understanding by Generative Pre-training." *BERT was introduced in Devlin et al. (2018),* "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," *which also reviews the fine-tuning protocol and probing methodology in detail. T5 is introduced in Raffel et al. (2020),* "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *The GLUE and SuperGLUE benchmarks are described in Wang et al. (2018, 2019). GPT-2 is introduced in Radford et al. (2019),* "Language Models are Unsupervised Multitask Learners." *Probing studies of BERT's representations include Tenney et al. (2019) and Jawahar et al. (2019). The analysis of what BERT knows is surveyed in Rogers et al. (2020),* "A Primer in BERTology."
