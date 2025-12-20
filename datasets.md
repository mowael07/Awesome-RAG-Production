# 💾 Datasets for RAG Benchmarking

You cannot improve what you cannot measure. Production-grade RAG requires rigorous evaluation against ground-truth datasets. This list covers general-purpose benchmarks and domain-specific corpora.

---

## 🏅 The Leaderboards

Before choosing a model, check these live leaderboards:
*   **[MTEB (Massive Text Embedding Benchmark)](https://huggingface.co/spaces/mteb/leaderboard)** - The gold standard for choosing an embedding model (Retrieval, Clustering, Reranking quality).
*   **[Open Compass](https://opencompass.org.cn/leaderboard-llm)** - Comprehensive LLM evaluation which includes retrieving capabilities.
*   **[Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)** - General LLM performance.

---

## 📚 General Knowledge (Open Domain QA)

| Dataset | Description | Size | Best For |
| :--- | :--- | :--- | :--- |
| **[MS MARCO](https://microsoft.github.io/msmarco/)** | Making AI the first truly conversational search engine. | 1M+ Queries | **Retrieval**. The baseline for almost all retriever/reranker training. |
| **[HotpotQA](https://hotpotqa.github.io/)** | Question answering requiring multi-hop reasoning. | 113k pairs | **Reasoning**. Testing if your RAG can combine facts from 2+ documents. |
| **[Natural Questions (NQ)](https://ai.google.com/research/NaturalQuestions)** | Real user queries issued to Google Search. | 300k+ | **Realism**. Real-world messy queries from actual users. |
| **[TriviaQA](https://nlp.cs.washington.edu/triviaqa/)** | Reading comprehension dataset containing triples of (question, answer, evidence). | 95k | **Factuality**. Checking if the model can pinpoint precise facts. |

---

## 📑 Long-Context & Document Understanding

| Dataset | Description | Best For |
| :--- | :--- | :--- |
| **[SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)** | Stanford Question Answering Dataset. Includes unanswerable questions. | **Hallucination Detection**. Can your model say "I don't know"? |
| **[Qasper](https://allenai.org/data/qasper)** | Question answering over NLP papers. | **Technical/Scientific**. RAG over dense, technical PDFs. |
| **[NarrativeQA](https://github.com/deepmind/narrativeqa)** | QA over collected stories (books and movie scripts). | **Long Context**. Retrieval from very long texts (summarization). |

---

## 🧪 Synthetic Data Generation

Don't have a dataset? Generate one from your own internal documents.
*   **[Ragas Synthetic Data Generator](https://docs.ragas.io/en/stable/concepts/testset_generation.html)** - create "Golden Datasets" (Question-Answer-Context triples) automatically.
*   **[LlamaIndex Data Generator](https://docs.llamaindex.ai/en/stable/module_guides/evaluator/usage_pattern.html)** - Built-in utils to generate questions from your indexed nodes.

---

<p align="right">(<a href="README.md#contents">back to main resource</a>)</p>
