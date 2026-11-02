# Intro to RAG

This workshop introduces retrieval-augmented generation (RAG) by showing why language models make up facts, how semantic search finds relevant text without exact keyword matches, where basic RAG breaks down, and how a multi-step agent loop fixes split clues.

## Notebooks

| Notebook | Use |
| --- | --- |
| [notebook.ipynb](notebook.ipynb) | Student-facing workshop notebook. |
| [notebook_solutions.ipynb](notebook_solutions.ipynb) | Completed reference notebook. |

The [lecture slides](intro-to-rag-lecture-slides.pdf) accompany these notebooks.

## What Students Build

- A side-by-side comparison of an ungrounded LLM hallucinating campus rules versus answering from injected context.
- A semantic search demo using `sentence-transformers` (`all-MiniLM-L6-v2`), including an exploration of why opposite sentences still score high cosine similarity.
- A document chunker with overlap and a normalized FAISS inner-product vector index.
- A basic RAG pipeline tested against normal queries, a poisoned wiki document (indirect prompt injection), and a multi-hop mystery.
- An agentic RAG loop that evaluates retrieved clues, rewrites follow-up queries, and connects facts across multiple documents.

## Run It

This workshop was designed for Kaggle or Colab.

1. Import `notebook.ipynb` from GitHub.
2. Turn on internet access so the notebook can download `all-MiniLM-L6-v2` and `Qwen/Qwen2.5-1.5B-Instruct` from Hugging Face.
3. Enable a GPU (optional, but speeds up local LLM generation).
4. Run the cells in order, complete the `TODO` prompts, and check `notebook_solutions.ipynb` if you get stuck.

All corpus documents are embedded directly in the notebook, so no external dataset attachment is needed.

## Related Workshops

- [Language Models](../intro-to-large-language-models/README.md): Introduces word embeddings, tokenization, BERT fine-tuning, and GPT-2 generation.
- [Notes to Podcast](../../PracticalWorkshops/Notes-To-Podcast/README.md): Applies RAG over university course notes to generate a two-host audio podcast.

## Credits

Created by Shayaan Absar for Edinburgh AI workshops. If you reuse it, please credit Edinburgh AI.
