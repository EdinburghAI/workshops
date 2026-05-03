# Language Models

This workshop introduces language models through embeddings first, then fine-tuning, then generation. It is designed to make the "ChatGPT" idea less magical by showing the pieces underneath: vectors, similarity search, pretrained models, and text generation.

## Notebooks

| Notebook | Use |
| --- | --- |
| [LanguageModels.ipynb](LanguageModels.ipynb) | Student-facing workshop notebook. |
| [LanguageModels-Solved.ipynb](LanguageModels-Solved.ipynb) | Completed reference notebook. |

## What Students Build

- A word-vector exploration using `gensim`.
- A sentence-embedding semantic-search system.
- A DistilBERT sentiment classifier over movie reviews.
- A GPT-2 generation demo.

## Run It

This workshop needs internet access because it downloads pretrained models and datasets.

For Kaggle:

1. Import the notebook from GitHub.
2. Turn on internet access.
3. Run the cells in order, especially the package-install cell for `sentence-transformers`.

The heavier sections use Hugging Face `datasets` and `transformers`, so local runs need those packages installed as well.

## Credits

This notebook was created by Edinburgh AI for use in its workshops. If you reuse it, please credit Edinburgh AI.
