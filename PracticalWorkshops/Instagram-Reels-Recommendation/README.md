# Instagram Reels Recommendation

This workshop explores sequential recommendation through short-form video interactions. The beginner route uses a live reels app and an AI chat tool without coding. The advanced route implements and trains a Hierarchical Sequential Transduction Unit (HSTU) model in PyTorch, then uses its trained parameters to recommend from a separate workshop reel catalogue.

## Notebooks and Materials

| File | Use |
| --- | --- |
| [BEGINNER-ediai-reels-workshop.ipynb](BEGINNER-ediai-reels-workshop.ipynb) | No-code activity guide using a browser and an AI chat tool. |
| [UNSOLVED-advanced-ediai-reels-workshop.ipynb](UNSOLVED-advanced-ediai-reels-workshop.ipynb) | Student coding notebook. |
| [SOLVED-advanced-edi-reels-workshop.ipynb](SOLVED-advanced-edi-reels-workshop.ipynb) | Completed coding reference; `edi` is the actual filename spelling. |
| [ig-reels-workshop-lecture-slides.pdf](ig-reels-workshop-lecture-slides.pdf) | Workshop slides. |

## What Students Build

- Beginner route: distinguish history, candidate inputs and prediction targets; copy a scrolling log into an AI chat tool; score candidate reels and explore Hit@10.
- Advanced route: load events and content embeddings, form interaction sequences and create padded PyTorch batches.
- Implement item/event encoders and HSTU blocks, train a next-item recommender and evaluate it with validation and test ranking metrics.
- Export `reels_recommender_transferable_state.pt`, load workshop reel embeddings, and generate ten recommendations from a copied scrolling log.

## Run It

For the beginner route, open the beginner guide, the [workshop reels app](https://edi-ai-app-reels-app-g9hix2dhtyjvntnkkvnzcd.streamlit.app/) and a web-based AI chat tool. Follow the guide's prompt and copy/paste activities; no Python execution or Kaggle dataset attachment is needed.

For the advanced route:

1. Import the unsolved advanced notebook into Kaggle and enable a GPU if available.
2. Attach [EdiAI Reels Adapted Short Video 2025 Dataset](https://www.kaggle.com/datasets/conoroshea46/ediai-reels-adapted-short-video-2025-dataset) for training. It supplies `events.parquet`, `items.parquet`, `content_embeddings.npy` and `item_embedding_index.csv`.
3. Run the training and evaluation sections in order, completing the `TODO` prompts.
4. For Section 4, also attach [EdiAI Workshop Reels 2026](https://www.kaggle.com/datasets/conoroshea46/ediai-workshop-reels-2026), which supplies `reels.parquet`, `reel_content_embeddings.npy` and `reel_embedding_index.csv`.
5. Record interactions with at least 20 reels in the app, copy the log into the notebook as directed, then copy the recommended IDs back into the app.

The two datasets have separate embedding indices; keep each index paired with its own embeddings. For local use, set `DATASET_DIR_PATH` and `WORKSHOP_DATASET_DIR_PATH` to the respective extracted folders. The coding notebooks use PyTorch, NumPy, pandas, scikit-learn, Matplotlib and tqdm; pandas also needs a Parquet reader such as pyarrow. These external datasets are not bundled in this folder.

## Credits

Created by Conor O'Shea for Edinburgh AI, with the beginner activity adapted from that workshop. The advanced notebooks reference Meta's HSTU research and the ShortVideo dataset and include Apache License, Version 2.0 information. Please retain their credits, source attribution and licence information when reusing the material.
