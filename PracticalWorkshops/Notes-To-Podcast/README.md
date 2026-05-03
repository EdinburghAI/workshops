# Notes To Podcast

This practical workshop builds a local podcast generator over university notes. Students embed the notes, retrieve relevant context with vector search, generate a two-host conversation with a local LLM, and synthesize audio using an open text-to-speech model.

<audio controls src="assets/notebook-audio-01.wav"></audio>

## Notebooks And Data

| File | Use |
| --- | --- |
| [notes-to-podcast-all-levels.ipynb](notes-to-podcast-all-levels.ipynb) | Student-facing workshop notebook. |
| [notes-to-podcast-solutions.ipynb](notes-to-podcast-solutions.ipynb) | Completed reference notebook. |
| [notes_dataset.csv](notes_dataset.csv) | Notes dataset used for retrieval. |

## What Students Build

- A small vector database over university notes using sentence embeddings and FAISS.
- A retrieval-augmented prompt for answering questions from those notes.
- A two-host conversation generated with Ollama and Qwen.
- A synthesized podcast sample using Kokoro TTS.

## Run It

This workshop was designed for Kaggle.

1. Import `notes-to-podcast-all-levels.ipynb` from GitHub.
2. Turn on internet access in Kaggle.
3. Run the setup cells that install `uv`, FAISS, Ollama, Kokoro, and `espeak-ng`.
4. Run the notebook from top to bottom.

The notebook downloads the notes CSV from this repo, so no separate Kaggle dataset is required for the current version.

## Credits

This notebook was created by Edinburgh AI for use in its workshops. If you reuse it, please credit Edinburgh AI.
