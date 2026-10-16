# Notes To Podcast

This practical workshop builds a local podcast generator over university notes. Students embed the notes, retrieve relevant context with vector search, generate a two-host conversation with a local LLM, and synthesize audio using an open text-to-speech model.

<audio controls src="assets/notebook-audio-01.wav"></audio>

[Listen to the example podcast](assets/notebook-audio-01.wav) if the embedded player is unavailable.

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
3. Ensure `uv` is available before the first `uv pip` cell; the notebooks use it but do not install it. Run the setup cells for FAISS, Ollama, Kokoro and `espeak-ng`.
4. Run the notebook from top to bottom.

The notebook downloads the notes CSV from this repo, so no separate Kaggle dataset is required for the current version.

The reference notebook uses `all-MiniLM-L6-v2` sentence embeddings, FAISS retrieval, Ollama's `qwen3:1.7b` and two Kokoro voices. Internet access is needed for the initial package, model and CSV downloads. A GPU helps with embedding and conversation generation.

For local use, the setup cells assume a Linux environment with `apt` and shell tools. Provide an Ollama server and the notebook's Python and speech dependencies, and use the included CSV if you do not download it. These cells are not directly portable to a standard Windows Jupyter environment.

## Related Workshop

For the embeddings refresher referenced by an outdated notebook link, use the current [Language Models solved notebook](../../TheoreticalWorkshops/intro-to-large-language-models/LanguageModels-Solved.ipynb).

## Credits

This notebook was created by Edinburgh AI for use in its workshops. If you reuse it, please credit Edinburgh AI.
