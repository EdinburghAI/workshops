# RNNs and LSTMs (Legacy)

This is the earlier frame-level workshop, retained for reference. Use the [current main workshop](../README.md) for the beginner activity and the version 2 clip-level classification notebooks. The datasets and prediction targets differ between the two versions.

This workshop introduces recurrent models through a time-series task: detecting throw attempts in judo pose sequences. Instead of classifying a single image, students work with ordered sequences of skeleton keypoints.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/The_LSTM_cell.png/1200px-The_LSTM_cell.png" alt="LSTM cell diagram" width="520">

## Notebooks

| Notebook | Use |
| --- | --- |
| [LSTM.ipynb](LSTM.ipynb) | Student-facing workshop notebook. |
| [LSTM-Solved.ipynb](LSTM-Solved.ipynb) | Completed reference notebook. |
| [intro-to-rnn-lstm-lecture-slides.pdf](intro-to-rnn-lstm-lecture-slides.pdf) | Lecture slides for this legacy workshop. |

## What Students Build

- A dataset loader for `.npy` sequence arrays.
- Train, validation, and test splits that preserve sequence-label alignment.
- PyTorch dataloaders for time-series data.
- RNN and LSTM models that predict whether each frame contains a throw attempt.

## Run It

This workshop was designed for Kaggle.

1. Import the notebook from GitHub.
2. Attach the Kaggle dataset [Judo Throw Interval Dataset](https://www.kaggle.com/datasets/conoroshea46/judo-throw-interval-dataset).
3. Turn on internet access.
4. Enable a GPU for training.
5. Run the notebook from top to bottom.

This version loads `input_data.npy`, `labels.npy` and `attempt_ids.npy` from the dataset's `throw-locator-model-data-final/` folder, plus its `config.yaml` and an example GIF. It uses a custom dataset with padding masks, a masked and class-weighted binary loss, and per-frame model evaluation. Do not substitute the version 2 clip-level dataset.

For local use, provide PyTorch, NumPy, scikit-learn, PyYAML, Matplotlib, tqdm and IPython, then update all dataset, configuration and GIF paths. The dataset is external and is not included in this folder.

## Supporting Notes

For notebook links that still use the previous repository layout, see the current [batching notes](../../yolo-single-raccoon-box-bounding/yolo-workshop-api/Additional-notes/Batching.md), [glossary](../../yolo-single-raccoon-box-bounding/yolo-workshop-api/Additional-notes/Glossary.md), [training-loop guide](../../yolo-single-raccoon-box-bounding/yolo-workshop-api/Additional-notes/Training%20Loops%20in%20PyTorch.md) and [computational training graphs](../../yolo-single-raccoon-box-bounding/yolo-workshop-api/Additional-notes/Computational%20Training%20Graphs.md).

## Credits

This notebook was created by Conor O'Shea for Edinburgh AI workshops. If you reuse it, please credit Edinburgh AI.
