# RNNs And LSTMs

This workshop introduces recurrent models through a time-series task: detecting throw attempts in judo pose sequences. Instead of classifying a single image, students work with ordered sequences of skeleton keypoints.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/The_LSTM_cell.png/1200px-The_LSTM_cell.png" alt="LSTM cell diagram" width="520">

## Notebooks

| Notebook | Use |
| --- | --- |
| [LSTM.ipynb](LSTM.ipynb) | Student-facing workshop notebook. |
| [LSTM-Solved.ipynb](LSTM-Solved.ipynb) | Completed reference notebook. |

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

## Credits

This notebook was created by Conor O'Shea for Edinburgh AI workshops. If you reuse it, please credit Edinburgh AI.
