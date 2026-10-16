# Sports Vision with RNNs and LSTMs

This workshop explores how ordered pose sequences can help identify judo throw attempts. The current coding notebooks classify an entire seven-second clip as attempt or no attempt, using 210 frames with 68 joint-coordinate features per frame. A separate beginner activity explains the ideas without coding.

## Notebooks and Materials

| File | Use |
| --- | --- |
| [BEGINNER-sports-vision-rnn-lstm.ipynb](BEGINNER-sports-vision-rnn-lstm.ipynb) | Browser-based activity guide; no coding or maths background required. |
| [UNSOLVED-advanced-sports-vision-rnn-lstm.ipynb](UNSOLVED-advanced-sports-vision-rnn-lstm.ipynb) | Current student coding notebook. |
| [SOLVED-advanced-sports-vision-rnn-lstm.ipynb](SOLVED-advanced-sports-vision-rnn-lstm.ipynb) | Completed reference notebook. |

## What Students Build

- Beginner route: inspect video frames, pixels and skeletons, invent detection rules, and explore a fair final test in the workshop app.
- Coding route: load pose arrays and supplied split manifests into PyTorch dataloaders.
- Compare an RNN and an LSTM using weighted binary loss, gradient clipping and validation-based checkpoint and threshold selection.
- Evaluate the selected model on the test set, inspect video predictions and save `judo_workshop_model.pt`.
- Optionally explore a bidirectional LSTM using validation data.

Pose extraction has already been performed; students do not need to run YOLO or a tracker.

## Run It

For the beginner route, read the beginner notebook alongside the [Judo Sports Vision workshop app](https://ediai-rnn-lstm-workshop-app.streamlit.app/). No Python environment or dataset download is required for that activity.

For the coding route:

1. Import the unsolved notebook into Kaggle.
2. Attach [Version 2 RNN/LSTM Workshop Judo Data](https://www.kaggle.com/datasets/conoroshea46/version2-rnn-lstm-workshop-judo-data).
3. Enable a GPU if available, then run the cells in order and complete the `TODO` prompts.
4. Use validation results for model and threshold choices before running the final test.

The setup cell discovers the dataset through `lstm_inputs.npy` and `split_manifest.csv`. It also uses `labels.npy`, `training_manifest.csv`, example videos and an illustrative GIF from that dataset. For local use, download the complete version 2 dataset and set `DATASET_DIR` to its extracted folder. The coding notebooks use PyTorch, NumPy, pandas, Matplotlib, scikit-learn and IPython.

## Legacy Version

The [legacy workshop](legacy/README.md) contains `LSTM.ipynb`, `LSTM-Solved.ipynb` and the earlier lecture slides. It predicts frame-level throw labels using the older Judo Throw Interval Dataset, custom padding masks and a YAML configuration. That dataset and task are incompatible with the current clip-level notebooks; use the version 2 dataset above for the main workshop.

## Credits

Created by Conor O'Shea for Edinburgh AI. The current coding notebooks credit Ultralytics YOLO11x-pose and ByteTrack for pose extraction and state an Apache License, Version 2.0 for the workshop code. Please retain the notebook's credits and licence information when reusing it.
