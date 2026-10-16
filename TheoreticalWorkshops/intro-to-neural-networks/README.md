# Neural Networks

This workshop moves from a single neuron to training a small neural network in PyTorch, then applies the same ideas to handwritten-digit classification with MNIST. It builds on the ideas in [Intro to Machine Learning](../intro-to-machine-learning/README.md).

## Notebooks and Materials

| File | Use |
| --- | --- |
| [NeuralNetworks-Beginner.ipynb](NeuralNetworks-Beginner.ipynb) | Starter version with more direct guidance. |
| [NeuralNetworks-Intermediate.ipynb](NeuralNetworks-Intermediate.ipynb) | Student version with implementation gaps. |
| [NeuralNetworks-Expert.ipynb](NeuralNetworks-Expert.ipynb) | More independent implementation. |
| [NeuralNetworks-Solved.ipynb](NeuralNetworks-Solved.ipynb) | Completed reference notebook. |
| [EdiAI Workshop 2 Slides.pdf](EdiAI%20Workshop%202%20Slides.pdf) | Workshop slides; the original filename is retained. |

## What Students Build

- A single linear neuron with `torch.nn.Linear` and a multilayer network with `nn.Sequential`.
- A training loop using a loss function, backpropagation and an optimiser.
- An MNIST classifier with PyTorch datasets and dataloaders.
- Prediction visualisations and an accuracy evaluation.

## Run It

1. Import a student notebook into Kaggle, or open it locally in Jupyter.
2. Enable internet access for the MNIST download.
3. Run the cells in order, filling in the `TODO` prompts. Use the solved notebook as a reference for missing imports and implementations.

Local runs need PyTorch, torchvision, Matplotlib, scikit-learn and tqdm. MNIST is downloaded by torchvision into `data/`; there is no separate Kaggle dataset to attach.

## Legacy Version

Use the notebooks beside this README for the current main version. The [legacy version](legacy/v1-intro-to-neural-networks/README.md) retains the earlier beginner, intermediate, expert and solved notebooks for reference.

## Credits

Created by Edinburgh AI for its workshops. Please credit Edinburgh AI if you reuse or adapt the material.
