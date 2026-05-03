# Edinburgh AI Workshop Resources

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://scikit-learn.org/stable/"><img alt="Scikit-Learn" src="https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white"></a>
<a href="https://pandas.pydata.org/"><img alt="Pandas" src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white"></a>
<a href="https://www.kaggle.com/"><img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-20BEFF?logo=kaggle&logoColor=white"></a>

Edinburgh AI runs beginner-friendly AI workshops at the University of Edinburgh. This repo contains the notebooks, datasets, helper code, and solution files used across different weekly sessions.

The workshops are deliberately self-contained. Each one was delivered in a different week, usually around one concrete build: a first ML classifier, a CNN image model, a language model demo, a podcast generator, an object detector, a sign-language interpreter, a Flappy Bird reinforcement-learning agent, or an LSTM sequence model.

Most sessions were designed around Kaggle because it gave students the easiest shared environment: browser notebooks, dataset attachment, optional GPUs, and minimal local setup.

## Workshop Catalogue

| Area | Workshop | What students build | Main stack |
| --- | --- | --- | --- |
| Foundations | [Intro to Machine Learning](TheoreticalWorkshops/Sem1Workshop1/IntroToML) | Linear regression and decision-tree classifiers on synthetic datasets | pandas, scikit-learn |
| Foundations | [Neural Networks](TheoreticalWorkshops/Sem1Workshop2) | A first PyTorch neural network and MNIST classifier | PyTorch, torchvision |
| Foundations | [Computer Vision with CNNs](TheoreticalWorkshops/Sem1Workshop3) | Convolution demos, an image classifier, and face segmentation | PyTorch, torchvision, transformers |
| Foundations | [Language Models](TheoreticalWorkshops/Sem1Workshop4) | Embeddings, semantic search, sentiment fine-tuning, GPT-2 generation | gensim, sentence-transformers, transformers |
| Theoretical | [Single-Object YOLO Detection](TheoreticalWorkshops/Sem2Workshop1) | A custom single-object detector with bounding-box loss and IoU evaluation | PyTorch, pandas, YAML |
| Theoretical | [RNNs and LSTMs](TheoreticalWorkshops/Sem2Workshop2) | A time-series model for detecting throw attempts in judo pose sequences | PyTorch, NumPy, scikit-learn |
| Practical | [Notes to Podcast](PracticalWorkshops/Notes-To-Podcast) | A local RAG pipeline that turns university notes into a two-host podcast | FAISS, Ollama, Kokoro TTS |
| Practical | [ASL Interpreter](PracticalWorkshops/ASL-Interpreter) | A sign-language classifier using hand landmarks and image models | MediaPipe, OpenCV, scikit-learn, TensorFlow |
| Practical | [Flappy Bird RL](PracticalWorkshops/Flappy-Bird) | A DQN agent trained to play Flappy Bird | Gymnasium, PyTorch |

## How To Run A Workshop

The normal route is Kaggle:

1. Go to [kaggle.com](https://kaggle.com) and create an account.
2. Create a new notebook.
3. Select `File -> Import Notebook -> GitHub`.
4. Search for `EdinburghAI/workshops`.
5. Pick the workshop notebook you want.
6. Turn on internet access in the Kaggle notebook settings when the workshop asks for it.
7. Attach the dataset listed in that workshop's README.
8. Run the cells from top to bottom.

Each workshop folder has its own README with the exact notebooks, datasets, outputs, and takeaways for that session.

## Repository Layout

```text
TheoreticalWorkshops/
  Sem1Workshop1/IntroToML/
  Sem1Workshop2/
  Sem1Workshop3/
  Sem1Workshop4/
  Sem2Workshop1/
  Sem2Workshop2/

PracticalWorkshops/
  Notes-To-Podcast/
  ASL-Interpreter/
  Flappy-Bird/
```

Notebook filenames are mostly preserved from the live workshop material so old Kaggle imports and shared links do not break. The README files use consistent labels such as student notebook, solution notebook, beginner, intermediate, and advanced.

## Credits

Created by [Pierre Mackenzie](https://pierre.wiki) and [Leo Camacho](https://www.leocamacho.co), with help from Valentin Magis, Niall Meagher, Finlay Ross, Conor O'Shea, and Edinburgh AI workshop contributors. Please credit Edinburgh AI if you use or adapt this material.
