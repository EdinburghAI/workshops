# Intro to Machine Learning (Legacy v1)

This is the earlier workshop retained for reference. Start with the [current main workshop](../../../README.md) for the latest version, which embeds its data and focuses on regression. The decision-tree exercises and CSV setup below belong to this legacy version.

This first workshop introduces machine learning through small, visual examples. Students fit a line with linear regression, test how models behave outside the data they saw, and then move into decision trees for classification.

![Linear-regression workshop output](assets/notebook-output-01.png)

## Notebooks

| Notebook | Use |
| --- | --- |
| [IntroToML-Beginner.ipynb](IntroToML-Beginner.ipynb) | Starter version with more scaffolding. |
| [IntroToML-Intermediate.ipynb](IntroToML-Intermediate.ipynb) | Legacy student version with implementation gaps. |
| [IntroToML-Expert.ipynb](IntroToML-Expert.ipynb) | More gaps and extension prompts. |
| [IntroToML-Solved.ipynb](IntroToML-Solved.ipynb) | Completed reference notebook. |

## What Students Build

- A linear-regression model over a deliberately fake student dataset.
- A polynomial feature version that shows how linear models can express curved relationships.
- A decision-tree classifier for a synthetic internship-selection problem.
- A discussion around why automated decision systems can be misleading even when they score well.

![Decision-tree workshop output](assets/notebook-output-05.png)

## Run It

This workshop was designed for Kaggle.

1. Import one of the notebooks from GitHub into Kaggle.
2. Add the [EdinburghAI Workshop1 dataset](https://www.kaggle.com/datasets/pierrelardet/edinburghai-workshop1). The notebooks change directory to `/kaggle/input/edinburghai-workshop1`.
3. Run the cells from top to bottom.

The CSV files are also included locally in [data](data), so the notebook can be adapted to run outside Kaggle by changing the data path.

That folder contains `bigcheese.csv`, `bigcheese-quadratic.csv`, `googleinternship_simple.csv` and `googleinternship_big.csv`. For local use, replace the Kaggle directory-change cell with a path to this folder and provide pandas, NumPy, Matplotlib and scikit-learn. The [generation scripts](../generation/) show how the synthetic datasets were created.

## Credits

This notebook was created by Edinburgh AI for use in its workshops. If you reuse it, please credit Edinburgh AI.
