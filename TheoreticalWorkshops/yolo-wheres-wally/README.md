# Where's Wally? Single-Object Detection

This workshop builds a CNN from scratch to locate one Wally in an image. Students generate synthetic labelled images by pasting transparent Wally cut-outs onto backgrounds, train a heatmap-based bounding-box model and evaluate its predictions with intersection over union (IoU).

## Notebooks and Materials

| File | Use |
| --- | --- |
| [UNSOLVED-wheres-wally-workshop.ipynb](UNSOLVED-wheres-wally-workshop.ipynb) | Student notebook with implementation prompts. |
| [SOLVED-wheres-wally-workshop.ipynb](SOLVED-wheres-wally-workshop.ipynb) | Completed reference notebook. |
| [yolo-wheres-wally-lecture-slides.pdf](yolo-wheres-wally-lecture-slides.pdf) | Workshop slides. |
| [notebook-dataset/backgrounds/](notebook-dataset/backgrounds/) | 286 background images included locally. |
| [notebook-dataset/waldos/](notebook-dataset/waldos/) | Five transparent Wally PNG cut-outs included locally. |

## What Students Build

- An augmentation and image-generation pipeline with automatically calculated bounding boxes.
- A 5,000-image dataset split into 4,000 training and 1,000 test images, with `labels.csv` files.
- A PyTorch dataset, dataloaders and a CNN producing a location heatmap and four box coordinates.
- An MSE loss, Adam optimiser and training loop; by default, training uses the same 1,000-image subset for 10 epochs.
- Learning curves, IoU summaries, predicted-box overlays and inspection of difficult examples.

The model always predicts one box. The exercises discuss the limitations of images containing zero or multiple Wallys and of transferring from pasted images to real book pages.

## Run It

1. Import the unsolved notebook into Kaggle.
2. Attach the [Waldo Dataset by zekikam](https://www.kaggle.com/datasets/zekikam/waldo-dataset).
3. Check `background_directory` and `waldo_directory` against the attached dataset paths. Generated images and labels are written under `/kaggle/working/generated images`.
4. Enable a GPU if available, then run the cells in order and complete the prompts.

The notebooks use PyTorch, torchvision, Pillow, NumPy and Matplotlib. For local use, set the two input directories to the included `notebook-dataset/backgrounds` and `notebook-dataset/waldos` folders and choose a writable output directory. The generation cell clears and recreates that output directory on each run, so use a dedicated folder for generated images.

## Credits

Part of the Edinburgh AI workshop materials. The notebooks use the Waldo Dataset by zekikam. Please credit Edinburgh AI and retain source dataset attribution when reusing the material.
