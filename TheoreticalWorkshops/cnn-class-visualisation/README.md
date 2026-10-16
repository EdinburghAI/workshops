# CNN Class Visualisation

This workshop explores what a pretrained CNN responds to by optimising image pixels while keeping the network's weights frozen. Students use a ResNet50 pretrained on ImageNet to compare naive activation maximisation with regularised grayscale visualisations and inspect intermediate layers.

## Notebooks and Materials

| File | Use |
| --- | --- |
| [class_visualiser.ipynb](class_visualiser.ipynb) | Student notebook with implementation prompts. |
| [class_visualiser_solutions.ipynb.ipynb](class_visualiser_solutions.ipynb.ipynb) | Completed reference notebook; the double `.ipynb` extension is the actual filename. |
| [cnn-class-visualisation-lecture-slides.pdf](cnn-class-visualisation-lecture-slides.pdf) | Workshop slides. |

## What Students Build

- A naive visualiser that updates an input image to increase a chosen class score.
- A loss combining activation strength, pixel intensity and spatial smoothness.
- A grayscale image parameterisation and optimisation over increasing image sizes.
- Forward hooks for comparing a final class response with intermediate-layer channels, using random image transformations.

Familiarity with CNNs and backpropagation is useful; see [Computer Vision with CNNs](../intro-to-convolutional-neural-networks/README.md).

## Run It

1. Open the student notebook in Kaggle or a local Jupyter environment.
2. Provide PyTorch, torchvision, Matplotlib and NumPy.
3. Enable internet access for torchvision to download the pretrained ResNet50 weights. No separate training dataset is required.
4. Run the cells in order and complete the prompts, comparing the generated images with the reference notebook.

The notebook selects CUDA when available and otherwise uses the CPU. A GPU helps with repeated image-optimisation steps.

## Credits

Part of the Edinburgh AI workshop materials. Please credit Edinburgh AI if you reuse or adapt the material.
