# Single-Object YOLO Workshop Helpers

This folder contains Python helper implementations and explanatory notes for the [single-object raccoon detection workshop](../README.md). The helpers provide the longer dataset, convolutional-backbone, loss-wrapper, evaluation and plotting routines so students can focus on the detector head, loss logic and training loop. They are imported as Python modules rather than run as a web service.

## Files and Notes

| File | Use |
| --- | --- |
| [dataset_backbone.py](dataset_backbone.py) | `RaccoonDataset`: loads images and five-element confidence/box targets. |
| [dataset_backbone.md](dataset_backbone.md) | Dataset implementation notes. |
| [model_backbone.py](model_backbone.py) | Convolutional feature extractor extended by the notebook's model head. |
| [model_backbone.md](model_backbone.md) | Model implementation notes. |
| [loss_backbone.py](loss_backbone.py) | Wrapper applying the student's confidence and coordinate loss function. |
| [loss_function.md](loss_function.md) | Loss explanation. |
| [intersection_over_union.py](intersection_over_union.py) | Bounding-box conversion and batched IoU calculation. |
| [intersection_over_union.md](intersection_over_union.md) | IoU explanation. |
| [results_viewer.py](results_viewer.py) | Predicted and true bounding-box plots. |
| [config.yaml](config.yaml) | Training, image-size, loss-weight and evaluation settings. |
| [Overall-main-file-structure.md](Overall-main-file-structure.md) | Supporting file-structure notes. |
| [single-obj-yolo-presentation.pptx](single-obj-yolo-presentation.pptx) | Workshop presentation. |

## Additional Guides

- [Tensors](Additional-notes/Tensors.md)
- [Batching](Additional-notes/Batching.md)
- [Training Loops in PyTorch](Additional-notes/Training%20Loops%20in%20PyTorch.md)
- [Computational Training Graphs](Additional-notes/Computational%20Training%20Graphs.md)
- [Glossary](Additional-notes/Glossary.md)

## Using the Helpers

The main unsolved and final solved notebooks download these five Python modules and `config.yaml` into their working directory. For offline use, copy them from this folder into the notebook's working directory before importing them.

The code uses PyTorch, torchvision, pandas, Pillow and Matplotlib; the notebook also uses scikit-learn and PyYAML. `RaccoonDataset` hard-codes the Kaggle image root `/kaggle/input/racoon-detection/Racoon Images/images`. A local run needs that path adapted as well as the notebook's CSV and configuration paths. The external raccoon images and labels are not bundled here.

The earlier solved beta notebook uses a separate helper repository and configuration dataset. Follow the [main workshop README](../README.md) for the current notebook pair and dataset setup.

## Credits

Created for Edinburgh AI workshops. Please credit Edinburgh AI if you reuse or adapt the material.
