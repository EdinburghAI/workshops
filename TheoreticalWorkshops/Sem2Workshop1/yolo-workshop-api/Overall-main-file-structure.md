
1. **Parse your .yaml file**
2. **Hardware Selection** (Detect and define `device` as CPU or GPU/MPS)
3. **Split the dataset into 3 sub-datasets** - training, validation, testing
4. **Load the data** (Instantiate the `Dataset` class for the 3 sub-datasets)
5. **Instantiate a DataLoader object for the 3 sub datasets** (Handle batching, shuffling, and workers)
6. **Instantiate the model** (And move it `.to(device)`)
7. **Instantiate your loss object** - it can be custom or one that comes with torch
8. **Instantiate the optimizer object** (Register model parameters and set learning rate)
9. **Training Loop** (The nested Epoch/Batch loops with the 5-step update process)
10. **Inference and evaluation** (Testing performance using `model.eval()` and `torch.no_grad()`)
11. **Persistence** (Saving the `state_dict` to a file for future use)
