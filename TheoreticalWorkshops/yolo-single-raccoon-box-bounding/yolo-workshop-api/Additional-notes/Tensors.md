# Overview

Tensors are the fundamental building block of data in PyTorch (and almost all Deep Learning frameworks).

At their core, **a Tensor is just a container for numbers**, very similar to a NumPy array. However, they are designed with specific features that make them suitable for training deep neural networks.

# The Hierarchy of Dimensions

You can think of Tensors as a generalization of matrices to $N$-dimensions.

- **Rank 0 (Scalar):** A single number (e.g., loss value). Shape: `[]`
- **Rank 1 (Vector):** A 1D array (e.g., the output of a classification layer). Shape: `[5]`
- **Rank 2 (Matrix):** A 2D grid (e.g., a grayscale image). Shape: `[224, 224]`
- **Rank 3 (3D Tensor):** A volume (e.g., a color RGB image). Shape: `[3, 224, 224]`
- **Rank 4 (4D Tensor):** A batch of images. Shape: `[32, 3, 224, 224]`

# Why use Tensors instead of NumPy Arrays?

If they are just containers for numbers, why don't we just use `numpy.array`?

### 1. GPU Acceleration
NumPy arrays only live on the CPU. Tensors can be moved to the **GPU (Graphics Processing Unit)**.
Matrix multiplication on a GPU is significantly faster (often 50x-100x) than on a CPU because GPUs are designed to do thousands of tiny calculations in parallel.

```
# Moving a tensor to the GPU
x = torch.tensor([1, 2, 3])
x = x.to('cuda')
```

### 2. Autograd (Automatic Differentiation)

This is the "Magic" of PyTorch. A Tensor can keep track of every operation performed on it. This creates a computational graph, allowing PyTorch to calculate derivatives (gradients) automatically during backpropagation.

- **Without Tensors:** You have to manually calculate the calculus for `dW` and `db`.

- **With Tensors:** You just call `loss.backward()`.


# Important Attributes

When debugging, 90% of your errors will be related to mismatched attributes. Always check these three:

1. **`.shape`**: The dimensions.

    - _Error:_ `RuntimeError: size mismatch`
    
2. **`.dtype`**: The data type (e.g., `float32`, `int64`).

    - _Note:_ Neural network weights usually require `float32`. Labels usually require `long` (int64).
    
3. **`.device`**: Where the data lives (`cpu` or `cuda:0`).

    - _Error:_ `RuntimeError: Expected all tensors to be on the same device`





# What do the Units in a Tensor mean?

When you look at a tensor's shape (e.g., `[32, 3, 224, 224]`) or print its values, it can be abstract. Here is what those numbers actually represent in the physical world of Computer Vision.

## 1. The Dimensions (The Shape)

In PyTorch, image tensors are almost always **Rank 4**. The standard layout is **NCHW**:

**Shape:** `[Batch_Size, Channels, Height, Width]`



- **N (Batch Size):** How many images are you processing at once?
    - *Example:* `32` means you are feeding 32 separate photos into the model simultaneously.
- **C (Channels):** The color depth.
    - `3` = Red, Green, Blue (RGB).
    - `1` = Grayscale.
- **H (Height):** Vertical resolution in pixels.
- **W (Width):** Horizontal resolution in pixels.

*Note: TensorFlow/Keras often uses **NHWC** (Height/Width first, Channels last). PyTorch is strictly Channels first.*



## 2. The Values (The Data inside)

What do the actual floating-point numbers inside the tensor measure?

### A. Input Tensors (Images)
**Unit:** Pixel Intensity.

- **Raw Data (Int8):** usually `0` (Black) to `255` (Full Color).
- **Normalized (Float32):** usually `0.0` (Black) to `1.0` (Full Color).
    - *Why?* Neural networks math works better with small numbers between 0 and 1.
- **Standardized:** sometimes `-1.0` to `1.0`.
    - *Why?* This centers the data around zero (mean subtraction), which helps gradients flow better.

### B. Output Tensors (YOLO Predictions)
**Unit:** Relative Ratio (Normalized).

If your model outputs a bounding box `[x, y, w, h]`, these are rarely pixels. They are usually **normalized to the image size**.

- **x = 0.5:** The center of the box is at 50% of the image width.
- **w = 0.2:** The box is 20% as wide as the total image.

**To convert back to "Pixels":**
$$\text{Pixel } x = \text{Normalized } x \times \text{Image Width}$$
$$\text{Pixel } w = \text{Normalized } w \times \text{Image Width}$$

### C. Confidence / Class Scores
**Unit:** Probability (or Logits).

- **After Sigmoid/Softmax:** The unit is "Probability" (0.0 to 1.0).
    - `0.95` = "I am 95% sure."
- **Before Activation (Logits):** The unit is "Raw Score" ($-\infty$ to $+\infty$).
    - These numbers are hard for humans to interpret until passed through an activation function.