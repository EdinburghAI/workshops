

# Epoch

Epochs are a unit of how many times you pass in the entire dataset to the training loop.

e.g. 1 epoch represents passing the entire dataset through the model once, 2 epochs represents passing the entire dataset through the model twice, etc.



# Optimizer

Optimizers are just the functions that define how the weights in a network should be updated.

e.g. For stochastic Gradient Descent

new_weight = weight − (learning_rate × gradient)



# Stochastic Gradient Descent (SGD)

Stochastic Gradient Descent is an example of an optimizer that attempts to move closer to the minimum of the error function created by the current batch.

It has the weight update rule:

new_weight = weight − (learning_rate × gradient)


# Learning Rate 

*Learning Rate is an example of a [[Glossary#Hyper parameters|hyper parameter]]*

The learning rate (often denoted by $\eta$) allows you to have some control over how sharp the weight updates are. In plain English "It allows you to control how fast the network learns".

It is simply just a parameter that is usually defined by whoever creates the model (although in some advanced networks the model may learn the learning rate)

- A higher learning rate means big weight updates 
- A lower learning rate means smaller weight updates


**What is a good learning rate?**

It depends highly on the type of model, size of model and [[Glossary#Optimizer|optimizer]] you're using but its common to use a learning rate on the interval **0.01 to 0.0001**.




# VRAM (Video Random Access Memory)

VRAM is your gpu's internal random access memory.
You can think of it as being like your regular DRAM but just specialized for your GPU.

Only your GPU can compute on vram.

You can technically run your models using virtual memory, however, the speed difference between DRAM and VRAM for machine learning is immense and most practitioners prefer their models to instead crash if they run out of vram (instead of using some addresses in main memory)

*Note: e.g. Apple don't actually use VRAM, they use a unified memory structure across the entire machine which gives them very very high throughput since they can use the normal memory hierarchy. However, if your using an NVIDIA gpu, you will be using VRAM*



# GPU (Graphical Processing Unit)

The GPU is usually used for processing all of the graphics on your screen.

However, lots of the maths required for graphical processing is very similar to machine learning maths in the sense that they both use lots of matrix multiplications in parallel. 
As a result, the gpu is highly specialized for matrix multiplications and parallel processing.
AI researchers noticed this and started offloading computation for neural networks to the gpu as it's way more efficient.

*Note: Google recently developed a Tensor Processing Unit (TPU) but it's quite proprietary and specialized for google infrastructure. The TPU is purely for machine learning.* 


**Some benchmarks**

Llama 3 (a Large Language Model developed by Meta) was trained on 24,000 GPU's.
If we were to train it on other machinery, this would've been the outcome:

![[Pasted image 20260118190621.png]]




# Inference

In the context of machine learning, **inference** is the phase where you put a trained model to work.

If training is the "learning" phase, inference is the "testing" or "application" phase. It is the process of passing new, unseen data into a model to get a prediction or a result.



# Hyper parameters

In machine learning, **hyperparameters** are the external "settings" or "knobs" that you configure **before** the training process begins.

Unlike standard **parameters** (the weights and biases), which the model learns on its own during training, hyperparameters are chosen by _you_, the engineer. They control the overall behavior of the learning algorithm and determine how the model is structured.

e.g

- learning rate
- Batch Size
- Epochs
- Hidden Layers
- Dropout Rate




# Dropout

Dropout is a regularization technique used in deep learning to prevent overfitting.

The computer randomly selects random neurons to not fire during each forward pass. On the next iteration, a **different** random set of neurons is dropped.

**Why use it?**

Dropout solves two major problems:

- **Co-adaptation:** Without dropout, neurons often become highly dependent on each other, "fixing" each other's mistakes rather than learning the actual features of the data. Dropout breaks these dependencies.

- **The "Ensemble" Effect:** By training on different random sub-sections of the network every time, you are essentially training thousands of smaller, different models. When you turn dropout off at the end, it’s like all those small models are "voting" together, which usually leads to a much more accurate result.


==Note: You should only ever use dropout during training, never during inference as during inference you want the full power of the network==



# Batch Normalization

**Batch Normalization** (or "Batch Norm") is a technique used to make neural networks faster and more stable by **re-centering and re-scaling** the inputs to each layer.

**How it Works**

For every batch of data passing through a layer, Batch Norm does the following:

1. **Calculate Mean and Variance:** It finds the average value of the current batch.
2. **Normalize:** It subtracts the mean and divides by the standard deviation. This forces the data to have a mean of 0 and a variance of 1.
3. **Scale and Shift:** It applies two "learnable" parameters (γ and β). This allows the model to decide if it _wants_ the data to be centered at 0, or if it would actually learn better if the data were shifted slightly elsewhere.
