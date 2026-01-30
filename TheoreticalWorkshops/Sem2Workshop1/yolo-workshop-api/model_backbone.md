
# Overview 

This file is for abstracting the code needed for the convolution layers of the model since convolutions were already covered in detail in the semester 1 CNN workshop.



# Building a model class


# Key components

- Must be a subclass of torch.nn.Module
- Must have a constructor that calls the parent constructor and also defines the layers of the model
- Must have a function that defines the forward pass


# Key Things you DO NOT include in the model class

- The training loop
- The loading of the dataset




# The constructor

Within the constructor there are 2 essential components:

- We must call the super-constructor (the constructor of nn.Module)
- We must define the layers of the network

Additional components we can add:

- [[Glossary#Dropout|Dropout]]
- Activation functions (e.g. ReLU)
- [[Glossary#Batch Normalization|Batch Normalization]]
- Pooling


**Why we need the superconstructor?**

The superconstructor will instantiate your model as part of the PyTorch framework.
Specifically, it enables:

- **Parameter Tracking:** It creates the internal dictionaries (`_parameters`, `_modules`, and `_buffers`) that PyTorch uses to keep track of every weight and bias in your network.

- **Recursive Sub-module Registration:** When you define a layer as `self.layer1 = nn.Linear(...)`, the `nn.Module` base class automatically detects it and adds its weights to the model’s total parameters.

- **State Management:** It enables the `.to(device)` (moving to GPU) and `.train()/.eval()` methods to work recursively. If you don't call the super-constructor, running `model.to('cuda')` will fail to move your layers to the GPU.




### Defining the Layers

PyTorch provides many types of Layers: https://docs.pytorch.org/docs/stable/nn.html

The main 3 types you may be familiar with are:

- **Linear Layer** 
- **Convolution Layers**
- **Transformer Layers**


If we want to chain layers together (i.e. the output of 1 layer is the input to another and so on so forth) we have 2 main options:

- **nn.Sequential** - useful for defining a block of layers. However, if you want the output of a later layer in the block to go into an earlier layer in the block, this wont work as it literally just computers the layers sequentially (hence why it's called "Sequential"). 
- **Using variables that reference layer objects (within the constructor body)** - can be a bit messier but they give you all of the flexibility that nn.Sequential can't offer.

In practice, it's best to use a combination of both.

==Note: you can put activation functions between the layer variables in nn.Sequential. You can also just define them as objects==




### Example of a Constructor for a model


```
class HybridCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 1. INDIVIDUAL VARIABLES
        # We define these separately because we need to access them 
        # individually in the forward pass to create a "Skip Connection"
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 2. SEQUENTIAL BLOCK
        # We group the "Downsampling" logic into a single block.
        # This only runs if the input size doesn't match the output size.
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # 3. ACTIVATION
        self.relu = nn.ReLU()
```


### Explanation of parameters for Convolution Layers

- **Input channels** - The number of feature map inputs (e.g. for an rgb image it would be 3)
- **Output Channels** - the number of feature map outputs
- **Kernel Size** - the number of rows in each kernel (since kernels are a square, num_rows = num_columns)
- **Stride** - The number of units you move across for each convolution operation
- **Padding** - the number of units of padding per kernel






# The Forward Pass

The forward pass details how the input flows through the model.
i.e. The constructor just defines the components and they can be in any order they like (apart from sequential blocks which will operate in the order of their arguments). It is the forward pass that defines what order they actually go in.



### Code example - simple

You can see we just iterate through the layers we defined in the constructor.

```

def forward(self, x):
        # Save input for the residual connection
        identity = x
        
        # Main Path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Shortcut Path (Dimensions matching)
        identity = self.shortcut(identity)
        
        # Skip Connection & Final Activation
        out += identity
        out = self.relu(out)
        
        return out

```



Here we can use the same variable named out and keep updating it's state since the input goes into all layers regardless. However, in more complex networks you may have to track multiple variables (e.g. [Temporal Fusion Transformer](https://arxiv.org/pdf/1912.09363))


### Shape errors

You need to be very careful with passing tensor sizes between layers.
e.g. if a layer expects a tensor of size `[5,4,2]` then the layer before that layer must output a tensor of that size.

This issue can also occur if you're adding tensors.

There are a bunch of tricks to fixing these sorts of issues but it's really situation dependent



### Skip Connections 

A **Skip Connection** (also known as a **Shortcut Connection** or **Residual Connection**) is a structural tweak where you take the output of one layer and "skip" it forward, adding it directly to the output of a deeper layer.

Instead of the data traveling strictly through every single layer in a straight line, it has a "fast lane" to bypass certain transformations.

==Note: it's the network that learns how often it wants to skip those layers==

*You'll see these crop up in more advanced architectures such as ResNet*