
# The library in Torch that handles this

The autograd library in PyTorch handles all of the implementations of functions necessary for building the computational graphs and giving them to the back-propagation elements of your code.

It's very complex under the hood but the documentation is here:
https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html


# What are they in torch?

Anytime you invoke the forward() function, torch builds a tree (represented by a graph) representing the operations executed during that forward pass.

As your code executes line-by-line in the forward method, PyTorch looks for any Tensors that have requires_grad=True (attributes with this value are essentially just tensors containing weights but there are a few other cases. When it sees an operation (like an addition or a convolution) involving these tensors, it creates a Function Node in a [[Computational Training Graphs#Directed Acyclic Graphs|directed acyclic graph]].

The nodes are an instance of the class `torch.autograd.Function`.

Using these nodes , torch builds a directed acyclic graph (a bit like a linked list). This means that when you're training models using the backwards function, the backwards function can just look at the tree and see e.g. this layer and this hidden state caused this state which went into another layer and caused this state. It can then apply the chain rule to this linked list in reverse to calculate the weight updates for all weights in the network.

==The computational tree is always destroyed after a backwards function is invoked and finished its work==


**Example of a computational tree**

`Layer 1 Weights` → `Hidden State A` → `Layer 2 Weights` → `Hidden State B`





### Hidden states (Hidden State vectors) 

Hidden state vectors are just the output of a given layer after an activation function has been applied to the output.

e.g.  
$$\text{Hidden State} = ReLU(\omega x + b)$$






# Directed Acyclic Graphs

They are just directed graphs that have a couple extra rules:

- The graph must be acyclic - No paths lead back to the starting node; you can't follow arrows and end up where you began (e.g., no X → Y → X).
- The relationships between connected nodes must be causal - the child node's state must depend on its parent node's state.


![[Pasted image 20260118163626.png]]



## How torch implements the graphs

- **Nodes** are **Operations** (Functions).
- **Edges** are **Tensors** (Data/States).
- **Nodes store pointers** to their parents, creating the "graph" structure.


### The implementation of the nodes in PyTorch

The goal of the nodes is essentially just to tell the backwards function what operations caused what states.

**Nodes store pointers** to their parents, creating the "graph" structure.

If you were to inspect a node in your IDE, it would look like this:

```
Node (MulBackward)
├── next_functions: [Pointer to x's creator, Pointer to w's creator]
├── saved_tensors: [x, w]  <-- This is why memory usage is high!
└── backward_logic: "grad_output * other_tensor"
```

We need to store what operations caused the given values which is why we store information on the backward logic. e.g. was this tensor multiplied by the previous hidden state tensor or was it added (this can affect the weight updates).



#### Example tree of nodes

For example, if we have a simple multi-layer perceptron consisting of:

- 1 input layer with 5 neurons
- 1 hidden layer with 8 neurons
- 1 output layer with 4 neurons

*Torch would store the nodes:*


**1. The Input → Hidden Transformation**

- **The Node:** `AddmmBackward` (Addition + Matrix Multiplication).
- **What it represents:** The math that calculates the 8 hidden neurons from the 5 input neurons.   
- **Saved State:** It captures the **Input Tensor** (1×5) and the **Weight Matrix** (5×8).

#### 2. The Activation Function

- **The Node:** `ReluBackward`.
- **What it represents:** The logic that "fires" the neurons (turning negative values to zero).
- **Saved State:** It saves the **result** of the previous math (the 1×8 vector) so it knows where the zeros are during back-propagation.

#### 3. The Hidden → Output Transformation

- **The Node:** Another `AddmmBackward`.
- **What it represents:** Calculating the 4 output neurons from the 8 hidden ones.
- **Saved State:** It saves the **Hidden State** vector (1×8) and the **Output Weight Matrix** (8×4).
- 