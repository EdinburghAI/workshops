
# Where we define the training loop

The training loop should always be the 2nd to last thing that we implement.

It doesn't need it's own class but it relies on both the model and the dataset already being instantiated



# What the training loop involves

The training loop is basically just a for loop that runs for as many [[Glossary#Epoch|epochs]] as you desire.

Its work includes:

- Iterating over the batches in the current epoch
- Computing the output matrix for the current datapoint in the current batch
- computing weight updates for the network and applying them for the next batch



# The flow of the training loop

1. **Fetch** - get the next batch (should be represented as a matrix)
2. **Clear** - Torch stores a buffer of the previous computed gradients. For most purposes we don't need them for the next batch. If we don't clear them we'll have the gradients from the last batch also added onto our weight updates which can cause exploding gradients
3. **Predict (forward pass)** - Run the batch through the model to get the output matrix
4. **Grade (compute loss)** - Calculate the error for the batch we just put through the forward pass
5. **Compute derivatives** - use back propagation to go through the network and compute the direction that the loss should go
6. **Update** - update the weights in the network using the [[Glossary#Optimizer|optimizer]]



# Example code

```
# The epoch loop (e.g., repeating the whole dataset 50 times)
for epoch in range(num_epochs):
    
    # Track performance for this epoch
    epoch_loss = 0.0

	# 1. FETCH
    # The batch loop (the DataLoader shuffles this for us every epoch)
    for inputs, targets in train_loader:
        
        # 2. RESET: Clear the "Report"
        # PyTorch accumulates gradients; we must clear them for every new batch.
        optimizer.zero_grad()
        
        # 3. PROCESS: Forward Pass
        # Pass the whole matrix [BatchSize, 5] through the model.
        outputs = model(inputs)
        
        # 4. EVALUATE: Compute Loss
        # Get a single scalar value representing the average error of the batch.
        loss = criterion(outputs, targets)
        
        # 5. ANALYZE: Backward Pass (The "Inspector")
        # Trace the graph back and write the derivatives into the weight                                                                  mailboxes (.grad).
        loss.backward()
        
        # 6. UPDATE: Optimizer Step (The "Plumber")
        # Read the .grad mailboxes and actually modify the weights in memory.
        optimizer.step()
        
        # LOGGING: Keep a running tally
        # .item() is vital; it prevents the whole graph from staying in RAM.
        epoch_loss += loss.item()

    # End of Epoch Summary
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
```




## Fetch in PyTorch

We generally have a `DataLoader` instance for loading the batches of data.
You can simply just loop over that instance.

e.g. Where train_loader is an instance of DataLoader

```
for inputs, targets in train_loader:
```

In this code snippet, inputs will be your batch ==and will be a PyTorch tensor==


# Clear in PyTorch

We instantiate our optimizer before the training loop but we need to clear it's gradient buffer in out training loop.

The zero_grad function sets all of the gradients within the gradient buffer (inside the optimizer object) to zero. 

```
optimizer.zero_grad()
```


**Why set them to zero?**

By default, when we use our optimizer to update our weights, the gradients in the buffer will be added on to our weight update.

If we set them to zero, we still do the addition but we're adding on nothing, which obviously will have no affect on the weight update.




# Forward Pass in PyTorch

We define [[Defining a model class#The Forward Pass|the forward pass]] implementation within our model class but we only actually invoke the function within our training loop (and for [[Glossary#Inference|inference]]).

However, when we invoke the function we don't actually do it the normal way we would with other objects.

e.g. if when we instantiate the model we do:

```
my_model = MyModelClass()
```

We apply the forward pass by doing:

```
output_forward_pass = my_model(input_tensor)
# input tensor is just a placeholder variable for the iterator variable you use for your batches
```


**Why is this the case?**

If we instead did

```
output_forward_pass = my_model.forward(input_tensor)
```

we can technically still do it this way, however, if you want to access the [[Computational Training Graphs]] feature (+ some other important PyTorch stuff), if you just used .forwards() the framework for all of this wouldn't actually be instantiated.

This is because under the hood in the nn.Module class, PyTorch uses a `__call__()` function wrapper for child implementations of the nn.Module class in order to instantiate the PyTorch framework required for training graphs. This `__call__()` function is only invoked if you do the `my_model(input_tensor)`way.


