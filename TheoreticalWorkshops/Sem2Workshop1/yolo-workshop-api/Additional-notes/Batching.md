
# What is it?

Batching allows us to move groups of datapoints into the gpu at the same time, allowing for parallel computation.

e.g. if we have a dataset with 128 data points, we could split it into 4 batches of 32.

We usually update the weights after each batch

## Making sure batches are different

For every [[Glossary#Epoch|epoch]], we should make sure the batches are different. e.g. for a dataset with 8 rows and batch size 2, epoch 1 we might have

- Batch 1 = data points 1 and 2
- Batch 2 = data points 3 and 4
- Batch 3 = data points 5 and 6
- Batch 4 = data points 7 and 8

It is very important that we batch the data differently for epoch 2: e.g. we should go for something like:

- Batch 1 = data points 1 and 7
- Batch 2 = data points 3 and 5
- Batch 3 = data points 2 and 4
- Batch 4 = data points 8 and 6

and so on so forth for the rest of the epochs.

**Why must we do this?**

The model may start learning that there's a correlation between groups of datapoints, instead of the actual relationship those datapoints have in the dataset as a whole.

==neural networks can be very lazy==

# Implications for error computation

When we use batch processing, if we're computing our error using mean squared error (mse), we take the mse over the batch instead of the entire dataset.

Notice that a nice side affect of this is that since we're taking the mean over the batch, we dont have to worry about

# Implications for computational training graphs

If we were to follow the same approach of storing vectors within the nodes we could struggle when batching. e.g. if we have a batch of 32, we would have to create and store 32 graphs in vram.

**A better idea**

We can instead store matrices at each node.

e.g. node 1 contains information on the transformation of the matrix of inputs, as opposed to just a singular input.

This means we only need to store one tree in memory, no matter the batch size.
