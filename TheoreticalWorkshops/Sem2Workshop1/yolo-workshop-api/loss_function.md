
# Overview

This workshop uses a simplified loss function since we're only detecting 1 object per image.

We use weighted Mean Square Error (MSE) for the confidence loss and something a bit non-trivial for the coordinate loss.

You should keep batching in mind when reading this file.


  
# Confidence Loss

*Weighted confidence loss is more important for multi-object detection but we'll look at it a bit here.*

The confidence loss is just the measuring the distance between the confidence score that the model predicted and the confidence score that it should've predicted.

We define this with the function for a single datapoint:

$\text{Confidence loss} = (Target−Prediction)^2$




#### Doing this for the batch

When we compute loss we often do it for the entire batch, not just per datapoint. Once again, this is mainly done for parallel computation purposes.

We want the average loss over the batch which ends up just giving us the MSE formula:

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (target - Prediction)^2$$

*Where n = the number of items in the batch*


# Coordinate Loss

We only apply the weighted coordinate loss on examples where there should be an object in that square (i.e. when the labelled confidence score is 1).
This is because it's not very valuable punishing loss for the size of a box which shouldn't exist. In this scenario we instead just punish the model for saying that there should be a box when there shouldn't be one.


**We spit the coordinate loss into 2 parts:**

- The positional loss for the center of the box prediction
- The loss for the prediction of the width and height of the box


**Positional Loss**

We simply just take the sum of the squares of the prediction errors like normal:

$$(x_{true} - x_{pred})^2 + (y_{true} - y_{pred})^2$$


**Height and Width loss**

We have to treat this a little differently since e.g. a 10 pixel loss means a lot more for a small box than it does for a big box.

Therefore, we first square root the pixel sizes before taking the sum of the squares of the errors:

$$(\sqrt{w_{true}} - \sqrt{w_{pred}})^2 + (\sqrt{h_{true}} - \sqrt{h_{pred}})^2$$

This makes it so that the loss will be scaled depending on how big the box is.





**Combining to get the total coordinate loss**

We simply add together the positional coordinate loss and the height and width coordinate loss to get the total coordinate loss:

$$L_{coord} = [(x - \hat{x})^2 + (y - \hat{y})^2] + [(\sqrt{w} - \sqrt{\hat{w}})^2 + (\sqrt{h} - \sqrt{\hat{h}})^2]$$



## Weighted Loss

**The Problem:**

It is often easier for the model to learn the Confidence score than the exact Coordinates.
*Neural networks can be really lazy*

If we treat these losses equally, the model might settle for a "good enough" box (e.g., loosely around the raccoon) just to maximize its Confidence score. The gradients from the Coordinate loss aren't strong enough to force the model to be pixel-perfect.

**The Solution: $\lambda_{coord}$**

We introduce a weight to strictly penalize sloppy bounding boxes. We multiply the coordinate loss by a factor (usually **5**).

$$Loss_{total} = \lambda_{coord} \times Loss_{coordinates} + Loss_{confidence}$$

- **$\lambda_{coord} = 5$:** "I care 5 times more about the box being tight than I do about the confidence score."





# Combining all of this to get the full loss function

The Total Loss function is just the sum of the coordinate loss and the confidence loss functions.


**For a single data point:**

Code snippet


$$\text{Total loss} = \text{coordinate loss} + \text{confidence loss}$$
$$= (\hat{c} - c)^2 + \lambda_{coord} \left[ (x - \hat{x})^2 + (y - \hat{y})^2 + (\sqrt{w} - \sqrt{\hat{w}})^2 + (\sqrt{h} - \sqrt{\hat{h}})^2 \right]$$

**For a batch of size n:**

$$\text{Total loss} = \frac{1}{n} \sum_{i=1}^{n} \left( (\hat{c}_i - c_i)^2 + \lambda_{coord} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right] \right)$$



