
# Overview

This file contains functionality for using the intersection over union metric for YOLO model evaluation.



# Note

In these notes, we visualize the origin as the **Bottom-Left** (like standard maths graphs). However, computer images actually use the **Top-Left** as the origin $(0,0)$.


**Does this change the math?**

- **Conceptually:** No. We are still finding the overlapping area.

- **Implementation:** Yes. In images, "Top" means a smaller Y value.


**The Golden Rule:** Regardless of the coordinate system, always calculate intersection based on **Start** (smaller number) and **End** (larger number):

- Intersection **Start** (Left/Top in images) = `max(start1, start2)`

- Intersection **End** (Right/Bottom in images) = `min(end1, end2)`



# Intersection over union (iou)

Intersection over union gives us a metric for quantifying how close our predicted box is to the one labeled for the datapoint.

It is computed literally as it is named: we divide the area where the truth box and predicted box overlap by the total area covered by the boxes


$$\frac{\text{Area of intersection}}{\text{Area covered total}}$$



## What this looks like


![[Pasted image 20260130204808.png]]



## How do we get the coordinates for the box of intersection?

The leftmost vertical edge will be the **bigger** x coordinate between these 2:

- The left edge of our predicted box
- The left edge the label box

The rightmost vertical edge will be the **smaller** x coordinate between these 2:

- The right edge of our predicted box
- The right edge the label box


The top of the intersection box will be the the **smaller** y coordinate between these 2:

- The top edge of our predicted box
- The top edge of the label box


The bottom of the intersection box will be the **bigger** y coordinate between these 2:

- The bottom edge of our predicted box
- The bottom edge of the label box




- **xstart​** = The further right of the two left edges →max(x1A​,x1B​)
    
- **ystart​** = The lower of the two top edges →max(y1A​,y1B​)
    
- **xend​** = The further left of the two right edges →min(x2A​,x2B​)
    
- **yend​** = The higher of the two bottom edges →min(y2A​,y2B​)



### Using the box to get the intersection

The idea is to compute the area of this intersection box, however, we need to watch out for cases where the area of intersection is zero.

With our current coordinate logic, if we have e.g.

If Box A is at $x=0$ to $10$, and Box B is at $x=20$ to $30$:

- `x_start` = 20

- `x_end` = 10

- `width` = `10 - 20` = **-10**

Your area will be negative, which will skew your Union calculation and ruin your training.

Therefore we need to do what's called clamping


#### Clamping

Clamping allows us to detect instances where the boxes don't overlap and stop them from returning a negative area.

We define the rules:

- **The Width Rule:** If `x_start` (intersection left) is greater than `x_end` (intersection right), it means the boxes do not overlap horizontally. The width should be **0**, not a negative number.

- **The Height Rule:** If `y_start` (intersection top) is greater than `y_end` (intersection bottom), it means the boxes do not overlap vertically. The height should be **0**.


**Mathematical Implementation:**

Instead of just subtracting, we compare the result against 0 and take the larger value.

$$\text{Intersection Width} = \max(0, x_{end} - x_{start})$$

$$\text{Intersection Height} = \max(0, y_{end} - y_{start})$$

**Why this works:**

If the boxes don't overlap in _either_ direction (width or height becomes 0), the total area calculation becomes `0 * something`, which results in a **0** intersection area. This correctly reflects that there is no overlap.




# Getting the Union

We need to use the inclusion exclusion principle as we don't want to double count the intersection area.

$$\text{Union Area} = \text{Area}_{\text{pred}} + \text{Area}_{\text{label}} - \text{Area}_{\text{intersection}}$$

#### Preventing division by zero

In the event that our union area is zero, we'd have division by zero.

We can prevent this by adding a tiny value (known as an epsilon value) to the Union area, just to make sure it's non-zero.

This gives us:

$$\text{IoU} = \frac{\text{Area}_{\text{inter}}}{\text{Area}_{\text{union}} + 1e^{-7}}$$




# What do we do with the intersection over union values?

We want to define a threshold value where if the intersection over union value is over this threshold, we count our predicted box as close enough.

Our threshold is a hyperparameter and should be config.yaml file.

*Note: the intersection over union score is on the interval `[0,1]`*


**What is a good threshold?**

The universal baseline tends to be 0.5, but it can be project dependent






# The iou_batch function

iou_batch takes a batch of 1d prediction and label tensors at a time and computes the intersection over union.

We do this in batches to allow the GPU to do this work in parallel