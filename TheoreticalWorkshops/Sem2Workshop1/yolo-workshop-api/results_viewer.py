import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import torch

def display_results(model, loader, device, title="Raccoon Detection"):
    """
    Hides all the matplotlib and tensor manipulation boilerplate.
    """
    model.eval()
    # 1. Grab a single batch
    images, targets = next(iter(loader))
    images, targets = images.to(device), targets.to(device)

    # 2. Get predictions
    with torch.no_grad():
        outputs = model(images)

    # 3. Choose the first image in the batch to plot
    image_tensor = images[0]
    pred_box = outputs[0]
    target_box = targets[0]

    # --- Internal Plotting Logic (The part you want to mask) ---
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    img = inv_normalize(image_tensor.cpu()).permute(1, 2, 0).numpy()
    img = img.clip(0, 1) 

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)
    h, w, _ = img.shape

    # Helper for drawing boxes
    def add_box(box, color, label):
        cx, cy, bw, bh = box[1].item() * w, box[2].item() * h, box[3].item() * w, box[4].item() * h
        rect = patches.Rectangle((cx - bw/2, cy - bh/2), bw, bh, 
                                 linewidth=2, edgecolor=color, facecolor='none', label=label)
        ax.add_patch(rect)

    add_box(target_box.cpu(), 'g', 'Ground Truth')
    add_box(pred_box.cpu(), 'r', 'Prediction (Confidence: {:.2f})'.format(pred_box[0].item()))

    plt.title(title)
    plt.legend()
    plt.axis('off')
    plt.show()