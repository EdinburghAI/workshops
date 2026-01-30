import pandas as pd
import torch
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms

class RaccoonDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.root_dir_images= "/kaggle/input/racoon-detection/Racoon Images/images"
        self.transform = transform
        

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        # extract labels
        row = self.df.iloc[index]

        label_features = [1, row['normalized_center_x'], row['normalized_center_y'], row['normalized_box_width'], row['normalized_box_height']]
        label_tensor = torch.tensor(label_features, dtype=torch.float32)

        # extract tensor for image
        img_path = os.path.join(self.root_dir_images, row['filename'])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image) # Image becomes a Tensor here
        
        return image, label_tensor
    