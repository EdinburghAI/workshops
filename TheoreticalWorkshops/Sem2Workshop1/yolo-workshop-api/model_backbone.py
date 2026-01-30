import torch.nn as nn   # rename the nn module for easier to read code
import pandas as pd

class RaccoonDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # we can re-use the activation function since it doesn't have any weights to learn
        self.activation = nn.ReLU()

        # num out channels determines how many kernels we have (since states how many output feature maps)

        # first layer
        # 3 input channels since RGB image
        # 16 output since want 16 kernels
        self.conv_layer_1 = nn.Conv2d(in_channels=3,out_channels=16, kernel_size=3, padding=1)

        # batch norm for layer 1
        self.layer_1_batch_norm = nn.BatchNorm2d(num_features=16)
        

        self.layer_1_pool = nn.MaxPool2d(kernel_size=2, stride=2)


        # Layer 2: 16 -> 32 kernels (Output: 112x112)
        self.conv_layer_2 = nn.Conv2d(16, 32, 3, padding=1)
        self.layer_2_batch_norm = nn.BatchNorm2d(32)
        self.layer_2_pool = nn.MaxPool2d(2, 2)

        # Layer 3: 32 -> 64 kernels (Output: 56x56)
        self.conv_layer_3 = nn.Conv2d(32, 64, 3, padding=1)
        self.layer_3_batch_norm = nn.BatchNorm2d(64)
        self.layer_3_pool = nn.MaxPool2d(2, 2)

        # Layer 4: 64 -> 128 kernels (Output: 28x28)
        self.conv_layer_4 = nn.Conv2d(64, 128, 3, padding=1)
        self.layer_4_batch_norm = nn.BatchNorm2d(128)
        self.layer_4_pool = nn.MaxPool2d(2, 2)

        # Layer 5: 128 -> 256 kernels (Output: 14x14)
        self.conv_layer_5 = nn.Conv2d(128, 256, 3, padding=1)
        self.layer_5_batch_norm = nn.BatchNorm2d(256)
        self.layer_5_pool = nn.MaxPool2d(2, 2)



        # now we need to define the head of the model
        # The Final Head
        self.flatten = nn.Flatten()
        
        self.head = None


    def forward(self,input_image):
        x = input_image

        # x starts as: [batch_size, 3, 448, 448]

        # Block 1 -> Output: [batch_size, 16, 224, 224]
        x = self.conv_layer_1(x)
        x = self.layer_1_batch_norm(x)
        x = self.activation(x)
        x = self.layer_1_pool(x)

        # Block 2 -> Output: [batch_size, 32, 112, 112]
        x = self.conv_layer_2(x)
        x = self.layer_2_batch_norm(x)
        x = self.activation(x) # You can reuse the same ReLU
        x = self.layer_2_pool(x)

        # Block 3 -> Output: [batch_size, 64, 56, 56]
        x = self.conv_layer_3(x)
        x = self.layer_3_batch_norm(x)
        x = self.activation(x)
        x = self.layer_3_pool(x)

        # Block 4 -> Output: [batch_size, 128, 28, 28]
        x = self.conv_layer_4(x)
        x = self.layer_4_batch_norm(x)
        x = self.activation(x)
        x = self.layer_4_pool(x)

        # Block 5 -> Output: [batch_size, 256, 14, 14]
        x = self.conv_layer_5(x)
        x = self.layer_5_batch_norm(x)
        x = self.activation(x)
        x = self.layer_5_pool(x)

        # --- Transition to Head ---
        
        # Flattens (256, 14, 14) into a single vector of 50,176 values
        x = self.flatten(x) 

        # the head of the model whould be implemented on kaggle 
        if self.head is None:
            raise NotImplementedError("You need to define 'self.head' in your model!")
        return self.head(x)

        