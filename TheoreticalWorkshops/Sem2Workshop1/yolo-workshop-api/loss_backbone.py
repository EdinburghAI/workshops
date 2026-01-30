import torch
import torch.nn as nn

class loss_backbone(nn.Module):
    def __init__(self, confidence_weight, coord_weight, loss_logic_fn):
        super().__init__()
        self.confidence_weight = confidence_weight
        self.coord_weight = coord_weight
        # We store the student's function here
        self.loss_logic_fn = loss_logic_fn
        self.mse = nn.MSELoss(reduction='none')


    def forward(self, predictions, targets):
        # Now the forward pass knows exactly which function to use
        return self.loss_logic_fn(predictions, targets, self.confidence_weight, self.coord_weight, self.mse)