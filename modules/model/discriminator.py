''' Basic discriminator class - just one ff layer to predict language type'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, hidden_dim, output_dim=1, initial_affine_size=-1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.initial_affine_size = initial_affine_size
        if (initial_affine_size > 0):
            self.affine_combo = nn.Linear(initial_affine_size, 1)

        self.ff_1 = nn.Linear(self.hidden_dim, self.hidden_dim //2)
        self.ff_2 = nn.Linear(self.hidden_dim //2, output_dim)

    def forward(self, x):
        if self.initial_affine_size > 0:
            # (batch_size, num_heads, hidden_dim)
            x = torch.squeeze(self.affine_combo(x.transpose(1,2)))
            assert(x.shape[1] == self.hidden_dim and len(x.shape) == 2), "Error in output of affine combination!"
            # output: (batch_size, hidden_dim)

        return self.ff_2(F.relu(self.ff_1(x)))
