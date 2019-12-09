''' Basic discriminator class - just one ff layer to predict language type'''

import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, hidden_dim, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ff_1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.ff_2 = nn.Linear(hidden_dim//2, output_dim)

    def forward(self, x):
        return self.ff_2(F.relu(self.ff_1(x)))
