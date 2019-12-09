''' Basic discriminator class - just one ff layer to predict language type'''

import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, hidden_dim, output_dim=1, initial_affine_size=-1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.initial_affine_size = initial_affine_size
        if (initial_affine_size > 1):
            self.affine_combo = nn.Linear(hidden_dim, hidden_dim//initial_affine_size)
            self.hidden_dim = self.hidden_dim//initial_affine_size
            self.batch_norm = nn.BatchNorm1d(hidden_dim)

        self.ff_1 = nn.Linear(self.hidden_dim, self.hidden_dim //2)
        self.ff_2 = nn.Linear(self.hidden_dim //2, output_dim)

    def forward(self, x):
        if self.initial_affine_size > 1:
            x = self.batch_norm(self.affine_combo(x))

        return self.ff_2(F.relu(self.ff_1(x)))
