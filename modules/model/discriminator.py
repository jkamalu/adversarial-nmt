''' Basic discriminator class - just one ff layer to predict language type'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    
    def __init__(self, regularization, discriminator_kwargs):
        super().__init__()

        self.regularization = regularization

        if regularization == "hidden":
            hidden_dim = discriminator_kwargs["d_model"]
        else:
            raise ValueError("Discriminator must be initialized with \'hidden\' regularization.")

        self.hidden_dim = hidden_dim
        self.output_dim = discriminator_kwargs["output_dim"]

        self.ff_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ff_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ff_3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ff_out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        """
        Accepts
            x: (batch size, num attention heads, hidden dim)
        Returns
            o: (batch size, hidden_dim)
        """
        x = torch.sum(x, dim=1)
        o = F.leaky_relu(self.ff_1(x))
        o = F.leaky_relu(self.ff_2(o))
        o = F.leaky_relu(self.ff_3(o))
        o = self.ff_out(o)
        return o
