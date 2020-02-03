#!/usr/bin/env python
# coding: utf-8

# ## Audio Conv2D Model

import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict


class AudioConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        HIDDEN_SIZES = [25, 50, 46*50]

        self.conv1 = nn.Conv2d(1, HIDDEN_SIZES[0], 3)
        self.conv2 = nn.Conv2d(HIDDEN_SIZES[0], HIDDEN_SIZES[1], 3)  
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(HIDDEN_SIZES[2], HIDDEN_SIZES[2])
    
    def forward(self, x):
        # x : (N, T:seq_len, F:num_features)
        x = x.unsqueeze(1) # -> (N, 1, T, F)
        x = F.relu(self.conv1(x)) # -> (N, C1, T-2, F-2)
        x = F.relu(self.conv2(x)) # -> (N, C2, 1, F-4)
        x = self.flatten(x) # -> (N, C2*F-4)
        x = F.relu(self.fc1(x)) # -> (N, C2*F-4)
        return x
