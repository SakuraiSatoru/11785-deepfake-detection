#!/usr/bin/env python
# coding: utf-8

# ## Multi-Modal Model
# Densenet + AudioConv2d

# In[1]:


import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict

import importlib

import dataloader
from audio_conv2d import AudioConv2d
from Densenet import densenet169


# In[2]:


class MultiModalSimpleConcatModel(nn.Module):
    HIDDEN = [1664+2300, 1664+2300, 1664+2300]
    def __init__(self):
        super(MultiModalSimpleConcatModel, self).__init__()
        self.densenet = densenet169(pretrained=True, progress=True, memory_efficient=False)
        self.audionet = AudioConv2d()
        self.linears = nn.Sequential(nn.Linear(self.HIDDEN[0], self.HIDDEN[1]),
                                    nn.Linear(self.HIDDEN[1], self.HIDDEN[2]))
        self.scoring = nn.Linear(self.HIDDEN[-1], 2)
    
    def forward(self, videos, audios):
        batch_size = videos.shape[0]
        assert(videos.shape == (batch_size, 3, 224, 224) and audios.shape == (batch_size, 5, 50))
        video_embed, _ = self.densenet(videos) # (N, C, H, W) -> (N, num_features:1664)
        audio_embed = self.audionet(audios) # -> (N, num_features:2300)
        concat = torch.cat((video_embed, audio_embed), dim=1) # (N, 1664+2300)
        
        final_embed = self.linears(concat) # (N, 1664+2300)
        classification = self.scoring(final_embed) # (N, 2)
        return final_embed, classification 

class MultiModalSimpleConcatDropoutModel(nn.Module):
    HIDDEN = [1664+2300, 1664+2300]
    def __init__(self):
        super(MultiModalSimpleConcatDropoutModel, self).__init__()
        self.densenet = densenet169(pretrained=True, progress=True, memory_efficient=False)
        self.audionet = AudioConv2d()
        self.audio_dropout = nn.Dropout(p=0.3)
        self.video_dropout = nn.Dropout(p=0.3)
        self.linears = nn.Sequential(nn.Linear(self.HIDDEN[0], self.HIDDEN[1]), nn.ReLU6())
        self.scoring = nn.Linear(self.HIDDEN[-1], 2)
    
    def forward(self, videos, audios):
        batch_size = videos.shape[0]
        assert(videos.shape == (batch_size, 3, 224, 224) and audios.shape == (batch_size, 5, 50))
        video_embed, _ = self.densenet(videos) # (N, C, H, W) -> (N, num_features:1664)
        video_embed = self.video_dropout(video_embed)
        audio_embed = self.audionet(audios) # -> (N, num_features:2300)
        audio_embed = self.audio_dropout(audio_embed)
        concat = torch.cat((video_embed, audio_embed), dim=1) # (N, 1664+2300)
        
        final_embed = self.linears(concat) # (N, 1664+2300)
        classification = self.scoring(final_embed) # (N, 2)
        return final_embed, classification 


# In[3]:


def load_data():
    BATCH_SIZE = 64

    train_video_dataset = dataloader.get_dataset(dataloader.TRAIN_JSON_PATH, dataloader.SINGLE_FRAME)
    train_audio_dataset = dataloader.AudioDataset()
    train_loader = dataloader.AVDataLoader(train_video_dataset, train_audio_dataset, batch_size=BATCH_SIZE, shuffle=True, single_frame=True)
    return train_loader


# In[4]:


def verify_model(train_loader):
    model = MultiModalSimpleConcatModel()

    for v, a, _, _, l in train_loader:
        print('videos shape:', v.shape) # batch_size*3(channel)*224*224
        print('audios shape:', a.shape) # batch_size*5*50(channel)
        print('labels shape:', l.shape) # batch_size

        _, classification = model(v, a)
        print(classification.shape)
        break


# In[5]:


if __name__ == "__main__":
    train_loader = load_data()
    verify_model(train_loader)

