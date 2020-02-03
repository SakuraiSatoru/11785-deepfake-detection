#!/usr/bin/env python
# coding: utf-8

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


class MultimodalAutoencoder(nn.Module):
    """
    Autoencoder for multi-modal data fusion. The purpose of the autoencoder
    is to generate a hidden representation that correlates audio, video features 
    with the binary classification.
    """
    ENCODER_HIDDEN = [2048, 1024]
    DECODER_A_HIDDEN = [1024]
    DECODER_V_HIDDEN = [1024]
    CLASSIFIER = [512, 64]
    P_DROPOUT = 0.2
    ZETA = 0.7
    
    def __init__(self, a_in_shape, v_in_shape):
        super().__init__()
        
        self.a_in = a_in_shape
        self.v_in = v_in_shape
        self.activation = nn.ReLU6()
        self.p_dropout = self.P_DROPOUT
        self.zeta = self.ZETA
        
        encoder_layer_sizes = [self.a_in + self.v_in] + self.ENCODER_HIDDEN
        decoder_a_layer_sizes = [self.ENCODER_HIDDEN[-1]] + self.DECODER_A_HIDDEN + [self.a_in]
        decoder_v_layer_sizes = [self.ENCODER_HIDDEN[-1]] + self.DECODER_V_HIDDEN + [self.v_in]
        classifier_layer_sizes = [self.ENCODER_HIDDEN[-1]] + self.CLASSIFIER

        self.encoder = nn.Sequential(*self.get_layers(encoder_layer_sizes, dropout=True))
        self.decoder_a = nn.Sequential(*self.get_layers(decoder_a_layer_sizes))
        self.decoder_v = nn.Sequential(*self.get_layers(decoder_v_layer_sizes))
        self.classifier = nn.Sequential(*self.get_layers(classifier_layer_sizes))
        self.classifier_scoring = nn.Linear(classifier_layer_sizes[-1], 2)
    
    def get_layers(self, layer_sizes, dropout=False):
        layers = []
        if dropout: layers.append(nn.Dropout(self.p_dropout))
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(self.activation)
        return layers
    
    def forward(self, a_in, v_in):
        batch_size = v_in.shape[0]
        assert(v_in.size() == (batch_size, 1664) and a_in.size() == (batch_size, 2300))
        x = torch.cat((a_in, v_in), dim=1)
        x = self.encoder(x)
        a_out = self.decoder_a(x)
        v_out = self.decoder_v(x)
        classification_embedding = self.classifier(x)
        binary_out = self.classifier_scoring(classification_embedding)
        return a_out, v_out, binary_out, classification_embedding
    
    def loss(self, model_output, target):
        model_v, model_a, model_binary = model_output
        target_v, target_a, target_binary = target
        
        batch_size = model_v.size(0)
        assert(model_v.shape == target_v.shape)
        assert(model_a.shape == target_a.shape)
        assert(model_binary.size() == (batch_size, 2) and target_binary.size() == (batch_size, ))

        v_loss = F.mse_loss(model_v, target_v)
        a_loss = F.mse_loss(model_a, target_a)
        binary_loss = F.cross_entropy(model_binary, target_binary)

        return (a_loss + v_loss)*(1-self.zeta) + binary_loss*self.zeta


# In[3]:


class MultiModalAEModel(nn.Module):
    AUDIO_DIM = 2300
    VIDEO_DIM = 1664
    def __init__(self):
        super(MultiModalAEModel, self).__init__()
        self.densenet = densenet169(pretrained=True, progress=True, memory_efficient=False)
        self.audionet = AudioConv2d()
        self.ae = MultimodalAutoencoder(self.AUDIO_DIM, self.VIDEO_DIM)
    
    def forward(self, videos, audios):
        batch_size = videos.shape[0]
        assert(videos.size() == (batch_size, 3, 224, 224) and audios.size() == (batch_size, 5, 50))
        video_embed, _ = self.densenet(videos) # (N, C, H, W) -> (N, num_features:1664)
        audio_embed = self.audionet(audios) # -> (N, num_features:2300)
        audio_out, video_out, binary_out, classification_embedding = self.ae(audio_embed, video_embed)
        
        assert(video_embed.shape == video_out.shape and audio_embed.shape == audio_out.shape)
        assert(binary_out.size() == (batch_size, 2) and classification_embedding.size() == (batch_size, self.ae.CLASSIFIER[-1]))
        return video_out, audio_out, binary_out, classification_embedding, video_embed, audio_embed
    
    def loss(self, *args):
        return self.ae.loss(*args)
        


# In[4]:


def load_data():
    BATCH_SIZE = 2

    train_video_dataset = dataloader.get_dataset(dataloader.TRAIN_JSON_PATH, dataloader.SINGLE_FRAME)
    train_audio_dataset = dataloader.AudioDataset()
    train_loader = dataloader.AVDataLoader(train_video_dataset, train_audio_dataset, batch_size=BATCH_SIZE, shuffle=True, single_frame=True)
    return train_loader


# In[5]:


def verify_model(train_loader):
    model = MultiModalAEModel()
#     print(model)

    for v, a, _, _, l in train_loader:
        print('videos shape:', v.shape) # batch_size*3(channel)*224*224
        print('audios shape:', a.shape) # batch_size*5*50(channel)
        print('labels shape:', l.shape) # batch_size

        video_out, audio_out, binary_out, classification_embedding, video_embed, audio_embed = model(v, a)
        print("out")
        print(video_out.shape)
        print(audio_out.shape)
        print(binary_out.shape)
        print(classification_embedding.shape)
        loss = model.loss((video_out, audio_out, binary_out), (video_embed, audio_embed, l))
        break


# In[6]:


if __name__ == "__main__":
    train_loader = load_data()
    verify_model(train_loader)

