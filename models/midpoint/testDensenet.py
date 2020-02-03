#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import torch
import torch.nn.functional as F
from torch import autograd, nn
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import time
import dataloader
import numpy as np
from tqdm.autonotebook import tqdm

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')

# def validation(model, val_loader):
#     model.eval()
#     test_loss = []
#     accuracy = 0
#     total = 0
#
#     for batch_num, (feats, labels) in enumerate(val_loader):
#
#         feats, labels = feats.to(DEVICE), labels.to(DEVICE)
#         outputs = model(feats)
#         _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
#         pred_labels = pred_labels.view(-1)
#         loss = criterion(outputs, labels.long())
#
#         accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
#         total += len(labels)
#         test_loss.extend([loss.item()]*feats.size()[0])
#         del feats
#         del labels
#         del loss
#         torch.cuda.empty_cache()
#         if batch_num % 5 == 0 and batch_num != 0:
#             break
#
#     model.train()
#     return np.mean(test_loss), accuracy/total
#
#
# def test(model, test_loader):
#     model.eval()
#     accuracy = 0
#     total = 0
#
#     for batch_num, (feats, labels) in enumerate(test_loader):
#         feats, labels = feats.to(DEVICE), labels.to(DEVICE)
#         outputs = model(feats)
#         _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
#         pred_labels = pred_labels.view(-1)
#
#         accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
#         total += len(labels)
#
#         if batch_num % 500 == 0 and batch_num != 0:
#             print('acc:', accuracy/total)
#         del feats
#         del labels
#         torch.cuda.empty_cache()
#     return accuracy/total

val_video_dataset = dataloader.get_dataset(dataloader.VAL_JSON_PATH, dataloader.SINGLE_FRAME)
val_audio_dataset = dataloader.AudioDataset()
val_loader = dataloader.AVDataLoader(val_video_dataset, val_audio_dataset, batch_size=64, shuffle=False, single_frame=True)

def _val_batch(model, videos, audios, labels, criterion):
    videos, audios, labels = videos.to(DEVICE), audios.to(DEVICE), labels.to(DEVICE)
    preds = model(videos)  # linear output
    # calculate loss
    batch_loss = criterion(preds, labels).item()
    # calculate acc
    _, max_preds = torch.max(preds, 1)
    batch_corrects = (max_preds == labels).sum().item()
    batch_size = labels.shape[0]
    return batch_loss, batch_corrects, batch_size

def val_epoch(model, val_loader, criterion):
    model.eval()
    epoch_loss = 0.0
    epoch_corrects = 0
    num_samples = 0
    num_batches = 0
    for videos, audios, _, _, labels in tqdm(val_loader):
        # print(num_batches,'/',len(val_loader))
        batch_loss, batch_corrects, batch_size = _val_batch(model, videos,
                                                                 audios,
                                                                 labels, criterion)
        epoch_loss += batch_loss
        epoch_corrects += batch_corrects
        num_samples += batch_size
        num_batches += 1
        # break

    epoch_loss /= num_batches
    print(epoch_loss)
    epoch_acc = epoch_corrects / num_samples * 100.0
    print(epoch_acc)


model = models.densenet169(pretrained=False, progress=True, memory_efficient=True)
criterion = nn.CrossEntropyLoss()
checkpoint = torch.load('densenet169.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
val_epoch(model, val_loader, criterion)






