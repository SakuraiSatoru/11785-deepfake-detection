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
from dataloader import *
import numpy as np

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')


def train(model, data_loader, val_loader):

    model.train()

    for epoch in range(numEpochs):
        avg_loss = 0.0

        train_loss = []
        accuracy = 0
        total = 0


        start_time = time.time()
        for i in range(5):
        #for batch_num, (feats, labels) in enumerate(data_loader):
            break
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()

            outputs = model(feats)


            _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
            pred_labels = pred_labels.view(-1)


            loss = criterion(outputs, labels.long())

            accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            train_loss.extend([loss.item()]*feats.size()[0])

            loss.backward()
            optimizer.step()

            
            avg_loss += loss.item() 


            if batch_num % 100 == 99:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/100))
                avg_loss = 0.0
                PATH = "./densenet169.pth"

                # print('trying to save model')
                # torch.save({
                # 'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler,
                #
                # }, PATH )
                # print('saved model')
                break



            if batch_num % 10 == 0:
                print(batch_num,'batches')
            
            del feats
            del labels
            del loss
            torch.cuda.empty_cache()
            # break   

        end_time = time.time()
         
        
        print(epoch, 'Epoches')
        #train_loss = np.mean(train_loss)
        #train_acc = accuracy/total
         
        val_loss, val_acc = validation(model, val_loader)
        print("after test classify")
        #print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
        #    format(train_loss, train_acc, val_loss, val_acc))
        print('Val Loss: {:.4f}\tVal Accuracy: {:.4f}'.
            format(val_loss, val_acc))
        print('Time: ',end_time - start_time, 's')

        scheduler.step(val_loss)
        # break




def validation(model, val_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(val_loader):
        print('batch:', batch_num)
        
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        outputs = model(feats)
        
        # _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        # pred_labels = pred_labels.view(-1)
        #
        # loss = criterion(outputs, labels.long())
        #
        #
        # accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        # total += len(labels)
        # test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels
        # del loss
        # del outputs
        # del pred_labels
        torch.cuda.empty_cache()
        if batch_num % 5 == 0 and batch_num != 0:
            break

    model.train()
    return np.mean(test_loss), accuracy/total






#train_dataset = get_dataset(TRAIN_JSON_PATH, SINGLE_FRAME)
#train_loader = DataLoader(train_dataset, batch_size=64, num_workers=8, shuffle=True)

train_loader = None
val_dataset = get_dataset(VAL_JSON_PATH, SINGLE_FRAME)
val_loader = DataLoader(val_dataset, batch_size=64, num_workers=0, shuffle=True)

#test_dataset = get_dataset(TEST_JSON_PATH, SINGLE_FRAME)
#test_loader = DataLoader(test_dataset, batch_size=64, num_workers=8, shuffle=False)


densenet = models.densenet169(pretrained=True, progress=True,memory_efficient=True)


learningRate = 1e-2 
weightDecay = 5e-5
numEpochs = 1



print('after building!')
# print(densenet)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(densenet.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

print('enter training!')

checkpoint = torch.load('densenet169.pth')
densenet.load_state_dict(checkpoint['model_state_dict'])

densenet.to(DEVICE)

train(densenet, train_loader, val_loader)



print("congrates!")






