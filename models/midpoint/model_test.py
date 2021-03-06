

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

# global viarable
DEFAULT_FRAME_SIZE = 10


def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)

# out_phonme how many
class GRULayer(nn.Module):
    def __init__(self, out_phome, frame_size, hidden_size):
        super(GRULayer, self).__init__()
        self.gru = nn.GRU(frame_size, hidden_size, num_layers=3, bidirectional=True)
        self.output = nn.Linear(hidden_size*2, out_phome)
    
    # def forward(self, X, length):
    def forward(self, X):
        # print('input size of GRU forward:', X.size())
        out = self.gru(X)[0]
        # print('output size of GRU forward1:', out.size())
        out = self.output(out).log_softmax(2)
        out = out.transpose(0, 1)
        out = out[:,-1,:]

        
        return out

class Model(nn.Module):


    def __init__(self, GRU_out_size, GRU_frame_size, GRU_hidden_size):
        super(Model, self).__init__()
        self.densenet = models.densenet169(pretrained=False, progress=True, memory_efficient=True)
        self.gru = GRULayer(GRU_out_size, GRU_frame_size, GRU_hidden_size)



    def forward(self, input_frames):
        # print("input_frames.size():", input_frames.size())

        batch_size, frames, C, H, W = input_frames.size()
        input_frames = input_frames.view(batch_size*frames, C, H, W)

        cnn_result = self.densenet(input_frames)
        # print("cnn_result:", cnn_result.size())

        input_to_gru = cnn_result[:,:5]

        for i in range(1, len(cnn_result)-4): 
            torch.cat((input_to_gru, cnn_result[:,i:i+5]),1)

        input_to_gru = input_to_gru.reshape(batch_size, frames, -1).transpose(0,1)

        output = self.gru(input_to_gru)
        # print("output inside forward:", output.size())

        return output


def train_epoch(model, criterion, optimizer, train_loader, val_loader):
    criterion = criterion.to(device)
    before = time.time()
    print("training", len(train_loader), "number of batches")
    model.train()
    for batch_idx, (inputs,targets) in enumerate(train_loader):
        # print("targets:",targets.size()) 
        # print("targets:",targets[0].size()) 
        if batch_idx == 0:
            first_time = time.time()
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        outputs= model(inputs)
        # print("output:", outputs.size())

        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx == 0:
            print("Time elapsed", time.time() - first_time)
        # if batch_idx == 1:
        #     print("Time elapsed", time.time() - first_time)
        #     break
            
        if batch_idx % 100 == 0 and batch_idx != 0:
            after = time.time()
            print('%d Batch:'%batch_idx)
            print('Train Loss: {:.4f}\tTime: {:.4f}'.
                  format(loss.item(), after - before))
            before = after



        if batch_idx % 5000 == 0 and batch_idx != 0:

            writeFile("./save/%d/loss.csv" % batch_idx, str(float(loss.item())))

            PATH = "./save/%d/rcnn2.pth" % batch_idx
            

            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler,
            }, PATH )

    val_loss = 0
    model.eval()
    for batch_idx, (inputs,targets) in enumerate(val_loader):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs,targets)
        val_loss =loss.item()
        print("\nValidation loss:",val_loss)
        break
    return val_loss


def test(model, test_loader):
    before = time.time()
    model.eval()
    batch = 0
    total_predictions = 0
    correct_predictions = 0

    y_label = []
    y_prob = []

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        batch += 1
        if batch_idx == 0:
            first_time = time.time()
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        outputs = model(inputs)
        o = outputs
        o = o.to('cpu').numpy()
        print('o', o)
        outputs = torch.argmax(outputs, dim=1)
        o = o[outputs]
        print(outputs)
        print(o)

        total_predictions += targets.size(0)
        correct_predictions += (outputs == targets).sum().item()
        acc = (correct_predictions/total_predictions)*100.0
        print('Testing Accuracy: ', acc)
        break
    acc = (correct_predictions/total_predictions)*100.0
    print('Testing Accuracy: ', acc)




    return acc

class VideoDataset(Dataset):
    
    def __init__(self, inputs, output):
        self.inputs = inputs
        self.output = output

    
    def __getitem__(self,i):
        video = self.inputs[i]
        label = self.output[i]
        return video, label

    def __len__(self):
        return self.inputs.shape[0]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('will start immediately!!!')

GRU_out_size = 2
GRU_frame_size = 5
GRU_hidden_size = 55


learningRate = 1e-2 
weightDecay = 5e-5

model = Model(GRU_out_size, GRU_frame_size, GRU_hidden_size)
print('after building the model!!!')
print(model)


optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
criterion = nn.CrossEntropyLoss()


checkpoint = torch.load('rcnn.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)


print('after loading model!!!')



model.to(device)

num_epochs = 20



# train_dataset = get_dataset(TRAIN_JSON_PATH, DEFAULT_FRAME_SIZE)
# train_loader = DataLoader(train_dataset, batch_size=5, num_workers=0, shuffle=True)
#
#
# val_dataset = get_dataset(VAL_JSON_PATH, DEFAULT_FRAME_SIZE)
# val_loader = DataLoader(val_dataset, batch_size=5, num_workers=0, shuffle=True)

test_dataset = get_dataset(TEST_JSON_PATH, DEFAULT_FRAME_SIZE)
test_loader = DataLoader(test_dataset, batch_size=5, num_workers=0, shuffle=True)



print('after loading the data!!!')
print('just before training!!!')




# for epoch in range(num_epochs):
#     print('%d Epoch:' % epoch)
#     val_loss = train_epoch(model, criterion, optimizer, train_loader, val_loader)
#     scheduler.step(val_loss)
#     break

test(model, test_loader)

print('congrates!!!')














