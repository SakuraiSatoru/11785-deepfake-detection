# out_phonme how many
class GRULayer(nn.Module):
    def __init__(self, out_phome, frame_size, hidden_size, nums_layer):
        super(GRULayer, self).__init__()
        self.nums_layer = nums_layer
        self.layers = []
        for layer in range(nums_layer):
            self.gru = nn.GRU(frame_size, hidden_size, num_layers=3, bidirectional=True)
            self.output = nn.Linear(hidden_size*2, out_phome)
            self.layers.append([self.gru, self.output])

       
    
    # def forward(self, X, length):
    def forward(self, X):
        result = []
        for layer in range(len(self.layers)):

            out = self.layers[layer][0](X[layer].tranpose(0,1))[0]
            out = self.layers[layer][1](out).log_softmax(2)
        
            resulta.append(out)
        return result


class MLP(nn.Module):
    def __init__(self, input_size, hidden_dims=None):
        super(MLP, self).__init__()
        layers = [input_size]
        in_size = input_size
        if hidden_dims is not None:
            for h in hidden_dims:
                layers.append(nn.Linear(in_size, h))
                layers.append(nn.Sigmoid())
                in_size = h
        layers.append(nn.Linear(in_size, 1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



class Model(nn.Module):


    def __init__(self,growth_rate, block_config, num_init_feature, GRU_out_size, GRU_frame_size, GRU_hidden_size \
        , GRU_nums_layer, MLP_input_size, MLP_dims):
        self.densenet = DesnetLayer(growth_rate, block_config, num_init_feature)
        self.gru = GRULayer(GRU_out_size, GRU_frame_size, GRU_hidden_size, GRU_nums_layer)
        self.mlp = MLPLayer(MLP_input_size, MLP_dims)


    def forward(self, input_frames):

        cnn_result = self.densenet(input_frame)

        gru_result = self.gru(cnn_result)

        output = self.mlp(gru_result)

        return output




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

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)




def train_epoch(model, optimizer, train_loader):
    criterion = nn.CrossEntropy()
    criterion = criterion.to(DEVICE)
    before = time.time()
    print("training", len(train_loader), "number of batches")
    model.train()
    for batch_idx, (inputs,targets) in enumerate(train_loader):
        if batch_idx == 0:
            first_time = time.time()
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        outputs= model(inputs)

        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx == 0:
            print("Time elapsed", time.time() - first_time)
            
        if batch_idx % 100 == 0 and batch_idx != 0:
            after = time.time()
            print("Time: ", after - before)
            print("Loss: ", loss.item())
            before = after

    val_loss = 0
    model.eval()
    for batch_idx, (inputs,targets) in enumerate(val_loader):
        
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        outputs, outputs_lens = model(inputs)
        loss = criterion(outputs,targets)
        val_loss =loss.item()
        print("\nValidation loss:",val_loss)





