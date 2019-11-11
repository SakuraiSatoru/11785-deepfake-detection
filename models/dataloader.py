import json
import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader


CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')

HOME_PATH = os.path.abspath(os.pardir)
JSON_PATH = os.path.join(HOME_PATH, 'dataset')
TRAIN_JSON_PATH = os.path.join(JSON_PATH, 'train.json')
VAL_JSON_PATH = os.path.join(JSON_PATH, 'val.json')
TEST_JSON_PATH = os.path.join(JSON_PATH, 'test.json')
ORIGINAL_SEQ_PATH = os.path.join(HOME_PATH, r'data/faceforensics/c40/original_sequences/youtube/c40/processed_images')
MANIPULATED_SEQ_PATH = os.path.join(HOME_PATH, r'data/faceforensics/c40/manipulated_sequences/Deepfakes/c40/processed_images')

SINGLE_FRAME = 0 # used for dataloader for CNN
DEFAULT_FRAME_SIZE = 50 # num of frames per sequence in dataloader
DEFAULT_SKIP_TOLERANCE = 1 # max num of missed frames per sequence in dataloader


class ImageDataset(Dataset):
    """Fake seq is labeled as 1, original seq as 0"""
    def __init__(self, folder_list, target_list, frame_size, skip_tolerance):
        """
            args:
                frame_size:     >=1: [frame_size] of frames as an utterance
                                            =0: flattened frames for CNN
                                            <0: whole video (not implemented)
        """
        if frame_size < 0:
            raise NotImplemented
        self.folder_list = folder_list
        self.target_list = target_list
        self.frame_size = frame_size
        self.img_list = []
        self.label_list = []
        for i,folder in enumerate(self.folder_list):
            imgs = self.get_imgs(folder, frame_size, skip_tolerance)
            self.img_list.extend(imgs)
            self.label_list.extend([self.target_list[i]]*len(imgs))
        if self.frame_size >0:
            print('loaded %d sets of %d-frame images '%(len(self.img_list), frame_size))
        elif self.frame_size == 0:
            print('loaded %d images '%len(self.img_list))
        else:
            pass

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        imgs = self.img_list[index]
        if self.frame_size != 0:
            out = []
            for i in range(self.frame_size):
                img = Image.open(imgs[i])
                out.append(torchvision.transforms.ToTensor()(img))
            out = torch.stack(out)
        else:
            assert len(imgs) == 1
            img = Image.open(imgs[0])
            out = torchvision.transforms.ToTensor()(img)
        label = self.label_list[index]
        return out, label
    
    def get_imgs(self, folder, n, t):
        if n == 0:
            n = 1
        s = dict()
        for (root,dirs,files) in os.walk(folder):
            for f in files:
                try:
                    s[int(f.split('.')[0])] = f
                except:
                    pass
            break
        out = []
        ptr = 0
        while ptr <= len(s) - n:
            frames = []
            tolerance = 0
            while tolerance<=t:
                if ptr in s:
                    frames.append(os.path.join(folder, s[ptr]))
                else:
                    tolerance += 1
                ptr += 1
                if len(frames) == n:
                    out.append(frames)
                    break
        return out


def get_dataset(json_dir, frame_size, skip_tolerance=DEFAULT_SKIP_TOLERANCE, 
                original_dir=ORIGINAL_SEQ_PATH, 
                manipulated_dir=MANIPULATED_SEQ_PATH):
    with open(json_dir, 'r') as f:
        l = json.load(f)
    folder_list = []
    target_list = []
    for f1,f2 in l:
        p0 = os.path.join(original_dir, f1)
        p1 = os.path.join(original_dir, f2)
        p2 = os.path.join(manipulated_dir, '%s_%s'%(f1, f2))
        p3 = os.path.join(manipulated_dir, '%s_%s'%(f2, f1))
        if os.path.isdir(p0):
            folder_list.append(p0)
            target_list.append(0)
        if os.path.isdir(p1):
            folder_list.append(p1)
            target_list.append(0)
        if os.path.isdir(p2):
            folder_list.append(p2)
            target_list.append(1)
        if os.path.isdir(p3):
            folder_list.append(p3)
            target_list.append(1)
    return ImageDataset(folder_list, target_list, frame_size, skip_tolerance)


if __name__ == '__main__':
	print('dataloader for batched frames (for training whole model)')
	train_dataset = get_dataset(TRAIN_JSON_PATH, DEFAULT_FRAME_SIZE)
	train_loader = DataLoader(train_dataset, batch_size=10, num_workers=0, shuffle=True)

	for data,label in train_loader:
	    print('data shape:', data.shape)
	    print('label shape:', label.shape)
	    break

	print('dataloader for single frames (for training CNN)')
	# dataloader for single frames
	train_dataset = get_dataset(TRAIN_JSON_PATH, SINGLE_FRAME)
	train_loader = DataLoader(train_dataset, batch_size=10, num_workers=0, shuffle=True)

	for data,label in train_loader:
	    print('data shape:', data.shape)
	    print('label shape:', label.shape)
	    break

