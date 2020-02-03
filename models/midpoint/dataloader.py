import os
import json
import math
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')

HOME_PATH = os.path.abspath(os.pardir)
JSON_PATH = os.path.join(HOME_PATH, 'dataset')
TRAIN_JSON_PATH = os.path.join(JSON_PATH, 'train.json')
VAL_JSON_PATH = os.path.join(JSON_PATH, 'val.json')
TEST_JSON_PATH = os.path.join(JSON_PATH, 'test.json')
ORIGINAL_SEQ_PATH = os.path.join(HOME_PATH,
                                 r'data/faceforensics/c40/original_sequences/youtube/c40/processed_images')
MANIPULATED_SEQ_PATH = os.path.join(HOME_PATH,
                                    r'data/faceforensics/c40/manipulated_sequences/Deepfakes/c40/processed_images')
AUDIO_PATH = os.path.join(HOME_PATH, r'data/faceforensics/audio/padded_audio.npy')
AUDIO_LEN_PATH = os.path.join(HOME_PATH, r'data/faceforensics/audio/padded_audio_lens.npy')

WHOLE_FRAME = -1
SINGLE_FRAME = 0  # used for dataloader for CNN
DEFAULT_FRAME_SIZE = 10  # num of frames per sequence in dataloader
DEFAULT_SKIP_TOLERANCE = 1  # max num of missed frames per sequence in dataloader

class ImageDataset(Dataset):
    """Fake seq is labeled as 1, original seq as 0"""

    def __init__(self, folder_list, target_list, frame_size):
        """
            args:
                frame_size:     >=1: [frame_size] of frames as an utterance
                                            =0: flattened frames for CNN
                                            <0: whole video (not implemented)
        """
        self.folder_list = folder_list
        self.target_list = target_list
        self.frame_size = frame_size
        self.img_list = []
        self.label_list = []
        self.index_list = []
        for i, folder in enumerate(self.folder_list):
            imgs = self.get_imgs(folder, frame_size)
            # assert len(imgs) == 1
            self.img_list.extend(imgs)
            self.label_list.extend([self.target_list[i]] * len(imgs))
            if self.frame_size == WHOLE_FRAME:
                self.index_list.extend([int(folder.split(r'/')[-1][:3])]*len(imgs))
            else:
                # print(imgs)
                # print([int(imgs[0][0].split(r'/')[-2]), int(imgs[0][0].split(r'/')[-1].split('.')[0])])
                self.index_list.extend(list(map(lambda x: None if x[0] is None else (int(x[0].split(r'/')[-2].split('_')[0]), int(x[0].split(r'/')[-1].split('.')[0])), imgs)))

        if self.frame_size > 0:
            print('loaded %d sets of %d-frame images ' % (len(self.img_list), frame_size))
        elif self.frame_size == 0:
            print('loaded %d images ' % len(self.img_list))
        else:
            print('loaded %d video sequences' % len(self.img_list))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        imgs = self.img_list[index]
        if self.frame_size != 0:
            out = []
            for i in range(len(imgs)):
                if imgs[i] is None:
                    out.append(torch.zeros(3, 224, 224, dtype=torch.float32))
                else:
                    try:
                        img = Image.open(imgs[i])
                        out.append(torchvision.transforms.ToTensor()(img))
                    except Exception as e:
                        print('error occurred while reading %s'%imgs[i])
                        print(e)
            # if self.frame_size > 0:
            #     while len(out) < self.frame_size:
            #         out.append(out[-1])
            out = torch.stack(out)
        else:
            assert len(imgs) == 1
            if imgs[0] is None:
                out = None
            else:
                try:
                    img = Image.open(imgs[0])
                    out = torchvision.transforms.ToTensor()(img)
                except:
                    out = None
        label = self.label_list[index]
        return out, label, self.index_list[index]

    def get_imgs(self, folder, n):
        if n == 0:
            n = 1
        s = dict()
        max_index = 0
        for (root, dirs, files) in os.walk(folder):
            for f in files:
                try:
                    d = f.split('.')[0]
                    if d.isdigit():
                        s[int(d)] = f
                        max_index = max(int(d), max_index)
                except Exception as e:
                    print(e, '\ncan not handle', s)
            break
        out = []
        frames = []
        for ptr in range(max_index+1):
            if ptr in s:
                frames.append(os.path.join(folder, s[ptr]))
            else:
                frames.append(None)
            if (self.frame_size < 0 and ptr == max_index) or (self.frame_size >= 0 and len(frames) == n):
                out.append(frames)
                frames = []
        return out

    def get_vid_num(self):
        return len(self.img_list)

# def collate(seq_list):
#     inputs, targets = zip(*seq_list)
#     len_input = torch.LongTensor(list(map(len, inputs)))
#     inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
#     return inputs, targets, len_input


def get_dataset(json_dir, frame_size,
                original_dir=ORIGINAL_SEQ_PATH,
                manipulated_dir=MANIPULATED_SEQ_PATH):
    with open(json_dir, 'r') as f:
        l = json.load(f)
    folder_list = []
    target_list = []
    for f1, f2 in l:
        p0 = os.path.join(original_dir, f1)
        p1 = os.path.join(original_dir, f2)
        p2 = os.path.join(manipulated_dir, '%s_%s' % (f1, f2))
        p3 = os.path.join(manipulated_dir, '%s_%s' % (f2, f1))
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
    return ImageDataset(folder_list, target_list, frame_size)


class AudioDataset(Dataset):

    def __init__(self, audio_path=AUDIO_PATH, lens_path=AUDIO_LEN_PATH):
        self.audio = np.load(audio_path, allow_pickle=True).transpose((1, 0, 2))
        self.lens = np.load(lens_path, allow_pickle=True)

    def __len__(self):
        return self.lens.shape[0]

    def __getitem__(self, index):
        if isinstance(index, int):
            audio = self.audio[index]
            lens = self.lens[index]
            return audio, lens
        else:
            i, j = index
            lens = self.lens[i]
            if lens <= 1:
                return None, None
            return self.audio[i, j*5:j*5+5], None


class AVDataLoader(DataLoader):

    def __init__(self, videoDataset, audioDataset, batch_size, shuffle=True, single_frame=False):
        self.batch_size = batch_size
        self.videoDataset = videoDataset
        self.num_video = videoDataset.get_vid_num()
        self.audioDataset = audioDataset
        self.shuffle = shuffle
        self.single_frame = single_frame

    def __len__(self):
        return math.ceil(self.num_video/self.batch_size)

    def __iter__(self):
        indexList = list(range(self.num_video))
        if self.shuffle:
            np.random.shuffle(indexList)

        for i in range(0, self.num_video, self.batch_size):
            videos = []
            videos_lens = []
            labels = []
            audios = []
            audios_lens = []

            for j in range(self.batch_size):
                if i+j >= self.num_video:
                    break
                video_data, label, index = self.videoDataset[indexList[i+j]]
                if index is None or video_data is None:
                    continue
                audio_data, audio_len = self.audioDataset[index]
                if audio_data is None or (audio_len is not None and audio_len <= 1):
                    continue
                videos.append(video_data)
                labels.append(label)
                audios.append(torch.as_tensor(audio_data))
                if not self.single_frame:
                    videos_lens.append(torch.as_tensor(video_data.size(0)))
                    audios_lens.append(torch.as_tensor(audio_len))

            if not len(videos):
                continue

            videos = pad_sequence(videos, batch_first=True)
            audios = torch.stack(audios, dim=0)
            if self.single_frame:
                videos_lens = None
                audios_lens = None
            else:
                videos_lens = torch.stack(videos_lens, dim=0)
                audios_lens = torch.stack(audios_lens, dim=0)
            labels = torch.as_tensor(labels)
            # print('videos type(expect tensor):', type(videos))
            yield videos, audios, videos_lens, audios_lens, labels



if __name__ == '__main__':

    print('\ndataloader for whole videos')
    train_vid = get_dataset(TRAIN_JSON_PATH, WHOLE_FRAME)
    audio_dataset = AudioDataset()
    loader = AVDataLoader(train_vid, audio_dataset, batch_size=8, shuffle=True)
    for v, a, vl, al, l in loader:
        # note: batch_size might vary since some audio is missing
        print('videos shape:', v.shape) # batch_size*seq_length*3(channel)*224*224
        print('videos_lens shape:', vl.shape) # batch_size
        print('audios shape:', a.shape) # batch_size*9790(padded_length)*50(channel)
        print('audios_lens shape:', al.shape) # batch_size
        print('labels shape:', l.shape) # batch_size
        break


    print('\ndataloader for single frames')
    train_vid = get_dataset(TRAIN_JSON_PATH, SINGLE_FRAME)
    audio_dataset = AudioDataset()
    loader = AVDataLoader(train_vid, audio_dataset, batch_size=8, shuffle=True, single_frame=True)
    for v, a, _, _, l in loader:
        print('videos shape:', v.shape) # batch_size*3(channel)*224*224
        print('audios shape:', a.shape) # batch_size*5*50(channel)
        print('labels shape:', l.shape) # batch_size
        break
