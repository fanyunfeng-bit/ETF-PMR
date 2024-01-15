import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import random


class CramedDataset(Dataset):

    def __init__(self, args, mode='train', class_imbalanced=False):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode

        self.data_root = './data/'
        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        self.visual_feature_path = r'D:\DATASETS\CREMAD\CREMA-D'
        self.audio_feature_path = r'D:\DATASETS\CREMAD\CREMA-D\Audio-299\Audio-299'

        self.train_csv = os.path.join(self.data_root, args.dataset + '/train.csv')
        self.test_csv = os.path.join(self.data_root, args.dataset + '/test.csv')

        if mode == 'train':
            csv_file = self.train_csv
        else:
            csv_file = self.test_csv

        if class_imbalanced and mode == 'train':
            with open(self.train_csv, encoding='UTF-8-sig') as f:
                csv_reader = csv.reader(f)
                subset_class = [[] for _ in range(len(class_dict))]  # instances for each class
                class_sample_num = [0 for _ in range(len(class_dict))]
                for item in csv_reader:
                    subset_class[class_dict[item[1]]].append(item[0]+' '+item[1])
                    class_sample_num[class_dict[item[1]]] += 1

            max_num = class_sample_num[0]
            num_perclass = []
            for ii in range(len(class_dict), 0, -1):
                num_perclass.append(max_num//len(class_dict)*ii)

            imbalanced_dataset = []
            print(num_perclass)
            # [972, 810, 648, 486, 324, 162]

            for ii in range(len(class_sample_num)):
                random.shuffle(subset_class[ii])
                imbalanced_dataset += subset_class[ii][:num_perclass[ii]]

            random.shuffle(imbalanced_dataset)
            for item in imbalanced_dataset:
                audio_path = os.path.join(self.audio_feature_path, item.split(' ')[0] + '.pkl')
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(self.args.fps), item.split(' ')[0])

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[item.split(' ')[1]])
                else:
                    continue
        else:
            with open(csv_file, encoding='UTF-8-sig') as f2:
                csv_reader = csv.reader(f2)
                for item in csv_reader:
                    audio_path = os.path.join(self.audio_feature_path, item[0] + '.pkl')
                    visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(self.args.fps), item[0])

                    if os.path.exists(audio_path) and os.path.exists(visual_path):
                        self.image.append(visual_path)
                        self.audio.append(audio_path)
                        self.label.append(class_dict[item[1]])
                    else:
                        continue


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        # # audio
        # samples, rate = librosa.load(self.audio[idx], sr=22050)
        # resamples = np.tile(samples, 3)[:22050*3]
        # resamples[resamples > 1.] = 1.
        # resamples[resamples < -1.] = -1.
        #
        # spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        # spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        # #mean = np.mean(spectrogram)
        # #std = np.std(spectrogram)
        # #spectrogram = np.divide(spectrogram - mean, std + 1e-9)

        spectrogram = pickle.load(open(self.audio[idx], 'rb'))

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])        
        select_index = np.random.choice(len(image_samples), size=self.args.num_frame, replace=False)
        select_index.sort()
        images = torch.zeros((self.args.num_frame, 3, 224, 224))
        for i in range(self.args.num_frame):
            #  for i in select_index:
            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1,0,2,3))

        # label
        label = self.label[idx]

        return spectrogram, images, label