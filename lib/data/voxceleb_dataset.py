#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, xiaobing.                        #
# written by wangduomin@xiaobing.ai             #
#################################################

import os
import math
import numpy as np
from lib.data.AudioConfig import AudioConfig
import shutil
import cv2
import glob
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
# from lib.data.image_data.base_dataset import BaseDataset
from PIL import Image
from tqdm import tqdm
import json

def crop_img(img, crop_size, crop=True, crop_len=16):
    
    if crop:
        img = img[:crop_size - crop_len*2, crop_len:crop_size - crop_len]
        img = cv2.resize(img, (crop_size, crop_size))

    return img

class VOXTotalDataset(data.Dataset):

    def __init__(self, cfg):
        super(VOXTotalDataset, self).__init__()
        self.img_size = cfg.img_size
        self.crop = cfg.crop
        self.cfg = cfg
        self.samples = cfg.sample_num_per_video_in_one_epoch
        
        # read data file list
        self.data_path = cfg.train_lmdb_path

        self.num_bins_per_frame = int(cfg.audio.sample_rate / cfg.audio.hop_size / cfg.audio.frame_rate) # 16000 / 160 / 25 = 4
        self.num_bins_per_frame_ori = int(cfg.audio.sample_rate / cfg.audio.frame_rate) # 16000 / 25 = 640
        self.num_frames_per_clip = cfg.audio.num_frames_per_clip
        self.num_audio_bins = self.num_frames_per_clip * self.num_bins_per_frame
        self.num_audio_ori_bins = cfg.audio.num_frames_per_clip * self.num_bins_per_frame_ori
        
        self.audio = AudioConfig(num_frames_per_clip=5, hop_size=160)
        self.last_mp3_path = "None"
        self.wav_ori = None

        self.dataload_dict, self.dataset_size = self.get_data(self.data_path)
        # self.dict_len = len(self.dataload_dict)
        self.keys = self.dataload_dict.keys()
        # self.dataset_size = self.dict_len * self.samples

        self.transform = transforms.Compose([
            transforms.Resize(size=(self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            
        ])
    
    # def frame2audio_indexs(self, frame_inds):
    #     # each frame use clip of spectrograms which it lies in to represent audio signal.
    #     start_frame_ind = frame_inds - self.num_frames_per_clip // 2

    #     start_audio_inds = start_frame_ind * self.num_bins_per_frame
    #     return start_audio_inds
    
    def frame2audio_ori_indexs(self, frame_inds):
        start_frame_ind = np.array(frame_inds) - self.audio.num_frames_per_clip // 2

        start_audio_inds = start_frame_ind * 640

        return start_audio_inds

    def load_spectrogram(self, audio_ind, spectrograms_dict):
        spectrograms_path = spectrograms_dict["spectrograms_path"]
        
        spectrogram = np.load(spectrograms_path)
        mel_shape = spectrogram.shape

        if (audio_ind + self.num_audio_bins) <= mel_shape[0] and audio_ind >= 0:                
            spect = np.array(spectrogram[audio_ind:audio_ind + self.num_audio_bins, :]).astype('float32')
        else:
            print('(audio_ind {} + opt.num_audio_bins {}) > mel_shape[0] {} '.format(audio_ind, self.num_audio_bins,
                                                                                    mel_shape[0]))
            if audio_ind > 0:
                spect = np.array(spectrogram[audio_ind:audio_ind + self.num_audio_bins, :]).astype('float32')
                
            else:
                spect = np.zeros((self.num_audio_bins, mel_shape[1])).astype(np.float16).astype(np.float32)

        spect = torch.from_numpy(spect)
        spect = spect.unsqueeze(0)

        spect = spect.transpose(-2, -1)
        return spect
    
    def load_wav_ori(self, audio_ind, wav_ori):
        mel_shape = wav_ori.shape
        # print(mel_shape)

        if (audio_ind + self.num_audio_ori_bins) <= mel_shape[0] and audio_ind >= 0:
            spect = np.array(wav_ori[audio_ind:audio_ind + self.num_audio_ori_bins]).astype('float32')
        else:
            print('(audio_ind {} + opt.num_audio_bins {}) > mel_shape[0] {} '.format(audio_ind, self.num_audio_ori_bins, mel_shape[0]))
            
            # spect = np.zeros((self.num_audio_ori_bins, mel_shape[1])).astype(np.float16).astype(np.float32)

            if audio_ind < 0:
                # spect1 = np.zeros((abs(audio_ind), mel_shape[1])).astype(np.float16).astype(np.float32)
                spect1 = np.zeros(abs(audio_ind)).astype(np.float16).astype(np.float32)
                spect2 = np.array(wav_ori[0:self.num_audio_ori_bins + audio_ind]).astype('float32')
                spect = np.concatenate([spect1, spect2], 0)
            else:
                spect1 = np.array(wav_ori[audio_ind:audio_ind + self.num_audio_ori_bins]).astype('float32')
                if spect1.shape[0] < self.num_audio_ori_bins:
                    spect2 = np.zeros(self.num_audio_ori_bins - spect1.shape[0]).astype(np.float16).astype(np.float32)
                    spect = np.concatenate([spect1, spect2], 0)
                else:
                    spect = spect1

        spect = torch.from_numpy(spect)
        # spect = spect.unsqueeze(0)

        return spect
    
    def get_data(self, json_path):
        print("loading data ...")
        
        with open(json_path, "r") as _file:
            data_dict = json.load(_file)
            
        if "data" in data_dict:
            data_dict = data_dict["data"]
        
        keys = list(data_dict)
        
        res_dict = {}
        index = 0
        for key in keys:
            id_label = data_dict[key]['label']
            
            res_dict[key] = data_dict[key]
            index += len(data_dict[key]["target_frame_inds"])
            
        print("Data loaded! total {} samples".format(len(res_dict)))
        return res_dict, index
    
    def __get_index(self, index):
        count = (index + 1)
        key_on = None
        for key in self.keys:
            data_len = len(self.dataload_dict[key]["target_frame_inds"])
            if count > data_len:
                count -= data_len
            else:
                key_on = key
                break
        
        return count - 1, key_on
        
        

    def __getitem__(self, index):
        
        in_data_index, on_data_index = self.__get_index(index)
            
        img_index = self.dataload_dict[on_data_index]["target_frame_inds"][in_data_index]
        mel_index = self.dataload_dict[on_data_index]["audio_inds"][in_data_index]
        id_img_path = self.dataload_dict[on_data_index]['id_img_path']
        mel_ori_index = self.frame2audio_ori_indexs(self.dataload_dict[on_data_index]["target_frame_inds"])[in_data_index]

        img_path = os.path.join(self.dataload_dict[on_data_index]["images_dir"], "{:0>4d}.jpg".format(img_index))
        
        audio_path = self.dataload_dict[on_data_index]["spectrograms_path"].replace("spectrograms.npy", "audio.mp3")
        if audio_path == self.last_mp3_path:
            wav_ori = self.wav_ori
        else:
            wav_ori = self.audio.load_wav(audio_path)
            self.wav_ori = wav_ori
        
        self.last_mp3_path = audio_path
        
        input_img = Image.open(img_path)
        id_img = Image.open(id_img_path)

        spectrograms = self.load_spectrogram(mel_index, self.dataload_dict[on_data_index])
        # wavs = self.load_wav_ori(mel_ori_index, wav_ori)

        if self.transform is not None:
            id_img = np.asarray(id_img).astype(np.uint8)
            input_img = np.asarray(input_img).astype(np.uint8)
            id_img = crop_img(id_img, self.img_size, self.crop)
            input_img = crop_img(input_img, self.img_size, self.crop)
            id_img = Image.fromarray(id_img.astype(np.uint8))
            input_img = Image.fromarray(input_img.astype(np.uint8))
            id_img = self.transform(id_img)
            input_img = self.transform(input_img)
        id_img = id_img.type(torch.FloatTensor)
        input_img = input_img.type(torch.FloatTensor)
        
        sp_list = id_img_path.split('/')
        # print(img_path)
        # img_idx = int(sp_list[-1].split('.')[0])
        img_idx = in_data_index + 2
        vid_idx = int(sp_list[-3])
        # id_idx = int(sp_list[-4][2:])
        id_idx = sp_list[-4]
        
        return id_img, input_img, spectrograms, id_idx, vid_idx, img_idx
            

    def __len__(self):
        return self.dataset_size

