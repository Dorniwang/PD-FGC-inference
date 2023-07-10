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
import torch.nn as nn

crop_ = True
# crop_ = False

class VOXTestDataset(nn.Module):

    def load_img(self, image_path, M=None, crop=crop_, crop_len=16, resize_size=224):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (resize_size, resize_size))
        # print("#####################################################", image_path)

        if img is None:
            print("dddd#####################################################", image_path)
            raise Exception('None Image')

        if M is not None:
            img = cv2.warpAffine(img, M, (self.crop_size, self.crop_size), borderMode=cv2.BORDER_REPLICATE)

        if crop:
            img = img[:self.crop_size - crop_len*2, crop_len:self.crop_size - crop_len]
            img = cv2.resize(img, (resize_size, resize_size))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def frame2audio_indexs(self, frame_inds):
        start_frame_ind = frame_inds - self.audio.num_frames_per_clip // 2

        start_audio_inds = start_frame_ind * self.audio.num_bins_per_frame
        return start_audio_inds

    def to_Tensor(self, img):
        if img.ndim == 3:
            wrapped_img = img.transpose(2, 0, 1) / 255.0
        elif img.ndim == 4:
            wrapped_img = img.transpose(0, 3, 1, 2) / 255.0
        else:
            wrapped_img = img / 255.0
        wrapped_img = torch.from_numpy(wrapped_img).float()

        return wrapped_img * 2 - 1

    def __init__(self, img_size):
        
        self.crop_size = img_size
        self.audio = AudioConfig(num_frames_per_clip=5, hop_size=160)
        self.num_audio_bins = self.audio.num_frames_per_clip * self.audio.num_bins_per_frame

    def set_data(self, audio_path, id_image_path, driving_dir):
        ### audio process
        wav = self.audio.read_audio(audio_path)
        self.spectrogram = self.audio.audio_to_spectrogram(wav)

        self.target_frame_inds = np.arange(2, len(self.spectrogram) // self.audio.num_bins_per_frame - 2)
        self.audio_inds = self.frame2audio_indexs(self.target_frame_inds)
        
        ### id image process
        if os.path.isdir(id_image_path):
            self.id_img = []
            item_list = list(os.listdir(id_image_path))
            for item in item_list[:50]:
                self.id_img.append(self.load_img(os.path.join(id_image_path, item)))
        else:
            self.id_img = self.load_img(id_image_path)

        ### driving image process
        if not driving_dir is None:
            self.drive_img = []
            if isinstance(driving_dir, list):
                for ddir in driving_dir:
                    item_list = list(os.listdir(ddir))
                    item_list.sort()
                    # print(ddir, len(item_list))
                    drive_img_temp = []
                    for item in item_list:
                        drive_img_temp.append(self.load_img(os.path.join(ddir, item)))
                    self.drive_img.append(drive_img_temp)
            else:
                item_list = list(os.listdir(driving_dir))
                item_list.sort()
                for item in item_list:
                    self.drive_img.append(self.load_img(os.path.join(driving_dir, item)))

    def load_spectrogram(self, audio_ind):
        mel_shape = self.spectrogram.shape
        # print("spectrogram shape: ", mel_shape)

        if (audio_ind + self.num_audio_bins) <= mel_shape[0] and audio_ind >= 0:
            spectrogram = np.array(self.spectrogram[audio_ind:audio_ind + self.num_audio_bins, :]).astype('float32')
        else:
            print('(audio_ind {} + opt.num_audio_bins {}) > mel_shape[0] {} '.format(audio_ind, self.num_audio_bins,
                                                                                     mel_shape[0]))
            if audio_ind > 0:
                spectrogram = np.array(self.spectrogram[audio_ind:audio_ind + self.num_audio_bins, :]).astype('float32')
            else:
                spectrogram = np.zeros((self.num_audio_bins, mel_shape[1])).astype(np.float16).astype(np.float32)

        spectrogram = torch.from_numpy(spectrogram)
        spectrogram = spectrogram.unsqueeze(0)

        spectrogram = spectrogram.transpose(-2, -1)
        return spectrogram

    def get_data(self, audio_path, id_image_path, driving_dir):
        
        self.set_data(audio_path, id_image_path, driving_dir)
        
        ### get id images
        if isinstance(self.id_img, list):
            id_img = []
            for iimg in self.id_img:
                id_img.append(self.to_Tensor(iimg[None, ...]))
            id_img = torch.cat(id_img, 0)
        else:
            id_img = self.to_Tensor(self.id_img[None, ...])
        
        ### get audio data
        spectrograms = []
        for index in range(self.audio_inds.shape[0]):
            mel_index = self.audio_inds[index]
            spectrogram = self.load_spectrogram(mel_index)
            spectrograms.append(spectrogram)

        spectrograms = torch.stack(spectrograms, 0)

        ### get drving images
        if not driving_dir is None:
            if isinstance(driving_dir, list):
                drive_img = []
                for drive_img_t in self.drive_img:
                    drive_img_temp = []
                    for iimg in drive_img_t:
                        drive_img_temp.append(self.to_Tensor(iimg[None, ...]))
                    drive_img_temp = torch.cat(drive_img_temp, 0)
                    drive_img.append(drive_img_temp)
            else:
                drive_img = []
                for iimg in self.drive_img:
                    drive_img.append(self.to_Tensor(iimg[None, ...]))
                drive_img = torch.cat(drive_img, 0)
        else:
            drive_img = None

        return id_img, spectrograms, drive_img


