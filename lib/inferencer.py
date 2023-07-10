#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, xiaobing.                        #
# written by wangduomin@xiaobing.ai             #
#################################################

import os
import cv2
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.model_builder import make_model
from lib.data.voxtest_dataset import VOXTestDataset


def post_process(images_tensor_list):
    # shape of [n, 4, b, c, h, w]
    len_ = len(images_tensor_list)
    num = len(images_tensor_list[0])
    images_list = [[] for i in range(num)]
    for i in range(len_):
        for j in range(num):
            images_list[j].append(images_tensor_list[i][j])

    for i in range(num):
        images_list[i] = torch.cat(images_list[i], 0)
    
    images_list = torch.cat(images_list, 3)

    images_return_list = []
    for i in range(images_list.shape[0]):
        img_tensor = images_list[i] # [b, c, h, w]
        imgs = img_tensor.detach().cpu().numpy()
        imgs = (np.transpose(imgs, (1, 2, 0)) + 1) / 2.0 * 255.0
        imgs = imgs.astype(np.uint8)
        images_return_list.append(imgs[:, :, [2, 1, 0]])
    return images_return_list


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def freeze_model(net):
    for param in net.parameters():
        param.requires_grad = False
    net.apply(fix_bn)

def conv_feat(features, k_size, weight=None, sigma=1.0):
    c = features.shape[1]
    if weight is None:
        pad = k_size // 2
        k = np.zeros(k_size).astype(np.float)
        for x in range(-pad, k_size-pad):
            k[x+pad] = np.exp(-x**2 / (2 * (sigma ** 2)))
        k = k / k.sum()
        print(k)
    else:
        k_size = len(weight)
        k = np.array(weight)
        pad = k_size // 2
        print(k)
    
    k = torch.from_numpy(k).to(features.device).float().unsqueeze(0).unsqueeze(0)
    k = k.repeat(c, 1, 1)
    features = features.unsqueeze(0).permute(0, 2, 1) # [1, 512, n]
    features = F.conv1d(features, k, padding=pad, groups=c)
    features = features.permute(0, 2, 1).squeeze(0)

    return features

class Tester(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.img_size = self.cfg.img_size
        self.batch_size = 8

        self.net_appearance, self.net_motion, self.net_audio, self.net_generator = make_model(self.cfg)
        self.net_appearance.cuda()
        self.net_motion.cuda()
        self.net_audio.cuda()
        self.net_generator.cuda()

        ### set eval and freeze models
        freeze_model(self.net_appearance)
        freeze_model(self.net_motion)
        freeze_model(self.net_audio)
        freeze_model(self.net_generator)

        self.net_appearance.eval()
        self.net_motion.eval()
        self.net_audio.eval()
        self.net_generator.eval()

        self.mouth = self.cfg.mouth
        self.blink = self.cfg.blink
        self.gaze = self.cfg.gaze
        self.emo = self.cfg.emo
        self.headpose = self.cfg.headpose

        self.dataset = VOXTestDataset(self.img_size)

        torch.manual_seed(self.cfg.random_seed)
        torch.cuda.manual_seed(self.cfg.random_seed)
        torch.cuda.manual_seed_all(self.cfg.random_seed)
        np.random.seed(self.cfg.random_seed)
        random.seed(self.cfg.random_seed)


    def reset_cfg(self, cfg):
        self.cfg = cfg

        self.mouth = self.cfg.mouth
        self.blink = self.cfg.blink
        self.gaze = self.cfg.gaze
        self.emo = self.cfg.emo
        self.headpose = self.cfg.headpose
        if self.cfg.all:
            self.mouth = self.blink = self.gaze = self.emo = self.headpose = True

    def generate_fake(self, id_feature, pose_feature):
        pose_feature = pose_feature.view(-1, pose_feature.shape[-1])
        style = torch.cat([id_feature[0].repeat(pose_feature.shape[0], 1), pose_feature], 1)
        style = [style]
        fake_image, style_rgb = self.net_generator(style, identity_style=id_feature[1].repeat(pose_feature.shape[0], 1, 1, 1))

        fake_image = fake_image.view(-1, 3, self.img_size, self.img_size)

        return fake_image, style_rgb

    def inference(self, audio_obj, id_img_obj, driving_obj=None):
        return self.__test(audio_obj, id_img_obj, driving_obj)
    
    def __test(self, audio_obj, id_img_obj, driving_obj):

        test_start = time.time()
        id_img, spectrograms, driving_imgs = self.dataset.get_data(audio_obj, id_img_obj, driving_obj)

        id_img = id_img.cuda()
        spectrograms = spectrograms.cuda()
        if not driving_obj is None:
            if isinstance(driving_obj, list):
                driving_imgs = [driving_img[2:].cuda() for driving_img in driving_imgs]
            else:
                driving_imgs = driving_imgs.cuda()

        emo_mem = None

        total_size = spectrograms.shape[0]
        dri_len0 = driving_imgs[0].shape[0]
        dri_len1 = driving_imgs[1].shape[0]
        dri_len2 = driving_imgs[2].shape[0]
        with torch.no_grad():
            ### net_appearance forward
            id_feature, id_scores = self.net_appearance(id_img)

            mouth_feat_list = []
            pose_feat_list = []
            eye_feat_list = []
            emo_feat_list = []
            # print(total_size)
            for iters in range(int(round(total_size / self.batch_size + 0.4999999))):
                spectrogram = spectrograms[iters * self.batch_size:min((iters + 1) * self.batch_size, total_size)]
                # print(spectrogram.shape)
                A_mouth_feature, A_mouth_embed = self.net_audio.forward(spectrogram)
                A_mouth_embed = A_mouth_embed * 1.5
                mouth_feat_list.append(A_mouth_embed)

                dri_start0 = (iters * self.batch_size) % dri_len0
                dri_end0 = ((iters + 1) * self.batch_size) % dri_len0
                if dri_start0 < dri_end0:
                    driving_img0 = driving_imgs[0][dri_start0:dri_end0]
                else:
                    driving_img0 = torch.cat([driving_imgs[0][dri_start0:], driving_imgs[0][:dri_end0]], 0)
                driving_img0 = driving_img0[:min(self.batch_size, abs(total_size - iters * self.batch_size))]
                
                dri_start1 = (iters * self.batch_size) % dri_len1
                dri_end1 = ((iters + 1) * self.batch_size) % dri_len1
                if dri_start1 < dri_end1:
                    driving_img1 = driving_imgs[1][dri_start1:dri_end1]
                else:
                    driving_img1 = torch.cat([driving_imgs[1][dri_start1:], driving_imgs[1][:dri_end1]], 0)
                driving_img1 = driving_img1[:min(self.batch_size, abs(total_size - iters * self.batch_size))]

                dri_start2 = (iters * self.batch_size) % dri_len2
                dri_end2 = ((iters + 1) * self.batch_size) % dri_len2
                if dri_start2 < dri_end2:
                    driving_img2 = driving_imgs[2][dri_start2:dri_end2]
                else:
                    driving_img2 = torch.cat([driving_imgs[2][dri_start2:], driving_imgs[2][:dri_end2]], 0)
                driving_img2 = driving_img2[:min(self.batch_size, abs(total_size - iters * self.batch_size))]

                V_headpose_embed, V_eye_embed, emo_feat, mouth_feat = self.net_motion(torch.cat([driving_img0, driving_img1, driving_img2], 0))
                
                pose_feat_list.append(V_headpose_embed[:A_mouth_embed.shape[0]])
                eye_feat_list.append(V_eye_embed[A_mouth_embed.shape[0]*2:])
                emo_feat_list.append(emo_feat[A_mouth_embed.shape[0]:A_mouth_embed.shape[0]*2])

            mouth_feat_list = torch.cat(mouth_feat_list, 0)
            pose_feat_list = torch.cat(pose_feat_list, 0)
            eye_feat_list = torch.cat(eye_feat_list, 0)
            emo_feat_list = torch.cat(emo_feat_list, 0)
            
            pose_feat_smooth_list = []
            for i in range(total_size):
                if i == 0:
                    pose_feat_smooth_list.append(pose_feat_list[i:i+2].mean(0)[None,:])
                elif i == total_size - 1:
                    pose_feat_smooth_list.append(pose_feat_list[i-1:].mean(0)[None,:])
                else:
                    pose_feat_smooth_list.append(pose_feat_list[i-1:i+2].mean(0)[None,:])
            pose_feat_smooth_list = torch.cat(pose_feat_smooth_list, 0)
            # pose_feat_smooth_list = torch.cat(pose_feat_smooth_list, 0) + torch.from_numpy(np.array([0.8, 0,  1.2, -0.8, 0, -0.8]))[None].cuda().float()


            mouth_feat_list = conv_feat(mouth_feat_list, k_size=3, sigma=1)

            gen_images_list = []
            for iters in range(int(round(total_size / self.batch_size + 0.4999999))):
                
                dri_start0 = (iters * self.batch_size) % dri_len0
                dri_end0 = ((iters + 1) * self.batch_size) % dri_len0
                if dri_start0 < dri_end0:
                    driving_img0 = driving_imgs[0][dri_start0:dri_end0]
                else:
                    driving_img0 = torch.cat([driving_imgs[0][dri_start0:], driving_imgs[0][:dri_end0]], 0)
                driving_img0 = driving_img0[:min(self.batch_size, abs(total_size - iters * self.batch_size))]

                dri_start1 = (iters * self.batch_size) % dri_len1
                dri_end1 = ((iters + 1) * self.batch_size) % dri_len1
                if dri_start1 < dri_end1:
                    driving_img1 = driving_imgs[1][dri_start1:dri_end1]
                else:
                    driving_img1 = torch.cat([driving_imgs[1][dri_start1:], driving_imgs[1][:dri_end1]], 0)
                driving_img1 = driving_img1[:min(self.batch_size, abs(total_size - iters * self.batch_size))]

                dri_start2 = (iters * self.batch_size) % dri_len2
                dri_end2 = ((iters + 1) * self.batch_size) % dri_len2
                if dri_start2 < dri_end2:
                    driving_img2 = driving_imgs[2][dri_start2:dri_end2]
                else:
                    driving_img2 = torch.cat([driving_imgs[2][dri_start2:], driving_imgs[2][:dri_end2]], 0)
                driving_img2 = driving_img2[:min(self.batch_size, abs(total_size - iters * self.batch_size))]

                
                A_mouth_embed = mouth_feat_list[iters * self.batch_size:min((iters + 1) * self.batch_size, total_size)]
                V_eye_embed = eye_feat_list[iters * self.batch_size:min((iters + 1) * self.batch_size, total_size)] * 1.2
                pose_feat = pose_feat_smooth_list[iters * self.batch_size:min((iters + 1) * self.batch_size, total_size)]
                emo_feat = emo_feat_list[iters * self.batch_size:min((iters + 1) * self.batch_size, total_size)] * 1.2

                if not self.mouth:
                    A_mouth_embed = torch.zeros_like(A_mouth_embed).cuda()
                if not self.headpose:
                    pose_feat = torch.zeros_like(pose_feat).cuda()
                if not self.blink:
                    V_eye_embed = V_eye_embed * torch.from_numpy(np.array([[0,0,1,1,0,0]])).cuda()
                if not self.gaze:
                    V_eye_embed = V_eye_embed * torch.from_numpy(np.array([[1,1,0,0,1,1]])).cuda()
                if not self.emo:
                    emo_feat = torch.zeros_like(emo_feat).cuda()

                pose_feature = torch.cat((A_mouth_embed, emo_feat, V_eye_embed, pose_feat), dim=1) # audio embed to drive mouth

                gen_images, _ = self.generate_fake(id_feature, pose_feature)
                save_img = [id_img.repeat(gen_images.shape[0], 1, 1, 1), gen_images, driving_img0, driving_img1, driving_img2]
                gen_images_list.append(save_img)
            
            gen_images = post_process(gen_images_list)
        
        test_end = time.time()
        print("total time is {:>03f}s".format(test_end - test_start))
        return gen_images
        
        
            
if "__main__" == __name__:
    tester = Tester()