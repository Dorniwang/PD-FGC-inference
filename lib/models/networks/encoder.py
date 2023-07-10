import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from util import util
from lib.models.networks.audio_network import ResNetSE, SEBasicBlock
from lib.models.networks.FAN_feature_extractor import FAN_use
from lib.models.networks.vision_network import ResNeXt50


class ResSEAudioEncoder(nn.Module):
    def __init__(self, opt, nOut=2048, n_mel_T=None):
        super(ResSEAudioEncoder, self).__init__()
        self.nOut = nOut
        self.opt = opt
        pose_dim = self.opt.model.net_motion.pose_dim
        eye_dim = self.opt.model.net_motion.eye_dim
        motion_dim = self.opt.model.net_motion.motion_dim
        # Number of filters
        num_filters = [32, 64, 128, 256]
        if n_mel_T is None: # use it when use audio identity
            n_mel_T = opt.model.net_audio.n_mel_T
        self.model = ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters, self.nOut, n_mel_T=n_mel_T)
        self.mouth_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, 512-pose_dim-eye_dim))
            

    def forward(self, x, _type=None):
        input_size = x.size()
        if len(input_size) == 5:
            bz, clip_len, c, f, t = input_size
            x = x.view(bz * clip_len, c, f, t)
        out = self.model(x)
        
        out = out.view(-1, out.shape[-1])
        mouth_embed = self.mouth_embed(out)
        return out, mouth_embed



class ResSESyncEncoder(ResSEAudioEncoder):
    def __init__(self, opt):
        super(ResSESyncEncoder, self).__init__(opt, nOut=512, n_mel_T=1)


class ResNeXtEncoder(ResNeXt50):
    def __init__(self, opt):
        super(ResNeXtEncoder, self).__init__(opt)


class FanEncoder(nn.Module):
    def __init__(self, opt):
        super(FanEncoder, self).__init__()
        self.opt = opt
        pose_dim = self.opt.model.net_motion.pose_dim
        eye_dim = self.opt.model.net_motion.eye_dim
        self.model = FAN_use()

        self.to_mouth = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512))
        self.mouth_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, 512-pose_dim-eye_dim))
        
        self.to_headpose = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512))
        self.headpose_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, pose_dim))

        self.to_eye = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512))
        self.eye_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, eye_dim))

        self.to_emo = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512))
        self.emo_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, 30))

    def forward_feature(self, x):
        net = self.model(x)
        return net

    def forward(self, x):
        x = self.model(x)
        mouth_feat = self.to_mouth(x)
        headpose_feat = self.to_headpose(x)
        headpose_emb = self.headpose_embed(headpose_feat)
        eye_feat = self.to_eye(x)
        eye_embed = self.eye_embed(eye_feat)
        emo_feat = self.to_emo(x)
        emo_embed = self.emo_embed(emo_feat)
        return headpose_emb, eye_embed, emo_embed, mouth_feat
            