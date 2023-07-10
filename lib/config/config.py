#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, xiaobing.                        #
# written by wangduomin@xiaobing.ai             #
#################################################

from .yacs import CfgNode as CN
import argparse
import os
import numpy as np
import pprint

cfg = CN()

### experiment name
cfg.exp_name = 'none'
cfg.task = 'none'
cfg.random_seed = 2021

### train type
cfg.trainer_type = "none"
cfg.dataset_type = "voxceleb2"

### train configuration for some specific
cfg.clip_len = 1
cfg.input_id_feature = True
cfg.num_clips = 1
cfg.num_inputs = 1

### network configuration and defination
cfg.model = CN()
cfg.model.vgg_path = ""
cfg.model.output_nc = 3
# identity network
cfg.model.net_appearance = CN()
cfg.model.net_appearance.model_name = "resnext"
cfg.model.net_appearance.model_type = "encoder"
cfg.model.net_appearance.load_pretrain = True
cfg.model.net_appearance.official_pretrain = ""
cfg.model.net_appearance.resume = False
cfg.model.net_appearance.pretrained_model = ""
cfg.model.net_appearance.init = "gaussian"
cfg.model.net_appearance.init_gain = 0.02

# non-identity network
cfg.model.net_motion = CN()
cfg.model.net_motion.model_name = "fan"
cfg.model.net_motion.model_type = "encoder"
cfg.model.net_motion.resume = False
cfg.model.net_motion.pretrained_model = ""
cfg.model.net_motion.init = "gaussian"
cfg.model.net_motion.init_gain = 0.02
cfg.model.net_motion.pose_dim = 12
cfg.model.net_motion.emo_dim = 30
cfg.model.net_motion.motion_dim = 0


# audio network
cfg.model.net_audio = CN()
cfg.model.net_audio.model_name = "ressesync"
cfg.model.net_audio.model_type = "encoder"
cfg.model.net_audio.resume = False
cfg.model.net_audio.pretrained_model = ""
cfg.model.net_audio.init = "gaussian"
cfg.model.net_audio.init_gain = 0.02
cfg.model.net_audio.n_mel_T = 4

# generator network
cfg.model.net_generator = CN()
cfg.model.net_generator.model_name = "modulate"
cfg.model.net_generator.model_type = "generator"
cfg.model.net_generator.resume = False
cfg.model.net_generator.pretrained_model = ""
cfg.model.net_generator.init = "gaussian"
cfg.model.net_generator.init_gain = 0.02
cfg.model.net_generator.style_dim = 2560
cfg.model.net_generator.feature_encoded_dim = 2560
cfg.model.net_generator.style_feature_loss = True

# discriminator network
cfg.model.net_discriminator = CN()
cfg.model.net_discriminator.model_name = "multiscale"
cfg.model.net_discriminator.model_type = "discriminator"
cfg.model.net_discriminator.resume = False
cfg.model.net_discriminator.pretrained_model = ""
cfg.model.net_discriminator.init = "gaussian"
cfg.model.net_discriminator.init_gain = 0.02
cfg.model.net_discriminator.n_layers_D = 4
cfg.model.net_discriminator.netD_subarch = "n_layer"
cfg.model.net_discriminator.D_input = "single"
cfg.model.net_discriminator.num_D = 2
cfg.model.net_discriminator.no_ganFeat_loss = False
cfg.model.net_discriminator.ndf = 64
cfg.model.net_discriminator.norm_D = "spectralinstance"
cfg.model.net_discriminator.label_nc = 1
cfg.model.net_discriminator.contain_dontcare_label = False
cfg.model.net_discriminator.no_instance = False


### data configuration
cfg.data = CN()
cfg.data.train_lmdb_path = "/mnt/wangduomin/data/voxceleb_processed/train_data/"
cfg.data.valid_lmdb_path = ""
cfg.data.num_workers = 8
cfg.data.num_classes = 5995
cfg.data.img_size = 224
# audio configuration
cfg.data.audio = CN()
cfg.data.audio.sample_rate = 16000
cfg.data.audio.hop_size = 160
cfg.data.audio.frame_rate = 25
cfg.data.audio.num_frames_per_clip = 5
# augmentation configuration
cfg.data.augitem = CN()
cfg.data.augitem.brightness = 0.5
cfg.data.augitem.contrast = 0.4
cfg.data.augitem.saturation = 0.4
cfg.data.augitem.hue = 0.4
cfg.data.augitem.distortion_scale = 0.1
cfg.data.augitem.perspective_prob = 0.9

cfg.data.augitem.mirror = False
cfg.data.augitem.blur = False
cfg.data.augitem.noise = False
cfg.data.augitem.noise_peak = 1

### train configuration
cfg.train = CN()
cfg.train.max_epoch = 50
cfg.train.max_iter = 500000
cfg.train.batch_size = 32 # batch size on each gpu
cfg.train.eval_batch_size = 100
cfg.train.num_eval_imgs = 50000

# net_appearance
cfg.train.net_appearance = CN()
cfg.train.net_appearance.weight_decay = 0.001
cfg.train.net_appearance.optimize = 'adam'
cfg.train.net_appearance.optimize_sam = False
cfg.train.net_appearance.optim_beta1 = 0
cfg.train.net_appearance.optim_beta2 = 0.99
cfg.train.net_appearance.loss = "ce"
cfg.train.net_appearance.lr = 0.0001 # identity learning rate
cfg.train.net_appearance.lr_scheduler = CN()
cfg.train.net_appearance.lr_scheduler.type = 'step'
cfg.train.net_appearance.lr_scheduler.step_size = 80000
cfg.train.net_appearance.lr_scheduler.gamma = 0.93
cfg.train.net_appearance.params = CN()
cfg.train.net_appearance.params.train = "all" # all | no
cfg.train.net_appearance.params.freeze_bn = False 
cfg.train.net_appearance.params.freeze_bn_part = "no" # all | no

# net_motion
cfg.train.net_motion = CN()
cfg.train.net_motion.weight_decay = 0.001
cfg.train.net_motion.optimize = 'adam'
cfg.train.net_motion.optimize_sam = False
cfg.train.net_motion.optim_beta1 = 0
cfg.train.net_motion.optim_beta2 = 0.99
cfg.train.net_motion.loss = ""
cfg.train.net_motion.lr = 0.0001 # nonidentity learning rate
cfg.train.net_motion.lr_scheduler = CN()
cfg.train.net_motion.lr_scheduler.type = 'step'
cfg.train.net_motion.lr_scheduler.step_size = 80000
cfg.train.net_motion.lr_scheduler.gamma = 0.93
cfg.train.net_motion.params = CN()
cfg.train.net_motion.params.train = "all" # all | encoder | pose | to_mouth
cfg.train.net_motion.params.freeze_bn = False 
cfg.train.net_motion.params.freeze_bn_part = "no" # encoder | no

# net_audio
cfg.train.net_audio = CN()
cfg.train.net_audio.weight_decay = 0.001
cfg.train.net_audio.optimize = 'adam'
cfg.train.net_audio.optimize_sam = False
cfg.train.net_audio.optim_beta1 = 0
cfg.train.net_audio.optim_beta2 = 0.99
cfg.train.net_audio.loss = "softmax_contrastive"
cfg.train.net_audio.lr = 0.0001 # audio learning rate
cfg.train.net_audio.lr_scheduler = CN()
cfg.train.net_audio.lr_scheduler.type = 'step'
cfg.train.net_audio.lr_scheduler.step_size = 80000
cfg.train.net_audio.lr_scheduler.gamma = 0.93
cfg.train.net_audio.no_cross_modal = False
cfg.train.net_audio.params = CN()
cfg.train.net_audio.params.train = "all" # all | no
cfg.train.net_audio.params.freeze_bn = False 
cfg.train.net_audio.params.freeze_bn_part = "no" # all | no

# net_generator
cfg.train.net_generator = CN()
cfg.train.net_generator.weight_decay = 0.001
cfg.train.net_generator.optimize = 'adam'
cfg.train.net_generator.optimize_sam = False
cfg.train.net_generator.optim_beta1 = 0
cfg.train.net_generator.optim_beta2 = 0.99
cfg.train.net_generator.loss = "hinge"
cfg.train.net_generator.lr = 0.0001 # generator learning rate
cfg.train.net_generator.lr_scheduler = CN()
cfg.train.net_generator.lr_scheduler.type = 'step'
cfg.train.net_generator.lr_scheduler.step_size = 80000
cfg.train.net_generator.lr_scheduler.gamma = 0.93
cfg.train.net_generator.params = CN()
cfg.train.net_generator.params.train = "all"
cfg.train.net_generator.params.freeze_bn = False 
cfg.train.net_generator.params.freeze_bn_part = "no" 

# net_discriminator
cfg.train.net_discriminator = CN()
cfg.train.net_discriminator.weight_decay = 0.001
cfg.train.net_discriminator.optimize = 'adam'
cfg.train.net_discriminator.optimize_sam = False
cfg.train.net_discriminator.optim_beta1 = 0
cfg.train.net_discriminator.optim_beta2 = 0.99
cfg.train.net_discriminator.loss = "hinge"
cfg.train.net_discriminator.lr = 0.0001 # discriminator learning rate
cfg.train.net_discriminator.lr_scheduler = CN()
cfg.train.net_discriminator.lr_scheduler.type = 'step'
cfg.train.net_discriminator.lr_scheduler.step_size = 80000
cfg.train.net_discriminator.lr_scheduler.gamma = 0.93
cfg.train.net_discriminator.params = CN()
cfg.train.net_discriminator.params.train = "all"
cfg.train.net_discriminator.params.freeze_bn = False 
cfg.train.net_discriminator.params.freeze_bn_part = "no" 

# other configuration for train
cfg.train.lambda_image = 1.0
cfg.train.lambda_feat = 10
cfg.train.lambda_vggface = 5.0
cfg.train.lambda_vgg = 10.0
cfg.train.lambda_rotate_D = 0.1
cfg.train.lambda_D = 1
cfg.train.lambda_softmax = 1000000
cfg.train.lambda_crossmodal = 1
cfg.train.lambda_contrastive = 100

cfg.train.resume_model = ""
cfg.train.resume = False

### recoder
cfg.record_dir = ''
cfg.val_freq = 100
cfg.log_iter = 1000
cfg.image_save_iter = 2000
cfg.display_size = 8
cfg.snapshot_save_iter = 20000

def parse_cfg(cfg, args):
    if cfg.task is None:
        raise ValueError("task must be specified")

    # cfg.img_type = cfg.data.img_type
    cfg.img_size = cfg.data.img_size
    cfg.freeze_bn = args.freeze_bn
    # assign the gpus
    cfg.local_rank = args.local_rank
    cfg.distributed = args.distributed

    cfg.mouth = args.mouth
    cfg.blink = args.blink
    cfg.gaze = args.gaze
    cfg.emo = args.emo
    cfg.headpose = args.headpose
    cfg.all = args.all
    
    cfg.name = args.name
    cfg.audio_path = args.audio_path
    cfg.app_img_path = args.app_img_path
    cfg.pose_path = args.pose_path
    cfg.exp_path = args.exp_path
    cfg.eye_path = args.eye_path

def make_cfg(args):
    # args.cfg_file = "configs/inference.yaml"
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    parse_cfg(cfg, args)
    print("exp config is {}, generate video {}".format(args.cfg_file, args.name))
    return cfg

parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/inference.yaml", type=str)
parser.add_argument("--name", default="", type=str)
parser.add_argument("--audio_path", default="", type=str)
parser.add_argument("--app_img_path", default="", type=str)
parser.add_argument("--pose_path", default="", type=str)
parser.add_argument("--exp_path", default="", type=str)
parser.add_argument("--eye_path", default="", type=str)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--freeze_bn", action='store_true')
parser.add_argument("--distributed", action='store_true')
parser.add_argument("--mouth", action='store_true', default=False)
parser.add_argument("--blink", action='store_true', default=False)
parser.add_argument("--gaze", action='store_true', default=False)
parser.add_argument("--emo", action='store_true', default=False)
parser.add_argument("--headpose", action='store_true', default=False)
parser.add_argument("--all", action='store_true', default=False)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

cfg = make_cfg(args)