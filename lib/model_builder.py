#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, xiaobing.                        #
# written by wangduomin@xiaobing.ai             #
#################################################

import math
import torch
import torch.nn.init as init
import numpy as np
# from thop import profile, clever_format
import lib.models.networks as models

def make_model(cfg):
    # net object init
    net_appearance = None
    net_motion = None
    net_audio = None
    net_generator = None

    ############################## build model #############################################################
    
    # create identity model
    net_appearance = models.define_networks(
        cfg, 
        cfg.model.net_appearance.model_name, 
        cfg.model.net_appearance.model_type
    )
    # create non-identity model
    net_motion = models.define_networks(
        cfg, 
        cfg.model.net_motion.model_name, 
        cfg.model.net_motion.model_type
    )
    # create audio model
    net_audio = models.define_networks(
        cfg, 
        cfg.model.net_audio.model_name, 
        cfg.model.net_audio.model_type
    )
    # create generator
    net_generator = models.define_networks(
        cfg, 
        cfg.model.net_generator.model_name, 
        cfg.model.net_generator.model_type
    )

    ########################## initialize model ###############################################################
    # load pretrained or random initialize net_appearance model
    return_list = []
    if not net_appearance is None:
        if cfg.model.net_appearance.resume:
            # load pretrained model
            model_file = cfg.model.net_appearance.pretrained_model
            # init_dict = {k:v for k, v in torch.load(model_file).items()}
            model_dict = torch.load(model_file)
            init_dict = net_appearance.state_dict()
            for k in model_dict:
                if "fc" in k and not "model" in k:
                    continue
                init_dict[k] = model_dict[k]
                # print("initializing net_appearance's parameters {} from {}".format(k, model_file))
            net_appearance.load_state_dict(init_dict)
            
        else:
            print("ERROR: identity model not be loaded!")
        return_list.append(net_appearance)
        print("identity model loaded!")
    else:
        print("ERROR: identity model needed!")

    # load pretrained or random initialize net_motion model
    if not net_motion is None:
        if cfg.model.net_motion.resume:
            # load pretrained model
            model_file = cfg.model.net_motion.pretrained_model
            model_dict = torch.load(model_file)
            init_dict = net_motion.state_dict()
            key_list = list(set(model_dict.keys()).intersection(set(init_dict.keys())))
            for k in key_list:
                
                if "mouth_fc" in k or "headpose_fc" in k or "classifier" in k or "to_feature" in k or "to_embed" in k:
                    continue
                    
                init_dict[k] = model_dict[k]
            net_motion.load_state_dict(init_dict)
        else:
            print("ERROR: non-identity model not be loaded!")
        return_list.append(net_motion)
        print("non-identity model loaded!")
    else:
        print("ERROR: non-identity model needed!")

    # load pretrained or random initialize net_audio model
    if not net_audio is None:
        if cfg.model.net_audio.resume:
            # load pretrained model
            model_file = cfg.model.net_audio.pretrained_model
            model_dict = torch.load(model_file)
            init_dict = net_audio.state_dict()
            for k in model_dict:
                if "fc" in k and not "model" in k:
                    continue
                if "model.layer4_exp" in k or "model.attention_exp" in k or "model.fc_exp" in k or "exp_embed" in k:
                    continue
                init_dict[k] = model_dict[k]

            net_audio.load_state_dict(init_dict)
        else:
            print("ERROR: audio model not be loaded!")
        return_list.append(net_audio)
        print("audio model loaded!")
    else:
        print("ERROR: audio model needed!")

    # load pretrained or random initialize net_audio model
    if not net_generator is None:
        if cfg.model.net_generator.resume:
            # load pretrained model
            model_file = cfg.model.net_generator.pretrained_model
            init_dict = net_generator.state_dict()
            model_dict = torch.load(model_file)
            for k in init_dict:
                init_dict[k] = model_dict[k]
                
            net_generator.load_state_dict(init_dict)
        else:
            print("ERROR: generator model not be loaded!")
        return_list.append(net_generator)
        print("generator model loaded!")
    else:
        print("ERROR: generator model needed!")
        
    print("all models load sucessfully!")

    return return_list
