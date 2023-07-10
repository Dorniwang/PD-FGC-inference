#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, xiaobing.                        #
# written by wangduomin@xiaobing.ai             #
#################################################

import os
import cv2
import numpy as np
from lib.config.config import cfg
from lib.inferencer import Tester

save_dir = "test/images/"
mp4_dir = "test/mp4s/"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(mp4_dir, exist_ok=True)

if __name__ == "__main__":

    tester = Tester(cfg)

    tester.reset_cfg(cfg)

    audio_path = cfg.audio_path
    app_img_path = cfg.app_img_path
    driving_path = [cfg.pose_path, cfg.exp_path, cfg.eye_path]

    _save_dir = "test/images/{}".format(cfg.name)
    os.makedirs(_save_dir, exist_ok=True)
    
    images_return = tester.inference(audio_path, app_img_path, driving_path)
    for idx, img in enumerate(images_return):
        cv2.imwrite(os.path.join(_save_dir, "{:0>4d}.jpg".format(idx)), img)
    
    save_mp4 = "test/mp4s/{}.mp4".format(cfg.name)
    command = "ffmpeg -y -r 25 -i {}/%04d.jpg -i {} {}".format(_save_dir, audio_path, save_mp4)
    os.system(command)
