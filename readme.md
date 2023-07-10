#### for inference test data of audio model and emo model
# --name set the save dir, you can make it to be the experiments name

#### for instance:

## run emotion generation
#   CUDA_VISIBLE_DEVICES=0 python inference.py --name test_emo

## run audio generation
#   CUDA_VISIBLE_DEVICES=0 python inference.py --name test_audio --audio_only

#### config file inference_emo.yaml
# 1. you can change data file in train_lmdb_path from test_data.json to train_data_stage3.json
# 2. you can change the value of data.sample_num_per_video_in_one_epoch, it represents the the number of consecutive frames in one 
#  video clip to be inference, you can use on_index and in_index to know different video and different frames independently
