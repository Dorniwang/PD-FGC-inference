
### case 1
CUDA_VISIBLE_DEVICES=0 python inference.py --all --name test1 \
    --audio_path "data/audios/audio_longer.mp3" \
    --app_img_path "data/apps/anne.png" \
    --pose_path "data/motions/pose/681600002/" \
    --exp_path "data/motions/exp/vox2_id01228_00309/" \
    --eye_path "data/motions/eye/id01333_00040/images/"

### case 2
CUDA_VISIBLE_DEVICES=0 python inference.py --all --name test2 \
    --audio_path "data/audios/audio_longer.mp3" \
    --app_img_path "data/apps/anne.png" \
    --pose_path "data/motions/pose/681600002/" \
    --exp_path "data/motions/exp/vox2_id01333_00040/" \
    --eye_path "data/motions/eye/id01333_00040/images/"

### case 3
CUDA_VISIBLE_DEVICES=0 python inference.py --all --name test3 \
    --audio_path "data/audios/audio_longer.mp3" \
    --app_img_path "data/apps/anne.png" \
    --pose_path "data/motions/pose/681600002/" \
    --exp_path "data/motions/exp/vox2_id01224_00294/" \
    --eye_path "data/motions/eye/id01333_00040/images/"

### case 4
CUDA_VISIBLE_DEVICES=0 python inference.py --all --name test4 \
    --audio_path "data/audios/audio_longer.mp3" \
    --app_img_path "data/apps/anne.png" \
    --pose_path "data/motions/pose/681600002/" \
    --exp_path "data/motions/exp/vox2_id01000_00077/" \
    --eye_path "data/motions/eye/id01333_00040/images/"

### case 5
CUDA_VISIBLE_DEVICES=0 python inference.py --all --name test5 \
    --audio_path "data/audios/playaudio_20s.mp3" \
    --app_img_path "data/apps/anne.png" \
    --pose_path "data/motions/pose/681600002/" \
    --exp_path "data/motions/exp/vox2_id01228_00309/" \
    --eye_path "data/motions/eye/id01333_00040/images/"

### case 6
CUDA_VISIBLE_DEVICES=0 python inference.py --all --name test6 \
    --audio_path "data/audios/playaudio_20s.mp3" \
    --app_img_path "data/apps/anne.png" \
    --pose_path "data/motions/pose/681600002/" \
    --exp_path "data/motions/exp/vox2_id01333_00040/" \
    --eye_path "data/motions/eye/id01333_00040/images/"

### case 7
CUDA_VISIBLE_DEVICES=0 python inference.py --all --name test7 \
    --audio_path "data/audios/playaudio_20s.mp3" \
    --app_img_path "data/apps/anne.png" \
    --pose_path "data/motions/pose/681600002/" \
    --exp_path "data/motions/exp/vox2_id01224_00294/" \
    --eye_path "data/motions/eye/id01333_00040/images/"

### case 8
CUDA_VISIBLE_DEVICES=0 python inference.py --all --name test8 \
    --audio_path "data/audios/playaudio_20s.mp3" \
    --app_img_path "data/apps/anne.png" \
    --pose_path "data/motions/pose/681600002/" \
    --exp_path "data/motions/exp/vox2_id01000_00077/" \
    --eye_path "data/motions/eye/id01333_00040/images/"