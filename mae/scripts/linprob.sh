# MODEL_TYPE="finetuned-clip"
# MODEL_TYPE="clip"

# MODEL_TYPE="mae"
# CKPT_PATH=/scratch/choi/model/mae/mae_pretrain_vit_huge.pth

# MODEL_NAME="clip"
# CKPT_PATH=/scratch/choi/model/CLIP-ViT-H-14-laion2B-s32B-b79K/

MODEL_NAME="finetuned-clip_full_image-only_imagenet100_timestep-400-600_clip-loss-ratio-1_epoch-7"
# MODEL_NAME="finetuned-clip_full_image-text_imagenet100_timestep-400-600_epoch-1"
# CKPT_PATH=/scratch/choi/output/Diff-Rep/ViT-H/ip-adapter/full_image-only_mscoco_04-27_20:20/checkpoint-2000
CKPT_PATH=/scratch/choi/output/Diff-Rep/ViT-H/ip-adapter/full_image-text_imagenet100_timestep-400-600_clip-loss-ratio-1.0_05-17_16:10/checkpoint_42231-step_7-epoch


# DATASET_TYPE=in1k
DATASET_TYPE=in100

if [ "$DATASET_TYPE" = "in1k" ]; then
  DATA_PATH=/scratch/choi/dataset/ImageNet1K/
elif [ "$DATASET_TYPE" = "in100" ]; then
  DATA_PATH=/scratch/choi/dataset/ImageNet100
else
  echo "DATASET TYPE INVALID"
fi

# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=3 main_linprobe.py \
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=3 --master_port=29600 main_linprobe.py \
    --batch_size 512 \
    --model vit_huge_patch14 --cls_token \
    --finetune ${CKPT_PATH} \
    --epochs 51 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval --data_path ${DATA_PATH} \
    --output_dir /scratch/choi/output/Diff-Rep/ViT-H/linprob/${MODEL_NAME}/ \
    --log_dir /scratch/choi/output/Diff-Rep/ViT-H/linprob/${MODEL_NAME}/ \
    --model_type ${MODEL_NAME} \
    --linprob_dataset ${DATASET_TYPE} \
    --collect_features

    # --finetune /scratch/choi/output/Diff-Rep/mae/mae_pretrain_vit_base.pth \
    # --finetune /scratch/choi/output/Diff-Rep/clip/vit_b_16-laion400m_e32-55e67d44.pt \


# batch size 1024
# [22:05:11.591968] Epoch: [0]  [  0/417]  eta: 5:47:40  lr: 0.000000  loss: 6.9509 (6.9509)  time: 50.0249  data: 22.8922  max mem: 6322
# [22:07:21.951770] Epoch: [0]  [ 20/417]  eta: 0:56:49  lr: 0.005755  loss: 6.9537 (6.9549)  time: 6.5171  data: 4.9466  max mem: 6335
# [22:10:36.298865] Epoch: [0]  [ 40/417]  eta: 0:57:23  lr: 0.011511  loss: 6.9483 (6.9526)  time: 9.7063  data: 7.7207  max mem: 6335
# [22:15:21.300027] Epoch: [0]  [ 60/417]  eta: 1:04:19  lr: 0.017266  loss: 6.9453 (6.9496)  time: 14.2492  data: 12.8121  max mem: 6335


# batch size 512
# [22:17:04.517992] Epoch: [0]  [  0/834]  eta: 4:31:10  lr: 0.000000  loss: 6.9536 (6.9536)  time: 19.5086  data: 11.3293  max mem: 3332
# [22:17:15.743119] Epoch: [0]  [ 20/834]  eta: 0:19:51  lr: 0.001439  loss: 6.9553 (6.9547)  time: 0.5611  data: 0.0012  max mem: 3344
# [22:17:31.407465] Epoch: [0]  [ 40/834]  eta: 0:14:58  lr: 0.002878  loss: 6.9497 (6.9524)  time: 0.7831  data: 0.1356  max mem: 3344
# [22:17:51.068604] Epoch: [0]  [ 60/834]  eta: 0:13:58  lr: 0.004317  loss: 6.9520 (6.9518)  time: 0.9830  data: 0.2394  max mem: 3344
# [22:18:10.835114] Epoch: [0]  [ 80/834]  eta: 0:13:18  lr: 0.005755  loss: 6.9476 (6.9511)  time: 0.9882  data: 0.3118  max mem: 3344
# [22:18:29.818572] Epoch: [0]  [100/834]  eta: 0:12:41  lr: 0.007194  loss: 6.9424 (6.9494)  time: 0.9491  data: 0.2909  max mem: 3344

# batch size 2048
# RuntimeError: DataLoader worker (pid 40931) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit. 