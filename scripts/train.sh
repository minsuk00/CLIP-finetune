CLIP_PATH=/scratch/choi/model/IP-Adapter/models/image_encoder
IP_ADAPTER_PATH=/scratch/choi/model/IP-Adapter/models/ip-adapter_sd15.bin
SD_PATH=/scratch/choi/model/stable-diffusion-v1-5
OUT_DIR=/scratch/choi/output/Diff-Rep/ViT-H/ip-adapter

###################### CUSTOM CONFIG ######################
#mscoco, imagenet1k
# DATASET_TYPE=imagenet1k
# DATASET_TYPE=mscoco
DATASET_TYPE=imagenet100

#full, only-clip
# TRAIN_TYPE=only-clip 
TRAIN_TYPE=full

#image-only, text-image
TRAIN_MODALITY=image-text
# TRAIN_MODALITY=image-only

# TIME_STEP=all
TIMESTEP=400-600

RESUME_PATH=_
# RESUME_PATH=/scratch/choi/output/Diff-Rep/ViT-H/ip-adapter/full_image-text_imagenet100_timestep-400-600_clip-loss-ratio-0.999_05-19_15:51/checkpoint_42231-step_7-epoch

# CLIP_LOSS_RATIO=0.999
# CLIP_LOSS_RATIO=1.0
CLIP_LOSS_RATIO=0.0
###########################################################

if [ "$DATASET_TYPE" = "mscoco" ]; then
  DATA_JSON_FILE=/scratch/choi/dataset/mscoco/annotations/_img_text_pair_train2017.json
  DATA_ROOT_PATH=/scratch/choi/dataset/mscoco/train2017
elif [ "$DATASET_TYPE" = "imagenet1k" ]; then
  DATA_JSON_FILE=/scratch/choi/dataset/ImageNet1K/_img_text_pair_train.json
  DATA_ROOT_PATH=/scratch/choi/dataset/ImageNet1K/train
elif [ "$DATASET_TYPE" = "imagenet100" ]; then
  DATA_JSON_FILE=/scratch/choi/dataset/ImageNet100/_img_text_pair_train.json
  DATA_ROOT_PATH=/scratch/choi/dataset/ImageNet100/train
else
  echo "DATASET TYPE INVALID"
fi

accelerate launch --num_processes=3 --multi_gpu --main_process_port=29700 --mixed_precision "fp16" \
  tutorial_train.py \
--pretrained_model_name_or_path ${SD_PATH} \
  --pretrained_ip_adapter_path ${IP_ADAPTER_PATH} \
  --image_encoder_path ${CLIP_PATH} \
  --resume_path ${RESUME_PATH} \
  --train_type ${TRAIN_TYPE} \
  --data_json_file ${DATA_JSON_FILE} \
  --data_root_path ${DATA_ROOT_PATH} \
  --dataset_type ${DATASET_TYPE} \
  --train_modality ${TRAIN_MODALITY} \
  --timestep ${TIMESTEP} \
  --mixed_precision="fp16" \
  --resolution=224 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-4 \
  --weight_decay=0.01 \
  --output_dir ${OUT_DIR} \
  --save_steps=2000 \
  --eval_epoch=1 \
  --clip_loss_ratio=${CLIP_LOSS_RATIO} \
  --num_train_epochs 100 \
  --train_batch_size=8 \
  --gradient_accum_step=32
 

  # --gradient_accum_step=32

# --logging_dir ${LOG_DIR} \
# --train_batch_size=8 \
# --mixed_precision="fp16" \

  # --train_batch_size=7 \
  # --gradient_accum_step=18
  # --weight_decay=0.01 \
  # --resolution=512 \
  # --resolution=224 \


# 14682MiB / 32510MiB  for batch 1 imagenet1k
# 14682MiB / 32510MiB for batch 1 mscoco

#  31002MiB / 32510MiB for batch 7

#1-NN accuracy: 0.922415494918823200 default