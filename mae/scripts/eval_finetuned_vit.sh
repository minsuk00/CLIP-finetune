CKPT_PATH=/scratch/choi/output/Diff-Rep/mae/mae_finetuned_vit_base.pth
IMAGENET_DIR=/scratch/choi/dataset/ImageNet1K

python main_finetune.py --eval --resume ${CKPT_PATH} --model vit_base_patch16 --batch_size 16 --data_path ${IMAGENET_DIR}
