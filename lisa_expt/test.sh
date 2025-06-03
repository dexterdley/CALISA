CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

deepspeed ./lisa_expt/train_ds.py \
  --version='xinlai/LISA-13B-llama2-v1' \
  --dataset_dir='dataset' \
  --val_dataset="ReasonSeg|val" \
  --dataset_dir='./dataset' \
  --exp_name="lisa-7b" \
  --vision_pretrained="./lisa_expt/sam_vit_h_4b8939.pth" \
  --eval_only