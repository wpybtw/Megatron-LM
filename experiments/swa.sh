

# rm 

bash experiments/prepare.sh

bash experiments/swa_vs_full/train_qwen3_1.7b_h100.sh
MASTER_PORT=6002 CUDA_VISIBLE_DEVICES=6,7  bash experiments/swa_vs_full/train_qwen3_1.7b_h100.sh SWA