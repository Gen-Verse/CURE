cd /mnt/shared-storage-user/zhupengyu1/tangling/grpo/CURE
source /mnt/shared-storage-user/zhupengyu1/anaconda3/bin/activate CURE

export VLLM_CUDART_SO_PATH=/mnt/shared-storage-user/zhupengyu1/anaconda3/envs/CURE/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
export CUDA_LAUNCH_BLOCKING=1
LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"
export CURE_LOG_FILE="$LOG_FILE"
python run.py 2>&1 | tee "$LOG_FILE"
