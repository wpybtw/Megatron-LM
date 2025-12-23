
PROFILE_NAME="megatron_simple_profile"

NSYS_CMD="nsys profile \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --sample=cpu \
    --output=${PROFILE_NAME} \
    --force-overwrite=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --gpu-metrics-device=all \
    --python-backtrace=cuda"

$NSYS_CMD python -m torch.distributed.run --nproc_per_node=2 experiments/simple/run_simple_mcore_train_loop.py
