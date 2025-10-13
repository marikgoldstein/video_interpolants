cd /mnt/home/mgoldstein/video_interpolants/
module load python/3.12.9 cuda cudnn
source ../gameflow/.venv/bin/activate
GPUS=1
BP_TORCHRUN="torchrun --standalone --nnodes=1 --nproc_per_node=${GPUS}"
FI_TORCHRUN="python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=${GPUS}"
cmd="${FI_TORCHRUN} main.py --overfit 1 --smoke_test 0"
echo ${cmd}
eval ${cmd}  
