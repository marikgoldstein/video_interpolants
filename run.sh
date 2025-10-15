# Specific to MG's environment, just remove and replace with your own
cd /mnt/home/mgoldstein/video_interpolants/
module load python/3.12.9 cuda cudnn
source ../gameflow/.venv/bin/activate

GPUS=1
BP_TORCHRUN="torchrun --standalone --nnodes=1 --nproc_per_node=${GPUS}"
FI_TORCHRUN="python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=${GPUS}"
CKPT="/mnt/home/mgoldstein/ceph/video/ckpts/kth_model_ours.pt"
cmd="${FI_TORCHRUN} main.py --overfit batch --smoke_test 0 --check_nans 0 --interpolant_type ours"
#--load_model_ckpt_path ${CKPT}"
echo ${cmd}
eval ${cmd}  
