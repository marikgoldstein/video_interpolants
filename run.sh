# Specific to MG's environment, just remove and replace with your own
cd /mnt/home/mgoldstein/video_interpolants/
module load python/3.12.9 cuda cudnn
source ../gameflow/.venv/bin/activate

GPUS=1
NEW_TORCHRUN="torchrun --standalone --nnodes=1 --nproc_per_node=${GPUS}"
OLD_TORCHRUN="python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=${GPUS}"
WHICH="linear" # or "ours"
CKPT="/mnt/home/mgoldstein/ceph/video/ckpts/kth_model_${WHICH}.pt"
cmd="${OLD_TORCHRUN} main.py --overfit one --smoke_test 0 --check_nans 0 --interpolant_type ${WHICH} --load_model_ckpt_path ${CKPT}"
#--load_model_ckpt_path ${CKPT}"
echo ${cmd}
eval ${cmd}  
