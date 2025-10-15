import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import torch.distributed as dist
from torchvision import transforms
from PIL import Image
import os
import argparse
import hashlib
from typing import List
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import wandb
from einops import rearrange, repeat                                                                                              
from torchdiffeq import odeint

# local stuff
import utils
import configs
import trainer

if __name__ == "__main__":
    
    # some torch settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False 
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    # if debugging nans, also put torch anomaly mode here,
    # but remember to remove it because it slows down code, I think.

    # get command line args and merge with Config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model_ckpt_path', type=str, default=None) # train from scratch
    parser.add_argument('--wandb_entity', type = str, default = 'marikgoldstein')
    parser.add_argument('--wandb_project', type = str, default = 'videointerpolants')
    parser.add_argument('--interpolant_type', type = str, choices = ['linear','ours'], default = 'linear')
    parser.add_argument('--dataset', type = str, choices = ['kth', 'clevrer'], default = 'kth')
    parser.add_argument('--overfit', type = str, choices = ['none', 'batch', 'one'], default='none')
    parser.add_argument('--smoke_test', type = int, default = 0)
    parser.add_argument('--check_nans', type = int, default = 0)
    args = parser.parse_args()
    args.smoke_test = bool(args.smoke_test)
    args.check_nans = bool(args.check_nans)

    # overfit mode: trains on one batch over and over. good check to see if sampling produces the batch
    # smoke test mode: even more toy than overfitting. Just runs 100 training steps to make sure the whole training loop runs without crashing

    config = configs.Config(
        dataset = args.dataset,
        overfit = args.overfit, 
        smoke_test = args.smoke_test, 
        check_nans = args.check_nans,
        interpolant_type = args.interpolant_type,
        load_model_ckpt_path = args.load_model_ckpt_path,
        wandb_entity = args.wandb_entity,
        wandb_project = args.wandb_project,
    )
    del args

    # SETUP DDP
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    assert config.global_batch_size % world_size == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank() 
    device = rank % torch.cuda.device_count()
    local_seed = config.global_seed * world_size + rank
    torch.manual_seed(local_seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={local_seed}, world_size={world_size}.")
    local_batch_size = int(config.global_batch_size // world_size)
    print("local batch size is", local_batch_size)

    # SETUP DIRECTORIES, LOGGING, and WANDB
    if rank == 0:
        os.makedirs(config.results_dir, exist_ok=True)  
        # include things here to distinguish experiments. 
        # NOTE: this will overwrite previous things with same name.
        # if you want, add some kind of automatic ID generator
        experiment_name = f"{config.dataset}-{config.interpolant_type}"
        experiment_dir = f"{config.results_dir}/{experiment_name}"  
        checkpoint_dir = f"{experiment_dir}/checkpoints"  
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = utils.create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        wandb_dir = os.path.join(experiment_dir, 'wandb')
        utils.wandb_initialize(
            config,
            entity=config.wandb_entity, 
            project_name=config.wandb_project, 
            directory=wandb_dir
        )
    else:
        checkpoint_dir = None # non-zero ranks should never touch checkpoint dir
        logger = utils.create_logger(None)

    # INIT AND LAUNCH THE TRAINER!
    trainer = trainer.Trainer(
        config = config,
        rank = rank,
        device = device,
        local_seed = local_seed,
        checkpoint_dir = checkpoint_dir,
        logger = logger,
        local_batch_size = local_batch_size,
    )
    trainer.training_loop()
