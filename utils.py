# Some of this file is taken from https://github.com/Araachie/river

import wandb
import torch
from torchvision.utils import make_grid
import torch.distributed as dist
from PIL import Image
import os
import argparse
import hashlib
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from torchdiffeq import odeint
from typing import List
import torch





class TensorFolder:

    @staticmethod
    def flatten(tensor: torch.Tensor) -> torch.Tensor:
        """
        Flattens the first two dimensions of the tensor

        :param tensor: (dim1, dim2, ...) tensor
        :return: (dim1 * dim2, ...) tensor
        """

        tensor_size = list(tensor.size())
        flattened_tensor = torch.reshape(tensor, tuple([-1] + tensor_size[2:]))

        return flattened_tensor

    @staticmethod
    def flatten_list(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Applies flatten to all elements in the sequence
        See flatten for additional details
        """

        flattened_tensors = [TensorFolder.flatten(current_tensor) for current_tensor in tensors]
        return flattened_tensors

    @staticmethod
    def fold(tensor: torch.Tensor, second_dimension_size: torch.Tensor) -> torch.Tensor:
        """
        Separates the first tensor dimension into two separate dimensions of the given size

        :param tensor: (dim1 * second_dimension_size, ...) tensor
        :param second_dimension_size: the wished second dimension size for the output tensor
        :return: (dim1, second_dimension_size, ...) tensor
        """

        tensor_size = list(tensor.size())
        first_dimension_size = tensor_size[0]

        # Checks sizes
        if first_dimension_size % second_dimension_size != 0:
            raise Exception(f"First dimension {first_dimension_size} is not a multiple of {second_dimension_size}")

        folded_first_dimension_size = first_dimension_size // second_dimension_size
        tensor = torch.reshape(tensor, ([folded_first_dimension_size, second_dimension_size] + tensor_size[1:]))
        return tensor

    @staticmethod
    def fold_list(tensors: List[torch.Tensor], second_dimension_size: torch.Tensor) -> List[torch.Tensor]:
        """
        Applies fold to each element in the sequence
        See fold for additional details
        """

        folded_tensors = [TensorFolder.fold(current_tensor, second_dimension_size) for current_tensor in tensors]
        return folded_tensors


class SequenceConverter(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(SequenceConverter, self).__init__()

        self.backbone = backbone

    @staticmethod
    def convert(x, n):
        if isinstance(x, list):
            return [TensorFolder.fold(e, n) for e in x]
        elif x.dim() <= 1:
            return x
        return TensorFolder.fold(x, n)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        assert len(args) > 0

        observations_count = args[0].size(1)
        for sequences in args:
            assert sequences.size(1) == observations_count, "Incompatible observations count"

        xs = [TensorFolder.flatten(sequences) for sequences in args]
        x = self.backbone(*xs)

        if isinstance(x, dict):
            for k, v in x.items():
                x[k] = self.convert(v, observations_count)
        else:
            x = self.convert(x, observations_count)

        return x


class DictWrapper(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(DictWrapper, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        item = self.__getitem__(attr)
        return item

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DictWrapper, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DictWrapper, self).__delitem__(key)
        del self.__dict__[key]



@torch.no_grad()
def make_observations_grid(
        images: List[torch.Tensor],
        num_sequences: int) -> torch.Tensor:
    """
    Formats the observations into a grid.

    :param images: List of [bs, num_observations, 3, height, width]
    :param num_sequences: Number of sequences to log
    :return: The grid of observations for logging.
    """

    h = max([im.size(3) for im in images])
    w = max([im.size(4) for im in images])
    n = max([im.size(1) for im in images])

    images = [im[:num_sequences] for im in images]

    def pad(x):
        if x.size(1) == n:
            return x
        else:
            num_sequences_pad = min(x.size(0), num_sequences)
            return torch.cat([
                torch.zeros([num_sequences_pad, n - x.size(1), 3, h, w]).to(x.device),
                x
            ], dim=1)

    def resize(x):
        if x.size(3) == h and x.size(4) == w:
            return x
        else:
            cn = x.size(1)
            y = F.interpolate(
                TensorFolder.flatten(x),
                size=(h, w),
                mode="nearest")
            return TensorFolder.fold(y, cn)

    def add_channels(x):
        if x.size(2) == 1:
            return x.expand(-1, -1, 3, -1, -1)
        else:
            return x

    # Pad and resize images
    images = [to_image(pad(resize(add_channels(x)))) for x in images]

    # Put the observations one next to another
    stacked_observations = torch.stack(images, dim=1)
    flat_observations = TensorFolder.flatten(TensorFolder.flatten(stacked_observations))

    grid = make_grid(flat_observations, nrow=flat_observations.size(0) // (len(images) * num_sequences))
    grid = grid.permute(1, 2, 0)
    grid = grid.detach().cpu().numpy()

    return grid




@torch.no_grad()
def uncenter_video(x: torch.Tensor) -> np.array:
    return (((torch.clamp(x, -1., 1.) + 1.) / 2.).detach().cpu().numpy() * 255).astype(np.uint8)

def to_wandb_vid(x, fps=1):
    return wandb.Video(uncenter_video(x), fps = fps, format='gif')

def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(-1,1))
    x = x.mul(255).add_(0.5).clamp_(0,255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x

def wandb_arr_to_img(arr):
    return wandb.Image(array2grid(arr)) 


@torch.no_grad()
def to_image(x: torch.Tensor) -> torch.Tensor:
    return (((torch.clamp(x, -1., 1.) + 1.) / 2.) * 255).to(torch.uint8)

def copy_into_A_from_B(A, B):                                                                                                                                                                                                                 
    for k in B:
        assert k not in A
        A[k] = B[k]


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def ddp_cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def is_main_process():
    return dist.get_rank() == 0

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }

def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)

def wandb_initialize(args, entity, project_name, directory):
    config_dict = namespace_to_dict(args)
    #wandb.login(key=os.environ["WANDB_KEY"])
    wandb.init(
        entity=entity,
        project=project_name,
        config=config_dict,
        id=None,
        name=None,
        resume=None,
        dir=directory,
    )

def get_lr(update_step, base_lr, min_lr, num_training_steps, warmup_steps, schedule, is_resumed):
    
    if schedule == "constant":
        lr_no_warmup = base_lr
    elif schedule == "cosine":
        rel_step = max(0, update_step - warmup_steps)
        rel_total = max(1, num_training_steps - warmup_steps)
        ratio = rel_step / rel_total
        lr_no_warmup = 0.5 * (1 + math.cos(math.pi * ratio)) * base_lr
    elif schedule == 'linear': 
        multiplier = .992 ** (update_step // 1000)
        lr_no_warmup = multiplier * base_lr
    else:
        assert False

    # linear warm‑up -------------------------------------------------------
    if is_resumed or (warmup_steps < 1):
        warmup_coef = 1.0
    else:
        warmup_coef = update_step / warmup_steps if update_step < warmup_steps else 1.0
    lr = warmup_coef * lr_no_warmup
    return max(lr, min_lr)

