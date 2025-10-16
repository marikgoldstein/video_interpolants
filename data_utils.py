import os 
import torch
from torch.utils.data import DataLoader                                                                                                                                                                             
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

# local
import video_dataset

def setup_data(
    data_path, 
    input_size, 
    crop_size, 
    num_observations, 
    skip_frames, 
    data_random_horizontal_flip, 
    data_aug, 
    data_albumentations, 
    local_batch_size, 
    num_training_steps, 
    local_seed, 
    num_workers
):
    
    datasets = {}

    for split in ['train','val']:


        # optionally always shut off augmentation for validiation, here we keep whatever we use for train
        datasets[split] = video_dataset.VideoDataset(
            data_path=os.path.join(data_path, split),                                                                                                                                                   
            input_size=input_size,
            crop_size=crop_size,                                                                                                                                                                            
            frames_per_sample=num_observations,
            skip_frames=skip_frames,
            random_horizontal_flip=data_random_horizontal_flip,
            aug=data_aug,
            albumentations=data_albumentations,
        )

    # we use non-DDP Dataloader + per-rank-local-seed-Generator to support 
    # sampling with replacement during training
    # we do this because we get random subsets of each datapoint.

    train_sampler = torch.utils.data.RandomSampler(
        datasets["train"],
        replacement=True,
        num_samples=(local_batch_size * num_training_steps),
        generator=torch.Generator().manual_seed(local_seed)                                                                                                                                                
    )
    
    train_dataloader = DataLoader(
        dataset=datasets['train'],
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
    )

    val_sampler = torch.utils.data.SequentialSampler(datasets['val'])
    
    val_dataloader = DataLoader(
        dataset=datasets['val'],
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=val_sampler,
        pin_memory=True,
        drop_last=True,
    )

    # dont need datasets or sampler in this code
    # (usually you need to set_epoch() on sampler but not with the above approach)
    return train_dataloader, val_dataloader

