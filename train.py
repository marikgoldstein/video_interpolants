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
import interpolants
import utils
from utils import TensorFolder
import arch
import vqvae
import video_dataset
import configs

class Trainer:

    def __init__(self, args):
        self.args = args
        self.setup_ddp_and_device()
        self.setup_dirs_and_logging()
        self.setup_model()
        self.setup_data()
        self.train_steps = 0
        self.setup_interpolant()

    def setup_interpolant(self,):
        if self.args.interpolant_type == 'linear':
            self.interpolant = interpolants.LinearInterpolant()
        elif self.args.interpolant_type == 'ours':
            self.interpolant = interpolants.OurInterpolant()
        else:
            assert False

    def is_main(self,):
        return self.rank == 0

    def is_early(self,):
        return self.train_steps in [0,1,2,3,4,5,10,20,50]

    def time_to_print(self,):
        A = self.train_steps % self.args.print_every == 0
        B = self.is_early()
        return A or B

    def time_to_sample(self,):
        return self.train_steps % self.args.sample_every == 0

    def time_to_log(self,):
        log_every = self.args.log_every
        return (self.train_steps % log_every == 0)

    def time_to_update_ema(self,):
        A = self.train_steps >= self.args.update_ema_after
        B = self.train_steps %  self.args.update_ema_every == 0
        return (A and B)

    def time_to_save(self,):
        return self.train_steps % self.args.save_every == 0

    def time_to_save_most_recent(self,):
        return self.train_steps % self.args.save_most_recent_every == 0

    def setup_ddp_and_device(self,):
        assert torch.cuda.is_available(), "Training currently requires at least one GPU."
        # Setup DDP:
        dist.init_process_group("nccl")
        self.world_size = dist.get_world_size()
        assert self.args.global_batch_size % self.world_size == 0, f"Batch size must be divisible by world size."
        self.rank = dist.get_rank()
        self.device = self.rank % torch.cuda.device_count()
        self.local_seed = self.args.global_seed * self.world_size + self.rank
        torch.manual_seed(self.local_seed)
        torch.cuda.set_device(self.device)
        print(f"Starting rank={self.rank}, seed={self.local_seed}, world_size={self.world_size}.")
        self.local_batch_size = int(self.args.global_batch_size // self.world_size)
        print("local batch size is", self.local_batch_size)

    def setup_dirs_and_logging(self,):
        if self.rank == 0:
            os.makedirs(self.args.results_dir, exist_ok=True)  
            experiment_name = f"{self.args.dataset}-{self.args.interpolant_type}"
            experiment_dir = f"{self.args.results_dir}/{experiment_name}"  
            self.checkpoint_dir = f"{experiment_dir}/checkpoints"  
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.logger = utils.create_logger(experiment_dir)
            self.logger.info(f"Experiment directory created at {experiment_dir}")
            wandb_dir = os.path.join(experiment_dir, 'wandb')
            utils.wandb_initialize(
                self.args, 
                entity=self.args.wandb_entity, 
                project_name=self.args.wandb_project, 
                directory=wandb_dir
            )
        else:
            self.logger = utils.create_logger(None)

    def maybe_load(self,):
        if self.args.load_model_ckpt_path is not None:
            ckpt_path = self.args.load_ckpt_path
            state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(state_dict["model"])
            self.ema.load_state_dict(state_dict["ema"])
            self.opt.load_state_dict(state_dict["opt"])
            self.args = state_dict["args"]

    def setup_model(self,):
        # taming vae does ckpt load automatically. todo handle custom river vae too.
        self.vae = vqvae.build_vqvae(self.args)
        self.vae.eval()
        self.vae.to(self.device)
        self.model = arch.VectorFieldRegressor(
            state_size = self.args.model_state_size,
            state_res = self.args.model_state_res,
            inner_dim = self.args.model_inner_dim,
            depth = self.args.model_depth,
            mid_depth = self.args.model_mid_depth,
            out_norm = self.args.model_out_norm,
        )
        self.ema = deepcopy(self.model).to(self.device)  
        self.ema.eval()  
        utils.requires_grad(self.ema, False)
        self.model = DDP(self.model.to(self.device), device_ids=[self.rank])
        utils.update_ema(self.ema, self.model.module, decay=0)  
        self.opt = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.args.base_lr, 
            weight_decay=self.args.weight_decay
        )
        self.maybe_load()
        self.logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def get_video_datasets(self, split):
        return video_dataset.VideoDataset(
            data_path=os.path.join(self.args.data_path, split),
            input_size=self.args.input_size,
            crop_size=self.args.crop_size,
            frames_per_sample=self.args.num_observations,
            skip_frames=self.args.skip_frames,
            random_horizontal_flip=self.args.data_random_horizontal_flip,
            aug=self.args.data_aug,
            albumentations=self.args.data_albumentations,
        )

    def setup_data(self,):
        self.datasets = {}
        self.datasets['train'] = self.get_video_datasets(split = 'train')
        self.datasets['val'] = self.get_video_datasets(split = 'val')
        self.sampler = torch.utils.data.RandomSampler(
            self.datasets["train"],
            replacement=True,
            num_samples=(self.local_batch_size * self.args.num_training_steps),
            generator=torch.Generator().manual_seed(self.local_seed)
        )
        self.train_dataloader = DataLoader(
            dataset=self.datasets['train'],
            batch_size=self.local_batch_size,
            shuffle=False, 
            num_workers=self.args.num_workers,
            sampler=self.sampler,
            pin_memory=True,
            drop_last=True,
        )

        self.val_dataloader = DataLoader(
            dataset=self.datasets['val'],
            batch_size=self.local_batch_size,
            shuffle=False, 
            num_workers=self.args.num_workers,
            sampler=self.sampler,
            pin_memory=True,
            drop_last=True,
        )

    def array2grid(self, x):
        nrow = round(math.sqrt(x.size(0)))
        x = make_grid(x, nrow=nrow, normalize=True, value_range=(-1,1))
        x = x.mul(255).add_(0.5).clamp_(0,255).permute(1,2,0).to('cpu', torch.uint8).numpy()
        return x

    def wandb_arr_to_img(self, arr):
        return wandb.Image(self.array2grid(arr))
    
    # TODO
    #def plot_real_data(self,):
    #    if self.is_main():
    #        self.logger.info("Plotting real data...")
    #        wandb.log({'real_data' : self.wandb_arr_to_img(self.x_overfit)}, step=0)
    #    dist.barrier()
    
    def training_loop(self,):
        self.checkpoint(mode = 'init')
        # get 2nd batch, dont like first one 
        self.x_overfit = next(iter(self.train_dataloader))
        # TODO
        #self.plot_real_data()
        while self.train_steps < self.args.num_training_steps:
            self.do_epoch()
        self.checkpoint(mode = 'final')
        self.logger.info("Done!")
        utils.ddp_cleanup()
 
    def done_with_epoch(self, batch_num):
    
        done = False

        if self.args.limit_train_batches > 0: 
            if batch_num >= self.args.limit_train_batches:
                done = True

        if self.train_steps >= self.args.num_training_steps:
            done = True
            self.logger.info("Done with num training step")

        return done

        
    def do_epoch(self):
        # set_epoch() not needed for sampler since each rank
        # uses its own generator with local seed
        for batch_num, x in enumerate(self.train_dataloader):
            if self.done_with_epoch(batch_num):
                break
            self.do_step(batch_num, x)

    def clip_grads(self, x):
        return torch.nn.utils.clip_grad_norm_(x, self.args.grad_clip_norm).item()
    
    #def get_window(self, x, t, K):
    #    bsz, num_obs, C, H, W = x.shape
    #    assert C == self.C_data
    #    assert H == self.H_data
    #    assert W == self.W_data
    #    start_indices = t - K
    #    assert torch.all(start_indices >= 0)
    #    indices = start_indices[:, None] + torch.arange(K + 1)
    #    indices = indices.to(x.device)
    #    out = x[torch.arange(bsz)[:, None], indices]
    #    assert out.shape == (bsz, K+1, C, H, W)
    #    prefix = out[:, :-1, ...]
    #    data = out[:, -1:, ...]  # : after -1 makes sure dim is retained
    #    assert prefix.shape == (bsz, K, C, H, W)
    #    assert data.shape == (bsz, 1, C, H, W)
    #    return prefix, data

    #def get_random_data_and_cond(self, x, K):
    #    # THIS T IS NOT THE INTERPOLANT T
    #    # THIS T IS THE DATA INDEX
    #    bsz, num_obs, C, H, W = x.shape
    #    assert C == self.C_data
    #    assert H == self.H_data
    #    assert W == self.W_data
    #    assert num_obs >= K + 1
    #    t = self.randint(K, num_obs, bsz)
    #    cond, xt = self.get_window(x, t, K)
    #    return cond, xt

    #@torch.no_grad()
    #def encode_decode(self, X_series):
    #    b = X_series.shape[0]
    #    Z_series = self.encode(X_series)
    #    ae_ours = self.config["autoencoder"]["type"] == "ours"
    #    dec_fn = self.vae.backbone.decode_from_latents if ae_ours else self.vae.decode
    #    # Decode to image space
    #    Z_series = rearrange(Z_series, "b n c h w -> (b n) c h w")
    #    X_series = dec_fn(Z_series)
    #    X_series = rearrange(X_series, "(b n) c h w -> b n c h w", b=b)
    #    return X_series

    def randint(self, low, high, sz):
        return torch.randint(low = low, high = high, size = [sz])

    def batched_get_index(self, x, idx):
        return x[torch.arange(x.shape[0]), idx]

    def wide(self, x):
        return x[:, None, None, None]

    def get_data_cond_ref(self, x):
        bsz, num_obs, C, H, W = x.shape
        assert num_obs > 2
        data_idx = self.randint(2, num_obs, bsz)
        data = self.batched_get_index(x, data_idx)
        ref_idx = data_idx - 1
        ref = self.batched_get_index(x, ref_idx)
        cond_idx = torch.cat(
            [self.randint(0, s-1, 1) for s in data_idx], dim = 0
        )
        cond = self.batched_get_index(x, cond_idx)
        assert torch.all(ref_idx > cond_idx)
        for tensor in [data, cond, ref]:
            assert tensor.shape == (bsz, C, H, W)
        gap = (ref_idx - cond_idx).type_as(data)
        images = torch.stack([data, ref, cond], dim = 1)
        return images, gap

    def prepare_batch(self, x):
        images, gap = self.get_data_cond_ref(x)
        Z = self.vae_encode(images)
        z1, zref, zcond = Z[:,0], Z[:, 1], Z[:, 2]
        t = self.sample_time(Z.shape[0]).type_as(z1)
        if self.args.interpolant_type == 'linear':
            z0 = torch.randn_like(z1)
            zt = self.interpolant.compute_xt(x0=z0,x1=z1,t=t)
            velocity_target = self.interpolant.compute_xdot(x0=z0,x1=z1,t=t) 
            target = velocity_target
        elif slef.args.interpolant_type == 'ours':
            z0 = zref
            noise = torch.randn_like(z1)
            zt = self.interpolant.compute_xt(x0=z0,x1=z1,t=t, noise = noise)
            drift_target = self.interpolant.compute_drift_target(x0=z0,x1=z1,t=t, noise = noise) 
            target = drift_target
        else:
            assert False
        return zt, t, zref, zcond, gap, target

    def loss_fn(self, x):
        bsz = x.size(0)
        num_obs = x.size(1)
        bsz, num_obs, C, H, W = x.shape
        zt, t, zref, zcond, gap, target = self.prepare_batch(x)
        model_out = self.model(zt, t, zref, zcond, gap)
        assert model_out.shape == target.shape
        loss = (model_out - target).pow(2).sum(dim=[1,2,3]).mean()
        return loss

    def do_step(self, batch_num, x):
        if self.args.overfit:
            x = self.x_overfit

        # e.g. for kth, num_obs is 40, so this gets first 40 frames
        # in case the video is longer
        # do this before putting on device to save mem.
        x = x[: , : self.args.num_observations]
        self.model.train()
        x = x.to(self.device)
        loss = self.loss_fn(x)
        self.opt.zero_grad()
        loss.backward()
        
        # just some book keeping and EMA updates
        loss_item = loss.detach().item()
        log_dict = {'train_loss': loss_item}
        log_dict['grad_norm'] = self.clip_grads(self.model.parameters())
        self.opt.step()
        log_dict['lr'] = self.update_lr()
        utils.update_ema(self.ema, self.model.module)
        sample_dict = self.do_sampling(x)
        utils.copy_into_A_from_B(A=log_dict, B=sample_dict)
        if self.time_to_log() and self.is_main():
            wandb.log(log_dict,step=self.train_steps)
        if self.time_to_print():
            self.logger.info(f"(step={self.train_steps:07d}) Train Loss: {loss_item:.4f}")
        self.checkpoint()
        self.train_steps += 1

    def flatten(self, x):
        return rearrange(x, "b n c h w -> (b n) c h w")

    def unflatten(self, x, n = 3):
        return rearrange(x, "(b n) c h w -> b n c h w", n=n)

    @torch.no_grad()
    def vae_encode(self, x):
        assert len(x.shape) == 5
        bsz, frames, C, H, W = x.shape
        assert C == self.args.C_data
        assert H == self.args.H_data
        assert W == self.args.W_data
        self.vae.eval()
        # the repo uses two different classes for vqvaes.
        # the kth dataset uses one from Taming Transformers
        # the clevrer dataset uses one written by RIVER repo
        # they have slightly different interfaces
        # could hide some of this in vae class to clean up this script
        if self.args.vqvae_type == 'river':
            Z = self.vae(x).latents
        else:
            flat_Z = self.vae.encode(self.flatten(x))
            Z = self.unflatten(flat_Z, n = x.shape[1])
        return Z

    @torch.no_grad() 
    def vae_decode(self, Z):

        b, n, c, h, w = Z.shape

        if self.args.vqvae_type == 'river':
            dec_fn = self.vae.backbone.decode_from_latents
        else:
            dec_fn = self.vae.decode

        # Decode to image space
        # could hide some of this in the decoder method itself
        # to clean up this script.
        Z = rearrange(Z, "b n c h w -> (b n) c h w")
        X = dec_fn(Z)
        X = rearrange(X, "(b n) c h w -> b n c h w", b=b)
        return X



    def sample_time(self, bsz):
        t = torch.distributions.Uniform(
            low = self.args.time_min_training, 
            high = self.args.time_max_training
        ).sample((bsz,))
        return t

    def update_lr(self,):
        new_lr = utils.get_lr(
            update_step=self.train_steps, 
            base_lr = self.args.base_lr, 
            min_lr = self.args.min_lr, 
            num_training_steps = self.args.num_training_steps,
            warmup_steps = self.args.lr_warmup_steps, 
            schedule=self.args.lr_schedule
        )

        for pg in self.opt.param_groups:
            pg["lr"] = new_lr
        return new_lr

    def get_checkpoint_dict(self,):
        self.model.eval()
        self.ema.eval()
        return {
            "model": self.model.module.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
            "args": self.args
        }

    def save_ckpt_to_file(self, checkpoint, checkpoint_path):
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _checkpoint(self, mode=None):
        
        if mode == 'init':
            checkpoint = self.get_checkpoint_dict()
            checkpoint_path = f"{self.checkpoint_dir}/init.pt"
            self.save_ckpt_to_file(checkpoint, checkpoint_path)

        elif mode == 'final':
            checkpoint = self.get_checkpoint_dict()
            checkpoint_path = f"{self.checkpoint_dir}/final.pt"
            self.save_ckpt_to_file(checkpoint, checkpoint_path)
        
        else:

            if self.time_to_save():
                checkpoint = self.get_checkpoint_dict()
                checkpoint_path = f"{self.checkpoint_dir}/{self.train_steps:07d}.pt"
                self.save_ckpt_to_file(checkpoint, checkpoint_path)

            if self.time_to_save_most_recent():
                checkpoint = self.get_checkpoint_dict()
                checkpoint_path = f"{self.checkpoint_dir}/latest.pt"
                self.save_ckpt_to_file(checkpoint, checkpoint_path)


    def checkpoint(self, mode=None):
        A = (mode in ['init', 'final'])
        B = self.time_to_save()
        C = self.time_to_save_most_recent()
        if (A or B or C):
            if self.rank == 0:
                self._checkpoint(mode = mode)
            dist.barrier()
 
    @torch.no_grad()
    def do_sampling(self, x):   
        if self.time_to_sample():
            self.logger.info("Generating samples...")
            batch_size = x.shape[0]
            num_frames = x.shape[1]
            # only plot 4 videos at a time on 
            # Wandb so that they display in large size
            batch_size = min(4, batch_size)
            num_cond = self.args.condition_frames
            num_gen = self.args.frames_to_generate
            assert num_cond + num_gen == num_frames
            x_real = x[ : batch_size, : (num_cond + num_gen)]
            x_cond = x_real[:, : num_cond]
            x_hat = self.sample(x_cond, num_gen)
            assert x_hat.shape == x_real.shape
             
            to_wandb_vid = lambda x: wandb.Video(utils.uncenter_video(x), fps = 1)
            split = 'train'
            # Log images grid
            grid = utils.make_observations_grid([x_cond, x_hat], num_sequences = x_real.shape[0])
 
            # two real and two generated, side by side
            lst = [x_real[0],  x_real[1], x_hat[0], x_hat[1]]
            both_videos = torch.stack(lst, dim=0)

            D = {
                f"{split}/Media/reconstructed_observations":wandb.Image(grid),
                f"{split}/Media/real_videos": wandb_vid(x_real),
                f"{split}/Media/generated_videos":  wandb_vid(x_hat),
                f"{split}/Media/real_vs_generated": wandb_vid(both_videos)
            }
        else:
            D = {}

        return D

    
    @torch.no_grad()
    def sample(self, x_cond, num_gen):
        '''
        sample loop to generate num_gen frames
        '''
        Z = self.vae_encode(x_cond)

        # (batch size, num frames, num channels, H, W)
        b, original_n, c, h, w = Z.shape

        if original_n == 1:
            # duplicate the prev frame. there is no extra reference frame to condition on
            # but remember to remove at the end!!!
            Z = Z[:, [0,0]]
            n = Z.shape[1]
            assert n == 2
        else:
            n = original_n

        def get_random_cond(z):
            high = z.shape[1] - 1
            i = torch.randint(low = 0, high = high, size = [b])
            cond = self.batched_get_index(z, i)
            gap = (high - i).type_as(z)
            return cond, gap

        print("GENERATING FRAMES")
        # Generate future latents
        t_min = self.args.time_min_sample
        t_max = self.args.time_max_sample
        steps = self.args.num_sampling_steps
        t_grid = torch.linspace(t_min, t_max, steps).type_as(Z)
        ones = torch.ones(b,).type_as(Z)
  
        for k in range(num_gen):
             
            if k % 5 == 0:
                print(f"generating frame {k} out of {num_gen}")
        
            # every frame, need to get next zcond and z_ref
            z_cond, gap = get_random_cond(Z)

            z_ref = Z[:, -1]
   
            # z_cond and z_ref are (batch, C_latent, H_latent, W_latent)

            if self.args.interpolant_type == 'linear':
            
                # for linear, the model is trained for the ODE velocity.
                def f(t, zt):
                    t_arr = t * ones
                    return self.model(zt, t_arr, z_ref, z_cond, gap)
                z0 = torch.randn_like(z_ref)
                # get last timestep of integration
                z1 = odeint(f, z0, t_grid, method='euler')[-1] 
               
            elif self.args.interpolant_type == 'ous':
            
                # z_t = a(t)x^{t-1} + b(t)x^t + sigma(t)root(t)noise                                                                                   
                # bhat(z_t, t, cond) = b(z_t, t, x^{t-1}, x^j, (t-1)-j)

                # adot(t) x^{t-1} + bdot(t)x^{t} + sigmadot(t)root(t)noise

                # dZ^t_s = bhat(Z^t_s, t, Z^{t-1}, Z^{j}, (t-1)-j) ds + sigma(s)dW_s

                dt = tgrid[1] - tgrid[0]
                zt = z_ref
                for tscalar in tgrid:
                    
                    # Euler-Maruyama integration

                    t_arr = tscalar * ones

                    # sde drift
                    f = self.model(zt, t_arr, z_ref, z_cond, gap)

                    # sde diffusion coef
                    g = self.wide(self.interpolant.sigma(t_arr))

                    zt_mean = zt + f * dt
                    
                    diffusion_term = g * torch.randn_like(zt_mean) * torch.sqrt(dt)
                    
                    zt = zt_mean + diffusion_term

                # don't add diffusion term on the last step
                # common trick for EM integration in deep generative models
                z1 = zt_mean

            else:
                assert False

            # add a time dim for concat
            Z_next = z1[:, None, ...]
            
            # append and now continue to generate the next frame
            Z = torch.cat([Z, Z_next], dim=1)
            

        if original_n == 1:
            Z = Z[:, 1:] # remove duplicated frame

        X = self.vae_decode(Z)
        return X

if __name__ == "__main__":
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False 
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    parser = argparse.ArgumentParser()
     
    # paths
    parser.add_argument('--load_model_ckpt_path', type=str, default=None) # train from scratch
    parser.add_argument('--wandb_entity', type = str, default = 'marikgoldstein')
    parser.add_argument('--wandb_project', type = str, default = 'videointerpolants')
    parser.add_argument('--interpolant_type', type = str, choices = ['linear','ours'], default = 'linear')
    parser.add_argument('--dataset', type = str, choices = ['kth', 'clevrer'], default = 'kth')
    parser.add_argument('--overfit', type = int, default = 0)
    args = parser.parse_args()
    args.overfit = bool(args.overfit)
    config = configs.Config(
        dataset = args.dataset,
        overfit = args.overfit,
        interpolant_type = args.interpolant_type,
        load_model_ckpt_path = args.load_model_ckpt_path,
        wandb_entity = args.wandb_entity,
        wandb_project = args.wandb_project,
    )
    trainer = Trainer(config)
    trainer.training_loop()
