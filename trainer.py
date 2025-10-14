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

    def __init__(self, config, rank, device, local_seed, checkpoint_dir, logger, local_batch_size):
        
        self.config = config
        self.rank = rank
        self.device = device
        self.local_seed = local_seed
        self.checkpoint_dir = checkpoint_dir
        self.logger = logger
        self.local_batch_size = local_batch_size
        self.setup_model()
        self.setup_data()
        self.train_steps = 0
        self.setup_interpolant()


    def check_valid(self, x):
        if self.config.check_nans:
            if torch.any(torch.isnan(x)):
                print("X IS NAN")
                assert False
            if torch.any(torch.isinf(x)):
                print("X IS INF")
                assert False

    def setup_interpolant(self,):
        if self.config.interpolant_type == 'linear':
            self.interpolant = interpolants.LinearInterpolant()
        elif self.config.interpolant_type == 'ours':
            self.interpolant = interpolants.OurInterpolant()
        else:
            assert False

    def is_main(self,):
        return self.rank == 0

    def is_early(self,):
        return self.train_steps in [0,1,2,3,4,5,10,20,50]

    def time_to_print(self,):
        A = self.train_steps % self.config.print_every == 0
        B = self.is_early()
        return A or B

    def time_to_sample(self,):
        return self.train_steps % self.config.sample_every == 0

    def time_to_log(self,):
        log_every = self.config.log_every
        return (self.train_steps % log_every == 0)

    def time_to_update_ema(self,):
        A = self.train_steps >= self.config.update_ema_after
        B = self.train_steps %  self.config.update_ema_every == 0
        return (A and B)

    def time_to_save(self,):
        return self.train_steps % self.config.save_every == 0

    def time_to_save_most_recent(self,):
        return self.train_steps % self.config.save_most_recent_every == 0


    def strip_module(self, state):

        is_ddp = False
        for k in state:
            if k.startswith("module"):
                is_ddp = True
                break
        if is_ddp:
            state = {k.replace("module.", ""): v for k, v in state.items()}


        state = {
            k.replace('vector_field_regressor.', ''): v for k, v in state.items()
        }


        state = {k: v for k, v in state.items() if not k.startswith('ae')}

        return state

    def maybe_load(self,):  
        ckpt_path = self.config.load_model_ckpt_path
        if ckpt_path is not None:
            state_dict = torch.load(
                ckpt_path, 
                map_location='cpu', 
                weights_only=False
            )
            
            self.model.load_state_dict(self.strip_module(state_dict["model"]))

            if 'ema' in state_dict:
                self.ema.load_state_dict(state_dict["ema"])
                ema_restored = True
            else:
                ema_restored = False


            if 'opt' in state_dict:
                self.opt.load_state_dict(state_dict["opt"])

            #self.args = state_dict["args"]
            print("LOADED")
            model_restored = True
        else:
            model_restored = False
            ema_restored = False
        return model_restored, ema_restored

    def setup_model(self,):
        # taming vae does ckpt load automatically. todo handle custom river vae too.
        self.vae = vqvae.build_vqvae(self.config)
        self.vae.eval()
        self.vae.to(self.device)
        
        # init model, ema, opt
        self.model = arch.VectorFieldRegressor(
            state_size = self.config.model_state_size,
            state_res = self.config.model_state_res,
            inner_dim = self.config.model_inner_dim,
            depth = self.config.model_depth,
            mid_depth = self.config.model_mid_depth,
            out_norm = self.config.model_out_norm,
            check_nans = self.config.check_nans,
        )
        self.opt = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.base_lr, 
            weight_decay=self.config.weight_decay
        )

        self.ema = deepcopy(self.model).to(self.device)  
        self.ema.eval()  
        utils.requires_grad(self.ema, False)

        # maybe resume from ckpt for model, ema, opt
        self.is_resumed, self.ema_restored = self.maybe_load()

        # wrapp in DDP
        self.model = DDP(self.model.to(self.device), device_ids=[self.rank])
        
        # Then move all optimizer state tensors to that device
        for state in self.opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        # sync ema to model if new, but don't overwrite ema if resuming.
        if not self.ema_restored:
            utils.update_ema(self.ema, self.model.module, decay=0)

        self.logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")


    def get_video_datasets(self, split):
        return video_dataset.VideoDataset(
            data_path=os.path.join(self.config.data_path, split),
            input_size=self.config.input_size,
            crop_size=self.config.crop_size,
            frames_per_sample=self.config.num_observations,
            skip_frames=self.config.skip_frames,
            random_horizontal_flip=self.config.data_random_horizontal_flip,
            aug=self.config.data_aug,
            albumentations=self.config.data_albumentations,
        )

    def setup_data(self,):
        self.datasets = {}
        self.datasets['train'] = self.get_video_datasets(split = 'train')
        self.datasets['val'] = self.get_video_datasets(split = 'val')

        
        # we use non-DDP Dataloader + per-rank-local-seed-Generator to support 
        # sampling with replacement during training
        # we do this because we get random subsets of each datapoint.

        self.train_sampler = torch.utils.data.RandomSampler(
            self.datasets["train"],
            replacement=True,
            num_samples=(self.local_batch_size * self.config.num_training_steps),
            generator=torch.Generator().manual_seed(self.local_seed)
        )
        self.train_dataloader = DataLoader(
            dataset=self.datasets['train'],
            batch_size=self.local_batch_size,
            shuffle=False, 
            num_workers=self.config.num_workers,
            sampler=self.train_sampler,
            pin_memory=True,
            drop_last=True,
        )

        self.val_sampler = torch.utils.data.SequentialSampler(self.datasets['val'])
        self.val_dataloader = DataLoader(
            dataset=self.datasets['val'],
            batch_size=self.local_batch_size,
            shuffle=False, 
            num_workers=self.config.num_workers,
            sampler=self.val_sampler,
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
    
    def training_loop(self,):
        self.checkpoint(mode = 'init')
        self.x_overfit = next(iter(self.train_dataloader))
        if self.config.overfit_one:
            for i in range(1, self.x_overfit.shape[0]):
                self.x_overfit[i] = self.x_overfit[0]

        while self.train_steps < self.config.num_training_steps:
            self.do_epoch()
        self.checkpoint(mode = 'final')
        self.logger.info("Done!")
        utils.ddp_cleanup()
 
    def done_with_epoch(self, batch_num):
    
        done = False

        if self.config.limit_train_batches > 0: 
            if batch_num >= self.config.limit_train_batches:
                done = True

        if self.train_steps >= self.config.num_training_steps:
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
        return torch.nn.utils.clip_grad_norm_(x, self.config.grad_clip_norm).item()
    
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

    def wide(self, x):
        return x[:, None, None, None]
              
    def randint(self, low, high, sz, device):
        return torch.randint(low=low, high=high, size=(sz,), device=device)

    #def batched_get_index(self, x, idx):
    #    return x[torch.arange(x.shape[0]), idx]

    def batched_get_index(self, x, idx):
        return x[torch.arange(x.shape[0], device=x.device), idx]

    def get_random_triplet(self, x):
        """
        Picks:
            data_frame = Frame(t)
            reference_frame = Frame(t-1)
            context_frame = Frame(j), j < t-1
        Assumes num_obs >= 3.
        """
        bsz, num_obs, C, H, W = x.shape
        dev = x.device

        # t ∈ {2, ..., num_obs-1}
        data_idx = torch.randint(low=2, high=num_obs, size=(bsz,), device=dev)
        reference_idx = data_idx - 1  # ∈ {1, ..., num_obs-2}

        # j ∈ {0, ..., t-2}  (exclusive upper bound t-1)
        # vectorized: sample u ∈ [0,1), scale by (t-1), floor
        context_idx = (torch.rand(bsz, device=dev) * (data_idx - 1).float()).floor().long()

        data_frame      = self.batched_get_index(x, data_idx)
        reference_frame = self.batched_get_index(x, reference_idx)
        context_frame   = self.batched_get_index(x, context_idx)

        # integer gap ≥ 1
        gap = (reference_idx - context_idx)            # dtype: long, values in {1,2,...}
        if self.config.smoke_test:
            assert torch.all(gap >= 1)

        return data_frame, reference_frame, context_frame, gap



    def prepare_batch(self, x):
        data_frame, reference_frame, context_frame, gap = self.get_random_triplet(x)
        Z = self.vae_encode(torch.stack([data_frame, reference_frame, context_frame], dim=1))
        assert Z.shape[1] == 3
        z1, Z_reference, Z_context = Z[:,0], Z[:, 1], Z[:, 2]
        t = self.sample_time(Z.shape[0]).type_as(z1)
        # default to velocity target for linear
        # and drift target for ours.
        if self.config.interpolant_type == 'linear':
            z0 = torch.randn_like(z1)
            zt = self.interpolant.compute_xt(z0=z0,z1=z1,t=t)
            target = self.interpolant.compute_xdot(z0=z0,z1=z1,t=t) 
        elif self.config.interpolant_type == 'ours':
            z0 = Z_reference
            noise = torch.randn_like(z1)
            zt = self.interpolant.compute_xt(z0=z0,z1=z1,t=t, noise = noise)
            target = self.interpolant.compute_drift_target(z0=z0, z1=z1,t=t, noise=noise) 
        else:
            assert False
        return zt, t, Z_reference, Z_context, gap, target

    def loss_fn(self, x):
        bsz = x.size(0)
        num_obs = x.size(1)
        bsz, num_obs, C, H, W = x.shape
        zt, t, Z_reference, Z_context, gap, target = self.prepare_batch(x)
        model_out = self.model(z=zt, t=t, ref=Z_reference, context=Z_context, gap=gap)
        self.check_valid(model_out)
        assert model_out.shape == target.shape
        loss = (model_out - target).pow(2).sum(dim=[1,2,3]).mean()
        return loss

    def do_step(self, batch_num, x):
        if self.config.overfit:
            x = self.x_overfit
        # e.g. for kth, num_obs is 40, so this gets first 40 frames
        # in case the video is longer
        # do this before putting on device to save mem.
        x = x[: , : self.config.num_observations]
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
        sample_dict = self.sample(x)
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
        assert C == self.config.C_data
        assert H == self.config.H_data
        assert W == self.config.W_data
        self.vae.eval()
        # the repo uses two different classes for vqvaes.
        # the kth dataset uses one from Taming Transformers
        # the clevrer dataset uses one written by RIVER repo
        # they have slightly different interfaces
        # could hide some of this in vae class to clean up this script
        if self.config.vqvae_type == 'river':
            Z = self.vae(x).latents
        else:
            flat_Z = self.vae.encode(self.flatten(x))
            Z = self.unflatten(flat_Z, n = x.shape[1])
        return Z

    @torch.no_grad() 
    def vae_decode(self, Z):

        b, n, c, h, w = Z.shape

        if self.config.vqvae_type == 'river':
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
    ''' 
    def sample_time(self, bsz):
        t = torch.distributions.Uniform(
            low = self.config.time_min_training, 
            high = self.config.time_max_training
        ).sample((bsz,))
        return t
    '''

    def sample_time(self, bsz):
        tmin = self.config.time_min_training
        tmax = self.config.time_max_training
        return torch.rand(bsz, device=self.device, dtype=torch.float32) * (tmax - tmin) + tmin


    def update_lr(self,):
        new_lr = utils.get_lr(
            update_step=self.train_steps, 
            base_lr = self.config.base_lr, 
            min_lr = self.config.min_lr, 
            num_training_steps = self.config.num_training_steps,
            warmup_steps = self.config.lr_warmup_steps, 
            schedule=self.config.lr_schedule,
            is_resumed = self.is_resumed,
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
            "args": self.config
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

    def get_num_frames_for_sampling(self, x):
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        # only plot 4 videos at a time on 
        # Wandb so that they display in large size
        batch_size = min(4, batch_size)
        num_cond = self.config.condition_frames
        num_gen = self.config.frames_to_generate
        assert num_cond + num_gen == num_frames
        return batch_size, num_cond, num_gen

    @torch.no_grad()
    def sample(self, x, use_ema = False):   
        if self.time_to_sample():

            x = x.clone()[torch.randperm(x.shape[0])]
            self.logger.info("Generating samples...")
            batch_size, num_cond, num_gen = self.get_num_frames_for_sampling(x)
            X_real = x[ : batch_size, : (num_cond + num_gen)]
            # we start generating frames after this prefix
            X_cond = X_real[:, : num_cond] 
            X_hat = self._sample(X_cond, num_gen, use_ema)
            assert X_hat.shape == X_real.shape
            split = 'train'
            # Log images grid
            grid = utils.make_observations_grid([X_cond, X_hat], num_sequences = X_real.shape[0])
            # two real and two generated, side by side
            lst = [X_real[0],  X_real[1], X_hat[0], X_hat[1]]
            both_videos = torch.stack(lst, dim=0)
            fps = 3 # frames per second. reduce to 1 to see each video more clearly picture-by-picture
            D = {
                f"{split}/Media/reconstructed_observations":wandb.Image(grid),
                f"{split}/Media/real_videos": utils.to_wandb_vid(X_real, fps=fps),
                f"{split}/Media/generated_videos":  utils.to_wandb_vid(X_hat, fps=fps),
                f"{split}/Media/real_vs_generated": utils.to_wandb_vid(both_videos, fps=fps)
            }
        else:
            D = {}
        return D

    def sample_frame_ode(self, z0, Z_ref, Z_context, gap, t_grid, use_ema):


        if use_ema:
            model = self.ema
        else:
            model = self.model

        model.eval()

        assert self.config.interpolant_type == 'linear'
        ones = torch.ones(z0.shape[0]).type_as(z0)
        # for linear, the model is trained for the ODE velocity.
        def f(t, zt):
            t_arr = t * ones

            self.check_valid(zt)
            self.check_valid(t_arr)
            self.check_valid(Z_ref)
            self.check_valid(Z_context)
            self.check_valid(gap)

            model_out = model(z=zt, t=t_arr, ref=Z_ref, context=Z_context, gap=gap)
            self.check_valid(model_out)
            return model_out


        # get last timestep of integration
        z1 = odeint(f, z0, t_grid, method='euler')[-1] 

        self.check_valid(z1)
        return z1


    def sample_frame_sde(self, z0, Z_ref, Z_context, gap, t_grid, use_ema):

        if use_ema:
            model = self.ema
        else:
            model = self.model

        model.eval()



        assert self.config.interpolant_type == 'ours'
        assert len(t_grid) >= 2
        # below, dot means time derivative.
        # z0 := Frame(t-1)
        # z1 := Frame(t)
        # noise := randn_like(z1)
        # z_t := a(t) z0 + b(t) z1 + sigma(t) root(t) noise
        # j is a random num less than t-1 
        # Reference Frame = Frame(t-1)
        # Context Frame = Frame(j)
        # Context frame makes the sampler not fully markovian.
        # cond = ( Frame(t-1), Frame(j), (t-1)-j) ) =  (Ref, Context, Gap) 
        # b_hat(z_t, t, cond) = E[drift_target | z_t] 
        # drift target is adot(t) z0 + bdot(z1 + sigmadot(t)root(t) noise.
        # note that the drift target isn't just the time derivative of the velocity
        # but rather the time derivative of velocity + coef * score
        # finally:
        # dZ^t_s = b_hat ds + sigma(s)dW_s
        dt = (t_grid[1] - t_grid[0]).item()
        ones = torch.ones(z0.shape[0], device=z0.device, dtype=z0.dtype)

        sqrt_dt = math.sqrt(dt)  # or: torch.tensor(dt, device=z0.device, dtype=z0.dtype).sqrt()


        zt = Z_ref.clone()
        zt_mean = zt
        for tscalar in t_grid:
            
            # Euler-Maruyama integration
            t_arr = tscalar * ones

            # sde drift
            f = model(z=zt, t=t_arr, ref=Z_ref, context=Z_context, gap=gap)
            self.check_valid(f)

            # sde diffusion coef
            g = self.wide(self.interpolant.sigma(t_arr))
            self.check_valid(g)

            zt_mean = zt + f * dt

            self.check_valid(zt_mean)


            diffusion_term = g * torch.randn_like(zt_mean) * sqrt_dt
            self.check_valid(diffusion_term)

            zt = zt_mean + diffusion_term

            self.check_valid(zt)


        # don't add diffusion term on the last step
        # common trick for EM integration in deep generative models
        z1 = zt_mean
        return z1

    def _pick_ref_ctx_gap(self, Z):
        """
        Choose reference = last available frame (t-1),
        context = some j < (t-1),
        gap = (t-1) - j >= 1
        """
        assert Z.ndim == 5  # (b, n, c, h, w)
        b, n, _, _, _ = Z.shape
        # need at least 2 frames available (t-1 exists)
        assert n >= 2, "Need ≥2 frames in Z to pick (ref=t-1, ctx<ref)."
        device = Z.device
        dtype_long = torch.long

        # last available index = t-1
        ref_idx_scalar = n - 1

        # context ∈ {0, 1, ..., (t-1) - 1}  == [0, ref_idx_scalar)
        # torch.randint: high is exclusive, so this is perfect.
        ctx_idx = torch.randint(0, ref_idx_scalar, (b,), device=device, dtype=dtype_long)

        ref_idx = torch.full((b,), ref_idx_scalar, device=device, dtype=dtype_long)

        Z_ref = self.batched_get_index(Z, ref_idx)
        Z_ctx = self.batched_get_index(Z, ctx_idx)

        gap = (ref_idx - ctx_idx)           # shape (b,), dtype long
        if self.config.smoke_test:
            assert torch.all(gap >= 1)

        return Z_ref, Z_ctx, gap

    @torch.no_grad()
    def _sample(self, x_cond, num_gen, use_ema):
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

            
        print("GENERATING FRAMES")
        t_grid = torch.linspace(
            self.config.time_min_sample, 
            self.config.time_max_sample, 
            self.config.num_sampling_steps
        ).type_as(Z)
        ones = torch.ones(b,).type_as(Z)


        # below, I use lowercase z for the interpolant
        # and uppercase Z for the generated latent frames
        # and uppercase X for the pixel space frames
  
        for k in range(num_gen):
             
            if k % 5 == 0:
                print(f"generating frame {k} out of {num_gen}")
            '''
            current_idx = Z.shape[1] - 1
            reference_idx = current_idx - 1
            context_idx = self.randint(0, current_idx-1, sz=b, device=Z.device) 
            reference_idx = torch.ones(b,).type_as(context_idx) * reference_idx
            current_idx = torch.ones(b,).type_as(context_idx) * current_idx
            Z_reference = self.batched_get_index(Z, reference_idx)
            Z_context = self.batched_get_index(Z, context_idx)
            gap = (reference_idx - context_idx).to(Z.device)

            # current_idx is the last available frame index
            current_idx_scalar   = Z.shape[1] - 1
            assert current_idx_scalar >= 2, "Assumes there are always at least 3 frames available."

            reference_idx_scalar = current_idx_scalar - 1      # ∈ {1, 2, ...}
            context_high         = reference_idx_scalar        # exclusive upper bound

            b = Z.shape[0]
            reference_idx = torch.full((b,), reference_idx_scalar, device=Z.device, dtype=torch.long)

            # j ∈ {0, ..., reference_idx_scalar - 1}
            context_idx = torch.randint(0, context_high, (b,), device=Z.device, dtype=torch.long)

            Z_reference = self.batched_get_index(Z, reference_idx)
            Z_context   = self.batched_get_index(Z, context_idx)

            # integer gap ≥ 1
            gap = (reference_idx - context_idx)                # dtype: long, values in {1,2,...}
            # if your model wants float later, cast at point-of-use:
            # gap_float = gap.to(Z.dtype)


            if self.config.smoke_test:
                assert torch.all(gap > 0)
            '''            

            Z_reference, Z_context, gap = self._pick_ref_ctx_gap(Z)

            #print("[sampling] (context, ref, current)", context_idx, reference_idx, current_idx)
            if self.config.interpolant_type == 'linear':
                z0 = torch.randn_like(Z_reference)
                z1 = self.sample_frame_ode(z0, Z_reference, Z_context, gap, t_grid, use_ema=use_ema)

            elif self.config.interpolant_type == 'ours':        
                z0 = Z_reference
                z1 = self.sample_frame_sde(z0, Z_reference, Z_context, gap, t_grid, use_ema=use_ema)
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

