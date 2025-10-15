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
import checkpointing

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
        self.update_steps = 0
        self.interpolant = interpolants.get_interpolant(
            self.config.interpolant_type
        )
        self.print_logging_settings()
        self.print_extra_notes()

    def print_logging_settings(self,):
        self.logger.info(f"------Experiment monitoring settings --------")
        self.logger.info(f"Smoke test mode: {self.config.smoke_test} (run everything quickly to make sure script finishes without error)")
        self.logger.info(f"Overfit mode: {self.config.overfit} (train on a batch or a datapoint and sample just for that batch or that datapoint. none = regular training)")
        self.logger.info(f"Print every: {self.config.print_every} (will also print steps [0,1,2,3,4,5,10,20,50])")
        self.logger.info(f"Wandb every: {self.config.wandb_every}")
        self.logger.info(f"Save separate checkpoint every: {self.config.save_every} [step_xyz.pt]")
        self.logger.info(f"Save most recent every: {self.config.save_most_recent_every} [latest.pt]")
        self.logger.info(f"Sample every: {self.config.sample_every}")
        self.logger.info(f"Update EMA after: {self.config.update_ema_after}")
        self.logger.info(f"Update EMA every: {self.config.update_ema_every}")
        self.logger.info(f"Data path: {self.config.data_path}")
        self.logger.info(f"Checkpoint dir: {self.checkpoint_dir}")
        self.logger.info(f"Interpolant type: {self.config.interpolant_type}")
        self.logger.info(f"Model checkpoint loaded: {self.is_resumed}")
        self.logger.info(f"EMA checkpoint loaded: {self.ema_restored}")
        self.logger.info(f"Optimizer checkpoint loaded: {self.opt_restored}")
        self.logger.info(f"VQVAE checkpoint loaded: {self.vae_restored} (should always be true...)") 
        self.logger.info('----------------------------------------------')

    def print_extra_notes(self,):
        self.logger.info('-------- NOTES ----------------')
        self.logger.info('1) The VQVAEs are NOT perfect. If you are evaluting your flow model')
        self.logger.info('You should compare against the "ground truth" of xhat = Decode(Encode(X))')
        self.logger.info('Since this is the best thing your flow model could possibly capture (this is all it sees)')
        self.logger.info('- - - - - - - - - ')
        self.logger.info('2) MG has recently open sourced this code base and has checked it on small overfitting tests')
        self.logger.info('MG should still run a full set of experiments to make sure nothing small changed from the historical/messy code base')
        self.logger.info(' - - - - - - - - - ')
        self.logger.info('3) DDP setup in main.py and DDP init in trainer.py assume just one node (i.e. rank = device). Fix in both places if multinode needed')
        self.logger.info('-------------------------------')

    def check_valid(self, x):
        if self.config.check_nans:
            if torch.any(torch.isnan(x)):
                print("X IS NAN")
                assert False
            if torch.any(torch.isinf(x)):
                print("X IS INF")
                assert False

    def is_main(self,):
        return self.rank == 0

    def is_early(self,):
        return self.update_steps in [0,1,2,3,4,5,10,20,50]

    def time_to_print(self,):
        A = self.update_steps % self.config.print_every == 0
        B = self.is_early()
        return A or B

    def time_to_sample(self,):
        return self.update_steps % self.config.sample_every == 0

    def time_to_wandb(self,):
        wandb_every = self.config.wandb_every
        return (self.update_steps % wandb_every == 0)

    def time_to_update_ema(self,):
        A = self.update_steps >= self.config.update_ema_after
        B = self.update_steps %  self.config.update_ema_every == 0
        return (A and B)

    def setup_model(self,):
        # taming vae does ckpt load automatically. todo handle custom river vae too.
        self.vae, self.vae_restored = vqvae.build_vqvae(self.config, logger = self.logger)
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


        # Setup Checkpointer
        self.checkpointer = checkpointing.Checkpointer(
            checkpoint_dir = self.checkpoint_dir,
            logger = self.logger,
            model = self.model,
            ema = self.ema,
            opt = self.opt,
            config = self.config,
            rank = self.rank,
            save_every = self.config.save_every,
            save_most_recent_every = self.config.save_most_recent_every,
        )

        # maybe resume from ckpt for model, ema, opt
        self.is_resumed, self.ema_restored, self.opt_restored = self.checkpointer.maybe_load(
            ckpt_path = self.config.load_model_ckpt_path
        )

        # wrapp in DDP
        self.model = DDP(self.model.to(self.device), device_ids=[self.rank])
        
        # Just in case, ensure opt state on correct device after potential checkpoint load
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

    def training_loop(self,):
        self.checkpointer.checkpoint(
            update_steps = self.update_steps, mode = 'init'
        )
        self.x_overfit = next(iter(self.train_dataloader))
        if self.config.overfit == 'one':
            for i in range(1, self.x_overfit.shape[0]):
                self.x_overfit[i] = self.x_overfit[0]

        while self.update_steps < self.config.num_training_steps:
            self.do_epoch()
        self.checkpointer.checkpoint(
            update_steps = self.update_steps, mode = 'final'
        )
        self.logger.info("Done!")
        utils.ddp_cleanup()
 
    def done_with_epoch(self, batch_num):
    
        done = False

        if self.config.limit_train_batches > 0: 
            if batch_num >= self.config.limit_train_batches:
                done = True

        if self.update_steps >= self.config.num_training_steps:
            done = True
            self.logger.info(f"Done with {self.config.num_training_steps} steps.")
        return done
        
    def do_epoch(self):
        # set_epoch() not needed for sampler since each rank
        # uses its own generator with local seed
        for batch_num, x in enumerate(self.train_dataloader):
            if self.done_with_epoch(batch_num):
                break
            self.train_step(batch_num, x)

    def clip_grads(self, x):
        return torch.nn.utils.clip_grad_norm_(x, self.config.grad_clip_norm).item()

    def batched_get_index(self, x, idx):
        return x[torch.arange(x.shape[0], device=x.device), idx]

    def get_frames_for_training(self, x):
        '''
        get the data frame, 
        the previous "reference" frame, 
        and one random "context" frame from the past.
        '''
        batch_size, num_frames, C, H, W = x.shape
        assert num_frames > 2
       
        # Frame(t) where t ∈ {2, ..., num_obs-1}
        data_idx = torch.randint(low=2, high=num_frames, size=(batch_size,), device=self.device)

        # Frame(t-1) where (t-1) ∈ {1, ..., num_obs-2}
        reference_idx = data_idx - 1

        # Frame(j) with j < t-1
        # option 1:  faster but less readable
        # why correct: rand() in [0,1), mult by (data_idx-1) gives [0, data_idx-1), then floor and long give {0,1,...data_idx-2}
        context_idx = (torch.rand(batch_size, device=self.device) * (data_idx - 1).float()).floor().long()
        # if you want a more readable but slower version, use this and recall that randint is exclusive of its upper bound.
        #cond_idx = torch.cat(
        #    [torch.randint(low=0, high=s-1, size=1, device=self.device) for s in data_idx], dim = 0
        #)
        
        # get frames
        data_frame      = self.batched_get_index(x, data_idx)
        reference_frame = self.batched_get_index(x, reference_idx)
        context_frame   = self.batched_get_index(x, context_idx)
        
        # gap ≥ 1
        gap = (reference_idx - context_idx).type_as(x)

        return data_frame, reference_frame, context_frame, gap

    def prepare_batch(self, x):
        '''
        - get a frame triplet for training
        - encode the frames into the latent space
        - compute the noisy states and interpolant targets: 
        this code either trains the ODE velocity for the linear schedule
        or the SDE drift for the proposed schedule, though in theory 
        you could also use an SDE with the linear schedule
        '''
        data_frame, reference_frame, context_frame, gap = self.get_frames_for_training(x)

        # pack frames to encode all at once instead of with 3 passes of VAE
        stacked_frames = torch.stack([data_frame, reference_frame, context_frame], dim=1)
        Z = vqvae.encode(self.vae, stacked_frames, self.config.vqvae_type)
        assert Z.shape[1] == 3
        z1, Z_reference, Z_context = Z[:,0], Z[:, 1], Z[:, 2]

        # get random times
        t = torch.distributions.Uniform(
            low = self.config.time_min_training,
            high = self.config.time_max_training,
        ).sample((Z.shape[0],)).type_as(z1)
                       
        # get noisy states and loss targets
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
        batch_size = x.size(0)
        num_frames = x.size(1)
        batch_size, num_frames, C, H, W = x.shape
        zt, t, Z_reference, Z_context, gap, target = self.prepare_batch(x)
        model_out = self.model(z=zt, t=t, ref=Z_reference, context=Z_context, gap=gap)
        self.check_valid(model_out)
        assert model_out.shape == target.shape
        loss = (model_out - target).pow(2).sum(dim=[1,2,3]).mean()
        return loss


    def train_step(self, batch_num, x):
        '''
        do a training step
        '''
        if self.config.overfit in ['batch', 'one']:
            x = self.x_overfit
        # e.g. for kth, num_frame is 40, so this gets first 40 frames
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
        if self.time_to_update_ema():
            utils.update_ema(self.ema, self.model.module)
        sample_dict = self.sample(x)
        utils.copy_into_A_from_B(A=log_dict, B=sample_dict)
        if self.time_to_wandb() and self.is_main():
            wandb.log(log_dict,step=self.update_steps)
        if self.time_to_print():
            self.logger.info(f"(step={self.update_steps:07d}) Train Loss: {loss_item:.4f}")
        self.checkpointer.checkpoint(
            update_steps = self.update_steps
        )
        self.update_steps += 1

    def update_lr(self,):
        new_lr = utils.get_lr(
            update_step=self.update_steps,
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
    def sample(self, x):
       
        if self.time_to_sample():

            ret_dict = {}

            # this split name is just for logging keys.
            # if you want to sample from held out data, you will
            # need to pass a different x here.
            split = 'train'

            x = x.clone()[torch.randperm(x.shape[0])]
            self.logger.info("Generating samples...")
            batch_size, num_cond, num_gen = self.get_num_frames_for_sampling(x)
            X_real = x[ : batch_size, : (num_cond + num_gen)]
            # we start generating frames after this prefix
            X_cond = X_real[:, : num_cond] 
            
            for use_ema in [False, True]:

                ema_key = 'ema' if use_ema else 'nonema'
                self.logger.info(f"Sampling with {ema_key}")

                X_hat = self._sample(X_cond, num_gen, use_ema)
                # Log images grid
                grid = utils.make_observations_grid([X_cond, X_hat], num_sequences = X_real.shape[0])
                # two real and two generated, side by side
                real_and_fake = torch.stack(
                    [X_real[0],  X_real[1], X_hat[0], X_hat[1]], dim = 0
                )

                # if you want to see it picture-by-picture in a long scroll
                #ret_dict[f"{split}/Media/reconstructed_observations"] = wandb.Image(grid)
                fps = 3 # frames per second. reduce to 1 to see each video more clearly picture-by-picture
                ret_dict[f"{split}/Media/real_videos_{ema_key}"] = utils.to_wandb_vid(X_real, fps=fps)
                ret_dict[f"{split}/Media/generated_videos_{ema_key}"] = utils.to_wandb_vid(X_hat, fps=fps)
                ret_dict[f"{split}/Media/real_vs_generated_{ema_key}"] = utils.to_wandb_vid(real_and_fake, fps=fps)
        else:
            ret_dict = {}
        return ret_dict

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
            g = self.interpolant.sigma(t_arr)
            self.check_valid(g)

            zt_mean = zt + f * dt

            self.check_valid(zt_mean)


            diffusion_term = g[:, None, None, None] * torch.randn_like(zt_mean) * sqrt_dt
            self.check_valid(diffusion_term)

            zt = zt_mean + diffusion_term

            self.check_valid(zt)


        # don't add diffusion term on the last step
        # common trick for EM integration in deep generative models
        z1 = zt_mean
        return z1

    def get_conditioning_frames_for_sampling(self, Z):
        """
        Choose reference = last available frame (t-1),
        context = some j < (t-1),
        gap = (t-1) - j >= 1
        """
        assert Z.ndim == 5  # (b, n, c, h, w)
        b, num_frames, _, _, _ = Z.shape
        # need at least 2 frames available (t-1 exists)
        assert num_frames >= 2, "Need ≥2 frames in Z to pick (reference=t-1, context<ref)."
        device = Z.device
        dtype_long = torch.long

        # last available index = t-1
        reference_idx_scalar = num_frames - 1

        # context ∈ {0, 1, ..., (t-1) - 1}  (torch.randint: high is exclusive)
        # unlike training, here the whole batch has the same reference idx.
        context_idx = torch.randint(0, reference_idx_scalar, (b,), device=device, dtype=dtype_long)

        # turn ref idx into a vector, one for each batch element
        reference_idx = torch.full((b,), reference_idx_scalar, device=device, dtype=dtype_long)

        Z_reference = self.batched_get_index(Z, reference_idx)
        Z_context = self.batched_get_index(Z, context_idx)

        gap = (reference_idx - context_idx).type_as(Z) # (b,)

        return Z_reference, Z_context, gap

    def get_time_grid_for_sampling(self,):
        return torch.linspace(
            self.config.time_min_sample, 
            self.config.time_max_sample, 
            self.config.num_sampling_steps
        )

    @torch.no_grad()
    def _sample(self, X_cond, num_gen, use_ema):
        '''
        sample loop to generate num_gen frames
        X_cond is the whole conditioning history that we will append to
        '''
        Z = vqvae.encode(self.vae, X_cond, self.config.vqvae_type)

        # (batch size, num frames, num channels, H, W)
        b, original_n, c, h, w = Z.shape

        # If there is only one conditioning frame, duplicate it so that we have
        # both a reference frame and context frame. But remember to remove it later.
        if original_n == 1:
            Z = Z[:, [0,0]]
            n = Z.shape[1]
            assert n == 2
        else:
            n = original_n
 
        self.logger.info("GENERATING FRAMES")
        t_grid = self.get_time_grid_for_sampling().type_as(Z)
        ones = torch.ones(b,).type_as(Z)

        # below, I use lowercase z for the interpolant
        # and uppercase Z for the generated latent frames
        # and uppercase X for the pixel space frames
        for k in range(num_gen):
             
            if k % 5 == 0:
                self.logger.info(f"generating frame {k} out of {num_gen}")

            Z_reference, Z_context, gap = self.get_conditioning_frames_for_sampling(Z)

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

        X = vqvae.decode(self.vae, Z, self.config.vqvae_type)
        return X

