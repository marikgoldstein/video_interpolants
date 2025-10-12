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

# local stuff
import interpolants
import utils
from utils import TensorFolder
import arch
import vqvae
import video_dataset

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
        self.seed = self.args.global_seed * self.world_size + self.rank
        torch.manual_seed(self.seed)
        torch.cuda.set_device(self.device)
        print(f"Starting rank={self.rank}, seed={self.seed}, world_size={self.world_size}.")
        self.local_batch_size = int(self.args.global_batch_size // self.world_size)
        print("local batch size is", self.local_batch_size)

    def setup_dirs_and_logging(self,):
        if self.rank == 0:
            os.makedirs(self.args.results_dir, exist_ok=True)  
            experiment_name = f"{self.args.task}-{self.args.interpolant_type}"
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
        self.model = arch.VectorFieldRegressor(
            state_size = self.args.model_state_size,
            state_res = self.args.model_state_res,
            inner_dim = self.args.model_inner_dim,
            depth = self.args.model_depth,
            mid_depth = self.args.model_mid_depth,
            out_norm = self.args.model.out_norm,
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
        self.logger.info(f"SiT Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

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


            self.data_aug = False
                        self.data_random_horizontal_flip = False
                                    self.data_albumentations = True


    def setup_data(self,):
        self.datasets = {}
        self.datasets['train'] = self.get_video_datasets(split = 'train')
        self.datasets['val'] = self.get_video_datasets(split = 'val')
        self.sampler = torch.utils.data.RandomSampler(
            self.datasets["train"],
            replacement=True,
            num_samples=(self.local_batch_size * self.args.num_training_steps),
            generator=torch.Generator().manual_seed(self.seed)
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
        self.logger.info(f"Training for {args.epochs} epochs...")
        # get 2nd batch, dont like first one 
        self.x_overfit = next(iter(self.train_dataloader))
        # TODO
        #self.plot_real_data()
        for epoch_num in range(self.args.epochs):
            if self.train_steps >= self.args.num_training_steps:
                self.logger.info("Done. Breaking out of loop over epochs")
                break
            self.do_epoch(epoch_num)
        self.model.eval()
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

        
    def do_epoch(self, epoch_num):

        self.sampler.set_epoch(epoch_num)
        self.logger.info(f"Beginning epoch {epoch_num}...")

        for batch_num, (x, y) in enumerate(self.train_dataloader):

            if self.done_with_epoch(batch_num):
                break
            
            self.do_step(batch_num, x, y)
            
        self.logger.info(f"Done with epoch:{epoch_num}")
   
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

    def get(self, x, idx):
        return x[torch.arange(x.shape[0]), idx]

    def wide(self, x):
        return x[:, None, None, None]

    def get_data_cond_ref(self, x):
        bsz, num_obs, C, H, W = x.shape
        assert num_obs > 2
        data_idx = self.randint(2, num_obs, bsz)
        data = self.get(x, data_idx)
        ref_idx = data_idx - 1
        ref = self.get(x, ref_idx)
        cond_idx = torch.cat(
            [self.randint(0, s-1, 1) for s in data_idx], dim = 0
        )
        cond = self.get(x, cond_idx)
        assert torch.all(ref_idx > cond_idx)
        for tensor in [data, cond, ref]:
            assert tensor.shape == (bsz, C, H, W)
        gap = (ref_idx - cond_idx).type_as(data)[:, None]
        images = torch.stack([data, ref, cond], dim = 1)
        return images, gap

    def prepare_batch(self, x):
        images, gap = self.get_data_cond_ref(x)
        Z = self.vae_encode(images)
        z1, zref, zcond = Z[:,0], Z[:, 1], Z[:, 2]
        t = self.sample_time(bsz).type_as(z1)
        if self.args.interpolant_type == 'linear':
            z0 = torch.randn_like(z1)
            zt = self.interpolant.compute_xt(x0=z0,x1=z1,t=t)
            zt_dot = self.interpolant.compute_xdot(x0=z0,x1=z1,t=t) 
        elif slef.args.interpolant_type == 'ours':
            z0 = zref
            noise = torch.randn_like(z1)
            zt = self.interpolant.compute_xt(x0=z0,x1=z1,t=t, noise = noise)
            zt_dot = self.interpolant.compute_xdot(x0=z0,x1=z1,t=t, noise = noise) 
        return zt, t, zref, zcond, gap, target

    def loss_fn(self, x):
        bsz = x.size(0)
        num_obs = x.size(1)
        bsz, num_obs, C, H, W = x.shape
        zt, t, zref, zcond, gap, target = self.prepare_batch(x)
        vtheta = self.model(zt, t, zref, zcond, gap)
        assert vtheta.shape == zt_dot.shape
        loss = (vtheta - target).pow(2).sum(dim=[1,2,3]).mean()
        return loss




    def do_step(self, batch_num, x, y):
        if self.args.overfit:
            x = self.x_overfit
        x = batch[: , : self.args.num_observations]
        self.model.train()
        x = x.to(self.device)
        loss = self.loss_fn(x)
        self.opt.zero_grad()
        loss.backward()
        loss_item = loss.detach().item()
        log_dict = {'train_loss': loss_item}
        log_dict['grad_norm'] = self.clip_grads(self.model.parameters())
        self.opt.step()
        log_dict['lr'] = self.update_lr()
        utils.update_ema(self.ema, self.model.module)
        #sample_dict = self.do_sampling(x, y)
        #utils.copy_into_A_from_B(A=log_dict, B=sample_dict)
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
        if self.args.vqvae_type == 'river':
            Z = self.vae(xall).latents
        else:
            flat_xall = self.flatten(xall)
            flat_Z = self.vae.encode(flat_xall)
            Z = self.unflatten(flat_Z, n = xall.shape[1])
        return Z

    def sample_time(self, bsz):
        t = torch.distributions.Uniform(
            low = self.args.time_min_training, 
            high = self.args.time_max_training
        ).sample((bsz,))
        return t

    def update_lr(self,):
        new_lr = get_lr(
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
    def do_sampling(self, x, y):    
        assert False
        if self.time_to_sample():
            self.logger.info("Generating samples...")
            bsz, new_num_obs = x.shape[0]
            new_num_obs = x.shape[1]
            new_bsz = min(4, bsz)
            num_cond_frames = self.args.condition_frames
            frames2gen = self.args.frames_to_generate
            assert num_cond_frames + frames2gen == new_num_obs
            xreal = x[ : new_bsz, : num_cond_frames + frames2gen]
            xcond = xreal[ : , : num_cond_frames]
            xhat = dmodel.sample(X_series = xcond, num_frames = frames2gen) # TODO
            assert xhat.shape == xreal.shape
            D = {'xhat': xhat, 'xreal': xreal, 'xcond': xcond}
            mg_log_media(D, logger, 'Training') # TODO 
            return D
        else:
            return {}

    
    @torch.no_grad()
    def sample(self, X_series, num_frames):
        assert False
        Z_series = self.encode(X_series)

        # batch, num frames, num channels, height width
        b, n, c, h, w = Z_series.shape
        if n == 1:
            Z_series = Z_series[:, [0, 0]]

        def get_random_cond(Z_series):
            
            high = Z_series.size(1) - 1
            i = torch.randint(low = 0, high = high, size = [b])
            cond = self.get(Z_series, i)
            gap = (high - i).type_as(Z_series)[:, None]
            return cond, gap

        print("GENERATING FRAMES")
        # Generate future latents
        tmin = self.args.time_min_sample
        tmax = self.args.time_max_sample
        steps = self.args.num_sampling_steps
        tgrid = torch.linspace(tmin, tmax, steps).to(Z_series.device)
        ones = torch.ones(b).to(Z_series.device)
        
    
        for k in range(num_frames):
             
            if k % 5 == 0:
                print(f"generating frame {k} out of {num_frames}")
        
            # every frame, need to get next zcond and zref
            zcond, gap = get_random_cond(Z_series)
            
            zref = Z_series[:, -1]
    
            assert zref.shape == (b, self.args.C_latent, self.args.H_latent, self.args.W_latent)
            assert zcond.shape == (b, self.args.C_latent, self.args.H_latent, self.args.W_latent)
            if self.args.interpolant_type == 'linear':
                # ODE step
                def f(t, zt):
                    t_arr = t * ones
                    return self.model(zt, t_arr, zref, zcond, gap)
                z0 = torch.randn([b, c, h ,w]).to(Z_series.device)
                z1 = odeint(f, z0, tgrid, method='euler')[-1]
 
            elif self.args.interpolant_type == 'ous':
            
                # z_t = a(t)x^{t-1} + b(t)x^t + sigma(t)root(t)noise                                                                                   
                # bhat(z_t, t, cond) = b(z_t, t, x^{t-1}, x^j, (t-1)-j)

                # adot(t) x^{t-1} + bdot(t)x^{t} + sigmadot(t)root(t)noise

                # dZ^t_s = bhat(Z^t_s, t, Z^{t-1}, Z^{j}, (t-1)-j) ds + sigma(s)dW_s

                init_condition = zref
                dt = tgrid[1] - tgrid[0]
                zt = zref
                for tscalar in tgrid:
                    t_arr = tscalar * ones
                    f = self.model(zt, t_arr, zref, zcond, gap)
                    g = self.wide(self.interpolant.sigma(t_arr))
                    zt_mean = zt + f * dt
                    diffusion_term = g * torch.randn_like(zt_mean) * torch.sqrt(dt)
                    zt = zt_mean + diffusion_term
                z1 = zt_mean

            else:

                assert False

            Z_next = z1[:, None, ...]
            Z_series = torch.cat([Z_series, Z_next], dim = 1)

        if n == 1:
            Z_series = Z_series[:, 1:]

        ae_ours = (self.args.vqvae_type == 'river')
        dec_fn = self.vae.backbone.decode_from_latents if ae_ours else self.vae.decode
        # Decode to image space
        Z_series = rearrange(Z_series, "b n c h w -> (b n) c h w")
        X_series = dec_fn(Z_series)
        X_series = rearrange(X_series, "(b n) c h w -> b n c h w", b=b)
        return X_series

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
    args = set_args_from_river_repo(args)
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
