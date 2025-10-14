import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

class Checkpointer

    def __init__(self, model, ema, opt, config, rank, save_every, save_most_recent_every):        
        self.model = model
        self.ema = ema
        self.opt = opt
        self.config = config
        self.rank = rank
        self.save_every = save_every
        self.save_most_recent_every = save_most_recent_every

        # does not keep track of pretrained VAE, no need.

	def strip_ddp_prefix_from_checkpoint(self, state):

	    is_ddp = False
	    for k in state:
		if k.startswith("module"):
		    is_ddp = True
		    break
	    if is_ddp:
		state = {k.replace("module.", ""): v for k, v in state.items()}

	    return state

	def cleanup_old_checkpoint_names(state):
	    # old checkpoints had the velocity model as an attribute of an outer class.
            # now the model is only that inner model.
            state = {
		k.replace('vector_field_regressor.', ''): v for k, v in state.items()
	    }

            # old checkpoints contained the autoencoder. We now keep it separate
	    state = {k: v for k, v in state.items() if not k.startswith('ae')}

	    return state

        def time_to_save(self, update_steps):
            return update_steps % self.save_every == 0

        def time_to_save_most_recent(self, update_steps):
            return update_steps % self.save_most_recent_every == 0

        def get_checkpoint_dict(self,):

            # save models in eval mode
            self.model.eval()
            self.ema.eval()
 
            is_ddp = True if isinstance(model, DDP) else False

            return {
                "model": self.model.module.state_dict() if is_ddp else self.model.state_dict(), 
                "ema": self.ema.state_dict(),
                "opt": self.opt.state_dict(),
                "config": self.config
            }

        def save_ckpt_to_file(self, checkpoint, checkpoint_path):
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")

        def checkpoint(self, mode=None):
            A = (mode in ['init', 'final'])
            B = self.time_to_save(update_steps)
            C = self.time_to_save_most_recent(update_steps)
            if (A or B or C):
                if self.rank == 0:
                    self._checkpoint_no_blocking(update_steps = update_steps, mode = mode)
                dist.barrier()

        def _checkpoint_no_blocking(self, update_steps, mode = mode):
            
            if mode == 'init':
                checkpoint = self.get_checkpoint_dict()
                checkpoint_path = f"{self.checkpoint_dir}/init.pt"
                self.save_ckpt_to_file(checkpoint, checkpoint_path)

            elif mode == 'final':
                checkpoint = self.get_checkpoint_dict()
                checkpoint_path = f"{self.checkpoint_dir}/final.pt"
                self.save_ckpt_to_file(checkpoint, checkpoint_path)
            
            else:

                if self.time_to_save(update_steps):
                    checkpoint = self.get_checkpoint_dict()
                    checkpoint_path = f"{self.checkpoint_dir}/{self.update_steps:07d}.pt"
                    self.save_ckpt_to_file(checkpoint, checkpoint_path)

                if self.time_to_save_most_recent(update_steps):
                    checkpoint = self.get_checkpoint_dict()
                    checkpoint_path = f"{self.checkpoint_dir}/latest.pt"
                    self.save_ckpt_to_file(checkpoint, checkpoint_path)
    
        
        def maybe_load(self, ckpt_path):
            if ckpt_path is not None:
                state_dict = torch.load(
                    ckpt_path, 
                    map_location='cpu', 
                    weights_only=False
                )

                model_state = self.strip_ddp_prefix_from_checkpoint(state_dict['model'])
                model_state = self.cleanup_old_checkpoint_names(model_state)
                self.model.load_state_dict(model_state)
                print("CHECKPOINT LOADED MODEL")
                del model_state
                del state_dict['model']
                model_restored = True

                if 'ema' in state_dict:
                    self.ema.load_state_dict(state_dict["ema"])
                    ema_restored = True
                    print("CHECKPOINT LOADED EMA, MAKE SURE NOT TO CLEAR EMA IN MODEL SETUP, AFTER THIS LOAD.")
                    del state_dict['ema']
                else:
                    ema_restored = False

                if 'opt' in state_dict:
                    self.opt.load_state_dict(state_dict["opt"])
                    del state_dict['opt']
                    print("CHECKPOINT LOADED AN OPTIMIZER STATE")
                    opt_restored = True
                else:
                    print("CHECKPOINT DID NOT HAVE AN OPTIMIZER STATE")
                    opt_restored = False

                
                if 'config' in state_dict:
                    print("CHECKPOINT FOUND OLD CONFIG, BUT NOT STORING IT + NOT OVERWRITING CURRENT CONFIG")
                    del state_dict['config']
                else:
                    print("CHECKPOINT DID NOT FIND OLD CONFIG")

            else:
                model_restored = False
                ema_restored = False
                opt_restored = False

            return model_restored, ema_restored, opt_restored

