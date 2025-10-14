import torch
import vqvae_river
import vqvae_taming
from einops import rearrange, repeat                                                                                                                                                                                                                                            


def flatten(x):
    return rearrange(x, "b n c h w -> (b n) c h w")

def unflatten(x, n = 3):
    return rearrange(x, "(b n) c h w -> b n c h w", n=n)

@torch.no_grad()
def encode(vae, x, vqvae_type):
    assert len(x.shape) == 5
    bsz, frames, C, H, W = x.shape
    vae.eval()
    if vqvae_type == 'river':
        Z = vae(x).latents
    else:
        # taming transformers VAE has different layout.
        flat_X = flatten(x)
        flat_Z = vae.encode(flat_X)
        Z = unflatten(flat_Z, n = x.shape[1])
    return Z

@torch.no_grad()
def decode(vae, Z, vqvae_type):

    b, n, c, h, w = Z.shape

    if vqvae_type == 'river':
        dec_fn = vae.backbone.decode_from_latents
    else:
        dec_fn = vae.decode

    Z = rearrange(Z, "b n c h w -> (b n) c h w")
    X = dec_fn(Z)
    X = rearrange(X, "(b n) c h w -> b n c h w", b=b)
    return X


@torch.no_grad()
def encode_decode(vae, X, vqvae_type):
    b = X.shape[0]
    Z = encode(vae, X, vqvae_type)
    X_hat = decode(vae, Z, vqvae_type)
    return X_hat    

def build_vqvae(args):
    
    if args.vqvae_type == 'river':
        backbone = vqvae_river.VQVAE(
            args, args.load_vqvae_ckpt_path
        )
        vae = utils.SequenceConverter(backbone)
    else:

        if args.vqvae_config == 'f8':
            cfg = vqvae_taming.vq_f8_ddconfig
        elif args.vqvae_config == 'f8_small':
            cfg = vqvae_taming.vq_f8_small_ddconfig
        else:
            cfg = vqvae_taming.vq_f16_ddconfig

        vae = vqvae_taming.VQModelInterface(
            cfg, args.load_vqvae_ckpt_path
        )

    return vae
