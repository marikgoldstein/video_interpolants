
import vqvae_river
import vqvae_taming

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
