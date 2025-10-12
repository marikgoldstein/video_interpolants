
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
import torch
import torch.nn as nn
from einops import rearrange
from utils import TensorFolder


def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=(stride, stride), bias=False)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            downsample_factor: int = 1,
            last_affine: bool = True,
            drop_final_activation: bool = False,
            norm_layer=nn.BatchNorm2d):
        """

        :param in_planes: Input features to the module
        :param out_planes: Output feature
        :param downsample_factor: Reduction factor in feature dimension
        :param drop_final_activation: if True does not pass the final output through the activation function
        """

        super(ResidualBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, out_planes, stride=1)
        self.bn1 = norm_layer(out_planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        # Enable the possibility to force alignment to normal gaussian
        self.bn2 = norm_layer(out_planes, affine=last_affine)
        self.downsample_factor = downsample_factor
        self.drop_final_activation = drop_final_activation

        self.downsample = None
        if self.downsample_factor != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, out_planes, stride=1),
                nn.AvgPool2d(downsample_factor),
                # Enable the possibility to force alignment to normal gaussian
                norm_layer(out_planes, affine=last_affine)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = F.avg_pool2d(out, self.downsample_factor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if not self.drop_final_activation:
            out = self.relu(out)

        return out




class UpBlock(nn.Module):
    """
    Upsampling block.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            kernel_size: int = 3,
            padding: int = 1,
            scale_factor: int = 2,
            upscaling_mode: str = "nearest",
            late_upscaling: bool = False):
        """

        :param in_features: Input features to the module
        :param out_features: Output feature
        :param kernel_size: Size of the kernel
        :param padding: Size of padding
        :param scale_factor: Multiplicative factor such that output_res = input_res * scale_factor
        :param upscaling_mode: interpolation upscaling mode
        :param late_upscaling: if True upscaling is applied at the end of the block, otherwise it is applied at the beginning
        """

        super(UpBlock, self).__init__()

        self.in_planes = in_features
        self.out_planes = out_features

        self.scale_factor = scale_factor
        self.upscaling_mode = upscaling_mode
        self.late_upscaling = late_upscaling
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=(kernel_size, kernel_size),
            padding=(padding, padding),
            bias=False)
        self.norm = nn.BatchNorm2d(out_features, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x

        # By default apply upscaling at the beginning
        if not self.late_upscaling:
            out = F.interpolate(out, scale_factor=self.scale_factor, mode=self.upscaling_mode)

        out = self.conv(out)
        out = self.norm(out)
        out = F.leaky_relu(out, 0.2)

        # If upscaling is required at the end, apply it afterwards
        if self.late_upscaling:
            out = F.interpolate(out, scale_factor=self.scale_factor, mode=self.upscaling_mode)

        return out





class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        min_encodings = min_encodings.view(z.size(0), z.size(1), z.size(2), self.n_e)

        return loss, z_q, min_encodings

    def get_latents_from_ids(self, latents_ids: torch.Tensor) -> torch.Tensor:
        """

        :param latents_ids: [b, h, w, n_e] one-hot vectors
        :return: [b, e_dim, h, w]
        """

        b, h, _, _ = latents_ids.shape

        flat_latents_ids = rearrange(latents_ids, "b h w e -> (b h w) e").to(torch.float32)
        flat_latents = torch.matmul(flat_latents_ids, self.embedding.weight)
        latents = rearrange(flat_latents, "(b h w) e -> b e h w", b=b, h=h)

        return latents




def normalize(in_channels, **kwargs):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def swish(x):
    return x*torch.sigmoid(x)

class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = 32):
        super(Encoder, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)

        residual_layers = []
        ch_mult = [1, 2, 4, 8]
        for i in range(len(ch_mult) - 1):
            in_ch = ch_mult[i] * mid_channels
            out_ch = ch_mult[i + 1] * mid_channels
            residual_layers.append(ResidualBlock(
                in_ch, out_ch, downsample_factor=2, norm_layer=normalize))
        self.residuals = nn.Sequential(*residual_layers)

        attn_ch = ch_mult[-1] * mid_channels
        self.pre_attn_residual = ResidualBlock(attn_ch, attn_ch, downsample_factor=1, norm_layer=normalize)
        self.attn_norm = normalize(attn_ch)
        self.attn = nn.MultiheadAttention(embed_dim=attn_ch, num_heads=1, batch_first=True)
        self.post_attn_residual = ResidualBlock(attn_ch, attn_ch, downsample_factor=1, norm_layer=normalize)

        self.out_norm = normalize(attn_ch)
        self.out_conv = nn.Conv2d(attn_ch, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """

        :param images: [b, c, h, w]
        """

        x = self.conv_in(images)
        x = self.residuals(x)

        x = self.pre_attn_residual(x)
        z = self.attn_norm(x)
        h = z.size(2)
        z = rearrange(z, "b c h w -> b (h w) c")
        z, _ = self.attn(query=z, key=z, value=z)
        z = rearrange(z, "b (h w) c -> b c h w", h=h)
        x = x + z
        x = self.post_attn_residual(x)

        x = self.out_norm(x)
        x = swish(x)
        x = self.out_conv(x)

        return x



class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = 256):
        super(Decoder, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)

        self.pre_attn_residual = ResidualBlock(mid_channels, mid_channels, downsample_factor=1, norm_layer=normalize)
        self.attn_norm = normalize(mid_channels)
        self.attn = nn.MultiheadAttention(embed_dim=mid_channels, num_heads=1, batch_first=True)
        self.post_attn_residual = ResidualBlock(mid_channels, mid_channels, downsample_factor=1, norm_layer=normalize)

        residual_layers = []
        ch_div = [1, 2, 4, 8]
        for i in range(len(ch_div) - 1):
            in_ch = mid_channels // ch_div[i]
            out_ch = mid_channels // ch_div[i + 1]
            residual_layers.append(nn.Sequential(
                ResidualBlock(in_ch, out_ch, downsample_factor=1, norm_layer=normalize),
                UpBlock(out_ch, out_ch, scale_factor=2, upscaling_mode="nearest")))
        self.residuals = nn.Sequential(*residual_layers)

        out_ch = mid_channels // ch_div[-1]
        self.out_norm = normalize(out_ch)
        self.out_conv = nn.Conv2d(out_ch, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """

        :param images: [b, c, h, w]
        """

        x = self.conv_in(images)

        x = self.pre_attn_residual(x)
        z = self.attn_norm(x)
        h = z.size(2)
        z = rearrange(z, "b c h w -> b (h w) c")
        z, _ = self.attn(query=z, key=z, value=z)
        z = rearrange(z, "b (h w) c -> b c h w", h=h)
        x = x + z
        x = self.post_attn_residual(x)

        x = self.residuals(x)

        x = self.out_norm(x)
        x = swish(x)
        x = self.out_conv(x)

        return torch.tanh(x)


class VQVAE(nn.Module):
    def __init__(self, args):
        super(VQVAE, self).__init__()
	self.args = args
        self.encoder = Encoder(
            in_channels = args.vqgan_encoder_in_channels,
            out_channels = args.vqgan_encoder_out_channels,
        )
        self.decoder = Decoder(
            in_channels = args.vqgan_decoder_in_channels,
            out_channels = args.vqgan_decoder_out_channels,
        )
        self.vector_quantizer = VectorQuantizer(
            ne = args.vqgan_num_embeddings,
            e_dim = args.vqgan_embedding_dimension,
            beta = 0.25
        )

    def forward(self, images: torch.Tensor):
        # Encode images
        latents = self.encoder(images)

        # Quantize latent codes
        vq_loss, quantized_latents, quantized_latents_ids = self.vector_quantizer(latents)

        # Decode images
        reconstructed_images = self.decoder(quantized_latents)

        return utils.DictWrapper(
            # Input
            images=images,

            # Reconstruction
            reconstructed_images=reconstructed_images,

            # Aux output
            vq_loss=vq_loss,
            latents=latents,
            quantized_latents=quantized_latents,
            quantized_latents_ids=quantized_latents_ids,
        )

    def get_latents_from_ids(self, latents_ids: torch.Tensor) -> torch.Tensor:
        """

        :param latents_ids: [b, h, w, n_e] one-hot vectors
        :return: [b, c, h, w]
        """

        latents = self.vector_quantizer.get_latents_from_ids(latents_ids)  # [b, e_dim, h, w]

        return latents

    def decode(self, latents_ids: torch.Tensor) -> torch.Tensor:
        """

        :param latents_ids: [b, h, w, n_e] one-hot vectors
        :return: [b, c, h, w]
        """

        latents = self.get_latents_from_ids(latents_ids)  # [b, e_dim, h, w]
        decoded_images = self.decoder(latents)  # [b, c, h, w]

        return decoded_images

    def decode_from_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """

        :param latents: [b, c, h, w] latents
        :return: [b, 3, h, w]
        """

        # Quantize latent codes
        _, quantized_latents, _ = self.vector_quantizer(latents)

        # Decode images
        reconstructed_images = self.decoder(quantized_latents)

        return reconstructed_images


def build_vqvae(args, convert_to_sequence=False):
    backbone = VQVAE(args)
    if convert_to_sequence:
        return utils.SequenceConverter(backbone)
    else:
        return backbone

