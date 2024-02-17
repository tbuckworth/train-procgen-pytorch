import math

import numpy as np
from torch import jit

from .misc_util import orthogonal_init, xavier_uniform_init
import torch.nn as nn
import torch
from vector_quantize_pytorch import VectorQuantize, FSQ
from einops.layers.torch import Reduce


def flatten_features(features):
    coor = get_coor(features)
    proc_features = torch.concat([features, coor], axis=3)
    flattened_features = entities_flatten(proc_features)
    return flattened_features


def get_coor(input_tensor):
    """
    The output of cnn is tagged with two extra channels indicating the spatial position(x and y) of each cell

    :param input_tensor: (TensorFlow Tensor)  [B,Height,W,D]
    :return: (TensorFlow Tensor) [B,Height,W,2]
    """
    # batch_size = tf.shape(input_tensor)[0]
    batch_size, height, width, _ = input_tensor.shape
    # change to -1:+1 as in deep rl w/ inductive biases (ICLR 2019)?
    coor = [[[h / height, w / width] for w in range(width)] for h in range(height)]
    coor = torch.unsqueeze(torch.Tensor(coor), axis=0)
    # [1,Height,W,2] --> [B,Height,W,2]
    coor = torch.tile(coor, [batch_size, 1, 1, 1])
    return coor


def entities_flatten(input_tensor):
    """
    flatten axis 1 and axis 2
    :param input_tensor: (TensorFlow Tensor) The input tensor from NN [B,H,W,D]
    :return: (TensorFlow Tensor) [B,N,D]
    """
    _, h, w, channels = input_tensor.shape
    return torch.reshape(input_tensor, [-1, h * w, channels])


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class MlpModel(nn.Module):
    def __init__(self,
                 input_dims=4,
                 hidden_dims=[64, 64],
                 **kwargs):
        """
        input_dim:     (int)  number of the input dimensions
        hidden_dims:   (list) list of the dimensions for the hidden layers
        use_batchnorm: (bool) whether to use batchnorm
        """
        super(MlpModel, self).__init__()

        # Hidden layers
        hidden_dims = [input_dims] + hidden_dims
        layers = []
        for i in range(len(hidden_dims) - 1):
            in_features = hidden_dims[i]
            out_features = hidden_dims[i + 1]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        self.apply(orthogonal_init)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class NatureModel(nn.Module):
    def __init__(self,
                 in_channels,
                 **kwargs):
        """
        input_shape:  (tuple) tuple of the input dimension shape (channel, height, width)
        filters:       (list) list of the tuples consists of (number of channels, kernel size, and strides)
        use_batchnorm: (bool) whether to use batchnorm
        """
        super(NatureModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=64 * 7 * 7, out_features=512), nn.ReLU()
        )
        self.output_dim = 512
        self.apply(orthogonal_init)

    def forward(self, x):
        x = self.layers(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


scale = 1


class ImpalaModel(nn.Module):
    def __init__(self,
                 in_channels,
                 **kwargs):
        super(ImpalaModel, self).__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16 * scale)
        self.block2 = ImpalaBlock(in_channels=16 * scale, out_channels=32 * scale)
        self.block3 = ImpalaBlock(in_channels=32 * scale, out_channels=32 * scale)
        self.fc = nn.Linear(in_features=32 * scale * 8 * 8, out_features=256)

        self.output_dim = 256
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.gru = orthogonal_init(nn.GRU(input_size, hidden_size), gain=1.0)

    def forward(self, x, hxs, masks):
        # Prediction
        if x.size(0) == hxs.size(0):
            # input for GRU-CELL: (L=sequence_length, N, H)
            # output for GRU-CELL: (output: (L, N, H), hidden: (L, N, H))
            masks = masks.unsqueeze(-1)
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        # Training
        # We will recompute the hidden state to allow gradient to be back-propagated through time
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            # (can be interpreted as a truncated back-propagation through time)
            has_zeros = ((masks[1:] == 0.0) \
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class ImpalaVQModel(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(ImpalaVQModel, self).__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16 * scale)
        self.block2 = ImpalaBlock(in_channels=16 * scale, out_channels=32 * scale)
        self.block3 = ImpalaBlock(in_channels=32 * scale, out_channels=32 * scale)
        self.fc = nn.Linear(in_features=32 * scale * 8 * 8, out_features=256)
        self.vq = VectorQuantize(dim=256, codebook_size=128, decay=.8, commitment_weight=1.)
        # decay=.99,cc=.25 is the VQ-VAE values
        self.output_dim = 256
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        quantized, indices, commit_loss = self.vq(x)
        return quantized


class ImpalaFSQModel(nn.Module):
    def __init__(self, in_channels, device, use_mha=False, **kwargs):
        super(ImpalaFSQModel, self).__init__()
        self.use_mha = use_mha
        self.device = device
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16 * scale)
        self.block2 = ImpalaBlock(in_channels=16 * scale, out_channels=32 * scale)
        self.block3 = ImpalaBlock(in_channels=32 * scale, out_channels=3 * scale)
        levels = [8, 6, 5]
        self.vq = FSQ(levels)
        if self.use_mha:
            self.mha1 = GlobalSelfAttention(shape=(64, 5), num_heads=5, embed_dim=5, dropout=0.1)

        self.max_pool = nn.MaxPool1d(kernel_size=5, stride=1)
        self.fc = nn.Linear(in_features=64, out_features=256)

        # self.vq = VectorQuantize(dim=256, codebook_size=128, decay=.8, commitment_weight=1.)
        # decay=.99,cc=.25 is the VQ-VAE values
        self.output_dim = 256
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = x.permute(0, 2, 3, 1)

        # Extract coordinates
        coor = get_coor(x)
        # Flatten
        flat_coor = entities_flatten(coor).to(device=self.device)
        flattened_features = entities_flatten(x)

        # Quantize
        x, indices = self.vq(flattened_features)

        # Add Co-ordinates to quantized latents
        x = torch.concat([x, flat_coor], axis=2)

        # Add mha layers
        if self.use_mha:
            x = self.mha1(x)
        x = self.max_pool(x)
        x = x.permute(0, 2, 1)
        # print(x.shape)
        x = nn.ReLU()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x.squeeze()


class BaseAttention(nn.Module):
    def __init__(self, shape, **kwargs):
        super(BaseAttention, self).__init__()
        self.mha = nn.MultiheadAttention(batch_first=True, **kwargs)
        self.layernorm = nn.LayerNorm(normalized_shape=shape)
        # self.add = torch.add


class GlobalSelfAttention(BaseAttention):
    def forward(self, x):
        attn_output, _ = self.mha(
            query=x,
            value=x,
            key=x,
            need_weights=False
        )
        x = x.add(attn_output)
        x = self.layernorm(x)
        return x

    def get_attn_weights(self, x):
        attn_output, attn_weight = self.mha(
            query=x,
            value=x,
            key=x,
            need_weights=True,
            average_attn_weights=False,
        )
        return attn_weight

    def forward_plus_attn_weights(self, x):
        attn_output, attn_weight = self.mha(
            query=x,
            value=x,
            key=x,
            need_weights=True,
            average_attn_weights=False,
        )
        x = x.add(attn_output)
        x = self.layernorm(x)
        return x, attn_weight


def halve_rounding_n_times(x, n_impala_blocks):
    for _ in range(n_impala_blocks):
        x = math.ceil(x / 2)
    return x


class ImpalaCNN(nn.Module):
    def __init__(self, in_channels, mid_channels, latent_dim, n_impala_blocks=3):
        super(ImpalaCNN, self).__init__()
        blocks = []
        for i in range(n_impala_blocks):
            in_c = mid_channels
            out_c = mid_channels
            if i == 0:
                in_c = in_channels
                out_c = mid_channels
            elif i == n_impala_blocks - 1:
                in_c = latent_dim
                out_c = latent_dim - 2
            elif i == n_impala_blocks - 2:
                in_c = mid_channels
                out_c = latent_dim
            blocks.append(
                ImpalaBlock(in_channels=in_c, out_channels=out_c)
            )
        self.blocks = nn.ModuleList(blocks)

        # self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=mid_channels)
        # self.block2 = ImpalaBlock(in_channels=mid_channels, out_channels=latent_dim)
        # self.block3 = ImpalaBlock(in_channels=latent_dim, out_channels=latent_dim - 2)

    def get_n_latents(self, input_shape):
        return int(np.prod([halve_rounding_n_times(x, len(self.blocks)) for x in input_shape[1:]]))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        # x = self.block1(x)
        # x = self.block2(x)
        # x = self.block3(x)
        x = nn.ReLU()(x)
        return x


class QuantizedMHAModel(nn.Module):
    use_vq: jit.Final[bool]
    use_fq: jit.Final[bool]

    def __init__(self,
                 in_channels,
                 device,
                 ob_shape,
                 n_latents,
                 encoder,
                 quantizer=None,
                 mha_layers=2,
                 num_heads=4,
                 embed_dim=64,
                 output_dim=256,
                 reduce='feature_wise',
                 **kwargs):
        super(QuantizedMHAModel, self).__init__()

        self.use_vq = True if isinstance(quantizer, VectorQuantize) else False
        self.use_fq = True if isinstance(quantizer, FSQ) else False
        self.device = device
        self.ob_shape = ob_shape
        self.output_dim = output_dim

        self.encoder = encoder
        self.quantizer = quantizer
        self.MHA = MHAModel(n_latents, embed_dim, mha_layers, output_dim, num_heads, reduce)

    def forward_with_attn_indices(self, x):
        x = self.encoder(x)
        x, indices = self.flatten_and_append_coor(x, True)
        x, atn_list = self.MHA.forward_plus_attn(x)
        # x = self.MHA(x)

        return x, atn_list, indices

    def forward(self, x):
        x = self.encoder(x)
        x, commit_loss = self.flatten_and_append_coor(x)
        x = self.MHA(x)

        if self.use_vq:
            return x, commit_loss
        return x

    def flatten_and_append_coor(self, x, return_indices=False):
        flat_coor, flattened_features = self.flatten_and_get_coor(x)

        # Quantize
        if self.use_vq:
            x, indices, commit_loss = self.quantizer(flattened_features)
        elif self.use_fq:
            x, indices = self.quantizer(flattened_features)
        else:
            x = flattened_features

        # Add Co-ordinates to quantized latents
        x = torch.concat([x, flat_coor], axis=2)
        if return_indices:
            return x, indices
        if self.use_vq:
            return x, commit_loss
        return x, None

    def flatten_and_get_coor(self, x):
        # Move channels to end
        x = x.permute(0, 2, 3, 1)
        # Extract coordinates
        coor = get_coor(x)
        # Flatten
        flat_coor = entities_flatten(coor).to(device=self.device)
        flattened_features = entities_flatten(x)
        return flat_coor, flattened_features


class ImpalaVQMHAModel(QuantizedMHAModel):
    def __init__(self, in_channels, mha_layers, device, obs_shape, use_vq=True, num_heads=4, **kwargs):
        hid_channels = 16
        latent_dim = 32
        # self.output_dim = 256
        input_shape = obs_shape
        # Each impala block halves input height and width.
        # These are flattened before VQ (hence prod)
        n_impala_blocks = 4
        encoder = ImpalaCNN(in_channels, hid_channels, latent_dim, n_impala_blocks)
        n_latents = encoder.get_n_latents(input_shape)

        self.device = device
        self.use_vq = use_vq
        self.mha_layers = mha_layers

        quantizer = None
        if use_vq:
            quantizer = VectorQuantize(dim=latent_dim - 2, codebook_size=128, decay=.8, commitment_weight=1.)

        super(ImpalaVQMHAModel, self).__init__(in_channels, device, input_shape, n_latents, encoder, quantizer,
                                               mha_layers, num_heads=num_heads, embed_dim=latent_dim, output_dim=256)

    def print_if_nan(self, name, print_nans, x):
        if print_nans:
            print(f"{name}:{'nans' if x.isnan().any() else ''}\n{x}")


class FSQMHAModel(QuantizedMHAModel):
    def __init__(self, in_channels, hid_channels, mha_layers, device, obs_shape, reduce, encoder_constructor,
                 levels=[8, 5, 5, 5], n_blocks=3, **kwargs):
        input_shape = obs_shape
        self.device = device
        self.mha_layers = mha_layers

        levels = levels
        quantizer = FSQ(levels)
        latent_dim = len(levels) + 2
        encoder = encoder_constructor(in_channels, hid_channels, latent_dim, n_blocks)
        n_latents = encoder.get_n_latents(input_shape)

        super(FSQMHAModel, self).__init__(in_channels, device, input_shape, n_latents, encoder, quantizer,
                                                mha_layers, num_heads=latent_dim, embed_dim=latent_dim, output_dim=256,
                                                reduce=reduce)


class ImpalaFSQMHAModel(FSQMHAModel):
    def __init__(self, in_channels, mha_layers, device, obs_shape, reduce, n_impala_blocks=3, levels=[8, 5, 5, 5],
                 **kwargs):
        hid_channels = 16
        encoder_constructor = ImpalaCNN
        super(ImpalaFSQMHAModel, self).__init__(in_channels, hid_channels, mha_layers, device, obs_shape, reduce,
                                                encoder_constructor, levels, n_impala_blocks)


class RibFSQMHAModel(FSQMHAModel):
    def __init__(self, in_channels, mha_layers, device, obs_shape, reduce, levels=[8, 5, 5, 5], **kwargs):
        hid_channels = 12
        encoder_constructor = ribEncoder
        super(RibFSQMHAModel, self).__init__(in_channels, hid_channels, mha_layers, device, obs_shape, reduce,
                                             encoder_constructor, levels)


class ribEncoder(nn.Module):
    def __init__(self, in_channels, mid_channels, embed_dim, n_blocks=None):
        super(ribEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=embed_dim-2, kernel_size=2, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        return x

    def get_n_latents(self, obs_shape):
        return (obs_shape[-1] - 2) ** 2


class ribMHA(nn.Module):
    def __init__(self,
                 in_channels,
                 device,
                 ob_shape,
                 use_vq=False,
                 mha_layers=2,
                 num_heads=4,
                 **kwargs):
        super(ribMHA, self).__init__()

        self.use_vq = use_vq
        self.device = device
        self.ob_shape = ob_shape
        # add 2 for the padding and square because we flatten
        embed_dim = 64
        output_dim = 256

        encoder = ribEncoder(in_channels, mid_channels=12, embed_dim=embed_dim)
        n_latents = encoder.get_n_latents(ob_shape)
        quantizer = None
        if use_vq:
            quantizer = VectorQuantize(dim=embed_dim - 2, codebook_size=128, decay=.8, commitment_weight=1.)

        self.quantizedMHA = QuantizedMHAModel(in_channels, device, ob_shape, n_latents, encoder, quantizer,
                                              mha_layers, num_heads, embed_dim, output_dim)
        self.output_dim = output_dim
        self.apply(xavier_uniform_init)

    def forward(self, x):
        return self.quantizedMHA(x)


class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        # See Section 4.1 of "Neural Discrete Representation Learning".
        layers = []
        for i in range(num_residual_layers):
            layers.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=num_hiddens,
                        out_channels=num_residual_hiddens,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=num_residual_hiddens,
                        out_channels=num_hiddens,
                        kernel_size=1,
                    ),
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = h + layer(h)

        # ResNet V1-style.
        return torch.relu(h)


class Encoder(nn.Module):
    def __init__(
            self,
            in_channels,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
    ):
        super().__init__()
        # See Section 4.1 of "Neural Discrete Representation Learning".
        # The last ReLU from the Sonnet example is omitted because ResidualStack starts
        # off with a ReLU.
        conv = nn.Sequential()
        for downsampling_layer in range(num_downsampling_layers):
            if downsampling_layer == 0:
                out_channels = num_hiddens // 2
            elif downsampling_layer == 1:
                (in_channels, out_channels) = (num_hiddens // 2, num_hiddens)

            else:
                (in_channels, out_channels) = (num_hiddens, num_hiddens)

            conv.add_module(
                f"down{downsampling_layer}",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            )
            conv.add_module(f"relu{downsampling_layer}", nn.ReLU())

        conv.add_module(
            "final_conv",
            nn.Conv2d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=3,
                padding=1,
            ),
        )
        self.conv = conv
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens
        )

    def forward(self, x):
        h = self.conv(x)
        return self.residual_stack(h)


class Decoder(nn.Module):
    def __init__(
            self,
            embedding_dim,
            num_hiddens,
            num_upsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
    ):
        super().__init__()
        # See Section 4.1 of "Neural Discrete Representation Learning".
        self.conv = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1,
        )
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens
        )
        upconv = nn.Sequential()
        for upsampling_layer in range(num_upsampling_layers):
            if upsampling_layer < num_upsampling_layers - 2:
                (in_channels, out_channels) = (num_hiddens, num_hiddens)

            elif upsampling_layer == num_upsampling_layers - 2:
                (in_channels, out_channels) = (num_hiddens, num_hiddens // 2)

            else:
                (in_channels, out_channels) = (num_hiddens // 2, 3)

            upconv.add_module(
                f"up{upsampling_layer}",
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            )
            if upsampling_layer < num_upsampling_layers - 1:
                upconv.add_module(f"relu{upsampling_layer}", nn.ReLU())

        self.upconv = upconv

    def forward(self, x):
        h = self.conv(x)
        h = self.residual_stack(h)
        x_recon = self.upconv(h)
        return x_recon


class VQVAE(nn.Module):
    def __init__(
            self,
            in_channels,
            num_hiddens=128,
            num_downsampling_layers=4,
            num_residual_layers=2,
            num_residual_hiddens=32,
            embedding_dim=8,
            num_embeddings=128,
            use_ema=True,
            decay=.99,
            epsilon=1e-5,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
        )
        self.pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1
        )
        self.vq = VectorQuantize(
            dim=embedding_dim,
            codebook_size=num_embeddings,
            ema_update=use_ema,
            decay=decay,
            eps=epsilon,
            accept_image_fmap=True
        )
        self.decoder = Decoder(
            embedding_dim,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
        )

    def quantize(self, x):
        z = self.pre_vq_conv(self.encoder(x))
        (z_quantized, encoding_indices, loss) = self.vq(z)
        return (z_quantized, encoding_indices, loss)

    def forward(self, x):
        (z_quantized, indices, loss) = self.quantize(x)
        x_recon = self.decoder(z_quantized)
        return {
            "loss": loss,
            "x_recon": x_recon,
        }

    def extract_features(self, inputs):
        z = self.pre_vq_conv(self._encoder(inputs))
        vq_output = self.vq(z)
        return vq_output, z


def get_trained_vqvqae(in_channels, hyperparameters, device):
    model = VQVAE(in_channels,
                  hyperparameters["num_hiddens"],
                  hyperparameters["num_downsampling_layers"],
                  hyperparameters["num_residual_layers"],
                  hyperparameters["num_residual_hiddens"],
                  hyperparameters["embedding_dim"],
                  hyperparameters["num_embeddings"],
                  hyperparameters["use_ema"],
                  hyperparameters["decay"],
                  hyperparameters["epsilon"],
                  )
    model.load_state_dict(torch.load(hyperparameters["model_path"], map_location=device)["model_state_dict"])
    return model


class MHAModel(nn.Module):
    def __init__(self, n_latents, latent_dim, mha_layers, output_dim, num_heads=4, reduce='feature_wise'):
        super(MHAModel, self).__init__()
        self.mha_layers = mha_layers
        # self.vqvae = get_trained_vqvqae(in_channels, hyperparameters, model_path, device)

        # Maybe dropout should be 0.0 to make relations less entangled
        self.mha = GlobalSelfAttention(shape=(n_latents, latent_dim), num_heads=num_heads, embed_dim=latent_dim,
                                       dropout=0.1)
        if reduce == 'feature_wise':
            pool_reduction = 'w'
            fc_dim = latent_dim
        elif reduce == 'dim_wise':
            pool_reduction = 'h'
            fc_dim = n_latents
        else:
            raise NotImplementedError
        self.max_pool = Reduce(f'b h w -> b {pool_reduction}', 'max')

        self.fc1 = nn.Linear(fc_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, output_dim)

        self.output_dim = output_dim
        self.apply(xavier_uniform_init)

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        for _ in range(self.mha_layers):
            x = self.mha(x)
        x = self.max_pool(x)
        x = x.squeeze()
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        x = nn.ReLU()(x)
        return x

    def forward_plus_attn(self, x):
        # if n > self.mha_layers:
        #     raise IndexError(f"n: {n} > mha_layers: {self.mha_layers}")
        output = []
        for _ in range(self.mha_layers):
            x, atn = self.mha.forward_plus_attn_weights(x)
            output.append(atn)
        return x, output
        # return self.mha.get_attn_weights(x)
