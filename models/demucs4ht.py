import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import numpy as np
import torch
import json
from omegaconf import OmegaConf
from demucs.demucs import Demucs
from demucs.hdemucs import HDemucs

import math
from openunmix.filtering import wiener
from torch import nn
from torch.nn import functional as F
from fractions import Fraction
from einops import rearrange

from demucs.transformer import CrossTransformerEncoder

from demucs.demucs import rescale_module
from demucs.states import capture_init
from demucs.spec import spectro, ispectro
from demucs.hdemucs import pad1d, ScaledEmbedding, HEncLayer, MultiWrap, HDecLayer
from models.stft_utils import STFT_Process

class HTDemucs(nn.Module):
    """
    Spectrogram and hybrid Demucs model.
    The spectrogram model has the same structure as Demucs, except the first few layers are over the
    frequency axis, until there is only 1 frequency, and then it moves to time convolutions.
    Frequency layers can still access information across time steps thanks to the DConv residual.

    Hybrid model have a parallel time branch. At some layer, the time branch has the same stride
    as the frequency branch and then the two are combined. The opposite happens in the decoder.

    Models can either use naive iSTFT from masking, Wiener filtering ([Ulhih et al. 2017]),
    or complex as channels (CaC) [Choi et al. 2020]. Wiener filtering is based on
    Open Unmix implementation [Stoter et al. 2019].

    The loss is always on the temporal domain, by backpropagating through the above
    output methods and iSTFT. This allows to define hybrid models nicely. However, this breaks
    a bit Wiener filtering, as doing more iteration at test time will change the spectrogram
    contribution, without changing the one from the waveform, which will lead to worse performance.
    I tried using the residual option in OpenUnmix Wiener implementation, but it didn't improve.
    CaC on the other hand provides similar performance for hybrid, and works naturally with
    hybrid models.

    This model also uses frequency embeddings are used to improve efficiency on convolutions
    over the freq. axis, following [Isik et al. 2020] (https://arxiv.org/pdf/2008.04470.pdf).

    Unlike classic Demucs, there is no resampling here, and normalization is always applied.
    """

    @capture_init
    def __init__(
        self,
        sources,
        # Channels
        audio_channels=2,
        channels=48,
        channels_time=None,
        growth=2,
        # STFT
        nfft=4096,
        num_subbands=1,
        wiener_iters=0,
        end_iters=0,
        wiener_residual=False,
        cac=True,
        # Main structure
        depth=4,
        rewrite=True,
        # Frequency branch
        multi_freqs=None,
        multi_freqs_depth=3,
        freq_emb=0.2,
        emb_scale=10,
        emb_smooth=True,
        # Convolutions
        kernel_size=8,
        time_stride=2,
        stride=4,
        context=1,
        context_enc=0,
        # Normalization
        norm_starts=4,
        norm_groups=4,
        # DConv residual branch
        dconv_mode=1,
        dconv_depth=2,
        dconv_comp=8,
        dconv_init=1e-3,
        # Before the Transformer
        bottom_channels=0,
        # Transformer
        t_layers=5,
        t_emb="sin",
        t_hidden_scale=4.0,
        t_heads=8,
        t_dropout=0.0,
        t_max_positions=10000,
        t_norm_in=True,
        t_norm_in_group=False,
        t_group_norm=False,
        t_norm_first=True,
        t_norm_out=True,
        t_max_period=10000.0,
        t_weight_decay=0.0,
        t_lr=None,
        t_layer_scale=True,
        t_gelu=True,
        t_weight_pos_embed=1.0,
        t_sin_random_shift=0,
        t_cape_mean_normalize=True,
        t_cape_augment=True,
        t_cape_glob_loc_scale=[5000.0, 1.0, 1.4],
        t_sparse_self_attn=False,
        t_sparse_cross_attn=False,
        t_mask_type="diag",
        t_mask_random_seed=42,
        t_sparse_attn_window=500,
        t_global_window=100,
        t_sparsity=0.95,
        t_auto_sparsity=False,
        # ------ Particuliar parameters
        t_cross_first=False,
        # Weight init
        rescale=0.1,
        # Metadata
        samplerate=44100,
        segment=10,
        use_train_segment=False,
    ):
        """
        Args:
            sources (list[str]): list of source names.
            audio_channels (int): input/output audio channels.
            channels (int): initial number of hidden channels.
            channels_time: if not None, use a different `channels` value for the time branch.
            growth: increase the number of hidden channels by this factor at each layer.
            nfft: number of fft bins. Note that changing this require careful computation of
                various shape parameters and will not work out of the box for hybrid models.
            wiener_iters: when using Wiener filtering, number of iterations at test time.
            end_iters: same but at train time. For a hybrid model, must be equal to `wiener_iters`.
            wiener_residual: add residual source before wiener filtering.
            cac: uses complex as channels, i.e. complex numbers are 2 channels each
                in input and output. no further processing is done before ISTFT.
            depth (int): number of layers in the encoder and in the decoder.
            rewrite (bool): add 1x1 convolution to each layer.
            multi_freqs: list of frequency ratios for splitting frequency bands with `MultiWrap`.
            multi_freqs_depth: how many layers to wrap with `MultiWrap`. Only the outermost
                layers will be wrapped.
            freq_emb: add frequency embedding after the first frequency layer if > 0,
                the actual value controls the weight of the embedding.
            emb_scale: equivalent to scaling the embedding learning rate
            emb_smooth: initialize the embedding with a smooth one (with respect to frequencies).
            kernel_size: kernel_size for encoder and decoder layers.
            stride: stride for encoder and decoder layers.
            time_stride: stride for the final time layer, after the merge.
            context: context for 1x1 conv in the decoder.
            context_enc: context for 1x1 conv in the encoder.
            norm_starts: layer at which group norm starts being used.
                decoder layers are numbered in reverse order.
            norm_groups: number of groups for group norm.
            dconv_mode: if 1: dconv in encoder only, 2: decoder only, 3: both.
            dconv_depth: depth of residual DConv branch.
            dconv_comp: compression of DConv branch.
            dconv_attn: adds attention layers in DConv branch starting at this layer.
            dconv_lstm: adds a LSTM layer in DConv branch starting at this layer.
            dconv_init: initial scale for the DConv branch LayerScale.
            bottom_channels: if >0 it adds a linear layer (1x1 Conv) before and after the
                transformer in order to change the number of channels
            t_layers: number of layers in each branch (waveform and spec) of the transformer
            t_emb: "sin", "cape" or "scaled"
            t_hidden_scale: the hidden scale of the Feedforward parts of the transformer
                for instance if C = 384 (the number of channels in the transformer) and
                t_hidden_scale = 4.0 then the intermediate layer of the FFN has dimension
                384 * 4 = 1536
            t_heads: number of heads for the transformer
            t_dropout: dropout in the transformer
            t_max_positions: max_positions for the "scaled" positional embedding, only
                useful if t_emb="scaled"
            t_norm_in: (bool) norm before addinf positional embedding and getting into the
                transformer layers
            t_norm_in_group: (bool) if True while t_norm_in=True, the norm is on all the
                timesteps (GroupNorm with group=1)
            t_group_norm: (bool) if True, the norms of the Encoder Layers are on all the
                timesteps (GroupNorm with group=1)
            t_norm_first: (bool) if True the norm is before the attention and before the FFN
            t_norm_out: (bool) if True, there is a GroupNorm (group=1) at the end of each layer
            t_max_period: (float) denominator in the sinusoidal embedding expression
            t_weight_decay: (float) weight decay for the transformer
            t_lr: (float) specific learning rate for the transformer
            t_layer_scale: (bool) Layer Scale for the transformer
            t_gelu: (bool) activations of the transformer are GeLU if True, ReLU else
            t_weight_pos_embed: (float) weighting of the positional embedding
            t_cape_mean_normalize: (bool) if t_emb="cape", normalisation of positional embeddings
                see: https://arxiv.org/abs/2106.03143
            t_cape_augment: (bool) if t_emb="cape", must be True during training and False
                during the inference, see: https://arxiv.org/abs/2106.03143
            t_cape_glob_loc_scale: (list of 3 floats) if t_emb="cape", CAPE parameters
                see: https://arxiv.org/abs/2106.03143
            t_sparse_self_attn: (bool) if True, the self attentions are sparse
            t_sparse_cross_attn: (bool) if True, the cross-attentions are sparse (don't use it
                unless you designed really specific masks)
            t_mask_type: (str) can be "diag", "jmask", "random", "global" or any combination
                with '_' between: i.e. "diag_jmask_random" (note that this is permutation
                invariant i.e. "diag_jmask_random" is equivalent to "jmask_random_diag")
            t_mask_random_seed: (int) if "random" is in t_mask_type, controls the seed
                that generated the random part of the mask
            t_sparse_attn_window: (int) if "diag" is in t_mask_type, for a query (i), and
                a key (j), the mask is True id |i-j|<=t_sparse_attn_window
            t_global_window: (int) if "global" is in t_mask_type, mask[:t_global_window, :]
                and mask[:, :t_global_window] will be True
            t_sparsity: (float) if "random" is in t_mask_type, t_sparsity is the sparsity
                level of the random part of the mask.
            t_cross_first: (bool) if True cross attention is the first layer of the
                transformer (False seems to be better)
            rescale: weight rescaling trick
            use_train_segment: (bool) if True, the actual size that is used during the
                training is used during inference.
        """
        super().__init__()
        self.num_subbands = num_subbands
        self.cac = cac
        self.wiener_residual = wiener_residual
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.bottom_channels = bottom_channels
        self.channels = channels
        self.samplerate = samplerate
        self.segment = segment
        self.use_train_segment = use_train_segment
        self.nfft = nfft
        self.hop_length = nfft // 4
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.freq_emb = None
        assert wiener_iters == end_iters

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.tencoder = nn.ModuleList()
        self.tdecoder = nn.ModuleList()

        chin = audio_channels
        chin_z = chin  # number of channels for the freq branch
        if self.cac:
            chin_z *= 2
        if self.num_subbands > 1:
            chin_z *= self.num_subbands
        chout = channels_time or channels
        chout_z = channels
        freqs = nfft // 2

        for index in range(depth):
            norm = index >= norm_starts
            freq = freqs > 1
            stri = stride
            ker = kernel_size
            if not freq:
                assert freqs == 1
                ker = time_stride * 2
                stri = time_stride

            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            kw = {
                "kernel_size": ker,
                "stride": stri,
                "freq": freq,
                "pad": pad,
                "norm": norm,
                "rewrite": rewrite,
                "norm_groups": norm_groups,
                "dconv_kw": {
                    "depth": dconv_depth,
                    "compress": dconv_comp,
                    "init": dconv_init,
                    "gelu": True,
                },
            }
            kwt = dict(kw)
            kwt["freq"] = 0
            kwt["kernel_size"] = kernel_size
            kwt["stride"] = stride
            kwt["pad"] = True
            kw_dec = dict(kw)
            multi = False
            if multi_freqs and index < multi_freqs_depth:
                multi = True
                kw_dec["context_freq"] = False

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            enc = HEncLayer(
                chin_z, chout_z, dconv=dconv_mode & 1, context=context_enc, **kw
            )
            if freq:
                tenc = HEncLayer(
                    chin,
                    chout,
                    dconv=dconv_mode & 1,
                    context=context_enc,
                    empty=last_freq,
                    **kwt
                )
                self.tencoder.append(tenc)

            if multi:
                enc = MultiWrap(enc, multi_freqs)
            self.encoder.append(enc)
            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin
                if self.cac:
                    chin_z *= 2
                if self.num_subbands > 1:
                    chin_z *= self.num_subbands
            dec = HDecLayer(
                chout_z,
                chin_z,
                dconv=dconv_mode & 2,
                last=index == 0,
                context=context,
                **kw_dec
            )
            if multi:
                dec = MultiWrap(dec, multi_freqs)
            if freq:
                tdec = HDecLayer(
                    chout,
                    chin,
                    dconv=dconv_mode & 2,
                    empty=last_freq,
                    last=index == 0,
                    context=context,
                    **kwt
                )
                self.tdecoder.insert(0, tdec)
            self.decoder.insert(0, dec)

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            if freq:
                if freqs <= kernel_size:
                    freqs = 1
                else:
                    freqs //= stride
            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(
                    freqs, chin_z, smooth=emb_smooth, scale=emb_scale
                )
                self.freq_emb_scale = freq_emb

        if rescale:
            rescale_module(self, reference=rescale)

        transformer_channels = channels * growth ** (depth - 1)
        if bottom_channels:
            self.channel_upsampler = nn.Conv1d(transformer_channels, bottom_channels, 1)
            self.channel_downsampler = nn.Conv1d(
                bottom_channels, transformer_channels, 1
            )
            self.channel_upsampler_t = nn.Conv1d(
                transformer_channels, bottom_channels, 1
            )
            self.channel_downsampler_t = nn.Conv1d(
                bottom_channels, transformer_channels, 1
            )

            transformer_channels = bottom_channels

        if t_layers > 0:
            self.crosstransformer = CrossTransformerEncoder(
                dim=transformer_channels,
                emb=t_emb,
                hidden_scale=t_hidden_scale,
                num_heads=t_heads,
                num_layers=t_layers,
                cross_first=t_cross_first,
                dropout=t_dropout,
                max_positions=t_max_positions,
                norm_in=t_norm_in,
                norm_in_group=t_norm_in_group,
                group_norm=t_group_norm,
                norm_first=t_norm_first,
                norm_out=t_norm_out,
                max_period=t_max_period,
                weight_decay=t_weight_decay,
                lr=t_lr,
                layer_scale=t_layer_scale,
                gelu=t_gelu,
                sin_random_shift=t_sin_random_shift,
                weight_pos_embed=t_weight_pos_embed,
                cape_mean_normalize=t_cape_mean_normalize,
                cape_augment=t_cape_augment,
                cape_glob_loc_scale=t_cape_glob_loc_scale,
                sparse_self_attn=t_sparse_self_attn,
                sparse_cross_attn=t_sparse_cross_attn,
                mask_type=t_mask_type,
                mask_random_seed=t_mask_random_seed,
                sparse_attn_window=t_sparse_attn_window,
                global_window=t_global_window,
                sparsity=t_sparsity,
                auto_sparsity=t_auto_sparsity,
            )
        else:
            self.crosstransformer = None

        # Calculate win_length to match _ispec calculations
        # In _ispec: win_length = n_fft // (1 + pad) where pad = 0
        win_length = self.nfft  # This matches n_fft // (1 + 0)
        
        # Calculate max_frames based on actual segment length
        # segment is in seconds, samplerate gives samples per second
        # hop_length is the stride between frames
        audio_length_samples = int(self.segment * self.samplerate) // 2
        max_frames = (audio_length_samples // self.hop_length) + 1
        
        # Add some buffer for safety (e.g., padding, different scales, variable length inputs)
        max_frames = int(max_frames * 1.2)  # 20% buffer
        
        # Ensure minimum reasonable size
        max_frames = max(max_frames, 512)  # At least 512 frames

        print("STFT_Process initialization arguments:")
        print(f"model_type: istft_C")
        print(f"n_fft: {self.nfft}")
        print(f"win_length: {win_length}")
        print(f"hop_len: {self.hop_length}")
        print(f"max_frames: {max_frames}")
        print(f"window_type: hann")
        
        self.STFT_Process = STFT_Process(model_type='istft_C',
                                         n_fft=self.nfft, 
                                         win_length=win_length, 
                                         hop_len=self.hop_length, 
                                         max_frames=max_frames,
                                         window_type='hann'
                                         ).eval()

    def _precompute_istft_basis(self):
        """Precompute ISTFT basis matrices for CoreML compatibility"""
        n_fft = self.nfft
        hop_length = self.hop_length
        win_length = n_fft // (1 + 3)  # This matches the calculation in _ispec
        win_length = 2
        
        # Get device - use CPU if parameters not initialized yet
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        
        # Create window
        window = torch.hann_window(win_length, device=device).float()
        if win_length < n_fft:
            pad_left = (n_fft - win_length) // 2
            pad_right = n_fft - win_length - pad_left
            window = F.pad(window, (pad_left, pad_right), mode='constant', value=0)
        
        # Pre-compute fourier basis
        fourier_basis = torch.fft.fft(torch.eye(n_fft, dtype=torch.float32, device=device))
        fourier_basis = torch.vstack([
            torch.real(fourier_basis[:n_fft//2 + 1, :]),
            torch.imag(fourier_basis[:n_fft//2 + 1, :])
        ]).float()
        
        # Create inverse basis
        inverse_basis = window * torch.linalg.pinv((fourier_basis * n_fft) / hop_length).T.unsqueeze(1)
        
        # Calculate window sum for overlap-add
        # Use the same max_frames calculation as in __init__
        audio_length_samples = int(self.segment * self.samplerate) // 2
        max_frames = (audio_length_samples // self.hop_length) + 1
        max_frames = max(int(max_frames * 1.2), 512)  # Same buffer and minimum as __init__
        n = n_fft + hop_length * (max_frames - 1)
        window_sum = torch.zeros(n, dtype=torch.float32, device=device)
        
        # Get original window and normalize it
        original_window = torch.hann_window(win_length, device=device).float()
        window_normalized = original_window / original_window.abs().max()
        
        # Pad the window to n_fft for overlap-add calculation
        if win_length < n_fft:
            pad_left = (n_fft - win_length) // 2
            pad_right = n_fft - win_length - pad_left
            win_sq = F.pad(window_normalized ** 2, (pad_left, pad_right), mode='constant', value=0)
        else:
            win_sq = window_normalized ** 2
        
        # Calculate overlap-add weights
        for i in range(max_frames):
            sample = i * hop_length
            window_sum[sample: min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
        
        # Calculate window sum inverse
        window_sum_inv = n_fft / (window_sum * hop_length + 1e-7)
        
        # Register as buffers (these will be saved with the model)
        self.register_buffer("istft_inverse_basis", inverse_basis)
        self.register_buffer("istft_window_sum_inv", window_sum_inv)

    def _spec(self, x):
        hl = self.hop_length
        nfft = self.nfft
        x0 = x  # noqa

        # We re-pad the signal in order to keep the property
        # that the size of the output is exactly the size of the input
        # divided by the stride (here hop_length), when divisible.
        # This is achieved by padding by 1/4th of the kernel size (here nfft).
        # which is not supported by torch.stft.
        # Having all convolution operations follow this convention allow to easily
        # align the time and frequency branches later on.
        assert hl == nfft // 4
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        
        # Store original dtype and convert to float32 for padding
        original_dtype = x.dtype
        x = x.float()
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")
        
        # Convert to float32 for STFT operations
        z = spectro(x, nfft, hl)
        # Convert to real/imaginary parts before slicing
        z = torch.view_as_real(z)
        z = z[..., :-1, :, :]
        # Convert back to complex using our own implementation
        z_real = z[..., 0]
        z_imag = z[..., 1]
        z = torch.complex(z_real, z_imag)
        # Convert back to original dtype
        #z = z.to(torch.cfloat)
        
        assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
        # Split into real and imaginary parts before slicing
        z_real = z.real[..., 2: 2 + le]
        z_imag = z.imag[..., 2: 2 + le]
        # Recombine into complex tensor
        z = torch.complex(z_real, z_imag)
        return z

    def _ispec(self, z, length=None, scale=0):
        # print(scale, "SCALE")
        hl = self.hop_length // (4**scale)
        z = F.pad(z, (0, 0, 0, 1))
        z = F.pad(z, (2, 2))
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad

        ### BRING IN ISPECTRO IMPLEMENTATION
        hop_length = hl
        length_stft = le

        *other, freqs, frames = z.shape
        n_fft = 2 * freqs - 2
        z = z.view(-1, freqs, frames)
        
        # Convert complex tensor to real tensor with real/imag parts concatenated
        z_real = z.real
        z_imag = z.imag
        z_complex_as_real = torch.cat([z_real, z_imag], dim=1)

        # # Run both methods and compare
        x_stft_process = self.STFT_Process(z_complex_as_real, 
                                         length=length_stft, 
                                         hop_length=hop_length,
                                         n_fft=n_fft)
        x_stft_process = x_stft_process.squeeze(1)
        
        # Apply scaling correction to STFT_Process output
        x = x_stft_process * 64.0
        
        assert length_stft == x.shape[-1]
        x = x.view(*other, length_stft)

        #x = ispectro(z, hl, length=le)
        x = x[..., pad: pad + length]
        return x

    def _magnitude(self, z):
        # return the magnitude of the spectrogram, except when cac is True,
        # in which case we just move the complex dimension to the channel one.
        if self.cac:
            B, C, Fr, T = z.shape
            m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
            m = m.reshape(B, C * 2, Fr, T)
        else:
            m = z.abs()
        return m

    def _mask(self, z, m):
        # Apply masking given the mixture spectrogram `z` and the estimated mask `m`.
        # If `cac` is True, `m` is actually a full spectrogram and `z` is ignored.
        niters = self.wiener_iters
        if self.cac:
            B, S, C, Fr, T = m.shape # B, S, C, Fr, T
            #  print(m.shape, "M")
            out = m.view(B, S, -1, 2, Fr*T) ## becomes B, S, C/2, 2, Fr, T
            # print(out.shape, "OUT")

            out = out.permute(0, 1, 2, 4, 3) ## becomes B, S, C/2, Fr*T, 2
            # print(out.shape, "OUT PERMUTED")
            out_real = out[..., 0] ## B, S, C/2, Fr*T
            out_imag = out[..., 1] ## B, S, C/2, Fr*T

            out_real = out_real.view(B, S, -1, Fr, T) ## B, S, C/2, Fr, T
            out_imag = out_imag.view(B, S, -1, Fr, T) ## B, S, C/2, Fr, T

            out = torch.complex(out_real, out_imag).contiguous() ## B, S, C/2, Fr, T
            return out
        if self.training:
            niters = self.end_iters
        if niters < 0:
            z = z[:, None]
            return z / (1e-8 + z.abs()) * m
        else:
            return self._wiener(m, z, niters)

    def _wiener(self, mag_out, mix_stft, niters):
        # apply wiener filtering from OpenUnmix.
        init = mix_stft.dtype
        wiener_win_len = 300
        residual = self.wiener_residual

        B, S, C, Fq, T = mag_out.shape
        mag_out = mag_out.permute(0, 4, 3, 2, 1)

        print(mix_stft.shape)
        
        # Convert complex to real/imaginary parts
        mix_stft_real = mix_stft.real.permute(0, 3, 2, 1)
        mix_stft_imag = mix_stft.imag.permute(0, 3, 2, 1)
        mix_stft = torch.stack([mix_stft_real, mix_stft_imag], dim=-1)

        outs = []
        for sample in range(B):
            pos = 0
            out = []
            for pos in range(0, T, wiener_win_len):
                frame = slice(pos, pos + wiener_win_len)
                # Slice real and imaginary parts separately
                mag_out_frame = mag_out[sample, frame]
                mix_stft_frame = mix_stft[sample, frame]
                
                z_out = wiener(
                    mag_out_frame, mix_stft_frame, niters,
                    residual=residual)
                out.append(z_out.transpose(-1, -2))
            outs.append(torch.cat(out, dim=0))
        
        # Convert back to complex
        out = torch.stack(outs, 0)
        out = out.permute(0, 4, 3, 2, 1).contiguous()
        if residual:
            out = out[:, :-1]
        assert list(out.shape) == [B, S, C, Fq, T]
        return out.to(init)

    def valid_length(self, length: int):
        """
        Return a length that is appropriate for evaluation.
        In our case, always return the training length, unless
        it is smaller than the given length, in which case this
        raises an error.
        """
        if not self.use_train_segment:
            return length
        training_length = int(self.segment * self.samplerate)
        if training_length < length:
            raise ValueError(
                    f"Given length {length} is longer than "
                    f"training length {training_length}")
        return training_length

    def cac2cws(self, x):
        k = self.num_subbands
        b, c, f, t = x.shape
        x = x.reshape(b, c, k, f // k, t)
        x = x.reshape(b, c * k, f // k, t)
        return x

    def cws2cac(self, x):
        k = self.num_subbands
        b, c, f, t = x.shape
        x = x.reshape(b, c // k, k, f, t)
        x = x.reshape(b, c // k, f * k, t)
        return x

    def forward(self, mix):
        length = mix.shape[-1]
        # Detect the model's parameter precision instead of input precision
        model_dtype = next(self.parameters()).dtype
        #model_dtype = torch.float16
        length_pre_pad = None
        if self.use_train_segment:
            if self.training:
                self.segment = Fraction(mix.shape[-1], self.samplerate)
            else:
                training_length = int(self.segment * self.samplerate)
                if mix.shape[-1] < training_length:
                    length_pre_pad = mix.shape[-1]
                    mix = F.pad(mix, (0, training_length - length_pre_pad))
        z = self._spec(mix)
        mag = self._magnitude(z)
        x = mag
        # Ensure tensor maintains the model's precision after STFT
        x = x.to(model_dtype)

        if self.num_subbands > 1:
            x = self.cac2cws(x)

        B, C, Fq, T = x.shape

        # unlike previous Demucs, we always normalize because it is easier.
        # Convert to fp16 before computing statistics
        x = x.to(torch.float32)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        # std operation produces fp32, so explicitly convert to fp16
        std = x.std(dim=(1, 2, 3), keepdim=True).to(torch.float16)
        # Ensure epsilon is in fp16
        epsilon = torch.tensor(1e-5, dtype=torch.float16, device=x.device)
        x = (x - mean) / (epsilon + std)
        # Convert back to model dtype
        x = x.to(model_dtype)

        # Prepare the time branch input.
        xt = mix
        xt = xt.to(torch.float32)
        meant = xt.mean(dim=(1, 2), keepdim=True)
        # std operation produces fp32, so explicitly convert to fp16
        stdt = xt.std(dim=(1, 2), keepdim=True).to(torch.float16)
        # Ensure epsilon is in fp16
        epsilon = torch.tensor(1e-5, dtype=torch.float16, device=xt.device)
        xt = (xt - meant) / (epsilon + stdt)
        # Convert back to model dtype
        xt = xt.to(model_dtype)

        # okay, this is a giant mess I know...
        saved = []  # skip connections, freq.
        saved_t = []  # skip connections, time.
        lengths = []  # saved lengths to properly remove padding, freq branch.
        lengths_t = []  # saved lengths for time branch.
        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.tencoder):
                # we have not yet merged branches.
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt).to(model_dtype)
                if not tenc.empty:
                    # save for skip connection
                    saved_t.append(xt)
                else:
                    # tenc contains just the first conv., so that now time and freq.
                    # branches have the same shape and can be merged.
                    inject = xt
            x = encode(x, inject).to(model_dtype)
            if idx == 0 and self.freq_emb is not None:
                # add frequency embedding to allow for non equivariant convolutions
                # over the frequency axis.
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                emb = emb.to(model_dtype)
                emb_scaled = torch.tensor(self.freq_emb_scale, dtype=model_dtype, device=x.device) * emb
                x = x + emb_scaled

            saved.append(x)
        if self.crosstransformer:
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_upsampler(x).to(model_dtype)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.channel_upsampler_t(xt).to(model_dtype)

            # Ensure inputs to transformer are in correct precision
            x = x.to(model_dtype)
            xt = xt.to(model_dtype)
            
            x, xt = self.crosstransformer(x, xt)
            
            # Ensure transformer outputs are in correct precision
            x = x.to(model_dtype)
            xt = xt.to(model_dtype)

            if self.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_downsampler(x).to(model_dtype)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.channel_downsampler_t(xt).to(model_dtype)

        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            if torch.is_complex(skip):
                print(f"skip is complex in decoder {idx}")
            if torch.is_complex(x):
                print(f"x is complex before decoder {idx}")
            x, pre = decode(x, skip, lengths.pop(-1))
            if torch.is_complex(x):
                print(f"x is complex after decoder {idx}")
            if pre is not None:
                if torch.is_complex(pre):
                    print(f"pre is complex in decoder {idx}")
                pre = pre.to(model_dtype)
            x = x.to(model_dtype)
            # `pre` contains the output just before final transposed convolution,
            # which is used when the freq. and time branch separate.

            offset = self.depth - len(self.tdecoder)
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    assert pre.shape[2] == 1, pre.shape
                    pre = pre[:, :, 0]
                    if torch.is_complex(xt):
                        print(f"xt is complex before tdecoder {idx}")
                    xt, _ = tdec(pre, None, length_t)
                    if torch.is_complex(xt):
                        print(f"xt is complex after tdecoder {idx}")
                else:
                    skip = saved_t.pop(-1)
                    if torch.is_complex(xt):
                        print(f"xt is complex before tdecoder {idx}")
                    xt, _ = tdec(xt, skip, length_t)
                    if torch.is_complex(xt):
                        print(f"xt is complex after tdecoder {idx}")

        # Let's make sure we used all stored skip connections.
        assert len(saved) == 0
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        S = len(self.sources)

        if self.num_subbands > 1:
            x = x.view(B, -1, Fq, T)
            if torch.is_complex(x):
                print("x is complex after view 1")
            x = self.cws2cac(x)
            if torch.is_complex(x):
                print("x is complex after cws2cac")

        x = x.view(B, S, -1, Fq * self.num_subbands, T)
        x = x * std[:, None] + mean[:, None]
        if torch.is_complex(x):
            print("x is complex before final model_dtype conversion")
        x = x.to(model_dtype)
        if torch.is_complex(x):
            print("x is complex after final model_dtype conversion")

        zout = self._mask(z, x)
        if self.use_train_segment:
            if self.training:
                x = self._ispec(zout, length)
            else:
                x = self._ispec(zout, training_length)
        else:
            x = self._ispec(zout, length)

        if self.use_train_segment:
            if self.training:
                xt = xt.view(B, S, -1, length)
            else:
                xt = xt.view(B, S, -1, training_length)
        else:
            xt = xt.view(B, S, -1, length)
        xt = xt * stdt[:, None] + meant[:, None]

        print("num nonzeros in zout", torch.count_nonzero(zout))
        print("num nonzeros in x", torch.count_nonzero(x))
        print("num nonzeros in xt", torch.count_nonzero(xt))

        x = xt + x.to(xt.device)
        if length_pre_pad:
            x = x[..., :length_pre_pad]
        return x
        
        # x = xt + x
        # Ensure final output maintains the model's precision
        #x = x.to(model_dtype)
        print(length_pre_pad)
        if length_pre_pad:
            x = x[..., :length_pre_pad]
            xt = xt[..., :length_pre_pad]
            
        # Separate complex tensor into real and imaginary parts
        if torch.is_complex(x):
            print(x.shape, x.dtype, "X")
            x_real = x.real
            x_imag = x.imag
            # Convert both parts to model_dtype
            x_real = x_real.to(model_dtype)
            x_imag = x_imag.to(model_dtype)
            return x_real, x_imag, xt
        else:
            return x, xt  # Return both spectrogram and time branch


def get_model(args):
    extra = {
        'sources': list(args.training.instruments),
        'audio_channels': args.training.channels,
        'samplerate': args.training.samplerate,
        # 'segment': args.model_segment or 4 * args.dset.segment,
        'segment': args.training.segment,
    }
    klass = {
        'demucs': Demucs,
        'hdemucs': HDemucs,
        'htdemucs': HTDemucs,
    }[args.model]
    kw = OmegaConf.to_container(getattr(args, args.model), resolve=True)
    model = klass(**extra, **kw)
    return model


