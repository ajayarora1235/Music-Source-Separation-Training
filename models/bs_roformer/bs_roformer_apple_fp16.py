from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import Callable
from einops import pack, rearrange, unpack
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import Module, ModuleList
from torch.utils.checkpoint import checkpoint

from models.bs_roformer.attend import Attend
try:
    from models.bs_roformer.attend_sage import Attend as AttendSage
except:
    pass


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


class CustomNorm(Module):
    def __init__(self, dim, eps: float = 5.960464477539063e-08):  # 0x1p-24
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        l2_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        denom = torch.maximum(l2_norm, torch.full_like(l2_norm, self.eps))
        normalized_x = x / denom
        return normalized_x * self.scale * self.gamma


# attention


class RotaryEmbedding(nn.Module):
    def __init__(self, cos_emb, sin_emb):
        super().__init__()
        # both (seq_len_for_rotation, dim_head)
        self.cos_emb = cos_emb
        self.sin_emb = sin_emb

    def rotate_half(self, x):
        x = rearrange(x, "... (d r) -> ... d r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... d r -> ... (d r)")

    def forward(self, x):
        # x is (batch_eff, heads, seq_len_for_rotation, dim_head)
        cos_b = self.cos_emb.unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
        sin_b = self.sin_emb.unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)

        term1 = x * cos_b
        term2 = self.rotate_half(x) * sin_b

        sum = term1.to(torch.float32) + term2.to(torch.float32)
        return sum.to(x.dtype)


class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            CustomNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        shared_qkv_bias=None,
        shared_out_bias=None,
        rotary_embed: RotaryEmbedding | None = None,
        flash=True,
        sage_attention=False,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        if sage_attention:
            self.attend = AttendSage(flash=flash, dropout=dropout)  # type: ignore
        else:
            self.attend = Attend(flash=flash, dropout=dropout)  # type: ignore

        self.norm = CustomNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=(shared_qkv_bias is not None))
        if shared_qkv_bias is not None:
            self.to_qkv.bias = shared_qkv_bias

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=(shared_out_bias is not None)),
            nn.Dropout(dropout),
        )
        if shared_out_bias is not None:
            self.to_out[0].bias = shared_out_bias

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)

        if self.rotary_embed is not None:
            q = self.rotary_embed(q)
            k = self.rotary_embed(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        gate_act = gates.sigmoid()

        out = out * rearrange(gate_act, "b n h -> b h n 1")

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class LinearAttention(Module):
    """
    this flavor of linear attention proposed in https://arxiv.org/abs/2106.09681 by El-Nouby et al.
    """

    @beartype
    def __init__(
        self,
        *,
        dim,
        dim_head=32,
        heads=8,
        scale=8,
        flash=False,
        dropout=0.0,
        sage_attention=False,
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.norm = CustomNorm(dim)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange("b n (qkv h d) -> qkv b h d n", qkv=3, h=heads),
        )

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        if sage_attention:
            self.attend = AttendSage(scale=scale, dropout=dropout, flash=flash)  # type: ignore
        else:
            self.attend = Attend(scale=scale, dropout=dropout, flash=flash)

        self.to_out = nn.Sequential(
            Rearrange("b h d n -> b n (h d)"), nn.Linear(dim_inner, dim, bias=False)
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = self.to_qkv(x)

        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()

        out = self.attend(q, k, v)

        return self.to_out(out)


class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        norm_output=True,
        rotary_embed: RotaryEmbedding | None = None,
        flash_attn=True,
        linear_attn=False,
        sage_attention=False,
        shared_qkv_bias=None,
        shared_out_bias=None,
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            attn: LinearAttention | Attention
            if linear_attn:
                attn = LinearAttention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    flash=flash_attn,
                    sage_attention=sage_attention,
                )
            else:
                attn = Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    shared_qkv_bias=shared_qkv_bias,
                    shared_out_bias=shared_out_bias,
                    rotary_embed=rotary_embed,
                    flash=flash_attn,
                    sage_attention=sage_attention,
                )

            self.layers.append(
                ModuleList([attn, FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)])
            )

        self.norm = CustomNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:  # type: ignore
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# bandsplit module


class BandSplit(Module):
    @beartype
    def __init__(self, dim, dim_inputs: tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(CustomNorm(dim_in), nn.Linear(dim_in, dim))

            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)


def MLP(
    dim_in: int,
    dim_out: int,
    dim_hidden: int | None = None,
    depth: int = 1,
    activation=nn.Tanh,
):
    dim_hidden = dim_hidden or dim_in

    net = []
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    @beartype
    def __init__(self, dim, dim_inputs: tuple[int, ...], depth, mlp_expansion_factor=4):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth), nn.GLU(dim=-1)
            )

            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


# fmt: off
DEFAULT_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129
)
# fmt: on


class BSRoformer(Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=1,
        time_transformer_depth=2,
        freq_transformer_depth=2,
        linear_transformer_depth=0,
        freqs_per_bands: tuple[
            int, ...
        ] = DEFAULT_FREQS_PER_BANDS,  # in the paper, they divide into ~60 bands, test with 1 for starters
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        flash_attn=True,
        stft_n_fft=2048,
        stft_hop_length=512,  # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
        stft_win_length=2048,
        stft_normalized=False,
        stft_window_fn: Callable | None = None,
        mask_estimator_depth=2,
        multi_stft_resolution_loss_weight=1.0,
        multi_stft_resolutions_window_sizes: tuple[int, ...] = (
            4096,
            2048,
            1024,
            512,
            256,
        ),
        multi_stft_hop_size=147,
        multi_stft_normalized=False,
        multi_stft_window_fn: Callable = torch.hann_window,
        mlp_expansion_factor=4,
        use_torch_checkpoint=False,
        skip_connection=False,
        sage_attention=False,
        use_shared_bias=False,
        chunk_size: int = 588800,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection

        self.layers = ModuleList([])

        if sage_attention:
            print("Use Sage Attention")

        if use_shared_bias:
            dim_inner = heads * dim_head
            self.shared_qkv_bias = nn.Parameter(torch.ones(dim_inner * 3))
            self.shared_out_bias = nn.Parameter(torch.ones(dim))

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            norm_output=False,
            sage_attention=sage_attention,
            shared_qkv_bias=self.shared_qkv_bias,
            shared_out_bias=self.shared_out_bias,
        )

        t_frames = chunk_size // stft_hop_length + 1  # e.g. 588800 // 512 + 1 = 1151
        self.cos_emb_time = nn.Parameter(torch.zeros(t_frames, dim_head))
        self.sin_emb_time = nn.Parameter(torch.zeros(t_frames, dim_head))
        time_rotary_embed = RotaryEmbedding(cos_emb=self.cos_emb_time, sin_emb=self.sin_emb_time)

        num_bands = len(freqs_per_bands)  # e.g. 62
        self.cos_emb_freq = nn.Parameter(torch.zeros(num_bands, dim_head))
        self.sin_emb_freq = nn.Parameter(torch.zeros(num_bands, dim_head))
        freq_rotary_embed = RotaryEmbedding(cos_emb=self.cos_emb_freq, sin_emb=self.sin_emb_freq)

        for _ in range(depth):
            tran_modules = []
            if linear_transformer_depth > 0:
                tran_modules.append(
                    Transformer(
                        depth=linear_transformer_depth,
                        linear_attn=True,
                        **transformer_kwargs,
                    )
                )
            tran_modules.append(
                Transformer(
                    depth=time_transformer_depth,
                    rotary_embed=time_rotary_embed,
                    **transformer_kwargs,
                )
            )
            tran_modules.append(
                Transformer(
                    depth=freq_transformer_depth,
                    rotary_embed=freq_rotary_embed,
                    **transformer_kwargs,
                )
            )
            self.layers.append(nn.ModuleList(tran_modules))

        self.final_norm = CustomNorm(dim)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )

        self.stft_window_fn = partial(stft_window_fn or torch.hann_window, stft_win_length)

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )

            self.mask_estimators.append(mask_estimator)

        # for the multi-resolution stft loss

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size, normalized=multi_stft_normalized
        )

    def forward(self, raw_audio, target=None, return_loss_breakdown=False):
        """
        einops

        b - batch
        f - freq
        t - time
        s - audio channel (1 for mono, 2 for stereo)
        n - number of 'stems'
        c - complex (2)
        d - feature dimension
        """

        device = raw_audio.device

        # defining whether model is loaded on MPS (MacOS GPU accelerator)
        x_is_mps = True if device.type == "mps" else False

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, "b t -> b 1 t")

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), (
            "stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)"
        )

        # to stft

        raw_audio, batch_audio_channel_packed_shape = pack([raw_audio], "* t")

        stft_window = self.stft_window_fn(device=device)
        stft_window = stft_window.half()  # Ensure window is in half precision

        # RuntimeError: FFT operations are only supported on MacOS 14+
        # Since it's tedious to define whether we're on correct MacOS version - simple try-catch is used
        try:
            stft_repr = torch.stft(
                raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True
            )
        except Exception:
            stft_repr = torch.stft(
                raw_audio.cpu() if x_is_mps else raw_audio,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=True,
            ).to(device)
        stft_repr = torch.view_as_real(stft_repr)
        stft_repr = stft_repr.half()  # Ensure STFT output is in half precision

        stft_repr = unpack(stft_repr, batch_audio_channel_packed_shape, "* f t c")[0]

        # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting
        stft_repr = rearrange(stft_repr, "b s f t c -> b (f s) t c")

        x = rearrange(stft_repr, "b f t c -> b t (f c)")

        if torch.isnan(x).any() or torch.isinf(x).any():
            raise RuntimeError(
                f"NaN/Inf in x after stft: {x.isnan().sum()} NaNs, {x.isinf().sum()} Infs"
            )

        if self.use_torch_checkpoint:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        if torch.isnan(x).any() or torch.isinf(x).any():
            raise RuntimeError(
                f"NaN/Inf in x after band_split: {x.isnan().sum()} NaNs, {x.isinf().sum()} Infs"
            )

        # axial / hierarchical attention

        store = [None] * len(self.layers)
        for i, transformer_block in enumerate(self.layers):
            if len(transformer_block) == 3:
                linear_transformer, time_transformer, freq_transformer = transformer_block

                x, ft_ps = pack([x], "b * d")
                if self.use_torch_checkpoint:
                    x = checkpoint(linear_transformer, x, use_reentrant=False)
                else:
                    x = linear_transformer(x)
                (x,) = unpack(x, ft_ps, "b * d")
            else:
                time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                # Sum all previous
                for j in range(i):
                    x = x + store[j]

            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")

            if self.use_torch_checkpoint:
                x = checkpoint(time_transformer, x, use_reentrant=False)
            else:
                x = time_transformer(x)

            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")

            if self.use_torch_checkpoint:
                x = checkpoint(freq_transformer, x, use_reentrant=False)
            else:
                x = freq_transformer(x)

            (x,) = unpack(x, ps, "* f d")

            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)

        num_stems = len(self.mask_estimators)

        if self.use_torch_checkpoint:
            mask = torch.stack(
                [checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators],
                dim=1,
            )
        else:
            mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        mask = rearrange(mask, "b n t (f c) -> b n f t c", c=2)

        # modulate frequency representation

        stft_repr = stft_repr.squeeze(1)  # Remove the second dimension
        stft_repr = rearrange(stft_repr, "b f t c -> b 1 f t c")

        if self.use_torch_checkpoint:
            mask = torch.stack(
                [checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators],
                dim=1,
            )
        else:
            mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        mask = rearrange(mask, "b n t (f c) -> b n f t c", c=2)

        # modulate frequency representation

        stft_repr = stft_repr.squeeze(1)  # Remove the second dimension
        stft_repr = rearrange(stft_repr, "b f t c -> b 1 f t c")

        # complex number multiplication

        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask)

        stft_repr = stft_repr * mask

        # istft

        stft_repr = rearrange(stft_repr, "b n (f s) t -> (b n s) f t", s=self.audio_channels)

        # same as torch.stft() fix for MacOS MPS above
        try:
            recon_audio = torch.istft(
                stft_repr,
                **self.stft_kwargs,
                window=stft_window,
                return_complex=False,
                length=raw_audio.shape[-1],
            )
        except Exception:
            # Ensure all tensors are in the same precision before ISTFT     
            stft_repr = stft_repr.half() if stft_repr.dtype == torch.float32 else stft_repr
            stft_repr = stft_repr.to(torch.cfloat)

            stft_window = stft_window.half() if stft_window.dtype == torch.float32 else stft_window

            recon_audio = torch.istft(stft_repr.cpu() if x_is_mps else stft_repr, 
                                    **self.stft_kwargs, 
                                    window=stft_window.cpu() if x_is_mps else stft_window, 
                                    return_complex=False, 
                                    length=raw_audio.shape[-1]).to(device)

        recon_audio = rearrange(
            recon_audio, "(b n s) t -> b n s t", s=self.audio_channels, n=num_stems
        )

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, "b 1 s t -> b s t")

        # if a target is passed in, calculate loss for learning

        if target is None:
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 2:
            target = rearrange(target, "... t -> ... 1 t")

        target = target[..., : recon_audio.shape[-1]]  # protect against lost length on istft

        loss = F.l1_loss(recon_audio, target)

        multi_stft_resolution_loss = 0.0

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(
                    window_size, self.multi_stft_n_fft
                ),  # not sure what n_fft is across multi resolution stft
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(rearrange(recon_audio, "... s t -> (... s) t"), **res_stft_kwargs)
            target_Y = torch.stft(rearrange(target, "... s t -> (... s) t"), **res_stft_kwargs)

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        weighted_multi_resolution_loss = (
            multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight
        )

        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)


class BSRoformerCoreML(Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=1,
        time_transformer_depth=2,
        freq_transformer_depth=2,
        linear_transformer_depth=0,
        freqs_per_bands: tuple[
            int, ...
        ] = DEFAULT_FREQS_PER_BANDS,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        flash_attn=True,
        stft_n_fft=2048,
        stft_hop_length=512,  # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
        stft_win_length=2048,
        stft_normalized=False,
        stft_window_fn: Callable | None = None,
        mask_estimator_depth=2,
        multi_stft_resolution_loss_weight=1.0,
        multi_stft_resolutions_window_sizes: tuple[int, ...] = (
            4096,
            2048,
            1024,
            512,
            256,
        ),
        multi_stft_hop_size=147,
        multi_stft_normalized=False,
        multi_stft_window_fn: Callable = torch.hann_window,
        mlp_expansion_factor=4,
        use_torch_checkpoint=False,
        skip_connection=False,
        sage_attention=False,
        use_shared_bias=False,
        chunk_size: int = 588800,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection

        self.layers = ModuleList([])

        if sage_attention:
            print("Use Sage Attention")

        if use_shared_bias:
            dim_inner = heads * dim_head
            self.shared_qkv_bias = nn.Parameter(torch.ones(dim_inner * 3))
            self.shared_out_bias = nn.Parameter(torch.ones(dim))

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            norm_output=False,
            sage_attention=sage_attention,
            shared_qkv_bias=self.shared_qkv_bias if use_shared_bias else None,
            shared_out_bias=self.shared_out_bias if use_shared_bias else None,
        )

        # Calculate time dimension size for rotary embeddings
        t_frames = chunk_size // stft_hop_length + 1  # e.g. 588800 // 512 + 1 = 1151
        
        # Create rotary embeddings for time and frequency
        self.cos_emb_time = nn.Parameter(torch.zeros(t_frames, dim_head))
        self.sin_emb_time = nn.Parameter(torch.zeros(t_frames, dim_head))
        time_rotary_embed = RotaryEmbedding(cos_emb=self.cos_emb_time, sin_emb=self.sin_emb_time)

        num_bands = len(freqs_per_bands)
        self.cos_emb_freq = nn.Parameter(torch.zeros(num_bands, dim_head))
        self.sin_emb_freq = nn.Parameter(torch.zeros(num_bands, dim_head))
        freq_rotary_embed = RotaryEmbedding(cos_emb=self.cos_emb_freq, sin_emb=self.sin_emb_freq)

        for _ in range(depth):
            tran_modules = []
            if linear_transformer_depth > 0:
                tran_modules.append(
                    Transformer(
                        depth=linear_transformer_depth,
                        linear_attn=True,
                        **transformer_kwargs,
                    )
                )
            tran_modules.append(
                Transformer(
                    depth=time_transformer_depth,
                    rotary_embed=time_rotary_embed,
                    **transformer_kwargs,
                )
            )
            tran_modules.append(
                Transformer(
                    depth=freq_transformer_depth,
                    rotary_embed=freq_rotary_embed,
                    **transformer_kwargs,
                )
            )
            self.layers.append(nn.ModuleList(tran_modules))

        self.final_norm = CustomNorm(dim)

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=2,  # Fixed depth for mask estimator
                mlp_expansion_factor=mlp_expansion_factor,
            )
            self.mask_estimators.append(mask_estimator)

    def forward(self, stft_repr):
        """
        Forward pass that expects frequency domain input (STFT representation)
        
        Args:
            stft_repr: Complex STFT representation of shape (batch, freq, time) or (batch, channels, freq, time)
        
        Returns:
            Complex STFT representation of separated sources
        """
        device = stft_repr.device

        # Handle input shape
        if stft_repr.ndim == 3:
            stft_repr = stft_repr.unsqueeze(1)  # Add channel dimension
        
        channels = stft_repr.shape[1]
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), (
            "stereo needs to be set to True if passing in stereo STFT (channel dimension of 2). also need to be False if mono (channel dimension of 1)"
        )

        # Convert to real representation if complex
        if torch.is_complex(stft_repr):
            stft_repr = torch.view_as_real(stft_repr)
        
        # Ensure half precision
        stft_repr = stft_repr.half()

        # Merge stereo/mono into frequency dimension
        b, s, f, t, c = stft_repr.shape
        stft_repr = stft_repr.reshape(b, f * s, t, c)
        x = stft_repr.reshape(b, t, f * s * c)

        # Process through band split
        if self.use_torch_checkpoint:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        # if torch.isnan(x).any() or torch.isinf(x).any():
        #     raise RuntimeError(
        #         f"NaN/Inf in x after band_split: {x.isnan().sum()} NaNs, {x.isinf().sum()} Infs"
        #     )

        # Process through transformer layers
        store = [None] * len(self.layers)
        for i, transformer_block in enumerate(self.layers):
            if len(transformer_block) == 3:
                linear_transformer, time_transformer, freq_transformer = transformer_block
                x = linear_transformer(x)
            else:
                time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                for j in range(i):
                    x = x + store[j]

            # Time transformer
            b, t, f, d = x.shape
            x = x.permute(0, 2, 1, 3).reshape(-1, t, d)
            x = time_transformer(x)
            x = x.reshape(b, f, t, d).permute(0, 2, 1, 3)

            # Frequency transformer
            b, t, f, d = x.shape
            x = x.permute(0, 1, 2, 3).reshape(-1, f, d)
            x = freq_transformer(x)
            x = x.reshape(b, t, f, d)

            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)

        # Generate masks
        if self.use_torch_checkpoint:
            mask = torch.stack(
                [checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators],
                dim=1,
            )
        else:
            mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        
        b, n, t, f = mask.shape
        mask = mask.reshape(b, n, t, f // 2, 2)  # Split into real and imaginary parts

        # Apply masks to input STFT
        stft_repr = stft_repr.unsqueeze(1)  # Add stem dimension
        
        # Ensure shapes match for complex multiplication
        # stft_repr: [b, 1, f, t, 2] -> [b, 1, f, t, 2]
        # mask: [b, n, t, f, 2] -> [b, n, f, t, 2]
        mask = mask.permute(0, 1, 3, 2, 4)  # Swap frequency and time dimensions
        
        # Complex multiplication using real numbers
        # (a + bi) * (c + di) = (ac - bd) + i(ad + bc)
        stft_real = stft_repr[..., 0]
        stft_imag = stft_repr[..., 1]
        mask_real = mask[..., 0]
        mask_imag = mask[..., 1]
        
        # Compute real and imaginary parts of the product
        real_part = stft_real * mask_real - stft_imag * mask_imag
        imag_part = stft_real * mask_imag + stft_imag * mask_real
        
        # Combine real and imaginary parts
        stft_repr = torch.stack([real_part, imag_part], dim=-1)

        # Reshape for output
        b, n, f, t, _ = stft_repr.shape
        stft_repr = stft_repr.reshape(b, n * self.audio_channels, f // self.audio_channels, t, 2)

        return stft_repr