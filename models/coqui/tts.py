from .tts_config import *

import os
from dataclasses import dataclass

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from coqpit import Coqpit

#---------------------------TTS.tts.layers.xtts.gpt_inference.py-----------------------------------
from transformers import GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class GPT2InferenceModel(GPT2PreTrainedModel):
    """Override GPT2LMHeadModel to allow for prefix conditioning."""

    def __init__(self, config, gpt, pos_emb, embeddings, norm, linear, kv_cache):
        super().__init__(config)
        self.transformer = gpt       # GPT2Model
        self.pos_embedding = pos_emb # Mel pos Embedding(608, 1024)
        self.embeddings = embeddings # Mel Embedding(1026, 1024)
        self.final_norm = norm       # Mel Norm
        self.lm_head = nn.Sequential(norm, linear) # Mel head Linear(1024, 1026) 
        self.kv_cache = kv_cache

    def store_prefix_emb(self, prefix_emb):
        self.cached_prefix_emb = prefix_emb
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)  # usually None
        if not self.kv_cache:
            past_key_values = None

        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
    
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        assert self.cached_prefix_emb is not None
        assert inputs_embeds is None # Not supported by this inference model.
        assert labels is None # Training not supported by this inference model.
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # assert len(past_key_values) + len(input_ids) == attention_mask.shape[1]

        # Create embedding
        prefix_len = self.cached_prefix_emb.shape[1]
        if input_ids.shape[1] != 1:
            gen_inputs = input_ids[:, prefix_len:] # 1024
            gen_emb = self.embeddings(gen_inputs)   # [1, 1, 1024]
            gen_emb = gen_emb + self.pos_embedding(gen_emb) # [1, 1, 1024]
            if self.cached_prefix_emb.shape[0] != gen_emb.shape[0]:
                prefix_emb = self.cached_prefix_emb.repeat_interleave(
                    gen_emb.shape[0] // self.cached_prefix_emb.shape[0], 0
                )
            else:
                prefix_emb = self.cached_prefix_emb.to(gen_emb.dtype)
            emb = torch.cat([prefix_emb, gen_emb], dim=1) # [1, 52, 1024]
        else:
            emb = self.embeddings(input_ids) # input_ids: 784 -> [1, 1, 1024]
            emb = emb + self.pos_embedding.get_fixed_embedding(
                attention_mask.shape[1] - (prefix_len + 1), attention_mask.device
            )
        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]  # [1, 52, 1024] then [1, 1, 1024]
        lm_logits = self.lm_head(hidden_states) # [1, 52, 1026] then [1, 1, 1026]

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]
        
        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
    
    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

#--------------------------TTS.tts.layers.xtts.latent_encoder.py----------------------------------

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def normalization(channels):
    groups = 32
    if channels <= 16:
        groups = 8
    elif channels <= 64:
        groups = 16
    while channels % groups != 0:
        groups = int(groups / 2)
    assert groups > 2
    return GroupNorm32(groups, channels) # GN(32, 1024)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
    
    def forward(self, qkv, mask=None, qk_bias=0):
        """
        Apply QKV attention
        
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape # [1, 3072, 517]
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1) # [16, 64, 517]
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q*scale, k*scale) # More stable with f16 than dividing afterwards
        weight = weight + qk_bias # [16, 517, 517]
        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)
            weight[mask.logical_not()] = -torch.inf
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v) # [16, 64, 517]

        return a.reshape(bs, -1, length)

class AttentionBlock(nn.Module):
    """An attention block that allows spatial positions to attend to each other."""
    
    def __init__(self, channels, num_heads=1, num_head_channels=-1,
                 out_channels=None, do_activation=False):
        super().__init__()
        self.channels = channels # 1024
        out_channels = channels if out_channels is None else out_channels # 1024
        self.do_activation = do_activation
        if num_head_channels == -1:
            self.num_heads = num_heads # 16
        else:
            assert(
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels) # GN(32, 1024)
        self.qkv = conv_nd(1, channels, out_channels * 3, 1) # Conv1d(1024, 3072, 1)
        self.attention = QKVAttention(self.num_heads) # 16

        self.proj_out = zero_module(conv_nd(1, out_channels, out_channels, 1)) # Conv1d(1024, 1024, 1)
        self.x_proj = nn.Identity() if out_channels == channels else conv_nd(1, channels, out_channels, 1)

    def forward(self, x, mask=None, qk_bias=0):
        b, c, *spatial = x.shape # [1, 1024, 517]
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0).repeat(x.shape[0], 1, 1)
            if mask.shape[1] != x.shape[-1]:
                mask = mask[:, : x.shape[-1], : x.shape[-1]]
        
        x = x.reshape(b, c, -1) # [1, 1024, 517]
        x = self.norm(x)        # [1, 1024, 517]
        if self.do_activation:
            x = F.silu(x, inplace=True)
        qkv = self.qkv(x)       # [1, 3072, 517]
        h = self.attention(qkv, mask=mask, qk_bias=qk_bias) # [1, 1024, 517]
        h = self.proj_out(h)    # [1, 1024, 517]
        xp = self.x_proj(x)     
        return (xp + h).reshape(b, xp.shape[1], *spatial)

class ConditioningEncoder(nn.Module):
    def __init__(
        self,
        spec_dim,           # 80
        embedding_dim,      # 1024
        attn_blocks=6,      # 6
        num_attn_heads=4    # 16
    ):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1) # nn.Conv1d(80, 1024, 1)
        for _ in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim
    
    def forward(self, x):
        """
        x: (b, 80, s) e.g [1, 80, 517]
        """
        h = self.init(x) # [1, 1024, 517]
        h = self.attn(h) # [1, 1024, 517]
        return h

#---------------------------TTS.tts.layers.xtts.perceiver_encoder.py-----------------------------
from collections import namedtuple
from functools import wraps
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from packaging import version
from torch import einsum

def exists(val):
    return val is not None

def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner

print_once = once(print)


class Attend(nn.Module):
    def __init__(self, dropout=0.0, causal=False, use_flash=False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

        self.use_flash = use_flash
        assert not (
            use_flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "in order to use flash attention, you must be using pytorch 2.0 or above"

        # determine efficient attention configs for cuda and cpu
        self.config = namedtuple("EfficientAttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"])
        self.cpu_config = self.config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once("A100 GPU detected, using flash attention if input tensor is on cuda")
            self.cuda_config = self.config(True, False, False)
        else:
            print_once("Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda")
            self.cuda_config = self.config(False, True, True)

    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def flash_attn(self, q, k, v, mask=None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])
        if k.ndim == 3:
            k = rearrange(k, "b ... -> b 1 ...").expand_as(q)
        if v.ndim == 3:
            v = rearrange(v, "b ... -> b 1 ...").expand_as(q)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L
        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention
        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=self.causal
            )

        return out

    def forward(self, q, k, v, mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device = q.shape[-2], q.device
        scale = q.shape[-1] ** -0.5

        if self.use_flash:
            return self.flash_attn(q, k, v, mask=mask)

        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"
        # similarity
        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # key padding mask
        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask
        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values
        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out

def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class RMSNorm(nn.Module):
    def __init__(self, dim, scale=True, dim_cond=None):
        super().__init__()
        self.cond = exists(dim_cond)
        self.to_gamma_beta = nn.Linear(dim_cond, dim * 2) if self.cond else None

        self.scale = dim**0.5 # 32
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None
    
    def forward(self, x, cond=None):
        gamma = default(self.gamma, 1)
        out = F.normalize(x, dim=-1) * self.scale * gamma

        if not self.cond:
            return out
        
        assert exists(cond)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        gamma, beta = map(lambda t: rearrange(t, "b d -> b 1 d"), (gamma, beta))
        return out * gamma + beta

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (kernel_size,) = self.kernel_size
        (dilation,) = self.dilation
        (stride,) = self.stride

        assert stride == 1
        self.causal_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        causal_padded_x = F.pad(x, (self.causal_padding, 0), value=0.0)
        return super().forward(causal_padded_x)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4, causal_conv=False):
    dim_inner = int(dim * mult * 2 / 3) # 2730

    conv = None
    if causal_conv:
        conv = nn.Sequential(
            Rearrange("b n d -> b d n"),
            CausalConv1d(dim_inner, dim_inner, 3),
            Rearrange("b d n -> b n d"),
        )
    return Sequential(nn.Linear(dim, dim_inner * 2), GEGLU(), conv, nn.Linear(dim_inner, dim))


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_context=None,
        causal=False,
        dim_head=64,
        heads=8,
        dropout=0.0,
        use_flash=False,
        cross_attn_include_queries=False,
    ):
        super().__init__()
        self.scale = dim_head**-0.5 # 0.126
        self.heads = heads          # 8
        self.cross_attn_include_queries = cross_attn_include_queries

        dim_inner = dim_head * heads # 512
        dim_context = default(dim_context, dim) # 1024

        self.attend = Attend(causal=causal, dropout=dropout, use_flash=use_flash)
        self.to_q = nn.Linear(dim, dim_inner, bias=False) # (1024, 512)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias=False) # (1024, 1024)
        self.to_out = nn.Linear(dim_inner, dim, bias=False) # (512, 1024)

    def forward(self, x, context=None, mask=None):
        h, has_context = self.heads, exists(context) # 8, True

        context = default(context, x) # context: [1, 517, 1024], x: [1, 32, 1024]

        if has_context and self.cross_attn_include_queries:
            context = torch.cat((x, context), dim=-2) # [1, 549, 1024]

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1)) # q:[1,32,512], k,v:[1,549,512]
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v)) # [1,8,32,64],[1,8,549,64],[1,8,549,64]

        out = self.attend(q, k, v, mask=mask) # [1,8,32,64]

        out = rearrange(out, "b h n d -> b n (h d)") # [1,32,512]
        return self.to_out(out) # [1,32,1024]


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=2,
        dim_context=None,
        num_latents=32,
        dim_head=64,
        heads=8,
        ff_mult=4,
        use_flash_attn=False
    ):
        super().__init__()
        dim_context = default(dim_context, dim) # 1024

        self.proj_context = nn.Linear(dim_context, dim) if dim_context != dim else nn.Identity()

        self.latents = nn.Parameter(torch.randn(num_latents, dim)) # nn.Parameter(32, 1024)
        nn.init.normal_(self.latents, std=0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,            # 1024
                            dim_head=dim_head,  # 64
                            heads=heads,        # 8
                            use_flash=use_flash_attn,
                            cross_attn_include_queries=True,
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )
        self.norm = RMSNorm(dim)

    def forward(self, x, mask=None):
        batch = x.shape[0] # [1, 517, 1024]
        latents = repeat(self.latents, "n d -> b n d", b=batch) # [1, 32, 1024]

        x = self.proj_context(x)

        for attn, ff in self.layers:
            latents = attn(latents, x, mask=mask) + latents
            latents = ff(latents) + latents

        return self.norm(latents)


#----------------------------TTS.tts.layers.xtts.gpt.py---------------------------------------------
import functools
import random
from transformers import GPT2Config

def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)

class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02, relative=False):
        super().__init__()
        # nn.Embedding
        self.emb = nn.Embedding(seq_len, model_dim) # nn.Embedding(404, 1024)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)
        self.relative = relative
        self.seq_len = seq_len # 404

    def forward(self, x):
        sl = x.shape[1]
        if self.relative:
            start = random.randint(sl, self.seq_len) - sl # [0, seq_len-sl]
            return self.emb(torch.arange(start, start + sl, device=x.device))
        else:
            return self.emb(torch.arange(0, sl, device=x.device))
    
    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)

from transformers import GPT2Config, GPT2Model

def build_hf_gpt_transformer(
    layers,
    model_dim,
    heads,
    max_mel_seq_len,
    max_text_seq_len,
    max_prompt_len,
    checkpointing,
):
    """
    GPT-2 implemented by the HuggingFace library.
    """

    gpt_config = GPT2Config(
        vocab_size=256, # unused.
        n_positions=max_mel_seq_len + max_text_seq_len + max_prompt_len, # 1082
        n_ctx=max_mel_seq_len + max_text_seq_len + max_prompt_len,
        n_embd=model_dim,
        n_layer=layers,
        n_head=heads,
        gradient_checkpointing=checkpointing,
        use_cache=not checkpointing,
    )
    gpt = GPT2Model(gpt_config)
    # Override the built in positional embeddings
    del gpt.wpe
    gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
    # Built-in token embeddings are unused.
    del gpt.wte

    mel_pos_emb = (
        LearnedPositionEmbeddings(max_mel_seq_len, model_dim)
        if max_mel_seq_len != -1
        else functools.partial(null_position_embeddings, dim=model_dim)
    )
    text_pos_emb = (
        LearnedPositionEmbeddings(max_text_seq_len, model_dim)
        if max_mel_seq_len != -1
        else functools.partial(null_position_embeddings, dim=model_dim)
    )
    # gpt torch.compile(gpt, model="reduce-overhead", fullgraph=True)
    return gpt, mel_pos_emb, text_pos_emb, None, None

class GPT(nn.Module):
    def __init__(
        self,
        start_text_token=261,
        stop_text_token=0,
        layers=8,
        model_dim=512,
        heads=8,
        max_text_tokens=120,
        max_mel_tokens=250,
        max_prompt_tokens=70,
        max_conditioning_inputs=1,
        code_stride_len=1024,
        number_text_tokens=256,
        num_audio_tokens=8194,
        start_audio_token=8192,
        stop_audio_token=8193,
        train_solo_embeddings=False,
        checkpointing=False,
        average_conditioning_embeddings=False,
        label_smoothing=0.0,
        use_perceiver_resampler=False,
        perceiver_cond_length_compression=256,
    ):
        """
        Args:

        """
        super().__init__()

        self.label_smoothing = label_smoothing

        self.number_text_tokens = number_text_tokens # 6681
        self.start_text_token = start_text_token     # 261
        self.stop_text_token = stop_text_token       # 0

        self.num_audio_tokens = num_audio_tokens    # 1026
        self.start_audio_token = start_audio_token  # 1024
        self.stop_audio_token = stop_audio_token    # 1025
        self.start_prompt_token = start_audio_token # 1024
        self.stop_prompt_token = stop_audio_token   # 1025

        self.layers = layers                        # 30
        self.heads = heads                          # 16
        self.model_dim = model_dim                  # 1024
        self.max_conditioning_inputs = max_conditioning_inputs # 1
        self.max_gen_mel_tokens = max_mel_tokens - self.max_conditioning_inputs - 2 # 602 (605-1-2)
        self.max_mel_tokens = -1 if max_mel_tokens == -1 else max_mel_tokens + 2 + self.max_conditioning_inputs # 608
        self.max_text_tokens = -1 if max_text_tokens == -1 else max_text_tokens + 2 # 404 (402+2)
        self.max_prompt_tokens = max_prompt_tokens  # 70
        self.code_stride_len = code_stride_len      # 1024

        self.conditioning_encoder = ConditioningEncoder(80, model_dim, num_attn_heads=heads)
        self.conditioning_dropout = nn.Dropout1d(0.1)

        self.average_conditioning_embeddings = average_conditioning_embeddings # False
        self.use_perceiver_resampler = use_perceiver_resampler                 # True
        self.perceiver_cond_length_compression = perceiver_cond_length_compression # 256

        self.text_embedding = nn.Embedding(self.number_text_tokens, model_dim) # (6681, 1024)
        self.mel_embedding = nn.Embedding(self.num_audio_tokens, model_dim)    # (1026, 1024)

        (
            self.gpt,
            self.mel_pos_embedding,
            self.text_pos_embedding,
            self.mel_layer_pos_embedding,
            self.text_layer_pos_embedding,
        ) = build_hf_gpt_transformer(
            layers,
            model_dim,
            heads,
            self.max_mel_tokens,
            self.max_text_tokens,
            self.max_prompt_tokens,
            checkpointing,
        )
        if train_solo_embeddings:
            self.mel_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02, requires_grad=True)
            self.text_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02, requires_grad=True)
        else:
            self.mel_solo_embedding = 0
            self.text_solo_embedding = 0

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens) # nn.Linear(1024, 6681)
        self.mel_head = nn.Linear(model_dim, self.num_audio_tokens)    # nn.Linear(1024, 1026)

        if self.use_perceiver_resampler:
            # XTTS v2
            self.conditioning_perceiver = PerceiverResampler(
                dim=model_dim,
                depth=2,
                dim_context=model_dim,
                num_latents=32,
                dim_head=64,
                heads=8,
                ff_mult=4,
                use_flash_attn=True,
            )
        else:
            # XTTS v1
            self.prompt_embedding = nn.Embedding(self.num_audio_tokens, model_dim)
            self.prompt_pos_embedding = LearnedPositionEmbeddings(24 * 9, model_dim)

    def get_grad_norm_parameter_groups(self):
        return {
            "conditioning_encoder": list(self.conditioning_encoder.parameters()),
            "conditioning_perceiver": list(self.conditioning_perceiver.parameters())
            if self.use_perceiver_resampler
            else None,
            "gpt": list(self.gpt.parameters()),
            "heads": list(self.text_head.parameters()) + list(self.mel_head.parameters()),
        }

    def init_gpt_for_inference(self, kv_cache=True, use_deepspeed=False):
        seq_length = self.max_prompt_tokens + self.max_mel_tokens + self.max_text_tokens + 1 # 1083
        gpt_config = GPT2Config(
            vocab_size=self.max_mel_tokens, # 608
            n_positions=seq_length,         # 1083
            n_ctx=seq_length,               # 1083
            n_embd=self.model_dim,          # 1024
            n_layer=self.layers,            # 30
            n_head=self.heads,              # 16
            gradient_checkpointing=False,
            use_cache=True,
        )
        self.gpt_inference = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )
        self.gpt.wte = self.mel_embedding # Embedding(1026, 1024)

        if use_deepspeed:
            import deepspeed

            self.ds_engine = deepspeed.init_inference(
                model=self.gpt_inference.half(),  # Transformers models
                mp_size=1,  # Number of GPU
                dtype=torch.float32,  # desired data type of output
                replace_method="auto",  # Lets DS autmatically identify the layer to replace
                replace_with_kernel_inject=True,  # replace the model with the kernel injector
            )
            self.gpt_inference = self.ds_engine.module.eval()

    def set_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, code_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with stop_audio_token in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        # Set padding areas within MEL (currently it is coded with the MEL code for <zero>).
        for b in range(len(code_lengths)):
            actual_end = code_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_audio_token
        return mel_input_tokens

    def get_logits(
        self,
        first_inputs,         # [1, 19, 1024]       -> text_embedding
        first_head,           # Linear(1024, 6681)  -> text_head
        second_inputs=None,   # [1, 43, 1024]       -> audio_embedding
        second_head=None,     # Linear(1024, 1026)  -> audio_head
        prompt=None,          # [1, 32, 1024]       -> speaker_embedding output
        get_attns=False,
        return_latent=False,
        attn_mask_cond=None,
        attn_mask_text=None,
        attn_mask_mel=None,
    ):
        if prompt is not None:
            offset = prompt.shape[1]
            if second_inputs is not None:
                emb = torch.cat([prompt, first_inputs, second_inputs], dim=1) # [1, 94, 1024]
            else:
                emb = torch.cat([prompt, first_inputs], dim=1)

        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        attn_mask = None
        if attn_mask_text is not None:
            attn_mask = torch.cat([attn_mask_text, attn_mask_mel], dim=1)
            if prompt is not None:
                attn_mask_cond = torch.ones(prompt.shape[0], offset, dtype=torch.bool, device=emb.device)
                attn_mask = torch.cat([attn_mask_cond, attn_mask], dim=1)

        gpt_out = self.gpt(
            inputs_embeds=emb,
            return_dict=True,
            output_attentions=get_attns,
            attention_mask=attn_mask,
        )

        if get_attns:
            return gpt_out.attentions

        enc = gpt_out.last_hidden_state[:, offset:] # [1, 62, 1024]
        enc = self.final_norm(enc)

        if return_latent:
            return enc[:, : first_inputs.shape[1]], enc[:, -second_inputs.shape[1] :]

        first_logits = enc[:, : first_inputs.shape[1]]
        first_logits = first_head(first_logits)
        first_logits = first_logits.permute(0, 2, 1)
        if second_inputs is not None:
            second_logits = enc[:, -second_inputs.shape[1] :]
            second_logits = second_head(second_logits)
            second_logits = second_logits.permute(0, 2, 1)
            return first_logits, second_logits
        else:
            return first_logits

    def get_conditioning(self, speech_conditioning_input):
        speech_conditioning_input = (
            speech_conditioning_input.unsqueeze(1)
            if len(speech_conditioning_input.shape) == 3
            else speech_conditioning_input
        )
        conds = []
        for j in range(speech_conditioning_input.shape[1]):
            conds.append(self.conditioning_encoder(speech_conditioning_input[:, j]))
        conds = torch.stack(conds, dim=1)
        conds = conds.mean(dim=1)
        return conds

    def get_prompts(self, prompt_codes):
        """
        Create a prompt from the mel codes. This is used to condition the model on the mel codes.
        Pad the prompt with start and stop mel tokens.
        """
        prompt = prompt_codes
        if self.training:
            lengths = []
            # Compute the real prompt length based on the first encounter with the token 83 used for padding
            for i in range(prompt_codes.shape[0]):
                length = 0
                for j in range(prompt_codes.shape[1]):
                    if prompt_codes[i, j] == 83:
                        break
                    else:
                        length += 1
                lengths.append(length)

            # prompt_len = random.randint(1, 9)  # in secs
            prompt_len = 3
            prompt_len = prompt_len * 24  # in frames
            if prompt_codes.shape[-1] >= prompt_len:
                for i in range(prompt_codes.shape[0]):
                    if lengths[i] < prompt_len:
                        start = 0
                    else:
                        start = random.randint(0, lengths[i] - prompt_len)
                prompt = prompt_codes[:, start : start + prompt_len]

        # add start and stop tokens
        prompt = F.pad(prompt, (1, 0), value=self.start_prompt_token)
        prompt = F.pad(prompt, (0, 1), value=self.stop_prompt_token)
        return prompt

    def get_style_emb(self, cond_input, return_latent=False):
        """
        cond_input: (b, 80, s) or (b, 1, 80, s)
        conds: (b, 1024, s)
        """
        conds = None
        if not return_latent:
            if cond_input.ndim == 4:
                cond_input = cond_input.squeeze(1)
            conds = self.conditioning_encoder(cond_input)  # (b, d, s) e.g. [1, 1024, 517]
            if self.use_perceiver_resampler:
                conds = self.conditioning_perceiver(conds.permute(0, 2, 1)).transpose(1, 2)  # (b, d, 32) e.g [1,1024,32]
        else:
            # already computed
            conds = cond_input.unsqueeze(1)
        return conds

    def forward(
        self,
        text_inputs,    # [1, 18]
        text_lengths,   # 18
        audio_codes,    # [1, 47]
        wav_lengths,    # 47*1024
        cond_mels=None,
        cond_idxs=None,
        cond_lens=None,
        cond_latents=None,  # [1, 32, 1024]
        return_attentions=False,
        return_latent=False,
    ):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode
        (actuated by `text_first`).

        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)
        cond_mels: MEL float tensor, (b, 1, 80,s)
        cond_idxs: cond start and end indexs, (b, 2)

        If return_attentions is specified, only logits are returned.
        If return_latent is specified, loss & logits are not computed or returned. Only the predicted latents are returned.
        """
        # ❗ FIXIT
        if self.max_conditioning_inputs == 0:
            assert cond_mels is None, " ❗ cond_mels is not None, but max_conditioning_inputs == 0"

        max_text_len = text_lengths.max()
        code_lengths = torch.ceil(wav_lengths / self.code_stride_len).long() + 3

        if cond_lens is not None:
            if self.use_perceiver_resampler:
                cond_lens = cond_lens // self.perceiver_cond_length_compression
            else:
                cond_lens = cond_lens // self.code_stride_len

        if cond_idxs is not None:
            # recompute cond idxs for mel lengths
            for idx in range(cond_idxs.size(0)):
                if self.use_perceiver_resampler:
                    cond_idxs[idx] = cond_idxs[idx] // self.perceiver_cond_length_compression
                else:
                    cond_idxs[idx] = cond_idxs[idx] // self.code_stride_len

        # ensure that the cond_mel does not have padding
        # if cond_lens is not None and cond_idxs is None:
        #     min_cond_len = torch.min(cond_lens)
        #     cond_mels = cond_mels[:, :, :, :min_cond_len]

        # If len(codes) + 3 is larger than maxiumum allowed length, we truncate the codes.
        max_mel_len = code_lengths.max()

        if max_mel_len > audio_codes.shape[-1]:
            audio_codes = F.pad(audio_codes, (0, max_mel_len - audio_codes.shape[-1]))

        # 💖 Lovely assertions
        assert (
            max_mel_len <= audio_codes.shape[-1]
        ), f" ❗ max_mel_len ({max_mel_len}) > audio_codes.shape[-1] ({audio_codes.shape[-1]})"
        assert (
            max_text_len <= text_inputs.shape[-1]
        ), f" ❗ max_text_len ({max_text_len}) > text_inputs.shape[-1] ({text_inputs.shape[-1]})"

        # Append stop token to text inputs
        text_inputs = F.pad(text_inputs[:, :max_text_len], (0, 1), value=self.stop_text_token)

        # Append silence token to mel codes
        audio_codes = F.pad(audio_codes[:, :max_mel_len], (0, 1), value=self.stop_audio_token)

        # Pad mel codes with stop_audio_token
        audio_codes = self.set_mel_padding(
            audio_codes, code_lengths - 3
        )  # -3 to get the real code lengths without consider start and stop tokens that was not added yet

        # Build input and target tensors
        # Prepend start token to inputs and append stop token to targets
        text_inputs, text_targets = self.set_inputs_and_targets(
            text_inputs, self.start_text_token, self.stop_text_token
        )
        audio_codes, mel_targets = self.set_inputs_and_targets(
            audio_codes, self.start_audio_token, self.stop_audio_token
        )

        # Set attn_mask
        attn_mask_cond = None
        attn_mask_text = None
        attn_mask_mel = None
        if not return_latent:
            attn_mask_cond = torch.ones(
                cond_mels.shape[0],
                cond_mels.shape[-1],
                dtype=torch.bool,
                device=text_inputs.device,
            )
            attn_mask_text = torch.ones(
                text_inputs.shape[0],
                text_inputs.shape[1],
                dtype=torch.bool,
                device=text_inputs.device,
            )
            attn_mask_mel = torch.ones(
                audio_codes.shape[0],
                audio_codes.shape[1],
                dtype=torch.bool,
                device=audio_codes.device,
            )

            if cond_idxs is not None:
                # use masking approach
                for idx, r in enumerate(cond_idxs):
                    l = r[1] - r[0]
                    attn_mask_cond[idx, l:] = 0.0
            elif cond_lens is not None:
                for idx, l in enumerate(cond_lens):
                    attn_mask_cond[idx, l:] = 0.0

            for idx, l in enumerate(text_lengths):
                attn_mask_text[idx, l + 1 :] = 0.0

            for idx, l in enumerate(code_lengths):
                attn_mask_mel[idx, l + 1 :] = 0.0

        # Compute text embeddings + positional embeddings
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs) # [1,19,1024]

        # Compute mel embeddings + positional embeddings
        mel_emb = self.mel_embedding(audio_codes) + self.mel_pos_embedding(audio_codes) # [1,43,1024]

        # Compute speech conditioning input
        if cond_latents is None:
            cond_latents = self.get_style_emb(cond_mels).transpose(1, 2)

        # Get logits
        sub = -5  # don't ask me why 😄
        if self.training:
            sub = -1

        text_logits, mel_logits = self.get_logits(
            text_emb,          # [1, 19, 1024]
            self.text_head,    # Linear(1024, 6681)
            mel_emb,           # [1, 43, 1024]
            self.mel_head,     # Linear(1024, 1026)
            prompt=cond_latents, # [1, 32, 1024]
            get_attns=return_attentions,
            return_latent=return_latent,
            attn_mask_cond=attn_mask_cond,
            attn_mask_text=attn_mask_text,
            attn_mask_mel=attn_mask_mel,
        )
        if return_latent:
            return mel_logits[:, :sub]  # sub to prevent bla.

        if return_attentions:
            return mel_logits

        # Set paddings to -1 to ignore them in loss
        for idx, l in enumerate(text_lengths):
            text_targets[idx, l + 1 :] = -1

        for idx, l in enumerate(code_lengths):
            mel_targets[idx, l + 1 :] = -1

        # check if stoptoken is in every row of mel_targets
        assert (mel_targets == self.stop_audio_token).sum() >= mel_targets.shape[
            0
        ], f" ❗ mel_targets does not contain stop token ({self.stop_audio_token}) in every row."

        # ignore the loss for the segment used for conditioning
        # coin flip for the segment to be ignored
        if cond_idxs is not None:
            cond_start = cond_idxs[idx, 0]
            cond_end = cond_idxs[idx, 1]
            mel_targets[idx, cond_start:cond_end] = -1

        # Compute losses
        loss_text = F.cross_entropy(
            text_logits, text_targets.long(), ignore_index=-1, label_smoothing=self.label_smoothing
        )
        loss_mel = F.cross_entropy(
            mel_logits, mel_targets.long(), ignore_index=-1, label_smoothing=self.label_smoothing
        )
        return loss_text.mean(), loss_mel.mean(), mel_logits

    def inference(self, cond_latents, text_inputs, **hf_generate_kwargs):
        self.compute_embeddings(cond_latents, text_inputs)
        return self.generate(cond_latents, text_inputs, **hf_generate_kwargs)

    def compute_embeddings(
        self,
        cond_latents,
        text_inputs,
    ):
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs = F.pad(text_inputs, (1, 0), value=self.start_text_token)
        emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        emb = torch.cat([cond_latents, emb], dim=1)
        self.gpt_inference.store_prefix_emb(emb)
        gpt_inputs = torch.full(
            (
                emb.shape[0],
                emb.shape[1] + 1,  # +1 for the start_audio_token
            ),
            fill_value=1,
            dtype=torch.long,
            device=text_inputs.device,
        )
        gpt_inputs[:, -1] = self.start_audio_token
        return gpt_inputs

    def generate(
        self,
        cond_latents,
        text_inputs,
        **hf_generate_kwargs,
    ):
        gpt_inputs = self.compute_embeddings(cond_latents, text_inputs)
        gen = self.gpt_inference.generate(
            gpt_inputs,
            bos_token_id=self.start_audio_token,
            pad_token_id=self.stop_audio_token,
            eos_token_id=self.stop_audio_token,
            max_length=self.max_gen_mel_tokens + gpt_inputs.shape[-1],
            **hf_generate_kwargs,
        )
        if "return_dict_in_generate" in hf_generate_kwargs:
            return gen.sequences[:, gpt_inputs.shape[1] :], gen
        return gen[:, gpt_inputs.shape[1] :]

    def get_generator(self, fake_inputs, **hf_generate_kwargs):
        return self.gpt_inference.generate_stream(
            fake_inputs,
            bos_token_id=self.start_audio_token,
            pad_token_id=self.stop_audio_token,
            eos_token_id=self.stop_audio_token,
            max_length=self.max_gen_mel_tokens + fake_inputs.shape[-1],
            do_stream=True,
            **hf_generate_kwargs,
        )

#---------------------------------higigan_decoder.py--------------------------------------------
import torchaudio
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
import fsspec
from pathlib import Path
from typing import Any, Callable, Union

def get_user_data_dir(appname):
    TTS_HOME = os.environ.get("TTS_HOME")
    XDG_DATA_HOME = os.environ.get("XDG_DATA_HOME")
    if TTS_HOME is not None:
        ans = Path(TTS_HOME).expanduser().resolve(strict=False)
    elif XDG_DATA_HOME is not None:
        ans = Path(XDG_DATA_HOME).expanduser().resolve(strict=False)
    else:
        ans = Path.home().joinpath(".local/share")
    return ans.joinpath(appname)

def load_fsspec(
    path: str,
    map_location: Union[str, Callable, torch.device, Dict[Union[str, torch.device], Union[str, torch.device]]] = None,
    cache: bool = True,
    **kwargs,
) -> Any:
    """
    Like torch.load but can load from other locations (e.g. s3://, gs://)
    Args:
        path: Any path or url supported by fsspec.
        map_location: torch.device or str
        cache: If True, cache a remote file locally for subsequent calls. It is cached under
                `get_user_data_dir()/tts_cache`. Defaults to True.
        **kwargs: Keyword arguments forwarded to torch.load.
    Returns:
        Object stored in path.
    """
    is_local = os.path.isdir(path) or os.path.isfile(path)
    if cache and not is_local:
        with fsspec.open(
            f"filecache::{path}", filecache={"cache_storage": str(get_user_data_dir("tts_cache"))},
            mode="rb",
        ) as f:
            return torch.load(f, map_location=map_location, **kwargs)
    else:
        with fsspec.open(path, "rb") as f:
            return torch.load(f, map_location=map_location, **kwargs)

LRELU_SLOPE = 0.1


def get_padding(k, d):
    return int((k * d - d) / 2)


class ResBlock1(torch.nn.Module):
    """Residual Block Type 1. It has 3 convolutional layers in each convolutional block.

    Network::

        x -> lrelu -> conv1_1 -> conv1_2 -> conv1_3 -> z -> lrelu -> conv2_1 -> conv2_2 -> conv2_3 -> o -> + -> o
        |--------------------------------------------------------------------------------------------------|


    Args:
        channels (int): number of hidden channels for the convolutional layers.
        kernel_size (int): size of the convolution filter in each layer.
        dilations (list): list of dilation value for each conv layer in a block.
    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            Tensor: output tensor.
        Shapes:
            x: [B, C, T]
        """
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_parametrizations(l, "weight")
        for l in self.convs2:
            remove_parametrizations(l, "weight")


class ResBlock2(torch.nn.Module):
    """Residual Block Type 2. It has 1 convolutional layers in each convolutional block.

    Network::

        x -> lrelu -> conv1-> -> z -> lrelu -> conv2-> o -> + -> o
        |---------------------------------------------------|


    Args:
        channels (int): number of hidden channels for the convolutional layers.
        kernel_size (int): size of the convolution filter in each layer.
        dilations (list): list of dilation value for each conv layer in a block.
    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_parametrizations(l, "weight")


class HifiganGenerator(torch.nn.Module):
    def __init__(
        self,
        in_channels,            # 1024
        out_channels,           # 1
        resblock_type,          # '1'
        resblock_dilation_sizes,# [[1,3,5],[1,3,5],[1,3,5]]
        resblock_kernel_sizes,  # [3,7,11]
        upsample_kernel_sizes,  # [16,16,4,4]
        upsample_initial_channel, # 512
        upsample_factors,       # [8,8,2,2]
        inference_padding=5,    # 0
        cond_channels=0,        # 512
        conv_pre_weight_norm=True, # False
        conv_post_weight_norm=True,# False
        conv_post_bias=True,       # False
        cond_in_each_up_layer=False, # True
    ):
        r"""HiFiGAN Generator with Multi-Receptive Field Fusion (MRF)

        Network:
            x -> lrelu -> upsampling_layer -> resblock1_k1x1 -> z1 -> + -> z_sum / #resblocks -> lrelu -> conv_post_7x1 -> tanh -> o
                                                 ..          -> zI ---|
                                              resblockN_kNx1 -> zN ---'

        Args:
            in_channels (int): number of input tensor channels.
            out_channels (int): number of output tensor channels.
            resblock_type (str): type of the `ResBlock`. '1' or '2'.
            resblock_dilation_sizes (List[List[int]]): list of dilation values in each layer of a `ResBlock`.
            resblock_kernel_sizes (List[int]): list of kernel sizes for each `ResBlock`.
            upsample_kernel_sizes (List[int]): list of kernel sizes for each transposed convolution.
            upsample_initial_channel (int): number of channels for the first upsampling layer. This is divided by 2
                for each consecutive upsampling layer.
            upsample_factors (List[int]): upsampling factors (stride) for each upsampling layer.
            inference_padding (int): constant padding applied to the input at inference time. Defaults to 5.
        """
        super().__init__()
        self.inference_padding = inference_padding    # 0
        self.num_kernels = len(resblock_kernel_sizes) # 3
        self.num_upsamples = len(upsample_factors)    # 4
        self.cond_in_each_up_layer = cond_in_each_up_layer # True

        # initial upsampling layers
        self.conv_pre = weight_norm(Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2
        # upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i), # 512,256,128,64
                        upsample_initial_channel // (2 ** (i + 1)), # 256,128,64,32
                        k, # 16,16,4,4
                        u, #  8, 8,2,2
                        padding=(k - u) // 2, # 4,4,1,1
                    )
                )
            )
        # MRF blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1)) # 256,128,64,32
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))
        # post convolution layer
        self.conv_post = weight_norm(Conv1d(ch, out_channels, 7, 1, padding=3, bias=conv_post_bias)) # Conv1d(32,1,7,1,3)
        if cond_channels > 0:
            self.cond_layer = nn.Conv1d(cond_channels, upsample_initial_channel, 1) # Conv1d(512,512,1)

        if not conv_pre_weight_norm:
            remove_parametrizations(self.conv_pre, "weight")

        if not conv_post_weight_norm:
            remove_parametrizations(self.conv_post, "weight")

        if self.cond_in_each_up_layer:
            self.conds = nn.ModuleList()
            for i in range(len(self.ups)):
                ch = upsample_initial_channel // (2 ** (i + 1)) # ch=(256,128,64,32)
                self.conds.append(nn.Conv1d(cond_channels, ch, 1)) # Conv1d(512, ch, 1, 1)

    def forward(self, x, g=None):
        """
        Args:
            x (Tensor): feature input tensor.
            g (Tensor): global conditioning input tensor.

        Returns:
            Tensor: output waveform.

        Shapes:
            x: [B, C, T]
            Tensor: [B, 1, T]
        """
        o = self.conv_pre(x)  # x: [1024, 165] o: [512, 165]
        if hasattr(self, "cond_layer"):
            o = o + self.cond_layer(g)  # g, cond_layer(g): [1, 512, 1], o: [1, 512, 165]
        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o)  # [1,256,1320], [1,128,10560], [1,64,21120], [1,32,42240]

            if self.cond_in_each_up_layer:
                o = o + self.conds[i](g) # [1, 256, 1320]

            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i * self.num_kernels + j](o)
                else:
                    z_sum += self.resblocks[i * self.num_kernels + j](o)
            o = z_sum / self.num_kernels
        o = F.leaky_relu(o)
        o = self.conv_post(o) # [1, 1, 42240]
        o = torch.tanh(o)
        return o

    @torch.no_grad()
    def inference(self, c):
        """
        Args:
            x (Tensor): conditioning input tensor.

        Returns:
            Tensor: output waveform.

        Shapes:
            x: [B, C, T]
            Tensor: [B, 1, T]
        """
        c = c.to(self.conv_pre.weight.device)
        c = torch.nn.functional.pad(c, (self.inference_padding, self.inference_padding), "replicate")
        return self.forward(c)

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_parametrizations(l, "weight")
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_parametrizations(self.conv_pre, "weight")
        remove_parametrizations(self.conv_post, "weight")

    def load_checkpoint(
        self, config, checkpoint_path, eval=False, cache=False
    ):  # pylint: disable=unused-argument, redefined-builtin
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            assert not self.training
            self.remove_weight_norm()


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction), # Linear(32,4)
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel), # Linear(4,32)
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


def set_init_dict(model_dict, checkpoint_state, c):
    # Partial initialization: if there is a mismatch with new and old layer, it is skipped.
    for k, v in checkpoint_state.items():
        if k not in model_dict:
            print(" | > Layer missing in the model definition: {}".format(k))
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in checkpoint_state.items() if k in model_dict}
    # 2. filter out different size layers
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if v.numel() == model_dict[k].numel()}
    # 3. skip reinit layers
    if c.has("reinit_layers") and c.reinit_layers is not None:
        for reinit_layer_name in c.reinit_layers:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if reinit_layer_name not in k}
    # 4. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    print(" | > {} / {} layers are restored.".format(len(pretrained_dict), len(model_dict)))
    return model_dict


class PreEmphasis(nn.Module):
    def __init__(self, coefficient=0.97):
        super().__init__()
        self.coefficient = coefficient
        self.register_buffer("filter", torch.FloatTensor([-self.coefficient, 1.0]).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        assert len(x.size()) == 2

        x = F.pad(x.unsqueeze(1), (1, 0), "reflect")
        return F.conv1d(x, self.filter).squeeze(1)


class ResNetSpeakerEncoder(nn.Module):
    """This is copied from 🐸TTS to remove it from the dependencies."""

    # pylint: disable=W0102
    def __init__(
        self,
        input_dim=64,
        proj_dim=512,
        layers=[3, 4, 6, 3],
        num_filters=[32, 64, 128, 256],
        encoder_type="ASP",
        log_input=False,
        use_torch_spec=False,
        audio_config=None,
    ):
        super(ResNetSpeakerEncoder, self).__init__()

        self.encoder_type = encoder_type # 'ASP'
        self.input_dim = input_dim       # 64
        self.log_input = log_input       # True
        self.use_torch_spec = use_torch_spec # True
        self.audio_config = audio_config     # (512,400,160,16000,0.97,64)
        self.proj_dim = proj_dim         # 512

        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        self.inplanes = num_filters[0]
        self.layer1 = self.create_layer(SEBasicBlock, num_filters[0], layers[0])
        self.layer2 = self.create_layer(SEBasicBlock, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self.create_layer(SEBasicBlock, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self.create_layer(SEBasicBlock, num_filters[3], layers[3], stride=(2, 2))

        self.instancenorm = nn.InstanceNorm1d(input_dim)

        if self.use_torch_spec:
            self.torch_spec = torch.nn.Sequential(
                PreEmphasis(audio_config["preemphasis"]),
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=audio_config["sample_rate"], # 16000
                    n_fft=audio_config["fft_size"],          # 512
                    win_length=audio_config["win_length"],   # 400
                    hop_length=audio_config["hop_length"],   # 160
                    window_fn=torch.hamming_window,
                    n_mels=audio_config["num_mels"],         # 64
                ),
            )

        else:
            self.torch_spec = None

        outmap_size = int(self.input_dim / 8) # 8

        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1), # Conv1d(2048,128,1)
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1), # Conv1d(128,2048)
            nn.Softmax(dim=2),
        )

        if self.encoder_type == "SAP":
            out_dim = num_filters[3] * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3] * outmap_size * 2 # 4096
        else:
            raise ValueError("Undefined encoder")

        self.fc = nn.Linear(out_dim, proj_dim) # Linear(4096, 512)

        self._init_layers()

    def _init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def create_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # pylint: disable=R0201
    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x, l2_norm=False):
        """Forward pass of the model.

        Args:
            x (Tensor): Raw waveform signal or spectrogram frames. If input is a waveform, `torch_spec` must be `True`
                to compute the spectrogram on-the-fly.
            l2_norm (bool): Whether to L2-normalize the outputs.

        Shapes:
            - x: :math:`(N, 1, T_{in})` or :math:`(N, D_{spec}, T_{in})`
        """
        x.squeeze_(1)
        # if you torch spec compute it otherwise use the mel spec computed by the AP
        if self.use_torch_spec:
            x = self.torch_spec(x) # preemphesis, then compute MelSpec

        if self.log_input:
            x = (x + 1e-6).log()
        x = self.instancenorm(x).unsqueeze(1) # [1, 1, 64, 3001]

        x = self.conv1(x) # [1, 32, 64, 3001]
        x = self.relu(x)  # 
        x = self.bn1(x)

        x = self.layer1(x) # [1, 32, 64, 3001]
        x = self.layer2(x) # [1, 64, 32, 1501]
        x = self.layer3(x) # [1, 128, 16, 751]
        x = self.layer4(x) # [1, 256, 8, 376]

        x = x.reshape(x.size()[0], -1, x.size()[-1]) # [1, 2048, 376]

        w = self.attention(x) # [1, 2048, 376]

        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP": # Attentive Statistics Pooling
            mu = torch.sum(x * w, dim=2) # [1, 2048]
            sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5)) # [1, 2048]
            x = torch.cat((mu, sg), 1) # [1, 4096]

        x = x.view(x.size()[0], -1) # [1, 4096]
        x = self.fc(x)   # [1, 512]

        if l2_norm:
            x = F.normalize(x, p=2, dim=1) # [1, 512]
        return x

    def load_checkpoint(
        self,
        checkpoint_path: str,
        eval: bool = False,
        use_cuda: bool = False,
        criterion=None,
        cache=False,
    ):
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
        try:
            self.load_state_dict(state["model"])
            print(" > Model fully restored. ")
        except (KeyError, RuntimeError) as error:
            # If eval raise the error
            if eval:
                raise error

            print(" > Partial model initialization.")
            model_dict = self.state_dict()
            model_dict = set_init_dict(model_dict, state["model"])
            self.load_state_dict(model_dict)
            del model_dict

        # load the criterion for restore_path
        if criterion is not None and "criterion" in state:
            try:
                criterion.load_state_dict(state["criterion"])
            except (KeyError, RuntimeError) as error:
                print(" > Criterion load ignored because of:", error)

        if use_cuda:
            self.cuda()
            if criterion is not None:
                criterion = criterion.cuda()

        if eval:
            self.eval()
            assert not self.training

        if not eval:
            return criterion, state["step"]
        return criterion


class HifiDecoder(torch.nn.Module):
    def __init__(
        self,
        input_sample_rate=22050,
        output_sample_rate=24000,
        output_hop_length=256,
        ar_mel_length_compression=1024,
        decoder_input_dim=1024,
        resblock_type_decoder="1",
        resblock_dilation_sizes_decoder=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        resblock_kernel_sizes_decoder=[3, 7, 11],
        upsample_rates_decoder=[8, 8, 2, 2],
        upsample_initial_channel_decoder=512,
        upsample_kernel_sizes_decoder=[16, 16, 4, 4],
        d_vector_dim=512,
        cond_d_vector_in_each_upsampling_layer=True,
        speaker_encoder_audio_config={
            "fft_size": 512,
            "win_length": 400,
            "hop_length": 160,
            "sample_rate": 16000,
            "preemphasis": 0.97,
            "num_mels": 64,
        },
    ):
        super().__init__()
        self.input_sample_rate = input_sample_rate     # 22050
        self.output_sample_rate = output_sample_rate   # 24000
        self.output_hop_length = output_hop_length     # 256
        self.ar_mel_length_compression = ar_mel_length_compression # 1024
        self.speaker_encoder_audio_config = speaker_encoder_audio_config
        self.waveform_decoder = HifiganGenerator(
            decoder_input_dim,
            1,
            resblock_type_decoder,
            resblock_dilation_sizes_decoder,
            resblock_kernel_sizes_decoder,
            upsample_kernel_sizes_decoder,
            upsample_initial_channel_decoder,
            upsample_rates_decoder,
            inference_padding=0,
            cond_channels=d_vector_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
            cond_in_each_up_layer=cond_d_vector_in_each_upsampling_layer,
        )
        self.speaker_encoder = ResNetSpeakerEncoder(
            input_dim=64,
            proj_dim=512,
            log_input=True,
            use_torch_spec=True,
            audio_config=speaker_encoder_audio_config,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, latents, g=None):
        """
        Args:
            x (Tensor): feature input tensor (GPT latent).
            g (Tensor): global conditioning input tensor.

        Returns:
            Tensor: output waveform.

        Shapes:
            x: [B, C, T]
            Tensor: [B, 1, T]
        """

        z = F.interpolate(
            latents.transpose(1, 2),
            scale_factor=[self.ar_mel_length_compression / self.output_hop_length], # 4
            mode="linear",
        ).squeeze(1)        # [1, 1024, 152]
        # upsample to the right sr
        if self.output_sample_rate != self.input_sample_rate:
            z = torch.nn.functional.interpolate(
                z,
                scale_factor=[self.output_sample_rate / self.input_sample_rate],
                mode="linear",
            ).squeeze(0)    # [1024, 165]
        o = self.waveform_decoder(z, g=g)
        return o

    @torch.no_grad()
    def inference(self, c, g):
        """
        Args:
            x (Tensor): feature input tensor (GPT latent).
            g (Tensor): global conditioning input tensor.

        Returns:
            Tensor: output waveform.

        Shapes:
            x: [B, C, T]
            Tensor: [B, 1, T]
        """
        return self.forward(c, g=g)

    def load_checkpoint(self, checkpoint_path, eval=False):  # pylint: disable=unused-argument, redefined-builtin
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"))
        # remove unused keys
        state = state["model"]
        states_keys = list(state.keys())
        for key in states_keys:
            if "waveform_decoder." not in key and "speaker_encoder." not in key:
                del state[key]

        self.load_state_dict(state)
        if eval:
            self.eval()
            assert not self.training
            self.waveform_decoder.remove_weight_norm()

from .tts_stream_gen import *
from .tts_tokenizer import *

#--------------------------TTS.tts.layers.xtts.xtts_manager.py-----------------------------------
class SpeakerManager():
    def __init__(self, speaker_file_path=None):
        self.speakers = torch.load(speaker_file_path)
    
    @property
    def name_to_id(self):
        return self.speakers.keys()
    
    @property
    def num_speakers(self):
        return len(self.name_to_id)
    
    @property
    def speaker_names(self):
        return list(self.name_to_id.keys())



class LanguageManager():
    def __init__(self, config):
        self.langs = config["languages"]
    
    @property
    def name_to_id(self):
        return self.langs
    
    @property
    def num_languages(self):
        return len(self.name_to_id)
    
    @property
    def language_names(self):
        return list(self.name_to_id)

from .base_tts import *

import librosa

init_stream_support()

def wav_to_mel_cloning(
    wav,
    mel_norms_file="../experiments/clips_mel_norms.pth",
    mel_norms=None,
    device=torch.device("cpu"),
    n_fft=4096,
    hop_length=1024,
    win_length=4096,
    power=2,
    normalized=False,
    sample_rate=22050,
    f_min=0,
    f_max=8000,
    n_mels=80,
):
    """
    Convert waveform to mel-spectrogram with hard-coded parameters for cloning.

    Args:
        wav (torch.Tensor): Input waveform tensor.
        mel_norms_file (str): Path to mel-spectrogram normalization file.
        mel_norms (torch.Tensor): Mel-spectrogram normalization tensor.
        device (torch.device): Device to use for computation.

    Returns:
        torch.Tensor: Mel-spectrogram tensor.
    """
    mel_stft = torchaudio.transforms.MelSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=power,
        normalized=normalized,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        norm="slaney",
    ).to(device)
    wav = wav.to(device)
    mel = mel_stft(wav)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    if mel_norms is None:
        mel_norms = torch.load(mel_norms_file, map_location=device)
    mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)
    return mel


def load_audio(audiopath, sampling_rate):
    # better load setting following: https://github.com/faroit/python_audio_loading_benchmark

    # torchaudio should chose proper backend to load audio depending on platform
    audio, lsr = torchaudio.load(audiopath)

    # stereo to mono if needed
    if audio.size(0) != 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    # Check some assumptions about audio range. This should be automatically fixed in load_wav_to_torch, but might not be in some edge cases, where we should squawk.
    # '10' is arbitrarily chosen since it seems like audio will often "overdrive" the [-1,1] bounds.
    if torch.any(audio > 10) or not torch.any(audio < 0):
        print(f"Error with {audiopath}. Max={audio.max()} min={audio.min()}")
    # clip audio invalid values
    audio.clip_(-1, 1)
    return audio


def pad_or_truncate(t, length):
    """
    Ensure a given tensor t has a specified sequence length by either padding it with zeros or clipping it.

    Args:
        t (torch.Tensor): The input tensor to be padded or truncated.
        length (int): The desired length of the tensor.

    Returns:
        torch.Tensor: The padded or truncated tensor.
    """
    tp = t[..., :length]
    if t.shape[-1] == length:
        tp = t
    elif t.shape[-1] < length:
        tp = F.pad(t, (0, length - t.shape[-1]))
    return tp


class Xtts(BaseTTS):
    """ⓍTTS model implementation.

    ❗ Currently it only supports inference.

    Examples:
        >>> from TTS.tts.configs.xtts_config import XttsConfig
        >>> from TTS.tts.models.xtts import Xtts
        >>> config = XttsConfig()
        >>> model = Xtts.inif_from_config(config)
        >>> model.load_checkpoint(config, checkpoint_dir="paths/to/models_dir/", eval=True)
    """

    def __init__(self, config: Coqpit):
        super().__init__(config, ap=None, tokenizer=None)
        self.mel_stats_path = None
        self.config = config
        self.gpt_checkpoint = self.args.gpt_checkpoint
        self.decoder_checkpoint = self.args.decoder_checkpoint  # TODO: check if this is even needed
        self.models_dir = config.model_dir
        self.gpt_batch_size = self.args.gpt_batch_size

        self.tokenizer = VoiceBpeTokenizer()
        self.gpt = None
        # self.init_models()
        self.register_buffer("mel_stats", torch.ones(80))
        self.gpt_cond_latent = None
        self.speaker_embedding = None
        self.gpt_cond_latent_count = 0

    def init_models(self):
        """Initialize the models. We do it here since we need to load the tokenizer first."""
        if self.tokenizer.tokenizer is not None:
            self.args.gpt_number_text_tokens = self.tokenizer.get_number_tokens() # 6681
            self.args.gpt_start_text_token = self.tokenizer.tokenizer.token_to_id("[START]") # 261
            self.args.gpt_stop_text_token = self.tokenizer.tokenizer.token_to_id("[STOP]")   # 0

        if self.args.gpt_number_text_tokens:
            self.gpt = GPT(
                layers=self.args.gpt_layers,
                model_dim=self.args.gpt_n_model_channels,
                start_text_token=self.args.gpt_start_text_token,
                stop_text_token=self.args.gpt_stop_text_token,
                heads=self.args.gpt_n_heads,
                max_text_tokens=self.args.gpt_max_text_tokens,
                max_mel_tokens=self.args.gpt_max_audio_tokens,
                max_prompt_tokens=self.args.gpt_max_prompt_tokens,
                number_text_tokens=self.args.gpt_number_text_tokens,
                num_audio_tokens=self.args.gpt_num_audio_tokens,
                start_audio_token=self.args.gpt_start_audio_token,
                stop_audio_token=self.args.gpt_stop_audio_token,
                use_perceiver_resampler=self.args.gpt_use_perceiver_resampler,
                code_stride_len=self.args.gpt_code_stride_len,
            )

        self.hifigan_decoder = HifiDecoder(
            input_sample_rate=self.args.input_sample_rate,   # 22050
            output_sample_rate=self.args.output_sample_rate, # 24000
            output_hop_length=self.args.output_hop_length,   # 256
            ar_mel_length_compression=self.args.gpt_code_stride_len, # 1024
            decoder_input_dim=self.args.decoder_input_dim,   # 1024
            d_vector_dim=self.args.d_vector_dim,             # 512
            cond_d_vector_in_each_upsampling_layer=self.args.cond_d_vector_in_each_upsampling_layer, # True
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def get_gpt_cond_latents(self, audio, sr, length: int = 30, chunk_length: int = 6):
        """Compute the conditioning latents for the GPT model from the given audio.

        Args:
            audio (tensor): audio tensor.
            sr (int): Sample rate of the audio.
            length (int): Length of the audio in seconds. If < 0, use the whole audio. Defaults to 30.
            chunk_length (int): Length of the audio chunks in seconds. When `length == chunk_length`, the whole audio
                is being used without chunking. It must be < `length`. Defaults to 6.
        """
        if sr != 22050:
            audio = torchaudio.functional.resample(audio, sr, 22050)
        if length > 0:
            audio = audio[:, : 22050 * length]
        if self.args.gpt_use_perceiver_resampler:
            style_embs = []
            for i in range(0, audio.shape[1], 22050 * chunk_length):
                audio_chunk = audio[:, i : i + 22050 * chunk_length]

                # if the chunk is too short ignore it 
                if audio_chunk.size(-1) < 22050 * 0.33:
                    continue

                mel_chunk = wav_to_mel_cloning(
                    audio_chunk,
                    mel_norms=self.mel_stats.cpu(),
                    n_fft=2048,
                    hop_length=256,
                    win_length=1024,
                    power=2,
                    normalized=False,
                    sample_rate=22050,
                    f_min=0,
                    f_max=8000,
                    n_mels=80,
                )               # [1, 80, 517]
                style_emb = self.gpt.get_style_emb(mel_chunk.to(self.device), None)
                style_embs.append(style_emb)

            # mean style embedding
            cond_latent = torch.stack(style_embs).mean(dim=0)
        else:
            mel = wav_to_mel_cloning(
                audio,
                mel_norms=self.mel_stats.cpu(),
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                power=2,
                normalized=False,
                sample_rate=22050,
                f_min=0,
                f_max=8000,
                n_mels=80,
            )
            cond_latent = self.gpt.get_style_emb(mel.to(self.device)) # [1, 1024, 32]
        return cond_latent.transpose(1, 2) # [1, 32, 1024]

    @torch.inference_mode()
    def get_speaker_embedding(self, audio, sr):
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        return (
            self.hifigan_decoder.speaker_encoder.forward(audio_16k.to(self.device), l2_norm=True)
            .unsqueeze(-1)
            .to(self.device)
        )

    @torch.inference_mode()
    def get_conditioning_latents(
        self,
        audio_path,
        max_ref_length=30,
        gpt_cond_len=6,
        gpt_cond_chunk_len=6,
        librosa_trim_db=None,
        sound_norm_refs=False,
        load_sr=22050,
    ):
        """Get the conditioning latents for the GPT model from the given audio.

        Args:
            audio_path (str or List[str]): Path to reference audio file(s).
            max_ref_length (int): Maximum length of each reference audio in seconds. Defaults to 30.
            gpt_cond_len (int): Length of the audio used for gpt latents. Defaults to 6.
            gpt_cond_chunk_len (int): Chunk length used for gpt latents. It must be <= gpt_conf_len. Defaults to 6.
            librosa_trim_db (int, optional): Trim the audio using this value. If None, not trimming. Defaults to None.
            sound_norm_refs (bool, optional): Whether to normalize the audio. Defaults to False.
            load_sr (int, optional): Sample rate to load the audio. Defaults to 24000.
        """
        # deal with multiples references
        if not isinstance(audio_path, list):
            audio_paths = [audio_path]
        else:
            audio_paths = audio_path

        speaker_embeddings = []
        audios = []
        speaker_embedding = None
        for file_path in audio_paths:
            audio = load_audio(file_path, load_sr)
            audio = audio[:, : load_sr * max_ref_length].to(self.device)
            if sound_norm_refs:
                audio = (audio / torch.abs(audio).max()) * 0.75
            if librosa_trim_db is not None:
                audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0]

            # compute latents for the decoder
            speaker_embedding = self.get_speaker_embedding(audio, load_sr) # [1, 512, 1]
            speaker_embeddings.append(speaker_embedding)

            audios.append(audio)

        # merge all the audios and compute the latents for the gpt
        full_audio = torch.cat(audios, dim=-1)
        gpt_cond_latents = self.get_gpt_cond_latents(
            full_audio, load_sr, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len
        )  # [1, 32, 1024]

        if speaker_embeddings:
            speaker_embedding = torch.stack(speaker_embeddings)
            speaker_embedding = speaker_embedding.mean(dim=0)

        return gpt_cond_latents, speaker_embedding

    def synthesize(self, text, config, speaker_wav, language, speaker_id=None, **kwargs):
        """Synthesize speech with the given input text.

        Args:
            text (str): Input text.
            config (XttsConfig): Config with inference parameters.
            speaker_wav (list): List of paths to the speaker audio files to be used for cloning.
            language (str): Language ID of the speaker.
            **kwargs: Inference settings. See `inference()`.

        Returns:
            A dictionary of the output values with `wav` as output waveform, `deterministic_seed` as seed used at inference,
            `text_input` as text token IDs after tokenizer, `voice_samples` as samples used for cloning, `conditioning_latents`
            as latents used at inference.

        """
        assert (
            "zh-cn" if language == "zh" else language in self.config.languages
        ), f" ❗ Language {language} is not supported. Supported languages are {self.config.languages}"
        # Use generally found best tuning knobs for generation.
        settings = {
            "temperature": config.temperature,
            "length_penalty": config.length_penalty,
            "repetition_penalty": config.repetition_penalty,
            "top_k": config.top_k,
            "top_p": config.top_p,
        }
        settings.update(kwargs)  # allow overriding of preset settings with kwargs
        if speaker_id is not None:
            gpt_cond_latent, speaker_embedding = self.speaker_manager.speakers[speaker_id].values()
            return self.inference(text, language, gpt_cond_latent, speaker_embedding, **settings)
        settings.update({
            "gpt_cond_len": config.gpt_cond_len,
            "gpt_cond_chunk_len": config.gpt_cond_chunk_len,
            "max_ref_len": config.max_ref_len,
            "sound_norm_refs": config.sound_norm_refs,
        })
        return self.full_inference(text, speaker_wav, language, **settings)

    @torch.inference_mode()
    def full_inference(
        self,
        text,
        ref_audio_path,
        language,
        # GPT inference
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        speed=1.0,
        # Cloning
        gpt_cond_len=30,
        gpt_cond_chunk_len=6,
        max_ref_len=10,
        sound_norm_refs=False,
        **hf_generate_kwargs,
    ):
        """
        This function produces an audio clip of the given text being spoken with the given reference voice.

        Args:
            text: (str) Text to be spoken.

            ref_audio_path: (str) Path to a reference audio file to be used for cloning. This audio file should be >3
                seconds long.

            language: (str) Language of the voice to be generated.

            temperature: (float) The softmax temperature of the autoregressive model. Defaults to 0.65.

            length_penalty: (float) A length penalty applied to the autoregressive decoder. Higher settings causes the
                model to produce more terse outputs. Defaults to 1.0.

            repetition_penalty: (float) A penalty that prevents the autoregressive decoder from repeating itself during
                decoding. Can be used to reduce the incidence of long silences or "uhhhhhhs", etc. Defaults to 2.0.

            top_k: (int) K value used in top-k sampling. [0,inf]. Lower values mean the decoder produces more "likely"
                (aka boring) outputs. Defaults to 50.

            top_p: (float) P value used in nucleus sampling. (0,1]. Lower values mean the decoder produces more "likely"
                (aka boring) outputs. Defaults to 0.8.

            gpt_cond_len: (int) Length of the audio used for cloning. If audio is shorter, then audio length is used
                else the first `gpt_cond_len` secs is used. Defaults to 30 seconds.

            gpt_cond_chunk_len: (int) Chunk length used for cloning. It must be <= `gpt_cond_len`.
                If gpt_cond_len == gpt_cond_chunk_len, no chunking. Defaults to 6 seconds.

            hf_generate_kwargs: (**kwargs) The huggingface Transformers generate API is used for the autoregressive
                transformer. Extra keyword args fed to this function get forwarded directly to that API. Documentation
                here: https://huggingface.co/docs/transformers/internal/generation_utils

        Returns:
            Generated audio clip(s) as a torch tensor. Shape 1,S if k=1 else, (k,1,S) where S is the sample length.
            Sample rate is 24kHz.
        """
        if self.gpt_cond_latent_count == 0:
            (gpt_cond_latent, speaker_embedding) = self.get_conditioning_latents(
                audio_path=ref_audio_path,
                gpt_cond_len=gpt_cond_len,
                gpt_cond_chunk_len=gpt_cond_chunk_len,
                max_ref_length=max_ref_len,
                sound_norm_refs=sound_norm_refs,
            )
            self.gpt_cond_latent = gpt_cond_latent
            self.speaker_embedding = speaker_embedding
            self.gpt_cond_latent_count += 1

        return self.inference(
            text,
            language,
            self.gpt_cond_latent,
            self.speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            speed=speed,
            **hf_generate_kwargs,
        )

    @torch.inference_mode()
    def inference(
        self,
        text,               # ['Hi, how are you?']
        language,           # 'en'
        gpt_cond_latent,    # [1, 32, 1024]
        speaker_embedding,  # [1, 512, 1]
        # GPT inference
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        num_beams=1,
        speed=1.0,
        enable_text_splitting=True,
        **hf_generate_kwargs,
    ):
        language = language.split("-")[0]  # remove the country code
        length_scale = 1.0 / max(speed, 0.05)
        gpt_cond_latent = gpt_cond_latent.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)
        if enable_text_splitting:
            text = split_sentence(text, language, self.tokenizer.char_limits[language])
        else:
            text = [text]

        wavs = []
        gpt_latents_list = []
        for sent in text:
            sent = sent.strip().lower()
            text_tokens = torch.IntTensor(self.tokenizer.encode(sent, lang=language)).unsqueeze(0).to(self.device)

            assert (
                text_tokens.shape[-1] < self.args.gpt_max_text_tokens
            ), " ❗ XTTS can only generate text with a maximum of 400 tokens."

            with torch.no_grad():
                gpt_codes = self.gpt.generate(
                    cond_latents=gpt_cond_latent,
                    text_inputs=text_tokens,
                    input_tokens=None,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=self.gpt_batch_size,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    output_attentions=False,
                    **hf_generate_kwargs,
                )
                expected_output_len = torch.tensor(
                    [gpt_codes.shape[-1] * self.gpt.code_stride_len], device=text_tokens.device
                )

                text_len = torch.tensor([text_tokens.shape[-1]], device=self.device)
                gpt_latents = self.gpt(
                    text_tokens,
                    text_len,
                    gpt_codes,
                    expected_output_len,
                    cond_latents=gpt_cond_latent,
                    return_attentions=False,
                    return_latent=True,
                ) # [1, 38, 1024]

                if length_scale != 1.0:
                    gpt_latents = F.interpolate(
                        gpt_latents.transpose(1, 2), scale_factor=length_scale, mode="linear"
                    ).transpose(1, 2)

                gpt_latents_list.append(gpt_latents.cpu())
                wavs.append(self.hifigan_decoder(gpt_latents, g=speaker_embedding).cpu().squeeze())

        return {
            "wav": torch.cat(wavs, dim=0).numpy(),
            "gpt_latents": torch.cat(gpt_latents_list, dim=1).numpy(),
            "speaker_embedding": speaker_embedding,
        }

    def handle_chunks(self, wav_gen, wav_gen_prev, wav_overlap, overlap_len):
        """Handle chunk formatting in streaming mode"""
        wav_chunk = wav_gen[:-overlap_len]
        if wav_gen_prev is not None:
            wav_chunk = wav_gen[(wav_gen_prev.shape[0] - overlap_len) : -overlap_len]
        if wav_overlap is not None:
            # cross fade the overlap section
            if overlap_len > len(wav_chunk):
                # wav_chunk is smaller than overlap_len, pass on last wav_gen
                if wav_gen_prev is not None:
                    wav_chunk = wav_gen[(wav_gen_prev.shape[0] - overlap_len) :]
                else:
                    # not expecting will hit here as problem happens on last chunk
                    wav_chunk = wav_gen[-overlap_len:]
                return wav_chunk, wav_gen, None
            else:
                crossfade_wav = wav_chunk[:overlap_len]
                crossfade_wav = crossfade_wav * torch.linspace(0.0, 1.0, overlap_len).to(crossfade_wav.device)
                wav_chunk[:overlap_len] = wav_overlap * torch.linspace(1.0, 0.0, overlap_len).to(wav_overlap.device)
                wav_chunk[:overlap_len] += crossfade_wav

        wav_overlap = wav_gen[-overlap_len:]
        wav_gen_prev = wav_gen
        return wav_chunk, wav_gen_prev, wav_overlap

    @torch.inference_mode()
    def inference_stream(
        self,
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        # Streaming
        stream_chunk_size=20,
        overlap_wav_len=1024,
        # GPT inference
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        speed=1.0,
        enable_text_splitting=False,
        **hf_generate_kwargs,
    ):
        language = language.split("-")[0]  # remove the country code
        length_scale = 1.0 / max(speed, 0.05)
        gpt_cond_latent = gpt_cond_latent.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)
        if enable_text_splitting:
            text = split_sentence(text, language, self.tokenizer.char_limits[language])
        else:
            text = [text]

        for sent in text:
            sent = sent.strip().lower()
            text_tokens = torch.IntTensor(self.tokenizer.encode(sent, lang=language)).unsqueeze(0).to(self.device)

            assert (
                text_tokens.shape[-1] < self.args.gpt_max_text_tokens
            ), " ❗ XTTS can only generate text with a maximum of 400 tokens."

            fake_inputs = self.gpt.compute_embeddings(
                gpt_cond_latent.to(self.device),
                text_tokens,
            )
            gpt_generator = self.gpt.get_generator(
                fake_inputs=fake_inputs,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                do_sample=do_sample,
                num_beams=1,
                num_return_sequences=1,
                length_penalty=float(length_penalty),
                repetition_penalty=float(repetition_penalty),
                output_attentions=False,
                output_hidden_states=True,
                **hf_generate_kwargs,
            )

            last_tokens = []
            all_latents = []
            wav_gen_prev = None
            wav_overlap = None
            is_end = False

            while not is_end:
                try:
                    x, latent = next(gpt_generator)
                    last_tokens += [x]
                    all_latents += [latent]
                except StopIteration:
                    is_end = True

                if is_end or (stream_chunk_size > 0 and len(last_tokens) >= stream_chunk_size):
                    gpt_latents = torch.cat(all_latents, dim=0)[None, :]
                    if length_scale != 1.0:
                        gpt_latents = F.interpolate(
                            gpt_latents.transpose(1, 2), scale_factor=length_scale, mode="linear"
                        ).transpose(1, 2)
                    wav_gen = self.hifigan_decoder(gpt_latents, g=speaker_embedding.to(self.device))
                    wav_chunk, wav_gen_prev, wav_overlap = self.handle_chunks(
                        wav_gen.squeeze(), wav_gen_prev, wav_overlap, overlap_wav_len
                    )
                    last_tokens = []
                    yield wav_chunk

    def forward(self):
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://tts.readthedocs.io/en/dev/models/xtts.html#training"
        )

    def eval_step(self):
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://tts.readthedocs.io/en/dev/models/xtts.html#training"
        )

    @staticmethod
    def init_from_config(config: "XttsConfig", **kwargs):  # pylint: disable=unused-argument
        return Xtts(config)

    def eval(self):  # pylint: disable=redefined-builtin
        """Sets the model to evaluation mode. Overrides the default eval() method to also set the GPT model to eval mode."""
        self.gpt.init_gpt_for_inference()
        super().eval()

    def get_compatible_checkpoint_state_dict(self, model_path):
        checkpoint = load_fsspec(model_path, map_location=torch.device("cpu"))["model"]
        # remove xtts gpt trainer extra keys
        ignore_keys = ["torch_mel_spectrogram_style_encoder", "torch_mel_spectrogram_dvae", "dvae"]
        
        for key in list(checkpoint.keys()):
            # check if it is from the coqui Trainer if so convert it
            if key.startswith("xtts."):
                new_key = key.replace("xtts.", "")
                checkpoint[new_key] = checkpoint[key]
                del checkpoint[key]
                key = new_key

            # remove unused keys
            if key.split(".")[0] in ignore_keys:
                del checkpoint[key]

        return checkpoint

    def load_checkpoint(
        self,
        config,
        checkpoint_dir=None,
        checkpoint_path=None,
        vocab_path=None,
        eval=True,
        strict=False,
        use_deepspeed=False,
        speaker_file_path=None,
    ):
        """
        Loads a checkpoint from disk and initializes the model's state and tokenizer.

        Args:
            config (dict): The configuration dictionary for the model.
            checkpoint_dir (str, optional): The directory where the checkpoint is stored. Defaults to None.
            checkpoint_path (str, optional): The path to the checkpoint file. Defaults to None.
            vocab_path (str, optional): The path to the vocabulary file. Defaults to None.
            eval (bool, optional): Whether to set the model to evaluation mode. Defaults to True.
            strict (bool, optional): Whether to strictly enforce that the keys in the checkpoint match the keys in the model. Defaults to True.

        Returns:BaseTTS
            None
        """

        model_path = checkpoint_path or os.path.join(checkpoint_dir, "model.pth")
        vocab_path = vocab_path or os.path.join(checkpoint_dir, "vocab.json")
        speaker_file_path = speaker_file_path or os.path.join(checkpoint_dir, "speakers_xtts.pth")

        self.language_manager = LanguageManager(config)
        self.speaker_manager = None
        if os.path.exists(speaker_file_path):
            self.speaker_manager = SpeakerManager(speaker_file_path)

        if os.path.exists(vocab_path):
            self.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)

        self.init_models()

        checkpoint = self.get_compatible_checkpoint_state_dict(model_path)

        # deal with v1 and v1.1. V1 has the init_gpt_for_inference keys, v1.1 do not
        try:
            self.load_state_dict(checkpoint, strict=strict)
            # del checkpoint -> No effect on memory! DEBUG
            # print("Deleted checkpoint")
        except:
            if eval:
                self.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache)
            self.load_state_dict(checkpoint, strict=strict)

        if eval:
            self.hifigan_decoder.eval()
            self.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=use_deepspeed)
            self.gpt.eval()

    def train_step(self):
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://tts.readthedocs.io/en/dev/models/xtts.html#training"
        )

#--------------------------TTS.utils.synthesizer.py---------------------------------------------
import time
import pysbd
import json
# from TTS.tts.configs.vits_config import VitsConfig
# from TTS.tts.models.vits import Vits

# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import
# from TTS.tts.utils.synthesis import synthesis, transfer_voice, trim_silence
# from TTS.utils.audio import AudioProcessor
# from TTS.utils.audio.numpy_transforms import save_wav
# from TTS.vc.models import setup_model as setup_vc_model
# from TTS.vocoder.models import setup_model as setup_vocoder_model
# from TTS.vocoder.utils.generic_utils import interpolate_vocoder_input
import scipy
from io import BytesIO
import subprocess

def save_wav(*, wav: np.ndarray, path: str, sample_rate: int = None, pipe_out=None, **kwargs) -> None:
    """Save float waveform to a file using Scipy.

    Args:
        wav (np.ndarray): Waveform with float values in range [-1, 1] to save.
        path (str): Path to a output file.
        sr (int, optional): Sampling rate used for saving to the file. Defaults to None.
        pipe_out (BytesIO, optional): Flag to stdout the generated TTS wav file for shell pipe.
    """
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))

    wav_norm = wav_norm.astype(np.int16)
    if pipe_out:
        wav_buffer = BytesIO()
        scipy.io.wavfile.write(wav_buffer, sample_rate, wav_norm)
        wav_buffer.seek(0)
        pipe_out.buffer.write(wav_buffer.read())
    scipy.io.wavfile.write(path, sample_rate, wav_norm)

def save_mp3(*, wav: np.ndarray, path: str, sample_rate: int = None, **kwargs) -> None:
    """Save float waveform to a file using Scipy.
    Args:
        wav (np.ndarray): Waveform with float values in range [-1, 1] to save.
        path (str): Path to a output file.
        sr (int, optional): Sampling rate used for saving to the file. Defaults to None.
    """
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wav_norm = wav_norm.astype(np.int16)
    wav_bytes = wav_norm.tobytes()
    
    cmd = ["ffmpeg", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i","pipe:0", "-qscale:a", "0" , path]
    
    try:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        process.stdin.write(wav_bytes)
        process.stdin.close()
        process.wait()
    except subprocess.CalledProcessError as e:
        print(e)

def load_config(config_path):
    config_dict = {}
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    config_dict.update(data)
    model_name = config_dict["model"]
    if model_name == "xtts":
        config_class = XttsConfig
    config = config_class()
    config.from_dict(config_dict)
    return config


class Synthesizer(nn.Module):
    def __init__(
        self,
        tts_checkpoint: str = "",
        tts_config_path: str = "",
        tts_speakers_file: str = "",
        tts_languages_file: str = "",
        vocoder_checkpoint: str = "",
        vocoder_config: str = "",
        encoder_checkpoint: str = "",
        encoder_config: str = "",
        vc_checkpoint: str = "",
        vc_config: str = "",
        model_dir: str = "",
        voice_dir: str = None,
        use_cuda: bool = False,
    ) -> None:
        """General 🐸 TTS interface for inference. It takes a tts and a vocoder
        model and synthesize speech from the provided text.

        The text is divided into a list of sentences using `pysbd` and synthesize
        speech on each sentence separately.

        If you have certain special characters in your text, you need to handle
        them before providing the text to Synthesizer.

        TODO: set the segmenter based on the source language

        Args:
            tts_checkpoint (str, optional): path to the tts model file.
            tts_config_path (str, optional): path to the tts config file.
            vocoder_checkpoint (str, optional): path to the vocoder model file. Defaults to None.
            vocoder_config (str, optional): path to the vocoder config file. Defaults to None.
            encoder_checkpoint (str, optional): path to the speaker encoder model file. Defaults to `""`,
            encoder_config (str, optional): path to the speaker encoder config file. Defaults to `""`,
            vc_checkpoint (str, optional): path to the voice conversion model file. Defaults to `""`,
            vc_config (str, optional): path to the voice conversion config file. Defaults to `""`,
            use_cuda (bool, optional): enable/disable cuda. Defaults to False.
        """
        super().__init__()
        self.tts_checkpoint = tts_checkpoint
        self.tts_config_path = tts_config_path
        self.tts_speakers_file = tts_speakers_file
        self.tts_languages_file = tts_languages_file
        self.vocoder_checkpoint = vocoder_checkpoint
        self.vocoder_config = vocoder_config
        self.encoder_checkpoint = encoder_checkpoint
        self.encoder_config = encoder_config
        self.vc_checkpoint = vc_checkpoint
        self.vc_config = vc_config
        self.use_cuda = use_cuda

        self.tts_model = None
        self.vocoder_model = None
        self.vc_model = None
        self.speaker_manager = None
        self.tts_speakers = {}
        self.language_manager = None
        self.num_languages = 0
        self.tts_languages = {}
        self.d_vector_dim = 0
        self.seg = self._get_segmenter("en")
        self.use_cuda = use_cuda
        self.voice_dir = voice_dir
        if self.use_cuda:
            assert torch.cuda.is_available(), "CUDA is not availabe on this machine."

        if tts_checkpoint:
            self._load_tts(tts_checkpoint, tts_config_path, use_cuda)
            self.output_sample_rate = self.tts_config.audio["sample_rate"]

        # if vocoder_checkpoint:
        #     self._load_vocoder(vocoder_checkpoint, vocoder_config, use_cuda)
        #     self.output_sample_rate = self.vocoder_config.audio["sample_rate"]

        # if vc_checkpoint:
        #     self._load_vc(vc_checkpoint, vc_config, use_cuda)
        #     self.output_sample_rate = self.vc_config.audio["output_sample_rate"]

        # if model_dir:
        #     if "fairseq" in model_dir:
        #         self._load_fairseq_from_dir(model_dir, use_cuda)
        #         self.output_sample_rate = self.tts_config.audio["sample_rate"]
        #     else:
        #         self._load_tts_from_dir(model_dir, use_cuda)
        #         self.output_sample_rate = self.tts_config.audio["output_sample_rate"]

    @staticmethod
    def _get_segmenter(lang: str):
        """get the sentence segmenter for the given language.

        Args:
            lang (str): target language code.

        Returns:
            [type]: [description]
        """
        return pysbd.Segmenter(language=lang, clean=True)

    # def _load_vc(self, vc_checkpoint: str, vc_config_path: str, use_cuda: bool) -> None:
    #     """Load the voice conversion model.

    #     1. Load the model config.
    #     2. Init the model from the config.
    #     3. Load the model weights.
    #     4. Move the model to the GPU if CUDA is enabled.

    #     Args:
    #         vc_checkpoint (str): path to the model checkpoint.
    #         tts_config_path (str): path to the model config file.
    #         use_cuda (bool): enable/disable CUDA use.
    #     """
    #     # pylint: disable=global-statement
    #     self.vc_config = load_config(vc_config_path)
    #     self.vc_model = setup_vc_model(config=self.vc_config)
    #     self.vc_model.load_checkpoint(self.vc_config, vc_checkpoint)
    #     if use_cuda:
    #        self.vc_model.cuda()

    # def _load_fairseq_from_dir(self, model_dir: str, use_cuda: bool) -> None:
    #     """Load the fairseq model from a directory.

    #     We assume it is VITS and the model knows how to load itself from the directory and there is a config.json file in the directory.
    #     """
    #     self.tts_config = VitsConfig()
    #     self.tts_model = Vits.init_from_config(self.tts_config)
    #     self.tts_model.load_fairseq_checkpoint(self.tts_config, checkpoint_dir=model_dir, eval=True)
    #     self.tts_config = self.tts_model.config
    #     if use_cuda:
    #         self.tts_model.cuda()

    # def _load_tts_from_dir(self, model_dir: str, use_cuda: bool) -> None:
    #     """Load the TTS model from a directory.

    #     We assume the model knows how to load itself from the directory and there is a config.json file in the directory.
    #     """
    #     config = load_config(os.path.join(model_dir, "config.json"))
    #     self.tts_config = config
    #     #self.tts_model = setup_tts_model(config)
    #     self.tts_model = Xtts.init_from_config(config=config)
    #     self.tts_model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True)
    #     if use_cuda:
    #         self.tts_model.cuda()

    def _load_tts(self, tts_checkpoint: str, tts_config_path: str, use_cuda: bool) -> None:
        """Load the TTS model.

        1. Load the model config.
        2. Init the model from the config.
        3. Load the model weights.
        4. Move the model to the GPU if CUDA is enabled.
        5. Init the speaker manager in the model.

        Args:
            tts_checkpoint (str): path to the model checkpoint.
            tts_config_path (str): path to the model config file.
            use_cuda (bool): enable/disable CUDA use.
        """
        # pylint: disable=global-statement
        self.tts_config = load_config(tts_config_path)
        if self.tts_config["use_phonemes"] and self.tts_config["phonemizer"] is None:
            raise ValueError("Phonemizer is not defined in the TTS config.")

        # self.tts_model = setup_tts_model(config=self.tts_config)
        self.tts_model = Xtts.init_from_config(config=self.tts_config)

        if not self.encoder_checkpoint:
            self._set_speaker_encoder_paths_from_tts_config()

        tts_dir, filename = os.path.split(tts_checkpoint)

        self.tts_model.load_checkpoint(self.tts_config, checkpoint_dir=tts_dir, eval=True)
        if use_cuda:
            self.tts_model.cuda()

        if self.encoder_checkpoint and hasattr(self.tts_model, "speaker_manager"):
            self.tts_model.speaker_manager.init_encoder(self.encoder_checkpoint, self.encoder_config, use_cuda)

    def _set_speaker_encoder_paths_from_tts_config(self):
        """Set the encoder paths from the tts model config for models with speaker encoders."""
        if hasattr(self.tts_config, "model_args") and hasattr(
            self.tts_config.model_args, "speaker_encoder_config_path"
        ):
            self.encoder_checkpoint = self.tts_config.model_args.speaker_encoder_model_path
            self.encoder_config = self.tts_config.model_args.speaker_encoder_config_path

    # def _load_vocoder(self, model_file: str, model_config: str, use_cuda: bool) -> None:
    #     """Load the vocoder model.

    #     1. Load the vocoder config.
    #     2. Init the AudioProcessor for the vocoder.
    #     3. Init the vocoder model from the config.
    #     4. Move the model to the GPU if CUDA is enabled.

    #     Args:
    #         model_file (str): path to the model checkpoint.
    #         model_config (str): path to the model config file.
    #         use_cuda (bool): enable/disable CUDA use.
    #     """
    #     self.vocoder_config = load_config(model_config)
    #     self.vocoder_ap = AudioProcessor(verbose=False, **self.vocoder_config.audio)
    #     self.vocoder_model = setup_vocoder_model(self.vocoder_config)
    #     self.vocoder_model.load_checkpoint(self.vocoder_config, model_file, eval=True)
    #     if use_cuda:
    #         self.vocoder_model.cuda()

    def split_into_sentences(self, text) -> List[str]:
        """Split give text into sentences.

        Args:
            text (str): input text in string format.

        Returns:
            List[str]: list of sentences.
        """
        return self.seg.segment(text)

    def save_wav(self, wav: List[int], path: str, pipe_out=None) -> None:
        """Save the waveform as a file.

        Args:
            wav (List[int]): waveform as a list of values.
            path (str): output path to save the waveform.
            pipe_out (BytesIO, optional): Flag to stdout the generated TTS wav file for shell pipe.
        """
        # if tensor convert to numpy
        if torch.is_tensor(wav):
            wav = wav.cpu().numpy()
        if isinstance(wav, list):
            wav = np.array(wav)
        save_wav(wav=wav, path=path, sample_rate=self.output_sample_rate, pipe_out=pipe_out)
        # save_mp3(wav=wav, path=path, sample_rate=self.output_sample_rate)

    # def voice_conversion(self, source_wav: str, target_wav: str) -> List[int]:
    #     output_wav = self.vc_model.voice_conversion(source_wav, target_wav)
    #     return output_wav

    def tts(
        self,
        text: str = "",
        speaker_name: str = "",
        language_name: str = "",
        speaker_wav=None,
        style_wav=None,
        style_text=None,
        reference_wav=None,
        reference_speaker_name=None,
        split_sentences: bool = True,
        speed: float = 1.0,
        **kwargs,
    ) -> List[int]:
        """🐸 TTS magic. Run all the models and generate speech.

        Args:
            text (str): input text.
            speaker_name (str, optional): speaker id for multi-speaker models. Defaults to "".
            language_name (str, optional): language id for multi-language models. Defaults to "".
            speaker_wav (Union[str, List[str]], optional): path to the speaker wav for voice cloning. Defaults to None.
            style_wav ([type], optional): style waveform for GST. Defaults to None.
            style_text ([type], optional): transcription of style_wav for Capacitron. Defaults to None.
            reference_wav ([type], optional): reference waveform for voice conversion. Defaults to None.
            reference_speaker_name ([type], optional): speaker id of reference waveform. Defaults to None.
            split_sentences (bool, optional): split the input text into sentences. Defaults to True.
            **kwargs: additional arguments to pass to the TTS model.
        Returns:
            List[int]: [description]
        """
        start_time = time.time()
        wavs = []

        if not text and not reference_wav:
            raise ValueError(
                "You need to define either `text` (for sythesis) or a `reference_wav` (for voice conversion) to use the Coqui TTS API."
            )

        if text:
            sens = [text]
            if split_sentences:
                #print(" > Text splitted to sentences.")
                sens = self.split_into_sentences(text)
            #print(sens)

        # handle multi-speaker
        if "voice_dir" in kwargs:
            self.voice_dir = kwargs["voice_dir"]
            kwargs.pop("voice_dir")
        speaker_embedding = None
        speaker_id = None
        if self.tts_speakers_file or hasattr(self.tts_model.speaker_manager, "name_to_id"):
            if speaker_name and isinstance(speaker_name, str) and not self.tts_config.model == "xtts":
                if self.tts_config.use_d_vector_file:
                    # get the average speaker embedding from the saved d_vectors.
                    speaker_embedding = self.tts_model.speaker_manager.get_mean_embedding(
                        speaker_name, num_samples=None, randomize=False
                    )
                    speaker_embedding = np.array(speaker_embedding)[None, :]  # [1 x embedding_dim]
                else:
                    # get speaker idx from the speaker name
                    speaker_id = self.tts_model.speaker_manager.name_to_id[speaker_name]
            # handle Neon models with single speaker.
            elif len(self.tts_model.speaker_manager.name_to_id) == 1:
                speaker_id = list(self.tts_model.speaker_manager.name_to_id.values())[0]
            elif not speaker_name and not speaker_wav:
                raise ValueError(
                    " [!] Looks like you are using a multi-speaker model. "
                    "You need to define either a `speaker_idx` or a `speaker_wav` to use a multi-speaker model."
                )
            else:
                speaker_embedding = None
        else:
            if speaker_name and self.voice_dir is None:
                raise ValueError(
                    f" [!] Missing speakers.json file path for selecting speaker {speaker_name}."
                    "Define path for speaker.json if it is a multi-speaker model or remove defined speaker idx. "
                )

        # handle multi-lingual
        language_id = None
        if self.tts_languages_file or (
            hasattr(self.tts_model, "language_manager") 
            and self.tts_model.language_manager is not None
            and not self.tts_config.model == "xtts"
        ):
            if len(self.tts_model.language_manager.name_to_id) == 1:
                language_id = list(self.tts_model.language_manager.name_to_id.values())[0]

            elif language_name and isinstance(language_name, str):
                try:
                    language_id = self.tts_model.language_manager.name_to_id[language_name]
                except KeyError as e:
                    raise ValueError(
                        f" [!] Looks like you use a multi-lingual model. "
                        f"Language {language_name} is not in the available languages: "
                        f"{self.tts_model.language_manager.name_to_id.keys()}."
                    ) from e

            elif not language_name:
                raise ValueError(
                    " [!] Look like you use a multi-lingual model. "
                    "You need to define either a `language_name` or a `style_wav` to use a multi-lingual model."
                )

            else:
                raise ValueError(
                    f" [!] Missing language_ids.json file path for selecting language {language_name}."
                    "Define path for language_ids.json if it is a multi-lingual model or remove defined language idx. "
                )

        # compute a new d_vector from the given clip.
        if (
            speaker_wav is not None
            and self.tts_model.speaker_manager is not None
            and hasattr(self.tts_model.speaker_manager, "encoder_ap")
            and self.tts_model.speaker_manager.encoder_ap is not None
        ):
            speaker_embedding = self.tts_model.speaker_manager.compute_embedding_from_clip(speaker_wav)

        vocoder_device = "cpu"
        use_gl = self.vocoder_model is None
        if not use_gl:
            vocoder_device = next(self.vocoder_model.parameters()).device
        if self.use_cuda:
            vocoder_device = "cuda"

        if not reference_wav:  # not voice conversion
            for sen in sens:
                if hasattr(self.tts_model, "synthesize"):
                    outputs = self.tts_model.synthesize(
                        text=sen,
                        config=self.tts_config,
                        speaker_id=speaker_name,
                        voice_dirs=self.voice_dir,
                        d_vector=speaker_embedding,
                        speaker_wav=speaker_wav,
                        language=language_name,
                        speed=speed,
                        **kwargs,
                    )
                # else:
                #     # synthesize voice
                #     outputs = synthesis(
                #         model=self.tts_model,
                #         text=sen,
                #         CONFIG=self.tts_config,
                #         use_cuda=self.use_cuda,
                #         speaker_id=speaker_id,
                #         style_wav=style_wav,
                #         style_text=style_text,
                #         use_griffin_lim=use_gl,
                #         d_vector=speaker_embedding,
                #         language_id=language_id,
                #     )
                waveform = outputs["wav"]
                # if not use_gl:
                #     mel_postnet_spec = outputs["outputs"]["model_outputs"][0].detach().cpu().numpy()
                #     # denormalize tts output based on tts audio config
                #     mel_postnet_spec = self.tts_model.ap.denormalize(mel_postnet_spec.T).T
                #     # renormalize spectrogram based on vocoder config
                #     vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
                #     # compute scale factor for possible sample rate mismatch
                #     scale_factor = [
                #         1,
                #         self.vocoder_config["audio"]["sample_rate"] / self.tts_model.ap.sample_rate,
                #     ]
                #     if scale_factor[1] != 1:
                #         print(" > interpolating tts model output.")
                #         vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
                #     else:
                #         vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)  # pylint: disable=not-callable
                #     # run vocoder model
                #     # [1, T, C]
                #     waveform = self.vocoder_model.inference(vocoder_input.to(vocoder_device))
                if torch.is_tensor(waveform) and waveform.device != torch.device("cpu") and not use_gl:
                    waveform = waveform.cpu()
                if not use_gl:
                    waveform = waveform.numpy()
                waveform = waveform.squeeze()

                # trim silence
                # if "do_trim_silence" in self.tts_config.audio and self.tts_config.audio["do_trim_silence"]:
                #     waveform = trim_silence(waveform, self.tts_model.ap)

                wavs += list(waveform)
                # wavs += [0] * 10000
                wavs += [0]*2000
        # else:
        #     # get the speaker embedding or speaker id for the reference wav file
        #     reference_speaker_embedding = None
        #     reference_speaker_id = None
        #     if self.tts_speakers_file or hasattr(self.tts_model.speaker_manager, "name_to_id"):
        #         if reference_speaker_name and isinstance(reference_speaker_name, str):
        #             if self.tts_config.use_d_vector_file:
        #                 # get the speaker embedding from the saved d_vectors.
        #                 reference_speaker_embedding = self.tts_model.speaker_manager.get_embeddings_by_name(
        #                     reference_speaker_name
        #                 )[0]
        #                 reference_speaker_embedding = np.array(reference_speaker_embedding)[
        #                     None, :
        #                 ]  # [1 x embedding_dim]
        #             else:
        #                 # get speaker idx from the speaker name
        #                 reference_speaker_id = self.tts_model.speaker_manager.name_to_id[reference_speaker_name]
        #         else:
        #             reference_speaker_embedding = self.tts_model.speaker_manager.compute_embedding_from_clip(
        #                 reference_wav
        #             )
        #     outputs = transfer_voice(
        #         model=self.tts_model,
        #         CONFIG=self.tts_config,
        #         use_cuda=self.use_cuda,
        #         reference_wav=reference_wav,
        #         speaker_id=speaker_id,
        #         d_vector=speaker_embedding,
        #         use_griffin_lim=use_gl,
        #         reference_speaker_id=reference_speaker_id,
        #         reference_d_vector=reference_speaker_embedding,
        #     )
        #     waveform = outputs
        #     if not use_gl:
        #         mel_postnet_spec = outputs[0].detach().cpu().numpy()
        #         # denormalize tts output based on tts audio config
        #         mel_postnet_spec = self.tts_model.ap.denormalize(mel_postnet_spec.T).T
        #         # renormalize spectrogram based on vocoder config
        #         vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
        #         # compute scale factor for possible sample rate mismatch
        #         scale_factor = [
        #             1,
        #             self.vocoder_config["audio"]["sample_rate"] / self.tts_model.ap.sample_rate,
        #         ]
        #         if scale_factor[1] != 1:
        #             print(" > interpolating tts model output.")
        #             vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
        #         else:
        #             vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)  # pylint: disable=not-callable
        #         # run vocoder model
        #         # [1, T, C]
        #         waveform = self.vocoder_model.inference(vocoder_input.to(vocoder_device))
        #     if torch.is_tensor(waveform) and waveform.device != torch.device("cpu"):
        #         waveform = waveform.cpu()
        #     if not use_gl:
        #         waveform = waveform.numpy()
        #     wavs = waveform.squeeze()

        # compute stats
        process_time = time.time() - start_time
        audio_time = len(wavs) / self.tts_config.audio["sample_rate"]
        print(f" > Processing time: {process_time}")
        print(f" > Real-time factor: {process_time / audio_time}")
        return wavs

#-----------------------------TTS.api.py------------------------------------------------------
import tempfile
import warnings
from pathlib import Path
from typing import Union

# from TTS.utils.audio.numpy_transforms import save_wav


class TTS(nn.Module):
    """TODO: Add voice conversion and Capacitron support."""

    def __init__(
        self,
        model_name: str = "",
        model_path: str = None,
        config_path: str = None,
        vocoder_path: str = None,
        vocoder_config_path: str = None,
        progress_bar: bool = True,
        gpu=False,
    ):
        """🐸TTS python interface that allows to load and use the released models.

        Example with a multi-speaker model:
            >>> from TTS.api import TTS
            >>> tts = TTS(TTS.list_models()[0])
            >>> wav = tts.tts("This is a test! This is also a test!!", speaker=tts.speakers[0], language=tts.languages[0])
            >>> tts.tts_to_file(text="Hello world!", speaker=tts.speakers[0], language=tts.languages[0], file_path="output.wav")

        Example with a single-speaker model:
            >>> tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False, gpu=False)
            >>> tts.tts_to_file(text="Ich bin eine Testnachricht.", file_path="output.wav")

        Example loading a model from a path:
            >>> tts = TTS(model_path="/path/to/checkpoint_100000.pth", config_path="/path/to/config.json", progress_bar=False, gpu=False)
            >>> tts.tts_to_file(text="Ich bin eine Testnachricht.", file_path="output.wav")

        Example voice cloning with YourTTS in English, French and Portuguese:
            >>> tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=True)
            >>> tts.tts_to_file("This is voice cloning.", speaker_wav="my/cloning/audio.wav", language="en", file_path="thisisit.wav")
            >>> tts.tts_to_file("C'est le clonage de la voix.", speaker_wav="my/cloning/audio.wav", language="fr", file_path="thisisit.wav")
            >>> tts.tts_to_file("Isso é clonagem de voz.", speaker_wav="my/cloning/audio.wav", language="pt", file_path="thisisit.wav")

        Example Fairseq TTS models (uses ISO language codes in https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html):
            >>> tts = TTS(model_name="tts_models/eng/fairseq/vits", progress_bar=False, gpu=True)
            >>> tts.tts_to_file("This is a test.", file_path="output.wav")

        Args:
            model_name (str, optional): Model name to load. You can list models by ```tts.models```. Defaults to None.
            model_path (str, optional): Path to the model checkpoint. Defaults to None.
            config_path (str, optional): Path to the model config. Defaults to None.
            vocoder_path (str, optional): Path to the vocoder checkpoint. Defaults to None.
            vocoder_config_path (str, optional): Path to the vocoder config. Defaults to None.
            progress_bar (bool, optional): Whether to pring a progress bar while downloading a model. Defaults to True.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
        """
        super().__init__()
        self.config = load_config(config_path) if config_path else None
        self.synthesizer = None
        self.voice_converter = None
        self.model_name = ""
        if gpu:
            warnings.warn("`gpu` will be deprecated. Please use `tts.to(device)` instead.")

        # if model_name is not None and len(model_name) > 0:
        #     if "tts_models" in model_name:
        #         self.load_tts_model_by_name(model_name, gpu)
        #     elif "voice_conversion_models" in model_name:
        #         self.load_vc_model_by_name(model_name, gpu)
        #     else:
        #         self.load_model_by_name(model_name, gpu)

        if model_path:
            self.load_tts_model_by_path(
                model_path, config_path, vocoder_path=vocoder_path, vocoder_config=vocoder_config_path, gpu=gpu
            )

    @property
    def is_multi_speaker(self):
        if hasattr(self.synthesizer.tts_model, "speaker_manager") and self.synthesizer.tts_model.speaker_manager:
            return self.synthesizer.tts_model.speaker_manager.num_speakers > 1
        return False

    @property
    def is_multi_lingual(self):
        # Not sure what sets this to None, but applied a fix to prevent crashing.
        if (
            isinstance(self.model_name, str)
            and "xtts" in self.model_name
            or self.config
            and ("xtts" in self.config.model or len(self.config.languages) > 1)
        ):
            return True
        if hasattr(self.synthesizer.tts_model, "language_manager") and self.synthesizer.tts_model.language_manager:
            return self.synthesizer.tts_model.language_manager.num_languages > 1
        return False

    @property
    def speakers(self):
        if not self.is_multi_speaker:
            return None
        return self.synthesizer.tts_model.speaker_manager.speaker_names

    @property
    def languages(self):
        if not self.is_multi_lingual:
            return None
        return self.synthesizer.tts_model.language_manager.language_names


    # def load_vc_model_by_name(self, model_name: str, gpu: bool = False):
    #     """Load one of the voice conversion models by name.

    #     Args:
    #         model_name (str): Model name to load. You can list models by ```tts.models```.
    #         gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
    #     """
    #     self.model_name = model_name
    #     model_path, config_path, _, _, _ = self.download_model_by_name(model_name)
    #     self.voice_converter = Synthesizer(vc_checkpoint=model_path, vc_config=config_path, use_cuda=gpu)

    def load_tts_model_by_path(
        self, model_path: str, config_path: str, vocoder_path: str = None, vocoder_config: str = None, gpu: bool = False
    ):
        """Load a model from a path.

        Args:
            model_path (str): Path to the model checkpoint.
            config_path (str): Path to the model config.
            vocoder_path (str, optional): Path to the vocoder checkpoint. Defaults to None.
            vocoder_config (str, optional): Path to the vocoder config. Defaults to None.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
        """

        self.synthesizer = Synthesizer(
            tts_checkpoint=model_path,
            tts_config_path=config_path,
            tts_speakers_file=None,
            tts_languages_file=None,
            vocoder_checkpoint=vocoder_path,
            vocoder_config=vocoder_config,
            encoder_checkpoint=None,
            encoder_config=None,
            use_cuda=gpu,
        )

    def _check_arguments(
        self,
        speaker: str = None,
        language: str = None,
        speaker_wav: str = None,
        emotion: str = None,
        speed: float = None,
        **kwargs,
    ) -> None:
        """Check if the arguments are valid for the model."""
        # check for the coqui tts models
        if self.is_multi_speaker and (speaker is None and speaker_wav is None):
            raise ValueError("Model is multi-speaker but no `speaker` is provided.")
        if self.is_multi_lingual and language is None:
            raise ValueError("Model is multi-lingual but no `language` is provided.")
        if not self.is_multi_speaker and speaker is not None and "voice_dir" not in kwargs:
            raise ValueError("Model is not multi-speaker but `speaker` is provided.")
        if not self.is_multi_lingual and language is not None:
            raise ValueError("Model is not multi-lingual but `language` is provided.")
        if not emotion is None and not speed is None:
            raise ValueError("Emotion and speed can only be used with Coqui Studio models. Which is discontinued.")

    def tts(
        self,
        text: str,
        speaker: str = None,
        language: str = None,
        speaker_wav: str = None,
        emotion: str = None,
        speed: float = None,
        split_sentences: bool = True,
        **kwargs,
    ):
        """Convert text to speech.

        Args:
            text (str):
                Input text to synthesize.
            speaker (str, optional):
                Speaker name for multi-speaker. You can check whether loaded model is multi-speaker by
                `tts.is_multi_speaker` and list speakers by `tts.speakers`. Defaults to None.
            language (str): Language of the text. If None, the default language of the speaker is used. Language is only
                supported by `XTTS` model.
            speaker_wav (str, optional):
                Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
                Defaults to None.
            emotion (str, optional):
                Emotion to use for 🐸Coqui Studio models. If None, Studio models use "Neutral". Defaults to None.
            speed (float, optional):
                Speed factor to use for 🐸Coqui Studio models, between 0 and 2.0. If None, Studio models use 1.0.
                Defaults to None.
            split_sentences (bool, optional):
                Split text into sentences, synthesize them separately and concatenate the file audio.
                Setting it False uses more VRAM and possibly hit model specific text length or VRAM limits. Only
                applicable to the 🐸TTS models. Defaults to True.
            kwargs (dict, optional):
                Additional arguments for the model.
        """
        self._check_arguments(
            speaker=speaker, language=language, speaker_wav=speaker_wav, emotion=emotion, speed=speed, **kwargs
        )
        wav = self.synthesizer.tts(
            text=text,
            speaker_name=speaker,
            language_name=language,
            speaker_wav=speaker_wav,
            reference_wav=None,
            style_wav=None,
            style_text=None,
            reference_speaker_name=None,
            split_sentences=split_sentences,
            speed=speed,
            **kwargs,
        )
        return wav

    def tts_to_file(
        self,
        text: str,
        speaker: str = None,
        language: str = None,
        speaker_wav: str = None,
        emotion: str = None,
        speed: float = 1.0,
        pipe_out=None,
        file_path: str = "output.wav",
        split_sentences: bool = True,
        **kwargs,
    ):
        """Convert text to speech.

        Args:
            text (str):
                Input text to synthesize.
            speaker (str, optional):
                Speaker name for multi-speaker. You can check whether loaded model is multi-speaker by
                `tts.is_multi_speaker` and list speakers by `tts.speakers`. Defaults to None.
            language (str, optional):
                Language code for multi-lingual models. You can check whether loaded model is multi-lingual
                `tts.is_multi_lingual` and list available languages by `tts.languages`. Defaults to None.
            speaker_wav (str, optional):
                Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
                Defaults to None.
            emotion (str, optional):
                Emotion to use for 🐸Coqui Studio models. Defaults to "Neutral".
            speed (float, optional):
                Speed factor to use for 🐸Coqui Studio models, between 0.0 and 2.0. Defaults to None.
            pipe_out (BytesIO, optional):
                Flag to stdout the generated TTS wav file for shell pipe.
            file_path (str, optional):
                Output file path. Defaults to "output.wav".
            split_sentences (bool, optional):
                Split text into sentences, synthesize them separately and concatenate the file audio.
                Setting it False uses more VRAM and possibly hit model specific text length or VRAM limits. Only
                applicable to the 🐸TTS models. Defaults to True.
            kwargs (dict, optional):
                Additional arguments for the model.
        """
        self._check_arguments(speaker=speaker, language=language, speaker_wav=speaker_wav, **kwargs)

        wav = self.tts(
            text=text,
            speaker=speaker,
            language=language,
            speaker_wav=speaker_wav,
            split_sentences=split_sentences,
            speed=speed,
            **kwargs,
        )
        self.synthesizer.save_wav(wav=wav, path=file_path, pipe_out=pipe_out)
        return file_path

    # def voice_conversion(
    #     self,
    #     source_wav: str,
    #     target_wav: str,
    # ):
    #     """Voice conversion with FreeVC. Convert source wav to target speaker.

    #     Args:``
    #         source_wav (str):
    #             Path to the source wav file.
    #         target_wav (str):`
    #             Path to the target wav file.
    #     """
    #     wav = self.voice_converter.voice_conversion(source_wav=source_wav, target_wav=target_wav)
    #     return wav

    # def voice_conversion_to_file(
    #     self,
    #     source_wav: str,
    #     target_wav: str,
    #     file_path: str = "output.wav",
    # ):
    #     """Voice conversion with FreeVC. Convert source wav to target speaker.

    #     Args:
    #         source_wav (str):
    #             Path to the source wav file.
    #         target_wav (str):
    #             Path to the target wav file.
    #         file_path (str, optional):
    #             Output file path. Defaults to "output.wav".
    #     """
    #     wav = self.voice_conversion(source_wav=source_wav, target_wav=target_wav)
    #     save_wav(wav=wav, path=file_path, sample_rate=self.voice_converter.vc_config.audio.output_sample_rate)
    #     return file_path

    # def tts_with_vc(
    #     self,
    #     text: str,
    #     language: str = None,
    #     speaker_wav: str = None,
    #     speaker: str = None,
    #     split_sentences: bool = True,
    # ):
    #     """Convert text to speech with voice conversion.

    #     It combines tts with voice conversion to fake voice cloning.

    #     - Convert text to speech with tts.
    #     - Convert the output wav to target speaker with voice conversion.

    #     Args:
    #         text (str):
    #             Input text to synthesize.
    #         language (str, optional):
    #             Language code for multi-lingual models. You can check whether loaded model is multi-lingual
    #             `tts.is_multi_lingual` and list available languages by `tts.languages`. Defaults to None.
    #         speaker_wav (str, optional):
    #             Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
    #             Defaults to None.
    #         speaker (str, optional):
    #             Speaker name for multi-speaker. You can check whether loaded model is multi-speaker by
    #             `tts.is_multi_speaker` and list speakers by `tts.speakers`. Defaults to None.
    #         split_sentences (bool, optional):
    #             Split text into sentences, synthesize them separately and concatenate the file audio.
    #             Setting it False uses more VRAM and possibly hit model specific text length or VRAM limits. Only
    #             applicable to the 🐸TTS models. Defaults to True.
    #     """
    #     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
    #         # Lazy code... save it to a temp file to resample it while reading it for VC
    #         self.tts_to_file(
    #             text=text, speaker=speaker, language=language, file_path=fp.name, split_sentences=split_sentences
    #         )
    #     if self.voice_converter is None:
    #         self.load_vc_model_by_name("voice_conversion_models/multilingual/vctk/freevc24")
    #     wav = self.voice_converter.voice_conversion(source_wav=fp.name, target_wav=speaker_wav)
    #    return wav

    # def tts_with_vc_to_file(
    #     self,
    #     text: str,
    #     language: str = None,
    #     speaker_wav: str = None,
    #     file_path: str = "output.wav",
    #     speaker: str = None,
    #     split_sentences: bool = True,
    # ):
    #     """Convert text to speech with voice conversion and save to file.

    #     Check `tts_with_vc` for more details.

    #     Args:
    #         text (str):
    #             Input text to synthesize.
    #         language (str, optional):
    #             Language code for multi-lingual models. You can check whether loaded model is multi-lingual
    #             `tts.is_multi_lingual` and list available languages by `tts.languages`. Defaults to None.
    #         speaker_wav (str, optional):
    #             Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
    #             Defaults to None.
    #         file_path (str, optional):
    #             Output file path. Defaults to "output.wav".
    #         speaker (str, optional):
    #             Speaker name for multi-speaker. You can check whether loaded model is multi-speaker by
    #             `tts.is_multi_speaker` and list speakers by `tts.speakers`. Defaults to None.
    #         split_sentences (bool, optional):
    #             Split text into sentences, synthesize them separately and concatenate the file audio.
    #             Setting it False uses more VRAM and possibly hit model specific text length or VRAM limits. Only
    #             applicable to the 🐸TTS models. Defaults to True.
    #     """
    #     wav = self.tts_with_vc(
    #         text=text, language=language, speaker_wav=speaker_wav, speaker=speaker, split_sentences=split_sentences
    #     )
    #     save_wav(wav=wav, path=file_path, sample_rate=self.voice_converter.vc_config.audio.output_sample_rate)


if __name__ == "__main__":
    # config = load_config()
    # model = Xtts.init_from_config(config)
    # model.load_checkpoint(config, checkpoint_dir="coqui", eval=True)
    tts = TTS(model_path="coqui/model.pth", config_path="coqui/config.json", progress_bar=False, gpu=True)
    text = "Replace the sky with a deep blue sky, white cloud hovering then replace the mountain with himalayan mountain covered in snow reflecting sunlight."

    tts.tts_to_file(text=text, speaker="Gilberto Mathias", language="en",
                    file_path="prompt.wav", speed=1.30, split_sentences=True)
    # with open("transcript/huberman_summary.txt", "r") as f:
    #     text = f.read()
    #     text = text.replace('\n', '')
    # tts.tts_to_file(text=text, speaker_wav='mp3/output.wav',
    #                 language="en", file_path="huberman_clone_v1.wav", speed=1.30, split_sentences=True)