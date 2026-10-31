"""Core model architecture, token sequence construction, and confidence estimation for laya."""
import os
from contextlib import nullcontext
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .inference_common import (  # noqa: F401 -- compatibility reexports
    QTYPES,
    QTYPE_NAMES,
    _DEFAULT_NOUL_LABELS,
    _TOKENIZE_LOCK,
    _QUESTION_TOKEN_CACHE,
    encode_text,
    _reuse_question_tokens,
    _disable_question_token_reuse,
    _encode_question_text,
    serialize_state,
    render_criterion,
    _resolve_noul_labels,
    render_options,
    build_sequence,
    build_head,
    _QUESTION_HEADS,
    _QUESTION_HEADS_LOCK,
    _QUESTION_HEADS_MAX,
    _question_head,
    _build_question_head,
    parallel_layout,
    OPTION_LAYOUTS,
    uses_parallel_layout,
    state_room,
    _WINDOW_CLAMP_WARN_RATIO,
    _WINDOW_BATCH_BLOWUP,
    window_budget,
    window_batch_cap,
    collapsed_options,
    unpermute_probs,
    ece_score,
    answer_confidence,
    confidence_from_probs,
    temp_bucket,
    TEMP_MIN,
    TEMP_MAX,
    clamp_temperature,
    temperature_rejected,
    resolve_lang_temperatures,
)


def _autocast_enabled(device_type: str) -> bool:
    """Whether an autocast scope covers `device_type`.

    `torch.is_autocast_enabled` only accepted a `device_type` from torch 2.4; before that it
    reported CUDA alone and CPU and MPS had no autocast to report. This package declares
    `torch>=2.0.0`, so call the device-aware form where it exists and fall back otherwise,
    using torch's separate CPU predicate.
    """
    try:
        return torch.is_autocast_enabled(device_type)
    except TypeError:
        # torch < 2.4 took no `device_type`: the no-argument call reports CUDA alone, CPU has
        # its own predicate, and MPS had no autocast to report.
        if device_type == "cpu":
            return torch.is_autocast_cpu_enabled()
        return torch.is_autocast_enabled()


def parallel_option_masks(attention_mask: torch.Tensor, position_ids: torch.Tensor, option_ids: torch.Tensor,
                          sliding_window: Optional[int]) -> Dict[str, torch.Tensor]:
    """Per-layer-type boolean attention masks for the parallel layout (`parallel_layout`).

    A token attends to every real token except those of a *different* option. ModernBERT's local
    layers keep their window, measured in position ids rather than sequence index: options share
    positions, so a sequence-index window would let slot 0 see more of the state than slot 3.
    Returned in the `{"full_attention": ..., "sliding_attention": ...}` form ModernBERT takes
    in place of a padding mask; each mask is `[batch, 1, query, key]`, True where attention is allowed.
    """
    keys = attention_mask.bool()[:, None, :]
    other_option = (option_ids[:, :, None] > 0) & (option_ids[:, None, :] > 0) \
        & (option_ids[:, :, None] != option_ids[:, None, :])
    full = keys & ~other_option
    masks = {"full_attention": full[:, None]}
    if sliding_window is not None:
        near = (position_ids[:, :, None] - position_ids[:, None, :]).abs() <= sliding_window
        masks["sliding_attention"] = (full & near)[:, None]
    return masks


class _DynamicMultiheadAttention(nn.MultiheadAttention):
    """`nn.MultiheadAttention` that keeps the shapes it traces.

    The stock module reshapes the packed projection with sizes captured while tracing, so under
    the legacy ONNX exporter the traced sequence length becomes a constant and a model exported
    from a short dummy input only runs at that length. The parameters and the maths are the same
    here; the reshape uses only constant shape arguments (`chunk` / `unflatten` / `flatten`) and
    `scaled_dot_product_attention`, the kernel the stock path already uses when weights are not
    requested.
    """

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if (need_weights or self.in_proj_weight is None or self.bias_k is not None
                or self.bias_v is not None
                or (attn_mask is not None and attn_mask.dtype != torch.bool)):
            # Weight averaging, additive masks and the optional k/v bias are not on the traced
            # path; the stock implementation keeps them correct.
            return super().forward(query, key, value, key_padding_mask=key_padding_mask,
                                   need_weights=need_weights, attn_mask=attn_mask,
                                   average_attn_weights=average_attn_weights, is_causal=is_causal)
        if self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        # (T, B, E) from here, matching the stock module's internals; attention runs on (B, H, T, D).
        if query is key is value:
            q, k, v = (part.unflatten(-1, (self.num_heads, self.head_dim)).permute(1, 2, 0, 3)
                       for part in F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1))
        else:
            embed_dim = query.shape[-1]
            wq, wk, wv = self.in_proj_weight.split(embed_dim, dim=0)
            bq, bk, bv = ((None, None, None) if self.in_proj_bias is None
                          else self.in_proj_bias.split(embed_dim, dim=0))
            q, k, v = (
                F.linear(t, w, b).unflatten(-1, (self.num_heads, self.head_dim)).permute(1, 2, 0, 3)
                for t, w, b in ((query, wq, bq), (key, wk, bk), (value, wv, bv))
            )
        mask = None
        if attn_mask is not None:
            mask = ~attn_mask
        if key_padding_mask is not None:
            # `== 0` keeps this correct for a bool mask and for the 0 / -inf float mask the encoder
            # layer hands over (`F._canonical_mask`), where `~` would not be defined.
            keep = key_padding_mask[:, None, None, :] == 0
            mask = keep if mask is None else mask & keep
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal and mask is None,
                                              dropout_p=self.dropout if self.training else 0.0)
        attn = attn.permute(2, 0, 1, 3).flatten(-2)
        if self.batch_first:
            attn = attn.transpose(0, 1)
        return self.out_proj(attn), None


class DecisionModel(nn.Module):
    """Bidirectional transformer encoder backbone + typed decision head."""

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1,
                 no_init: bool = False):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        # `no_init` means the caller is about to load every parameter from a checkpoint, so the
        # head's initial values are pure overhead -- and not just time. transformers'
        # `no_init_weights()` patches the torch.nn.init functions, but something inside
        # nn.TransformerEncoderLayer draws from the RNG outside them, so building it still
        # advanced the global generator and made `load()` a visible side effect. On the meta
        # device no initialisation kernel runs at all; the layers are materialised empty and the
        # load fills them.
        with torch.device("meta") if no_init else nullcontext():
            nhead = max(1, d // 64)
            layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
            # The stock attention bakes the traced length into an exported graph; see
            # _DynamicMultiheadAttention. Same parameters, same maths, traceable shapes.
            layer.self_attn = _DynamicMultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
            self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
            self.type_emb = nn.Embedding(3, d)
            self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
            self.register_buffer("temperature", torch.ones(3))
        if no_init:
            # Only the modules created above are on the meta device; the encoder is already real
            # and may hold non-persistent buffers (RoPE frequencies) that to_empty would wipe.
            for module in (self.head, self.type_emb, self.scorer, self.act_head):
                if module is not None:
                    module.to_empty(device="cpu")
            self.temperature = torch.empty_like(self.temperature, device="cpu")
        self.head_checkpointing = False

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder: bool = False,
                position_ids=None, option_ids=None):
        if option_ids is None:
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            # Parallel option layout (`parallel_layout`). The head below has no positional
            # encoding, so with order-blind marker embeddings the logits permute with the options.
            masks = parallel_option_masks(attention_mask, position_ids, option_ids,
                                          getattr(self.encoder.config, "sliding_window", None))
            h = self.encoder(input_ids=input_ids, attention_mask=masks, position_ids=position_ids).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                if self.head_checkpointing and self.training and torch.is_grad_enabled():
                    # Non-reentrant checkpointing also trains the head when its input
                    # is frozen. Default RNG preservation keeps dropout consistent.
                    h = checkpoint(layer, h, src_key_padding_mask=pad, use_reentrant=False)
                else:
                    h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p.size(-1) >= 2:
            top2 = p.topk(2, -1).values
        else:
            # A single-option question has exactly one marker, so p.topk(2, ...)
            # has nothing to select for the second slot and raises. The answer
            # is still well-defined: softmax over one logit is 1.0 regardless of
            # its value, so pad the missing second entry with 0.0 - that gives
            # the act head top1 - top2 == 1.0, the same "fully decided" signal
            # it would see for any other unambiguous top-1-vs-rest gap.
            top1 = p.topk(1, -1).values
            top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_input = torch.cat([pooled, feats], -1)
        # Explicit-dtype exports run without autocast; keep the confidence maths in fp32,
        # then match the head's weights. Autocast already chooses the linear's input dtype.
        if not _autocast_enabled(h.device.type):
            act_input = act_input.to(self.act_head[0].weight.dtype)
        act_logits = self.act_head(act_input)
        return logits, act_logits


def _apply_rope_config(ecfg) -> None:
    """Carry transformers>=5 per-layer RoPE settings over to the attributes 4.x reads.

    A checkpoint re-saved by transformers 5 stores RoPE as
    `rope_parameters = {"full_attention": {"rope_theta": ...}, "sliding_attention": {...}}`.
    transformers 4.x does not know that key, so it keeps its own defaults (global 160000,
    local 10000) and any checkpoint whose sliding-attention theta differs silently runs the
    wrong RoPE base -- mmBERT is exactly that case, both of its thetas are 160000. Map the
    values onto `global_rope_theta` / `local_rope_theta`, which 4.x does read. On
    transformers 5 this is a no-op beyond re-setting the same numbers.
    """
    rope = getattr(ecfg, "rope_parameters", None)
    if not isinstance(rope, dict):
        return
    flat = rope.get("rope_theta")
    for layer_type, attr in (("full_attention", "global_rope_theta"),
                             ("sliding_attention", "local_rope_theta")):
        params = rope.get(layer_type)
        theta = params.get("rope_theta") if isinstance(params, dict) else flat
        if theta is not None and hasattr(ecfg, attr):
            setattr(ecfg, attr, float(theta))


def _no_init_weights():
    """`no_init_weights` lives in different modules across transformers versions."""
    try:
        from transformers.initialization import no_init_weights
    except ImportError:  # transformers 4.x
        from transformers.modeling_utils import no_init_weights
    return no_init_weights()


def build_model(cfg: Dict, encoder_dir: Optional[str] = None, pretrained: bool = True,
                revision: Optional[str] = None) -> DecisionModel:
    """Build the decision model described by `cfg`.

    With `pretrained=False`, or when `encoder_dir` holds a saved encoder config, nothing is
    downloaded and **no parameter is initialised**: the caller is expected to load a checkpoint
    into the result with `load_state_dict(..., strict=True)` immediately. Skipping initialisation
    keeps `load()` from spending time on, or consuming RNG for, weights it is about to overwrite.
    """
    from transformers import AutoConfig, AutoModel

    head_layers, n_act = cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1
    if not pretrained or (encoder_dir and os.path.exists(encoder_dir)):
        ecfg = AutoConfig.from_pretrained(encoder_dir or cfg["encoder"])
        _apply_rope_config(ecfg)
        with _no_init_weights():
            enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
        return DecisionModel(enc, head_layers, n_act, no_init=True)
    # Training-time Hub load of the base encoder; allow pinning it like the checkpoints.
    kw = {"attn_implementation": "sdpa"}
    if revision:
        kw["revision"] = revision
    enc = AutoModel.from_pretrained(cfg["encoder"], **kw)
    return DecisionModel(enc, head_layers, n_act)


def proper_reward(
    q: torch.Tensor,
    target: torch.Tensor,
    qtype: torch.Tensor,
    mask: torch.Tensor,
    w_sph: float = 0.5,
    w_rps: float = 1.0,
    log_floor: float = -9.21,
) -> torch.Tensor:
    """Strictly proper scoring rule reward: log score + spherical score + ranked probability score.

    q: [..., N, K] reported distributions
    target: [N, K] (one-hot or soft target distributions)
    """
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q = torch.cumsum(q, -1)
        cdf_t = torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r


def td_lambda_targets(p_true: torch.Tensor, batch: Dict, lam: float = 1.0) -> torch.Tensor:
    """TD(lambda) targets for multi-turn conversation trajectories."""
    target = batch["target"].clone()
    groups = batch.get("ep_group")
    if groups is None:
        return target
    for g in torch.unique(groups[groups >= 0]).tolist():
        idx = (groups == g).nonzero(as_tuple=True)[0]
        idx = idx[torch.argsort(batch["ep_step"][idx])]
        y = batch["target"][idx[-1], 1]
        G = y
        for j in range(len(idx) - 1, -1, -1):
            if j < len(idx) - 1:
                G = (1 - lam) * p_true[idx[j + 1]] + lam * G
            target[idx[j], 0], target[idx[j], 1] = 1 - G, G
    return target


def amp_dtype(name: Optional[str]) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def collate_items(batch, pad_id: int):
    items = [it for group in batch for it in group]
    if not items:
        return None
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = np.full((n, L), pad_id, dtype=np.int64)
    att = np.zeros((n, L), dtype=np.int64)
    mpos = np.zeros((n, kmax), dtype=np.int64)
    mmask = np.zeros((n, kmax), dtype=bool)
    has_target = any("target" in it for it in items)
    target = np.zeros((n, kmax), dtype=np.float32) if has_target else None

    for i, it in enumerate(items):
        it_ids = it["ids"]
        l_ids = len(it_ids)
        ids[i, :l_ids] = it_ids
        att[i, :l_ids] = 1
        markers = it["markers"]
        k = len(markers)
        mpos[i, :k] = markers
        mmask[i, :k] = True
        if has_target and "target" in it:
            t = it["target"]
            if len(t) > k:
                # Otherwise this lands as "The expanded size of the tensor (k) must match the
                # existing size (kmax)" from inside the assignment, which says nothing about the
                # actual mistake: a target with more entries than the item has options. The limit
                # is this item's own marker count, not the batch-wide kmax: in a mixed-width
                # batch a longer sibling row must not legitimise extra entries (#311).
                raise ValueError(
                    "collate_items: item %d has %d target entries but only %d marker positions; "
                    "a target needs one entry per option" % (i, len(t), k))
            target[i, : len(t)] = t

    res = {
        "input_ids": torch.from_numpy(ids),
        "attention_mask": torch.from_numpy(att),
        "marker_pos": torch.from_numpy(mpos),
        "marker_mask": torch.from_numpy(mmask),
        "qtype": torch.from_numpy(np.array([it["qtype"] for it in items], dtype=np.int64)),
        "label": torch.from_numpy(np.array([it.get("label", -1) for it in items], dtype=np.int64)),
        "meta": [{k: it[k] for k in it if k not in ("ids", "markers", "target", "layout")} for it in items],
    }
    if target is not None:
        res["target"] = torch.from_numpy(target)
    if any("layout" in it for it in items):
        # All or none: a batch that mixes layouts would run half its rows through the wrong masks
        if not all("layout" in it for it in items):
            raise ValueError("collate_items: some items carry a parallel layout and some do not")
        pos = np.zeros((n, L), dtype=np.int64)
        opt = np.zeros((n, L), dtype=np.int64)
        for i, it in enumerate(items):
            l_ids = len(it["ids"])
            pos[i, :l_ids] = it["layout"]["position_ids"]
            opt[i, :l_ids] = it["layout"]["option_ids"]
        res["position_ids"], res["option_ids"] = torch.from_numpy(pos), torch.from_numpy(opt)
    return res
