#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3.5-0.8B：safetensors(bf16) → PQWF 扩展格式（.pqw，混合 int8 / f32）。

PQWF 扩展（相对 0.6B 那份 v1）有三点：
1. **混合 dtype**：v1 的规则是"ndim==1 才存 F32"，但 Qwen3.5 的 `conv1d.weight`
   是 [6144,1,4]（3D）且**不能量化**（深度可分离卷积核，逐通道 4 个系数，
   per-channel absmax 会把整条通道压成 1 个值）。所以本版改成显式规则：
   只有 Linear/Embedding 的 2D `*.weight` 走 INT8_PC，其余全部 F32 原样存。
2. **零中心 RMSNorm 折叠**：`Qwen3_5RMSNorm` 的前向是 `x_norm * (1 + w)`，
   写盘时直接把权重存成 `1 + w`（f32），device 端就退化成普通 RMSNorm。
   `Qwen3_5RMSNormGated`（GDN 的输出 norm）是标准权重，不加 1。
3. **派生标量**：`A_log` 存成 `exp(A_log)`（f32），因为 device 端没有 exp 的
   标量指令也不需要——g 只需要 exp(A_log) 这个常量因子。

元数据里记录 shape、层类型、mrope 参数，convert_pqwf35.py 与 ref 都依赖它。
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_qwen35 import (CONVK, CONV_DIM, DK, DV, H, HD, INTER, LAYERS, NKV, NH,  # noqa: E402
                        ROT, ROPE_THETA, VOCAB, SafeTensors)

INT8_PC, F32 = 1, 0
PREFIX = "model.language_model."
FULL_LAYER = (3, 7, 11, 15, 19, 23)

# 哪些张量走 int8：Linear / Embedding 的 2D weight。其余（norm/conv/A_log/
# dt_bias）保持 f32 —— norm 只有 1024 个数，量化没有收益且会破坏 eps 行为。
QUANT_SUFFIX = (
    ".in_proj_qkv.weight", ".in_proj_z.weight", ".in_proj_b.weight",
    ".in_proj_a.weight", ".out_proj.weight",
    ".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight",
    ".gate_proj.weight", ".up_proj.weight", ".down_proj.weight",
    "embed_tokens.weight",
)


def should_quantize(name: str, shape) -> bool:
    return len(shape) == 2 and name.endswith(QUANT_SUFFIX)


def quant_int8_pc(w: np.ndarray):
    """per-channel 对称 int8（第 0 维 = 输出通道 / 词表行）。"""
    w = w.astype(np.float32)
    amax = np.max(np.abs(w), axis=1)
    scale = (amax * np.float32(1.0 / 127.0)).astype(np.float32)
    safe = np.where(scale == 0, np.float32(1.0), scale).astype(np.float32)
    q = np.clip(np.rint(w / safe[:, None]), -127, 127).astype(np.int8)
    q[scale == 0] = 0
    return q, scale


class Tensor:
    __slots__ = ("name", "dims", "dtype", "wbytes", "sbytes", "scount")

    def __init__(self, name, dims, dtype, wbytes, sbytes=b"", scount=0):
        self.name, self.dims, self.dtype = name, dims, dtype
        self.wbytes, self.sbytes, self.scount = wbytes, sbytes, scount


def collect(st: SafeTensors, limit_layers: int | None):
    """按顺序产出要写盘的张量（名字沿用 HQ 的命名，便于对照）。"""
    out: list[Tensor] = []

    def add(name, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        out.append(Tensor(name, list(arr.shape), F32, arr.tobytes()))

    def add_linear(name, w):
        q, s = quant_int8_pc(w)
        out.append(Tensor(name, [w.shape[0], int(np.prod(w.shape[1:]))], INT8_PC,
                          q.tobytes(), s.tobytes(), w.shape[0]))

    emb = st.get(PREFIX + "embed_tokens.weight")
    add_linear(PREFIX + "embed_tokens.weight", emb)
    add(PREFIX + "norm.weight", st.get(PREFIX + "norm.weight").astype(np.float32) + 1.0)

    n_layers = LAYERS if limit_layers is None else limit_layers
    for L in range(n_layers):
        pre = f"{PREFIX}layers.{L}."
        add(pre + "input_layernorm.weight",
            st.get(pre + "input_layernorm.weight").astype(np.float32) + 1.0)
        add(pre + "post_attention_layernorm.weight",
            st.get(pre + "post_attention_layernorm.weight").astype(np.float32) + 1.0)
        if L in FULL_LAYER:
            for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                add_linear(pre + f"self_attn.{n}.weight", st.get(pre + f"self_attn.{n}.weight"))
            add(pre + "self_attn.q_norm.weight",
                st.get(pre + "self_attn.q_norm.weight").astype(np.float32) + 1.0)
            add(pre + "self_attn.k_norm.weight",
                st.get(pre + "self_attn.k_norm.weight").astype(np.float32) + 1.0)
        else:
            for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                add_linear(pre + f"linear_attn.{n}.weight", st.get(pre + f"linear_attn.{n}.weight"))
            add(pre + "linear_attn.conv1d.weight",
                st.get(pre + "linear_attn.conv1d.weight").reshape(CONV_DIM, CONVK))
            add(pre + "linear_attn.norm.weight", st.get(pre + "linear_attn.norm.weight"))
            add(pre + "linear_attn.A_log_exp",
                np.exp(st.get(pre + "linear_attn.A_log").astype(np.float32)))
            add(pre + "linear_attn.dt_bias", st.get(pre + "linear_attn.dt_bias"))
        for n in ("gate_proj", "up_proj", "down_proj"):
            add_linear(pre + f"mlp.{n}.weight", st.get(pre + f"mlp.{n}.weight"))
    return out


def write_pqwf(path: Path, tensors: list[Tensor], meta: dict) -> None:
    meta_bytes = json.dumps(meta, ensure_ascii=False).encode()
    entries = bytearray()
    data = bytearray()
    for t in tensors:
        woff = len(data)
        data += t.wbytes
        while len(data) % 32:           # 权重块 32 字节对齐
            data += b"\x00"
        soff = len(data)
        data += t.sbytes
        name = t.name.encode()
        entries += struct.pack("<I", len(name)) + name
        entries += struct.pack("<I", len(t.dims)) + struct.pack(f"<{len(t.dims)}Q", *t.dims)
        entries += struct.pack("<I", t.dtype)
        entries += struct.pack("<QQQQ", woff, len(t.wbytes), soff, len(t.sbytes))
        entries += struct.pack("<II", t.scount, 0)
    header = struct.pack("<IIIIQQ", 0x46575150, 2, 1, 0, len(tensors), len(meta_bytes))
    body = header + meta_bytes + bytes(entries)
    while len(body) % 32:
        body += b"\x00"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(body)
        f.write(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../../qwen3.5-0.8b-bf16")
    ap.add_argument("--out", default="build/qwen35-0.8b-int8.pqw")
    ap.add_argument("--limit-layers", type=int, default=None,
                    help="只转换前 N 层（调试用，配合 --tiny-blob）")
    a = ap.parse_args()
    src = Path(a.src)
    st = SafeTensors(src / "model.safetensors-00001-of-00001.safetensors")
    tensors = collect(st, a.limit_layers)
    meta = {
        "format": "pqwf", "version": 2,
        "quantization": "int8 per-channel symmetric (axis=0) for 2D linear weights; f32 otherwise",
        "note": "PQWF 扩展：零中心 RMSNorm 已折叠为 (1+w)；A_log 存 exp(A_log)",
        "arch": "qwen3_5_text",
        "hidden_size": H, "intermediate_size": INTER, "vocab_size": VOCAB,
        "num_hidden_layers": LAYERS if a.limit_layers is None else a.limit_layers,
        "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD,
        "linear_num_key_heads": 16, "linear_num_value_heads": 16,
        "linear_key_head_dim": DK, "linear_value_head_dim": DV,
        "linear_conv_kernel_dim": CONVK, "rot_dim": ROT, "rope_theta": ROPE_THETA,
        "mrope_section": [11, 11, 10], "full_attention_interval": 4,
        "layer_types": ["full_attention" if i in FULL_LAYER else "linear_attention"
                        for i in range(LAYERS if a.limit_layers is None else a.limit_layers)],
        "tie_word_embeddings": True,
    }
    out = Path(a.out)
    write_pqwf(out, tensors, meta)
    total = sum(len(t.wbytes) + len(t.sbytes) for t in tensors)
    print(f"pqw: {out} ({out.stat().st_size} 字节, {len(tensors)} 个张量, 数据区 {total} B)")


if __name__ == "__main__":
    main()
