#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3.5-0.8B：PQWF 扩展(.pqw) → guest blob（扁平布局）+ 布局头 config/model_qwen35.h。

布局分三段：
  ① 低地址：控制块 / 输入 id / 位置 id / embedding（tied lm_head）/ final norm /
     M-RoPE 表 —— 都是"每层之前就要用"的东西；
  ② 权重区：GDN 层（18 个）与全注意力层（6 个）各一套 stride，
     层内再分"int8 权重+scale"和"f32 小张量"两个块；
  ③ 状态区 + scratch：GDN 状态 S（18 MB f32）、短卷积历史、KV cache、
     以及每层复用的激活缓冲。

为什么 GDN/全注意力用两套 stride：两层的权重集合不同（GDN 有 in_proj_qkv/
z/b/a/out + conv + norm + A_exp + dt_bias，全注意力有 q/k/v/o + q_norm/k_norm），
用两套按层类型索引的数组可以避免为两种层都预留最大尺寸（省 ~50 MB）。

设备端约定（见 src/main.cpp）：
  x31 = 本 blob 基址；CTRL = [n_prompt, n_gen, max_seq, mode]
  mode=0 内置生成循环；mode=1 单步（处理绝对位置 CTRL[0] 的 1 个 token）。
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_qwen35 import (CONVK, CONV_DIM, DK, DV, H, HD, INTER, LAYERS, NKV, NH,  # noqa: E402
                        ROT, ROPE_THETA, VOCAB)

INT8_PC, F32 = 1, 0
HALF = ROT // 2                 # 32 个旋转对
NQH = 16                        # GDN 的 key/value 头数
QW = NH * HD                    # 2048：全注意力 q/gate 的宽度（q 和 gate 各 2048）
KW = NKV * HD                   # 512：k/v 的宽度
KVW = NQH * DV                  # 2048：GDN 的 q/k/v 宽度
SEQ = 32                        # 每次前向的 token 块大小
MAXSEQ = 8192                   # KV cache 容量 / 总长度上限
FULL_LAYER = (3, 7, 11, 15, 19, 23)
GDN_LAYERS = LAYERS - len(FULL_LAYER)


def align(n, a=0x1000):
    return (n + a - 1) // a * a


# ---------------------------------------------------------------------------
# 布局
# ---------------------------------------------------------------------------
def layer_blocks():
    """层内相对偏移（int8 权重 + scale 紧跟其后）。"""
    gdn, pos = {}, 0

    def w(name, rows, cols):
        nonlocal pos
        gdn[name] = pos
        pos += rows * cols
        gdn[name + "_S"] = pos
        pos += rows * 4

    w("QKV", CONV_DIM, H)
    w("Z", KVW, H)
    w("BOUT", NQH, H)
    w("AOUT", NQH, H)
    w("OUT", H, KVW)
    w("G", INTER, H)
    w("U", INTER, H)
    w("D", H, INTER)
    gdn["STRIDE"] = pos

    # f32 小张量块
    n, npos = {}, 0

    def f(name, cnt):
        nonlocal npos
        n[name] = npos
        npos += cnt * 4

    f("N1", H)
    f("N2", H)
    f("NORMW", DV)
    f("CONVW", CONV_DIM * CONVK)
    f("AEXP", NQH)
    f("DTB", NQH)
    n["STRIDE"] = npos

    full, fpos = {}, 0

    def v(name, rows, cols):
        nonlocal fpos
        full[name] = fpos
        fpos += rows * cols
        full[name + "_S"] = fpos
        fpos += rows * 4

    v("QG", 2 * QW, H)
    v("K", KW, H)
    v("V", KW, H)
    v("O", H, QW)
    v("G", INTER, H)
    v("U", INTER, H)
    v("D", H, INTER)
    full["STRIDE"] = fpos

    fn, fnpos = {}, 0

    def g(name, cnt):
        nonlocal fnpos
        fn[name] = fnpos
        fnpos += cnt * 4

    g("N1", H)
    g("N2", H)
    g("QN", HD)
    g("KN", HD)
    fn["STRIDE"] = fnpos
    return gdn, n, full, fn


GDN_W, GDN_N, FULL_W, FULL_N = layer_blocks()


def layout():
    o = {}
    end = [0]

    def put(name, size):
        o[name] = end[0]
        end[0] += size

    put("IDS", MAXSEQ * 4)
    put("CTRL", 16)
    put("PIDS", 3 * MAXSEQ * 4)          # (t,h,w) 三个位置 id（纯文本三者相同）
    put("X", SEQ * H * 4)
    put("EMB", VOCAB * H + VOCAB * 4)    # tied：lm_head 复用这块
    # 别名（tied embedding）：只写偏移，不推进 end
    o["LM"] = o["EMB"]
    o["EMB_S"] = o["EMB"] + VOCAB * H
    o["LM_S"] = o["EMB_S"]
    put("NF", H * 4)
    put("COS", MAXSEQ * HALF * 4)
    put("SIN", MAXSEQ * HALF * 4)

    # ---- GDN 层权重 ----
    o["LGW_BASE"] = end[0]
    o["LGW_STRIDE"] = GDN_W["STRIDE"]
    for k, v in GDN_W.items():
        if k != "STRIDE":
            o["GW_" + k] = o["LGW_BASE"] + v
    end[0] += GDN_LAYERS * GDN_W["STRIDE"]
    o["LGN_BASE"] = end[0]
    o["LGN_STRIDE"] = GDN_N["STRIDE"]
    for k, v in GDN_N.items():
        if k != "STRIDE":
            o["GN_" + k] = o["LGN_BASE"] + v
    end[0] += GDN_LAYERS * GDN_N["STRIDE"]

    # ---- 全注意力层权重 ----
    o["LFW_BASE"] = end[0]
    o["LFW_STRIDE"] = FULL_W["STRIDE"]
    for k, v in FULL_W.items():
        if k != "STRIDE":
            o["FW_" + k] = o["LFW_BASE"] + v
    end[0] += len(FULL_LAYER) * FULL_W["STRIDE"]
    o["LFN_BASE"] = end[0]
    o["LFN_STRIDE"] = FULL_N["STRIDE"]
    for k, v in FULL_N.items():
        if k != "STRIDE":
            o["FN_" + k] = o["LFN_BASE"] + v
    end[0] += len(FULL_LAYER) * FULL_N["STRIDE"]

    # ---- 状态区 ----
    o["S_BASE"] = end[0]                 # GDN 递推状态：[18][16][128 dv][128 dk] f32
    o["S_LAYER"] = NQH * DV * DK * 4
    end[0] += GDN_LAYERS * o["S_LAYER"]
    o["HIST_BASE"] = end[0]              # 短卷积历史：[18][3][6144] f32
    o["HIST_LAYER"] = (CONVK - 1) * CONV_DIM * 4
    end[0] += GDN_LAYERS * o["HIST_LAYER"]
    # KV cache（6 层，k 段 + v 段）：int8 [MAXSEQ][KW] + f32 scale [MAXSEQ]
    kd = MAXSEQ * KW + MAXSEQ * 4
    o["KV_BASE"] = end[0]
    o["KV_VOFF"] = kd
    o["KV_STRIDE"] = 2 * kd
    o["KV_SOFF"] = MAXSEQ * KW
    end[0] += len(FULL_LAYER) * o["KV_STRIDE"]
    o["BLOCK_END"] = end[0]

    # ---- scratch（每层复用；按 64 KB 对齐便于调试）----
    base = [align(o["BLOCK_END"], 0x10000)]

    def sc(name, size, a=0x1000):
        o[name] = base[0]
        base[0] += align(size, a)

    sc("H2", SEQ * H * 4)                 # rmsnorm 输出 f32
    sc("H8", SEQ * H)                     # 量化后的激活
    sc("HS", SEQ * 4, 0x100)
    sc("QKV", (CONVK - 1 + SEQ) * CONV_DIM * 4)   # 短卷积输入（3 行历史 + 本块）
    sc("MIX", SEQ * CONV_DIM * 4)         # 卷积+silu 输出：q|k|v
    sc("Q8", SEQ * KVW)                   # per-(token,head) 量化
    sc("K8", SEQ * KVW)
    sc("QSCR", SEQ * NQH * 4)
    sc("KSCR", SEQ * NQH * 4)
    sc("O32", SEQ * KVW * 4)              # 递推输出 f32
    sc("Z", SEQ * KVW * 4)                # 输出门投影
    sc("BETAS", SEQ * NQH * 4)
    sc("ALPHAS", SEQ * NQH * 4)
    sc("QG", SEQ * 2 * QW * 4)            # q_proj 输出 [rows][4096]（每头 256 q + 256 gate）
    sc("QF", SEQ * QW * 4)                # 紧凑的 q（8 头 × 256），供量化/attention 用
    sc("KR", SEQ * KW * 4)                # k_proj 输出 [rows][512]（2 个 kv 头）
    sc("VR", SEQ * KW * 4)                # v_proj 输出
    sc("ATTN", SEQ * QW * 4)              # 全注意力输出（乘门之前/之后同址）
    sc("AT8", SEQ * QW)
    sc("AS", SEQ * 4, 0x100)
    sc("OUT", SEQ * H * 4)
    sc("LG", SEQ * INTER * 4)
    sc("LU", SEQ * INTER * 4)
    sc("LSW", SEQ * INTER * 4)
    sc("LS8", SEQ * INTER)
    sc("LSS", SEQ * 4, 0x100)
    sc("DUMP", 4 << 20, 0x10000)          # 调试转储区（只在 -DQ35_DUMP 构建里用）
    sc("LOGITS", VOCAB * 4)               # 只算最后一行
    sc("ARGF", 128, 0x100)
    sc("ARGI", 128, 0x100)
    sc("SCRATCH_END", 0)
    o["SCRATCH_END"] = base[0]
    return o


# ---------------------------------------------------------------------------
def read_entries(data: bytes):
    magic, version, quant, flags, count, meta_len = struct.unpack_from("<IIIIQQ", data, 0)
    assert magic == 0x46575150, "不是 PQWF 文件"
    meta = json.loads(data[32:32 + meta_len])
    pos = 32 + meta_len
    ents = {}
    for _ in range(count):
        nl = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        name = bytes(data[pos:pos + nl]).decode()
        pos += nl
        nd = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        dims = list(struct.unpack_from(f"<{nd}Q", data, pos))
        pos += 8 * nd
        dtype = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        woff, wsize, soff, ssize = struct.unpack_from("<QQQQ", data, pos)
        pos += 32
        scount = struct.unpack_from("<I", data, pos)[0]
        pos += 8
        ents[name] = dict(dims=dims, dtype=dtype, woff=woff, wsize=wsize,
                          soff=soff, ssize=ssize, scount=scount)
    return meta, ents, align(pos, 32)


def rope_tables(npos: int):
    """M-RoPE 表（纯文本 t=h=w=p，退化成 1D 部分旋转）。

    与 ref_qwen35.rope_tables 完全一致（同样的 f32 运算顺序），
    这样 device 的 rope 与 numpy 参考只差浮点顺序。
    """
    f32 = np.float32
    inv_freq = (f32(1.0) / (f32(ROPE_THETA) ** (np.arange(0, ROT, 2, dtype=f32) / f32(ROT)))).astype(f32)
    pos = np.arange(npos, dtype=f32)[:, None]
    ang = (pos * inv_freq[None, :]).astype(f32)
    return np.cos(ang).astype(f32), np.sin(ang).astype(f32)


def gdn_index(L: int) -> int:
    return L - (L // 4)


def full_index(L: int) -> int:
    return L // 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pqwf", default="build/qwen35-0.8b-int8.pqw")
    ap.add_argument("--header", default="config/model_qwen35.h")
    ap.add_argument("--blob", default="build/qwen35_data.bin")
    ap.add_argument("--max-seq", type=int, default=MAXSEQ)
    ap.add_argument("--seq", type=int, default=SEQ)
    ap.add_argument("--ids", default="")
    ap.add_argument("--n-gen", type=int, default=0)
    a = ap.parse_args()
    globals()["SEQ"] = a.seq
    globals()["MAXSEQ"] = a.max_seq
    o = layout()
    src = memoryview(Path(a.pqwf).read_bytes())
    meta, ents, data_start = read_entries(bytes(src))
    n_layers = meta["num_hidden_layers"]

    blob = bytearray(o["SCRATCH_END"])
    ids = [int(x) for x in a.ids.split(",")] if a.ids else \
        [760, 6511, 314, 9338, 369]     # 默认提示（"The capital of France is"）
    ids = ids[: a.max_seq]
    struct.pack_into(f"<{len(ids)}i", blob, o["IDS"], *ids)
    struct.pack_into("<4i", blob, o["CTRL"], len(ids), a.n_gen, a.max_seq, 0)
    pids = np.arange(a.max_seq, dtype=np.int32)
    for s in range(3):
        struct.pack_into(f"<{a.max_seq}i", blob, o["PIDS"] + s * a.max_seq * 4, *pids)

    cos_v, sin_v = rope_tables(a.max_seq)
    struct.pack_into(f"<{a.max_seq*HALF}f", blob, o["COS"], *cos_v.reshape(-1))
    struct.pack_into(f"<{a.max_seq*HALF}f", blob, o["SIN"], *sin_v.reshape(-1))

    def copy_w(name, off, rows, cols, scale_off=None, scale_size=0):
        # 注意：N 必须是 32 的倍数（一个 tile 装 32 个输出通道）；BOUT/AOUT 是 16 宽，
        # 凑不满一个 tile，所以那两个保持行主序（kernel 侧仍走 gemm_nt）。
        """2D INT8 权重按 **tile-major** 存：[N/32][K/32][32][32]，每 tile 连续 1 KiB。

        行主序时一个 32x32 tile 是"32 段 32 B、段间相距 K 字节"，在 64 B burst 的
        DDR 上每次只用半个 burst；预分块后一个 tile 是 16 个满 burst，且 32 行落在
        连续 1 KiB 里（DRAM 页局部性）。字节数不变（只是置换），所以布局偏移不用改。
        例外：embedding/lm_head 不能这样存（查表要"一行连续"），仍是行主序。
        """
        e = ents[name]
        s0 = data_start + e["woff"]
        assert rows % 32 == 0 and cols % 32 == 0, (name, rows, cols)
        w = np.frombuffer(src, dtype=np.uint8, count=rows * cols, offset=s0)
        tiled = np.ascontiguousarray(
            w.reshape(rows // 32, 32, cols // 32, 32).transpose(0, 2, 1, 3))
        blob[off:off + rows * cols] = tiled.tobytes()
        if scale_off is not None and e["ssize"]:
            s2 = data_start + e["soff"]
            blob[scale_off:scale_off + scale_size] = src[s2:s2 + scale_size]

    def copy(name, off, size, scale_off=None, scale_size=0):
        e = ents[name]
        s = data_start + e["woff"]
        blob[off:off + size] = src[s:s + size]
        if scale_off is not None and e["ssize"]:
            s2 = data_start + e["soff"]
            blob[scale_off:scale_off + scale_size] = src[s2:s2 + scale_size]

    p = "model.language_model."
    copy(p + "embed_tokens.weight", o["EMB"], VOCAB * H, o["EMB_S"], VOCAB * 4)
    copy(p + "norm.weight", o["NF"], H * 4)

    def f32(name, off, size):
        e = ents[name]
        s = data_start + e["woff"]
        blob[off:off + size] = src[s:s + size]

    for L in range(n_layers):
        pre = f"{p}layers.{L}."
        if (L in FULL_LAYER) if n_layers == LAYERS else (L % 4 == 3):
            fl = full_index(L)
            wb = o["LFW_BASE"] + fl * o["LFW_STRIDE"]
            nb = o["LFN_BASE"] + fl * o["LFN_STRIDE"]
            for key, sub, rows, cols in (("QG", "self_attn.q_proj", 2 * QW, H),
                                         ("K", "self_attn.k_proj", KW, H),
                                         ("V", "self_attn.v_proj", KW, H),
                                         ("O", "self_attn.o_proj", H, QW),
                                         ("G", "mlp.gate_proj", INTER, H),
                                         ("U", "mlp.up_proj", INTER, H),
                                         ("D", "mlp.down_proj", H, INTER)):
                copy_w(pre + sub + ".weight", wb + FULL_W[key], rows, cols,
                       wb + FULL_W[key + "_S"], rows * 4)
            f32(pre + "input_layernorm.weight", nb + FULL_N["N1"], H * 4)
            f32(pre + "post_attention_layernorm.weight", nb + FULL_N["N2"], H * 4)
            f32(pre + "self_attn.q_norm.weight", nb + FULL_N["QN"], HD * 4)
            f32(pre + "self_attn.k_norm.weight", nb + FULL_N["KN"], HD * 4)
        else:
            gl = gdn_index(L)
            wb = o["LGW_BASE"] + gl * o["LGW_STRIDE"]
            nb = o["LGN_BASE"] + gl * o["LGN_STRIDE"]
            for key, sub, rows, cols in (("QKV", "linear_attn.in_proj_qkv", CONV_DIM, H),
                                         ("Z", "linear_attn.in_proj_z", KVW, H),
                                         ("BOUT", "linear_attn.in_proj_b", NQH, H),
                                         ("AOUT", "linear_attn.in_proj_a", NQH, H),
                                         ("OUT", "linear_attn.out_proj", H, KVW),
                                         ("G", "mlp.gate_proj", INTER, H),
                                         ("U", "mlp.up_proj", INTER, H),
                                         ("D", "mlp.down_proj", H, INTER)):
                if key in ("BOUT", "AOUT"):
                    # N=16，凑不满一个 32 行 tile → 保持行主序，走 gemm_nt
                    copy(pre + sub + ".weight", wb + GDN_W[key], rows * cols,
                         wb + GDN_W[key + "_S"], rows * 4)
                else:
                    copy_w(pre + sub + ".weight", wb + GDN_W[key], rows, cols,
                           wb + GDN_W[key + "_S"], rows * 4)
            f32(pre + "input_layernorm.weight", nb + GDN_N["N1"], H * 4)
            f32(pre + "post_attention_layernorm.weight", nb + GDN_N["N2"], H * 4)
            f32(pre + "linear_attn.norm.weight", nb + GDN_N["NORMW"], DV * 4)
            f32(pre + "linear_attn.conv1d.weight", nb + GDN_N["CONVW"], CONV_DIM * CONVK * 4)
            f32(pre + "linear_attn.A_log_exp", nb + GDN_N["AEXP"], NQH * 4)
            f32(pre + "linear_attn.dt_bias", nb + GDN_N["DTB"], NQH * 4)

    Path(a.blob).parent.mkdir(parents=True, exist_ok=True)
    Path(a.blob).write_bytes(blob)

    lines = ["// 自动生成（tools/convert_pqwf35.py），不要手改。",
             "#pragma once", "namespace mb {"]
    for k, v in (("Hidden", H), ("Inter", INTER), ("Vocab", VOCAB), ("Layers", n_layers),
                 ("Heads", NH), ("KvHeads", NKV), ("HeadDim", HD), ("Rot", ROT), ("Half", HALF),
                 ("QW", QW), ("KW", KW), ("Kvw", KVW), ("ConvDim", CONV_DIM),
                 ("ConvK", CONVK), ("GdnHeads", NQH), ("Dv", DV), ("Dk", DK),
                 ("Seq", a.seq), ("MaxSeq", a.max_seq), ("GdnLayers", GDN_LAYERS)):
        lines.append(f"inline constexpr int k{k} = {v};")
    lines.append("// 层类型：每 4 层一个循环，第 3/7/... 层是全注意力")
    lines.append("inline constexpr int kFullLayer[6] = {3, 7, 11, 15, 19, 23};")
    for k, v in o.items():
        lines.append(f"inline constexpr long k{k} = {v};")
    lines.append("}")
    Path(a.header).parent.mkdir(parents=True, exist_ok=True)
    Path(a.header).write_text("\n".join(lines) + "\n")
    print(f"header: {a.header}")
    print(f"blob: {a.blob} ({len(blob)} 字节), ids={ids}")
    print(f"weights 区 {o['BLOCK_END']} B, scratch {o['SCRATCH_END'] - o['BLOCK_END']} B, "
          f"logits@{o['LOGITS']}")


if __name__ == "__main__":
    main()
