# -*- coding: utf-8 -*-
"""Qwen3.5-0.8B 文本塔的 numpy 参考实现（**按设备数值语义**，可选纯 f32）。

为什么不是"把 qwen35_dense_demo 的 torch 代码搬过来"
----------------------------------------------------
demo 里的 `qwen35/` 是教学版，和官方 `transformers` 的 `modeling_qwen3_5.py`
有几处**会影响数值**的差异，以官方为准（都已核对源码）：

1. **q 的缩放**：官方 GDN 在递推里对 query 做 `query / sqrt(d_k)`，demo 没做；
2. **零中心 RMSNorm**：`Qwen3_5RMSNorm` 的 weight 初始化为 0，前向是
   `x_norm * (1 + weight)` → 权重转换时必须 `+1`；
3. **q/gate 的交错**：`q_proj` 输出 4096，先 view 成 `[T, 8, 512]` 再 `chunk(2, -1)`
   → 每个头内部 `[256 q | 256 gate]`，**不是** `[2048 q][2048 gate]`；
4. **短卷积无 bias 且带 silu**：`F.conv1d(..., bias=None)`，卷积后过 `hidden_act`
   （silu）；demo 有 bias、无激活；
5. **l2norm** 用 `rsqrt(sum(x^2)+1e-6)`（不是 `max(||x||,eps)` 归一化）；
6. **GDN 输出门**：`norm(o) * silu(gate)`，norm 是标准（非零中心）RMSNorm。

数值模式
--------
`quant=True`（默认，与设备一致）：
  * 权重：per-channel 对称 int8（`scale = absmax/127`，RNE 到 [-127,127]）；
  * 激活：per-token 对称 int8（`scale = absmax * f32(1/127)`，除法 + RNE）；
  * GEMM：i32 精确累加 → f32 → ×a_scale(行) → ×w_scale(列)（顺序与设备一致）；
  * GDN 状态 S：f32 常驻，**读**（S·k / S·q）和**秩 1 写**都经过 i8（per-row scale），
    更新本身在 f32 里做 —— 对应 kernels/gdn.h 的实现；
`quant=False`：同样的算法，但不量化（纯 f32，权重用反量化值）。
两种模式共用同一份控制流，便于把"量化误差"和"实现误差"分开看。
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np

f32 = np.float32
F32 = np.float32
INV127 = F32(1.0) / F32(127.0)      # 设备端常量 kInv127（f32）

# ---- 模型常量（Qwen3.5-0.8B，与 config.json 的 text_config 一致）----
H = 1024
INTER = 3584
VOCAB = 248320
LAYERS = 24
NH = 8
NKV = 2
HD = 256
ROT = 64                    # partial_rotary_factor=0.25 → 64
GDN_KH = 16
GDN_VH = 16
DK = 128
DV = 128
CONVK = 4
CONV_DIM = 2 * GDN_KH * DK + GDN_VH * DV     # 6144
FULL_LAYER = (3, 7, 11, 15, 19, 23)
ROPE_THETA = 1e7
EPS = 1e-6


# ---------------------------------------------------------------------------
# bf16 / safetensors 读取（不依赖 torch）
# ---------------------------------------------------------------------------
def bf16_to_f32(raw: bytes) -> np.ndarray:
    """bf16 → f32：bf16 就是 f32 的高 16 位，左移 16 位即精确还原。"""
    u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32)


SA_DTYPE = {"F32": ("<f4", 4), "BF16": (None, 2), "F16": ("<f2", 2)}


class SafeTensors:
    """最小 safetensors 读取器（只支持 F32/BF16/F16，够这个模型用）。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
            self.base = 8 + n

    def names(self):
        return [k for k in self.header if k != "__metadata__"]

    def get(self, name: str) -> np.ndarray:
        meta = self.header[name]
        dtype, shape = meta["dtype"], meta["shape"]
        start, end = meta["data_offsets"]
        fmt, size = SA_DTYPE[dtype]
        with open(self.path, "rb") as f:
            f.seek(self.base + start)
            raw = f.read(end - start)
        if dtype == "BF16":
            arr = bf16_to_f32(raw)
        else:
            arr = np.frombuffer(raw, dtype=fmt).astype(np.float32)
        return arr.reshape(shape)


# ---------------------------------------------------------------------------
# 量化原语（严格复刻设备的 tquant / mma / epilogue 语义）
# ---------------------------------------------------------------------------
# True 时模拟设备 i8 读路径（S 与 q/k 都量化成 i8），False 时全 f32
QUANT_STATE = False
# i8 读路径里"读侧量化"的方案（用来评估降误差的代价）：
#   "row"  : S 每行(128)一个 scale、q/k 每头一个 scale      ← 当前设备实现
#   "blk"  : S 每 32 列一个 scale、q/k 每 32 列一个 scale   （mma 本来就是按 32 列块做的）
#   "two"  : S 两级量化（i8 主项 + i8 残差 ≈ 15 bit）
#   "two_blk": 上面两个都开
# 读侧量化方案（评估降误差用）。当前只实现了 "row"（= 设备的 i8 路径，已验证
# 与设备逐层吻合到 4 位有效数字）；"blk"/"two" 的分析见 README §7.1，代码未实现。
QUANT_MODE = "row"


def _quant_blocks(x, blk: int = 32):
    """按最后一维每 blk 个一组做对称 int8 量化。返回 (q i8, scale[...])。"""
    shp = x.shape
    v = x.reshape(*shp[:-1], shp[-1] // blk, blk)
    amax = np.max(np.abs(v), axis=-1).astype(F32)
    sc = (amax * INV127).astype(F32)
    safe = np.where(sc == 0, F32(1.0), sc).astype(F32)
    q = np.clip(np.rint(v / safe[..., None]), -127, 127).astype(np.int8)
    return q.reshape(shp), sc


def _deq_blocks(q, sc, blk: int = 32):
    shp = q.shape
    v = q.reshape(*shp[:-1], shp[-1] // blk, blk).astype(F32)
    return (v * sc[..., None]).reshape(shp).astype(F32)


def _q_rows(x: np.ndarray):
    """per-(token,head) 量化：x [H][DK] f32 → (i8, scale[H] f32)。"""
    amax = np.max(np.abs(x), axis=1).astype(F32)
    s = (amax * INV127).astype(F32)
    safe = np.where(s == 0, F32(1.0), s).astype(F32)
    q = np.clip(np.rint(x / safe[:, None]), -127, 127).astype(np.int8)
    return q, s


def q_int8_per_channel(w: np.ndarray):
    """权重 per-channel 对称 int8：行主序第 0 维为 channel（Linear 的 out 维）。"""
    w = w.astype(F32)
    amax = np.max(np.abs(w), axis=1)
    scale = (amax * INV127).astype(F32)
    safe = np.where(scale == 0, F32(1.0), scale).astype(F32)
    q = np.rint(w / safe[:, None]).astype(F32)
    q = np.clip(q, -127, 127).astype(np.int8)
    q[scale == 0] = 0
    return q, scale


def q_act_rows(x: np.ndarray):
    """激活 per-token 量化：x [T][C] f32 → (i8 [T][C], scale [T] f32)。"""
    x = x.astype(F32)
    amax = np.max(np.abs(x), axis=1)
    scale = (amax * INV127).astype(F32)
    safe = np.where(scale == 0, F32(1.0), scale).astype(F32)
    q = np.rint(x / safe[:, None])
    q = np.clip(q, -127, 127).astype(np.int8)
    q[scale == 0] = 0
    return q, scale


def gemm_nt(a8, w8, a_scale, w_scale, quant=True):
    """out = (a8 · w8ᵀ) * a_scale(行) * w_scale(列)，i32 精确累加。"""
    if quant:
        acc = a8.astype(np.int32) @ w8.astype(np.int32).T
        out = acc.astype(F32) * a_scale[:, None]
        return (out * w_scale[None, :]).astype(F32)
    return (a8 @ w8.T).astype(F32)


def gemm_f32(x, w):
    return (x.astype(F32) @ w.T).astype(F32)


def linear(x, wq, ws, quant=True):
    """量化 Linear：输入先 per-token 量化。"""
    if quant:
        x8, xs = q_act_rows(x)
        return gemm_nt(x8, wq, xs, ws, True)
    return gemm_f32(x, wq.astype(F32) * ws[:, None])


def rmsnorm_zero_centered(x, w_plus1):
    """Qwen3_5RMSNorm：x * rsqrt(mean(x^2)+eps) * (1 + w)。"""
    x = x.astype(F32)
    var = np.mean(x * x, axis=-1, keepdims=True).astype(F32)
    inv = (F32(1.0) / np.sqrt(var + F32(EPS))).astype(F32)
    return ((x * inv) * w_plus1).astype(F32)


def rmsnorm_gated(o, w, gate):
    """Qwen3_5RMSNormGated：norm(o) * silu(gate)，per head over d_v。

    o/gate: [T][H][DV]；w: [DV]（标准 RMSNorm，非零中心）。
    """
    o = o.astype(F32)
    var = np.mean(o * o, axis=-1, keepdims=True).astype(F32)
    inv = (F32(1.0) / np.sqrt(var + F32(EPS))).astype(F32)
    y = ((o * inv) * w.reshape(1, 1, -1)).astype(F32)
    return (y * silu(gate)).astype(F32)


def silu(x):
    x = x.astype(F32)
    return (x / (F32(1.0) + np.exp(-x))).astype(F32)


def softplus(x):
    x = x.astype(F32)
    m = np.maximum(x, F32(0.0))
    return (m + np.log1p(np.exp(-np.abs(x)))).astype(F32)


def sigmoid(x):
    x = x.astype(F32)
    return (F32(1.0) / (F32(1.0) + np.exp(-x))).astype(F32)


def l2norm(x):
    ss = np.sum(x * x, axis=-1, keepdims=True).astype(F32)
    return (x * (F32(1.0) / np.sqrt(ss + F32(EPS)))).astype(F32)


def rope_tables(npos: int):
    """M-RoPE 表（纯文本 t=h=w=p → 与 1D 部分旋转相同）。

    inv_freq[i] = theta^(-2i/rot)，i in [0,32)；angle_i = p * inv_freq[i]。
    """
    inv_freq = (F32(1.0) / (F32(ROPE_THETA) ** (np.arange(0, ROT, 2, dtype=F32) / F32(ROT)))).astype(F32)
    pos = np.arange(npos, dtype=F32)[:, None]
    ang = (pos * inv_freq[None, :]).astype(F32)
    return np.cos(ang).astype(F32), np.sin(ang).astype(F32)


def apply_rope(x, cos, sin):
    """部分旋转 RoPE：只转前 64 维（rotate-half 在 64 维内）。"""
    half = ROT // 2
    xr = x[..., :ROT]
    x0, x1 = xr[..., :half], xr[..., half:]
    c, s = cos[:, None, :], sin[:, None, :]
    out = np.concatenate([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1).astype(F32)
    return np.concatenate([out, x[..., ROT:]], axis=-1).astype(F32)


def conv_causal(x, w, hist):
    """depthwise 因果短卷积（kernel=4，无 bias），x [T][C] f32。

    官方 `F.conv1d(x, w, padding=3, groups=C)[:, :, :T]` 等价于
        out[t] = Σ_j w[c][j] * x[t-3+j]      (x[<0] 取 hist)
    hist: [3][C] 上一步的最后 3 行（首块为 0）。
    返回 out 与新的 hist（最后 3 行）。
    """
    T = x.shape[0]
    buf = np.concatenate([hist, x], axis=0)           # [3+T][C]
    out = np.zeros((T, x.shape[1]), dtype=F32)
    for j in range(CONVK):
        out += buf[j:j + T] * w[:, j][None, :]
    new_hist = buf[-3:].copy() if T >= 3 else np.concatenate([buf, np.zeros((3 - T, x.shape[1]), F32)])[-3:]
    return out.astype(F32), new_hist


# ---------------------------------------------------------------------------
# 权重装载
# ---------------------------------------------------------------------------
class Weights:
    """把 safetensors 变成设备需要的形态（int8 权重 + f32 小张量）。"""

    def __init__(self, st: SafeTensors, quant: bool = True):
        self.st = st
        self.quant = quant
        self.emb8 = self.emb_s = None
        self.layers = []
        self.nf = None

    def mat(self, name: str):
        w = self.st.get(name)
        if self.quant:
            return q_int8_per_channel(w.reshape(w.shape[0], -1))
        return w.reshape(w.shape[0], -1).astype(F32), np.ones(w.shape[0], F32)

    def load(self):
        p = "model.language_model."
        self.emb8, self.emb_s = None, None
        w = self.st.get(p + "embed_tokens.weight")
        if self.quant:
            self.emb8, self.emb_s = q_int8_per_channel(w)
        else:
            self.emb8, self.emb_s = w.astype(F32), np.ones(w.shape[0], F32)
        self.nf = self.st.get(p + "norm.weight").astype(F32) + F32(1.0)   # 零中心
        for L in range(LAYERS):
            pre = f"{p}layers.{L}."
            d = {
                "n1": self.st.get(pre + "input_layernorm.weight").astype(F32) + F32(1.0),
                "n2": self.st.get(pre + "post_attention_layernorm.weight").astype(F32) + F32(1.0),
                "gate": self.mat(pre + "mlp.gate_proj.weight"),
                "up": self.mat(pre + "mlp.up_proj.weight"),
                "down": self.mat(pre + "mlp.down_proj.weight"),
            }
            if L in FULL_LAYER:
                d["type"] = "full"
                d["q"] = self.mat(pre + "self_attn.q_proj.weight")
                d["k"] = self.mat(pre + "self_attn.k_proj.weight")
                d["v"] = self.mat(pre + "self_attn.v_proj.weight")
                d["o"] = self.mat(pre + "self_attn.o_proj.weight")
                d["qn"] = self.st.get(pre + "self_attn.q_norm.weight").astype(F32) + F32(1.0)
                d["kn"] = self.st.get(pre + "self_attn.k_norm.weight").astype(F32) + F32(1.0)
            else:
                d["type"] = "linear"
                d["qkv"] = self.mat(pre + "linear_attn.in_proj_qkv.weight")
                d["z"] = self.mat(pre + "linear_attn.in_proj_z.weight")
                d["b"] = self.mat(pre + "linear_attn.in_proj_b.weight")
                d["a"] = self.mat(pre + "linear_attn.in_proj_a.weight")
                d["out"] = self.mat(pre + "linear_attn.out_proj.weight")
                d["conv"] = self.st.get(pre + "linear_attn.conv1d.weight").reshape(CONV_DIM, CONVK).astype(F32)
                d["norm"] = self.st.get(pre + "linear_attn.norm.weight").astype(F32)
                d["alog_exp"] = np.exp(self.st.get(pre + "linear_attn.A_log").astype(F32)).astype(F32)
                d["dtb"] = self.st.get(pre + "linear_attn.dt_bias").astype(F32)
            self.layers.append(d)
        return self


# ---------------------------------------------------------------------------
# 前向
# ---------------------------------------------------------------------------
class Ref:
    def __init__(self, weights: Weights, quant: bool = True, seq: int = 32):
        self.w = weights
        self.quant = quant
        self.seq = seq
        self.cos_t, self.sin_t = rope_tables(4096)
        self.reset()

    # -- 状态 ---------------------------------------------------------------
    def reset(self):
        self.S = [None] * LAYERS          # GDN: [H][DV][DK] f32
        self.hist = [None] * LAYERS       # GDN conv 的 3 行历史
        self.kcache = [None] * LAYERS     # full attn: [T][KW]
        self.vcache = [None] * LAYERS
        self.kcs = [None] * LAYERS
        self.vcs = [None] * LAYERS
        self.pos = 0

    # -- 组件 ---------------------------------------------------------------
    def _quant_heads(self, x, nheads, width):
        """把 [T][nheads*width] 按 (token, head) 行做 per-row 量化。"""
        T = x.shape[0]
        v = x.reshape(T * nheads, width)
        q, s = q_act_rows(v)
        return q.reshape(T, nheads * width), s.reshape(T, nheads)

    def gdn(self, L, x, d):
        """Gated DeltaNet：投影 → 短卷积 → L2norm → 递推 → 门控 norm → o_proj。"""
        T = x.shape[0]
        qkv = linear(x, *d["qkv"], self.quant)          # [T, 6144]
        z = linear(x, *d["z"], self.quant)              # [T, 2048]
        b = linear(x, *d["b"], self.quant)              # [T, 16]
        a = linear(x, *d["a"], self.quant)              # [T, 16]

        hist = self.hist[L]
        if hist is None:
            hist = np.zeros((3, CONV_DIM), F32)
        qkv_c, self.hist[L] = conv_causal(qkv, d["conv"], hist)
        qkv_c = silu(qkv_c)

        q = qkv_c[:, :GDN_KH * DK].reshape(T, GDN_KH, DK)
        k = qkv_c[:, GDN_KH * DK:2 * GDN_KH * DK].reshape(T, GDN_KH, DK)
        v = qkv_c[:, 2 * GDN_KH * DK:].reshape(T, GDN_VH, DV)

        q = (l2norm(q) * F32(1.0 / math.sqrt(DK))).astype(F32)
        k = l2norm(k)

        beta = sigmoid(b)                                # [T,16]
        g = (-d["alog_exp"][None, :] * softplus(a + d["dtb"][None, :])).astype(F32)
        alpha = np.exp(g).astype(F32)

        # 递推：两种模式（QUANT_STATE 开关），与设备的两条路径一一对应
        #   False：q/k 用 f32（L2norm 后）、S 全程 f32   → kernels/gdn.h::gdn_recurrence_f32
        #   True ：q/k 按 (token,head) 量化成 i8、S 每行量化成 i8 后再"读"
        #          （更新仍在 f32）                      → gdn_recurrence_i8
        S = self.S[L]
        if S is None:
            S = np.zeros((GDN_VH, DV, DK), F32)
        o = np.zeros((T, GDN_VH, DV), F32)
        for t in range(T):
            if QUANT_STATE:
                # 读侧量化：q/k 按 (token,head)、S 按行（都 absmax/127，RNE）；
                # 更新仍在 f32（用 f32 的 k）。这套写法与设备 i8 路径逐层吻合。
                q8, s_q = _q_rows(q[t])
                k8, s_k = _q_rows(k[t])
                amax = np.max(np.abs(S), axis=2)
                s_s = (amax * INV127).astype(F32)
                S8 = np.clip(np.rint(S / np.where(s_s == 0, F32(1), s_s)[:, :, None]),
                             -127, 127).astype(np.int8)
                pred = np.einsum("hvd,hd->hv", S8.astype(np.int32), k8.astype(np.int32))
                pred = ((pred.astype(F32) * s_s) * s_k[:, None]).astype(F32)
                u = (beta[t][:, None] * (v[t] - pred)).astype(F32)
                S = (alpha[t][:, None, None] * S +
                     np.einsum("hv,hd->hvd", u, k[t])).astype(F32)
                amax = np.max(np.abs(S), axis=2)
                s_s = (amax * INV127).astype(F32)
                S8 = np.clip(np.rint(S / np.where(s_s == 0, F32(1), s_s)[:, :, None]),
                             -127, 127).astype(np.int8)
                out = np.einsum("hvd,hd->hv", S8.astype(np.int32), q8.astype(np.int32))
                o[t] = ((out.astype(F32) * s_s) * s_q[:, None]).astype(F32)
            else:
                pred = np.einsum("hvd,hd->hv", S, k[t])
                u = (beta[t][:, None] * (v[t] - pred)).astype(F32)
                S = (alpha[t][:, None, None] * S +
                     np.einsum("hv,hd->hvd", u, k[t])).astype(F32)
                o[t] = np.einsum("hvd,hd->hv", S, q[t])
        self.S[L] = S

        og = rmsnorm_gated(o, d["norm"], z.reshape(T, GDN_VH, DV))
        return linear(og.reshape(T, GDN_VH * DV), *d["out"], self.quant)

    def attn(self, L, x, d):
        """全注意力：q(+gate) / k / v → qk-norm → 部分 RoPE → GQA → 输出门。"""
        T = x.shape[0]
        qg = linear(x, *d["q"], self.quant)              # [T, 4096]
        qg = qg.reshape(T, NH, 2 * HD)
        q = qg[:, :, :HD].reshape(T, NH * HD)
        gate = qg[:, :, HD:].reshape(T, NH * HD)
        k = linear(x, *d["k"], self.quant)               # [T, 512]
        v = linear(x, *d["v"], self.quant)

        q = q.reshape(T, NH, HD)
        k = k.reshape(T, NKV, HD)
        v = v.reshape(T, NKV, HD)
        q = rmsnorm_zero_centered(q, d["qn"])
        k = rmsnorm_zero_centered(k, d["kn"])
        cos = self.cos_t[self.pos:self.pos + T]
        sin = self.sin_t[self.pos:self.pos + T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # KV cache（per-token int8）
        if self.quant:
            k8, ks = q_act_rows(k.reshape(T, NKV * HD))
            v8, vs = q_act_rows(v.reshape(T, NKV * HD))
            self.kcache[L] = k8 if self.kcache[L] is None else np.concatenate([self.kcache[L], k8])
            self.vcache[L] = v8 if self.vcache[L] is None else np.concatenate([self.vcache[L], v8])
            self.kcs[L] = ks if self.kcs[L] is None else np.concatenate([self.kcs[L], ks])
            self.vcs[L] = vs if self.vcs[L] is None else np.concatenate([self.vcs[L], vs])
            q8, qs = q_act_rows(q.reshape(T, NH * HD))
        else:
            self.kcache[L] = k.reshape(T, NKV * HD) if self.kcache[L] is None else \
                np.concatenate([self.kcache[L], k.reshape(T, NKV * HD)])
            self.vcache[L] = v.reshape(T, NKV * HD) if self.vcache[L] is None else \
                np.concatenate([self.vcache[L], v.reshape(T, NKV * HD)])
            q8 = q.reshape(T, NH * HD)

        KL = self.kcache[L].shape[0]
        group = NH // NKV
        out = np.zeros((T, NH, HD), F32)
        scale = F32(1.0 / math.sqrt(HD))
        for h in range(NH):
            kv = h // group
            if self.quant:
                qh = q8[:, h * HD:(h + 1) * HD].astype(np.int32)
                kh = self.kcache[L][:, kv * HD:(kv + 1) * HD].astype(np.int32)
                s = qh @ kh.T
                s = (s.astype(F32) * qs[:, None]) * self.kcs[L][None, :]
                s = (s * scale).astype(F32)
            else:
                # 注意：这里必须用 einsum 而不是 `@`。
                # 本机的 numpy 2.3.5 + OpenBLAS 0.3.30 在小矩阵（如 [15][15]@[15][256]）
                # 上会返回 NaN/垃圾（实测 100 次里 32 次非有限），einsum 走自己的
                # 循环、结果正确且确定。大 GEMM 没问题。
                s = (np.einsum("td,sd->ts", q8[:, h * HD:(h + 1) * HD],
                               self.kcache[L][:, kv * HD:(kv + 1) * HD]) * scale).astype(F32)
            # 因果 mask：query 的绝对位置 self.pos+t，key 的位置 0..KL
            qi = self.pos + np.arange(T)[:, None]
            ki = np.arange(KL)[None, :]
            s = np.where(ki <= qi, s, F32(-np.inf)).astype(F32)
            m = np.max(s, axis=1, keepdims=True)
            e = np.exp(s - m).astype(F32)
            den = np.sum(e, axis=1, keepdims=True).astype(F32)
            p = (e / den).astype(F32)
            if self.quant:
                # 与设备一致：把 v_scale 折进 P，再 per-token 量化 P
                ps = (p * self.vcs[L][None, :]).astype(F32)
                pmax = np.max(np.abs(ps), axis=1)
                pscale = (pmax * INV127).astype(F32)
                safe = np.where(pscale == 0, F32(1.0), pscale).astype(F32)
                p8 = np.clip(np.rint(ps / safe[:, None]), -127, 127).astype(np.int8)
                acc = p8.astype(np.int32) @ self.vcache[L][:, kv * HD:(kv + 1) * HD].astype(np.int32)
                o = acc.astype(F32) * pscale[:, None]
            else:
                o = np.einsum("ts,sv->tv", p, self.vcache[L][:, kv * HD:(kv + 1) * HD]).astype(F32)
            out[:, h, :] = o
        out = (out.reshape(T, NH * HD) * sigmoid(gate)).astype(F32)
        return linear(out, *d["o"], self.quant)

    def mlp(self, x, d):
        g = linear(x, *d["gate"], self.quant)
        u = linear(x, *d["up"], self.quant)
        return linear(silu(g) * u, *d["down"], self.quant)

    # -- 整模型 --------------------------------------------------------------
    def forward(self, ids, stages=None):
        ids = np.asarray(ids, dtype=np.int64)
        T = ids.shape[0]
        x = (self.w.emb8[ids].astype(F32) * self.w.emb_s[ids][:, None]).astype(F32)
        stages = stages if stages is not None else {}
        stages["embed"] = x.copy()
        for L in range(LAYERS):
            d = self.w.layers[L]
            h = rmsnorm_zero_centered(x, d["n1"])
            mix = self.gdn(L, h, d) if d["type"] == "linear" else self.attn(L, h, d)
            x = (x + mix).astype(F32)
            h2 = rmsnorm_zero_centered(x, d["n2"])
            x = (x + self.mlp(h2, d)).astype(F32)
            stages[f"layer{L}"] = x.copy()
        hf = rmsnorm_zero_centered(x, self.w.nf)
        if self.quant:
            h8, hs = q_act_rows(hf)
            logits = gemm_nt(h8, self.w.emb8, hs, self.w.emb_s, True)
        else:
            logits = gemm_f32(hf, self.w.emb8)
        stages["logits"] = logits
        self.pos += T
        return logits


def load(model_dir: str | Path, quant: bool = True) -> Ref:
    st = SafeTensors(Path(model_dir) / "model.safetensors-00001-of-00001.safetensors")
    return Ref(Weights(st, quant).load(), quant=quant)
