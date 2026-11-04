# Qwen3.5-0.8B 文本塔（Tile 层 + YAICore 后端）

在 `tile_api_llvm_ds/qwen35_ops/` 里用 Tile 层 API 实现 Qwen3.5-0.8B 的**完整文本塔**
（24 层 = 18 层 Gated DeltaNet + 6 层全注意力），权重走
`bf16 → int8(PQWF 扩展) → guest blob`，在 Tile-NPU-Sim 上跑通并与 numpy 参考对拍。

**当前状态一句话**（2026-11 在本机 Ubuntu 上重新验证，**默认 i8 状态读**）：
端到端跑通（prefill 32 = 10,865,392 条动态指令），**L0–L15 与参考逐位一致
（rel ≈ 1.2e-7 ~ 1.6e-6，f32 舍入量级）**，L16 起因 int8 量化决策在舍入边界上
翻转而发散到 ~1e-2，最终 logits 与参考 **argmax 一致、corr 0.996621、top-10 9/10**；
**贪心生成 10 个 token 与参考逐个相同**（"The capital of France is" →
" the capital of France.\nThe capital of France"）。
（对拍时参考必须开 `--quant-state`，否则两边状态读方式不同、L0 就差 5e-2。）

> **本轮（Ubuntu 迁移后）修了什么**：ISA 删除除法 / `mma.nn` / gather 索引改
> u32 之后，算子与参考里有 5 处没跟上；另外 GDN 状态读的默认值被写成了 i8
> （应为 f32）。修复清单与复现命令见 §0。

---

## 0. 本机（Ubuntu）复现与修复记录

环境：`../llvm-project/build/bin/clang++`（LLVM 21.1.7 + XAICore）、
`riscv64-unknown-elf-` 工具链、`../Tile-NPU-Sim` 的 `libsim_core.a`、系统
`python3`（numpy 1.x）。原 `npucc/sim_bridge` 已丢失，本仓库的
`tools/sim_bridge.cpp` 是重建版（guest 装 `0x80000000`、blob 装 `0x81000000`、
`x31` = blob 基址、按 `--dump FILE@OFF:SIZE` 回写）。

**算子/参考的修复（都是 ISA 变更后没跟上的地方）：**

| # | 位置 | 问题 | 修复 |
|---|---|---|---|
| 1 | `attn256.h::pv_block256` | 用 `mma.nn`（右操作数不转置）——ISA 已删除 nn | 转置视图 + `rt=true`（`mma.nt`） |
| 2 | `attn256.h`、`gdn.h`、`mlp.h` | 大量 `div`（sigmoid/softmax 分母/silu）——ISA 已删除法 | 改成 `rcp.approx` + `mul`（模拟器里 rcp 就是精确 1/x） |
| 3 | `embedding.h` | gather 索引向量 dtype 用 i32 | 新 ISA 要求 **u32**（`idv.set_dtype(U32)`） |
| 4 | `include/tile.h` | `Tile::dequant` 用旧的回值签名（builtin 已改成 out 参数） | `void dequant(scale, Acc&)` |
| 5 | **`gdn.h` / `check_qwen35.py`** | 设备用 i8 状态读、参考却用 f32 读（`QUANT_STATE=False`）——两边不一致，L0 就差 5e-2 | 默认保持 i8（任务书 6.2 的原设计），参考加 `--quant-state` 对齐；此时 L0–L15 是 1e-7 量级 |

**参考实现的修复**：`conv_causal` 在 `T<3` 时历史更新写错（把零补到末尾再取最后
3 行，decode 的 T=1 拿到 `[新token,0,0]`），导致"增量 decode"与"整段重算"从 L0
起就发散；改成 `buf[-3:]` 后两者一致（1e-7 级）。另外参考改为**直接读 .pqw**
（与设备同一份 int8 权重，mmap 零拷贝），启动从分钟级降到秒级；GEMM 用
float64 BLAS + `rint`（i8 乘积和 < 2^53，与整数路径逐位相同）。

**工具**：`tools/chat_qwen35.py`（文本进/文本出）、`--stages` 的逐层残差转储
（slot `900000 + L*6000`）、`make bridge/chat`。

---

## 1. 目录

```
qwen35_ops/
├── Makefile                    # pqw / header+blob / guest / stats / check
├── include/tile.h              # 直接拷 qwen3_ops 的 Tile 层 API
├── test_include/               # 裸机 C++ 头垫片
├── config/model_qwen35.h       # 自动生成的布局头（转换脚本产出）
├── kernels/
│   ├── config.h  io.h  matrix.h            # 拷 qwen3_ops（config 常量按 Qwen3.5 改）
│   ├── gemm.h  quant.h  rmsnorm.h  mlp.h   # 拷 qwen3_ops（未改逻辑）
│   ├── embedding.h                          # 拷 qwen3_ops
│   ├── rope_gqa.h                           # 拷 qwen3_ops（rotate-half，按 32 列分块）
│   ├── attn256.h     # 全注意力：hd=256 + GQA 4:1 + 输出门 + KV 分块 online softmax
│   ├── rope_mrope.h  # 部分旋转(25%) + M-RoPE（纯文本退化，见 §4）
│   ├── gdn.h         # GDN：门控 + 短卷积 + L2norm + delta 递推 + gated RMSNorm
│   └── qwen35.h      # 汇总头
├── src/
│   ├── crt0.S
│   └── main.cpp      # 24 层驱动 + prefill/decode + 贪心 argmax + 可选调试转储
└── tools/
    ├── download_qwen35.py   # 从 hf-mirror 下载（HF 主站不通）
    ├── quantize_qwen35.py   # safetensors(bf16) → PQWF 扩展 .pqw
    ├── convert_pqwf35.py    # .pqw → guest blob + config/model_qwen35.h
    ├── ref_qwen35.py        # numpy 参考（可切换 W8A8 / 纯 f32）
    ├── check_qwen35.py      # 模拟器对拍（端到端 + --stages 逐层）
    ├── gen_dump_notes.md    # 调试转储的槽位约定（-DQ35_DUMP）
    └── reference/           # 官方 modeling_qwen3_5.py 快照（只作语义对照）
```

## 2. 快速开始

```bash
make pqw                 # safetensors(1.7 GB) → build/qwen35-0.8b-int8.pqw (756 MB)
make header blob         # → build/qwen35_data.bin (788 MB) + config/model_qwen35.h
make guest               # 编译 guest（静态 59,395 条指令，237,600 B）
make bridge              # 重建模拟器桥（本机没有 npucc/）
make check               # 模拟器跑 + 与 numpy 参考对拍（PQW 参考，秒级）
make stats               # 只打印静态指令数
make dump                # -DQ35_DUMP 版 guest（逐层残差转储）
python3 tools/check_qwen35.py --stages --guest build/qwen35d.bin
python3 tools/chat_qwen35.py "你好，请用一句话介绍你自己。" --max-new 16
```

依赖变量：`CLANG=../llvm-project/build/bin/clang++`、`LINK_LD=link.ld`、
`SRC=../qwen3.5-0.8b-bf16`、`PY=python3`、`TILE_NPU_SIM=../Tile-NPU-Sim`。

模拟器桥（`npucc/`）是**本轮重建**的：原目录整份丢失，`build_sim.py` 与
`sim_bridge` 都不在了。重建版直接用系统 clang++ 编译 Tile-NPU-Sim 的库源码
（不需要 xmake / cli11），接口与旧脚本完全兼容（`guest data out param steps extra_mib`）。
重建后用 0.6B 基线验证过一致性（见 §7）。

## 3. 权重格式（PQWF 扩展）

PQWF v1 的规则是"ndim==1 才存 F32"，Qwen3.5 需要三处扩展：

1. **显式混合 dtype**：只有 Linear/Embedding 的 2D `*.weight` 走 `INT8_PC`
   （per-channel 对称，`scale=absmax/127`，RNE），其余全部 F32 原样存。
   典型反例是 `linear_attn.conv1d.weight`：形状 [6144,1,4] 但**不能量化**
   （逐通道只有 4 个系数，per-channel absmax 会把它压成 1 个值）。
2. **零中心 RMSNorm 折叠**：官方 `Qwen3_5RMSNorm` 前向是 `x_norm * (1 + w)`
   （weight 初始化为 0），写盘时直接存 `1+w`，device 端退化成普通 RMSNorm。
   `Qwen3_5RMSNormGated`（GDN 输出 norm）是标准权重，**不加 1**。
   （实测统计确认了这个区分：input_layernorm 均值 0.24、max 1.0；
   linear_attn.norm 均值 0.95、min 0.52。）
3. **派生标量**：`A_log` 存成 `exp(A_log)`，device 端只需要这个常量因子。

另外 M-RoPE 表、状态区、scratch 的布局都写在 blob 的布局头里（`convert_pqwf35.py` 生成），
C++ 侧不写任何硬编码地址。

## 4. 与 `qwen35_dense_demo` 的差异（以官方 `modeling_qwen3_5.py` 为准）

demo 的 torch 参考在本机跑不了（没装 torch），所以转 numpy 时以**官方实现**校对，
发现 demo 有 5 处会影响数值的偏差，均已按官方修正：

| # | 官方实现 | demo |
|---|---|---|
| 1 | GDN 递推里 `query = query / d_k**0.5` | 没做 |
| 2 | `Qwen3_5RMSNorm` 零中心（`1+w`） | 普通 RMSNorm |
| 3 | `q_proj` 输出 4096 后 `view(T,8,512).chunk(2,-1)` → **每头内部** `[256 q \| 256 gate]` | `[2048 q][2048 gate]` 顺序切 |
| 4 | 短卷积**无 bias** 且卷积后过 silu | 有 bias、无激活 |
| 5 | `l2norm` = `x*rsqrt(sum(x²)+1e-6)`（不除 dim） | `F.normalize`（等价但 eps 位置不同） |

M-RoPE：纯文本输入 position_ids 是 arange 复制成 4 行（t=h=w），
所以"按 [11,11,10] 交错取坐标"退化成普通 1D 部分旋转。因此 cos/sin 就是
`[max_seq][32]` 的 1D 表；kernel 侧不做区分（接视觉塔时才需要按位置三元组摊平）。

## 5. 实现要点

### 5.1 全注意力（6 层，`attn256.h`）
hd=256 时输出方向有 **8 个 32 宽块**，所以 r0..r7 八个常驻 acc；K 方向 8 个块
是循环里逐块载入的（一个 qt 反复用）——**不能把 q/k/v 各 8 个 tile 全塞寄存器
（24 > 16 会直接 `ran out of registers`）**。寄存器：acc 10 / tile 4 / vec 8。

输出门按头切片：`attn_out[t][h*256+d] *= sigmoid(q_proj_out[t][h*512+256+d])`。
`sigmoid` 用 `1/(1+e^-x)`（`fill(1)` + `div`）——**不能写成 `x/(1+e^-x)`（那是 silu）**，
这是本轮踩过的一个真 bug（乘上了带符号的门值）。

### 5.2 GDN（18 层，`gdn.h`）
数学按官方逐 token 递推：`S ← αS`，`pred = S·k`，`S += β(v−pred)⊗k`，`o = S·q`。

* **S 的布局**：官方是 (d_k,d_v)，这里存 **(d_v,d_k)**（转置）。递推式同形，但
  "读"从列归约变成行内点积 —— 硬件**只有行归约没有列归约**，这个转置是必须的。
* **S 按 32 行（d_v）× 128 列（d_k）的行块处理**：更新式里没有跨 d_v 的项，
  行块之间完全独立。
* **softplus 的 8 次多项式**：设备没有 log，用
  `softplus(x) = max(x,0) + log1p(e^{-|x|})`，`log1p` 在 [0,1] 上用 deg-8
  Chebyshev 拟合（f32 Horner 最大误差 1.1e-7）。实测 `beta/alpha` 与参考差
  7e-8 / 2e-8。
* 短卷积：因果 depthwise、kernel=4、**无 bias**、卷积后 silu；跨 chunk 的 3 行历史
  持久化在 blob 的 HIST 区（每层 3×6144 f32 = 72 KB）。

### 5.3 寄存器/描述符约定（沿用 0.6B 的实测结论）
* 描述符三层写法：`config_*`（循环外）→ `set_zone`（形状不变那层）→
  `load_*_at`（只写 base）。
* **新建的 Acc 没有 zone**：`fill/div/mul/quant_ext` 之前必须 `set_zone`，
  否则要么静默不干活（zone=0×0），要么 `invalid_configuration`。
* **mma 的目的 acc 必须是 i32**：`to_fp32()` 之后要重新 `set_dtype(I32)`，
  否则下一轮 mma 报 `invalid_configuration`（0.6B 的 `pv_block` 里有同样的注释）。
* 跨层复用**必须加层偏移**：`kGW_*`/`kGN_*` 这些常量是"第 0 层的绝对偏移"，
  每层要再加 `gi * kLGW_STRIDE`。漏加会让 L0 对、其余层全用 L0 的权重
  （本轮踩过，现象是"L0 逐位一致、L1 起完全对不上"）。

## 6. 实测数据

### 6.1 规模与指令数

| 项 | Qwen3.5-0.8B | 对照：Qwen3-0.6B |
|---|---|---|
| 静态指令数（`main.cpp` 汇编里 tab 开头的助记符行） | **54,771** | 40,178 |
| guest 二进制 | **219,104 B** | 160,732 B |
| 权重 blob | 787,797,248 B（其中权重区 777 MB） | 791 MB |
| prefill 32 token 动态指令数 | **10,865,392**（i8 状态读） | 2,369,787 |
| prefill 5 token 动态指令数 | 4,434,435 | — |
| 层数 / 参数量 | 24 层（18 GDN + 6 full），752.8 M | 28 层，596 M |

prefill-32 是 0.6B 的 **4.24×**，来源大致是三块：
24/28 层 × 更大的 MLP（inter 3584 vs 3072）× GDN 的逐 token 递推。
GDN 递推本身约 **2.4M** 条（= 编码器细节见下），约占 1/4。

### 6.2 逐层数值比对（device vs `ref_qwen35.py` 的 W8A8 参考）

| 层 | 相对误差 | 说明 |
|---|---|---|
| L0 / L1 / L2（GDN） | 1.18e-7 / 1.44e-7 / 1.84e-7 | f32 舍入量级（i8 读 vs i8 读） |
| L3（首个全注意力） | 1.90e-7 | 同上 |
| L15 | 1.59e-6 | 仍是 f32 舍入 |
| L16 | 5.78e-3 | 开始发散（int8 舍入决策翻转） |
| L19 / L23 | 3.74e-2 / 6.74e-2 | |

**L4 的跳变不是实现 bug**，是这套 W8A8 数值本身的灵敏度：

```
参考自身：把 embedding 扰动 1e-7（相对），其它一切不变
  → L1/L2/L3 变化 1.1e-7 / 1.0e-7 / 1.3e-7
  → L4 变化 9.297e-03     ← 与 device-vs-ref 的 9.297e-03 完全同量级
  → L23 变化 7.6e-02      ← 与 device-vs-ref 的 5.2e-02 同量级
```

也就是：从 L4 起，某个 per-token absmax 量化的**舍入决策**在 1e-7 的扰动下翻转，
误差被"记住"并缓慢放大。**device 与参考的差异已经落在参考自身的可复现极限上**——
再往下要求"逐位一致"，只能要求两边每一次 f32 运算的顺序完全相同（int8 GEMM 的
i32 累加部分已经是逐位一致的，见下）。

### 6.3 端到端 logits（5 token prompt，`check_qwen35.py`）

```
max|Δ| 1.077   mean|Δ| 0.1708   |Δ| 中位数 0.1427   corr 0.996621
argmax dev 279 / ref 279        top-10 重合 9/10
|Δ| > 1.0 的 token: 3 / 248320
```

贪心生成（prompt = `760, 6511, 314, 9338, 369`，即 "The capital of France is"）：
设备与参考 **10 个 token 逐个相同** → `The capital of France.\nThe capital of France`。

top-5 对照（`279` = `" the"`）：

```
dev: ' the' 14.86  ' located' 14.01  ' Paris' 13.86  ' a' 13.79  ':' 13.74
ref: ' the' 14.80  ' Paris'   14.25  ' located' 13.98  '\n' 13.73 ':' 13.60
```

即：**"The capital of France is" → " the"**（argmax 与参考一致），第二名在
`located`/`Paris` 之间互换（差 0.4 的 logit）。

`max|Δ| = 13` 是 24 个 token 里的极端值（个别 token 的量化决策翻转），
中位数 0.16、且它们都不在 top-k 里。

### 6.4 各分量与参考的对齐精度（layer 0/3 的逐阶段转储）

| 阶段 | 相对误差 | 备注 |
|---|---|---|
| embedding | **0**（逐位） | gather+widen+scale |
| qkv 投影（int8 GEMM） | 1.4e-7 | i32 累加逐位；只有 epilogue 的 f32 乘法 |
| 短卷积 + silu | 1.7e-7 | 含 3 行历史 |
| beta / alpha（门控） | 7e-8 / 2e-8 | softplus 多项式误差 |
| q/k 的 L2norm（+q 的 1/√128） | 1.5e-7 | |
| GDN 递推（pred/u/S/o） | 2e-7 | f32 精确读 |
| gated RMSNorm + silu | 2.1e-7 | |
| o_proj / MLP（gate/up/swiglu/down） | ≤2.1e-7 | |
| 全注意力 q/k/v/norm/rope | 7e-8 ~ 1e-7 | |
| 全注意力输出（含 int8 量化） | 1.1e-2 | 这一档是 int8 量化的固有代价 |

## 7. 与任务书设计决策的偏差（都是实测后改的，必须说明）

### 7.1 GDN 的"读"没有走 i8 mma，改成 f32 内存常驻

决策 6.2/6.3 要求：S 常驻 acc（f32）、每行一个 scale 量化成 i8 后走 mma 读、
外积也走 i8。第一版就是这么写的，但跑出来 `pred` 恒为 0（三个独立信号：
`row_absmax` 的结果、`tquantx` 出的 tile、mma 的 pred 全为 0，而 acc 里的 S 非零
且按 `S += βv⊗k` 的发散轨迹增长）。

**两条路径都跑通了**（同一个 `kGdnI8Read` 开关切换），实测对比：

| | i8 读版（决策 6.2 原设计） | f32 读版（**当前默认**） |
|---|---|---|
| S 的读 | 量化成 i8 → `mma.nt`（per-row scale） | f32 精确读（每 token 重载） |
| 状态流量 | **~2 MiB/层**（18 层 × 2 MiB = 36 MiB/chunk） | 72 MiB/token（prefill-32 = 2.2 GiB） |
| 静态指令数 | 53,954 | **53,355** |
| guest 大小 | 215,836 B | **213,440 B** |
| prefill-5 动态 | 4,356,825 | **4,122,375** |
| prefill-32 动态 | 10,506,211 | **9,898,510** |
| 逐层误差 (L0) | 5.2e-2 | **1.9e-7** |
| 逐层误差 (L23) | 1.1e-1 | **~5e-2** |
| 端到端 logits | corr 0.9935 / argmax 一致 | **corr 0.9959 / argmax 一致 / top-10 9/10** |

结论：**功能模拟器上 f32 版全面更好**（误差小 5 个数量级），因为模拟器只数
指令、不算带宽；而 i8 版的价值全在**带宽**上（36 MiB/chunk vs 2.2 GiB）。所以默认 f32，
i8 版保留一行开关，等有带宽模型或真机时切换。

> **默认是 i8 读**（`kGdnI8Read = true`，任务书 6.2 的原设计）。对拍时参考必须
> 开 `--quant-state`（`QUANT_STATE=True`）：两边都用 i8 读时 L0–L15 是 1e-7 量级、
> 端到端 corr 0.9966、top-10 9/10、生成 10/10 个 token 相同；若参考用 f32 读而
> 设备用 i8 读，L0 就差 5e-2（这是两条路径的差异，不是实现 bug）。
> 改成 f32 读只需把 `kGdnI8Read` 设为 `false`（数值更好：corr 0.9962、
> top-10 10/10，但状态访存大 48 倍）。

**这段排查的最大收获是：三个"以为是后端 bug"的现象，最后全是我们自己的字节偏移错。**
逐段验证（`sim_gdn_pred_guest.cpp`，384 条指令）证明指令语义都是对的：

| 段 | 做法 | 结果 |
|---|---|---|
| [1] | 4 个**常驻** acc 的合并 absmax（`def` + `vmax` 链） | **Δ = 0** ✅ |
| [2] | `tquantx`（`Acc::quant_ext`）+ `tstore` 直存 i8 | **Δ = 0** ✅ |
| [4] | `mma.nt`（1×32 k 块 ×4 累加）+ `acvt` + ×scale 反量化 | **相对 1e-4** ✅ |

配套的 5 变体对照（`sim_tquant_guest.cpp`，702 条指令）也是全 Δ=0：常驻/重载、
scale 装载/常量、tile strides 配与不配，四种组合都对。

### 7.1.1 i8 路径上找到的四个真 bug（都在我们自己的 kernel 里）

1. **更新步里 k 的 f32 向量偏移**把 `h*kDk`（i8 字节数）当 f32 偏移用了，漏乘 4；
2. **`quant_heads` 的输出偏移**用了绝对列 `col` 而不是块内相对列 `ocol`
   → h>0 的量化结果写进别的 token 的行；
3. **i8 tile 的写出跨步**用了 f32 的 `col_bytes`（4）而不是 i8 的 1；
4. **scale 写出时 lane 之间的步长**用了 `scr.col_bytes`（4）而不是 `scr.row_bytes`（64）
   → `[rows][16]` 的 scale 只有 (0,0) 落对位置，其余全错。**这个是 pred 错的最后一环**。

而更早误判过的两个"后端 bug"（"常驻 acc 上 reduce 返回 0"、后来"`tquantx` 写错
tile"）都是复现器自身的寻址 bug（acc 的 row stride 写成 128 字节、实际应是 512；
k 指针没按列块推进）造成的假象。

### 7.1.1b 访存流量：i8 读路径改变了什么（按代码算，不是估的）

GN D 递推里 S 的访存单位是"行块"= 32 行(d_v) × 128 列(d_k) f32 = 16 KiB = 4 个 acc。
每层有 16 头 × 4 行块 = 64 个 (head, 行块)。

| 路径 | 每个 (head, 行块) 的访存 | 每层每 chunk | 与 T 的关系 |
|---|---|---|---|
| f32 | **每 token** 12 × aload + 4 × astore（4 KiB 每个）= 64 KiB | 64 × T × 64 KiB = **T × 4 MiB** | 线性 |
| i8 | 整段 token **只** 4 × aload + 4 × astore = 32 KiB | 64 × 32 KiB = **2 MiB** + q8/k8 ≈ 0.6 MiB | **与 T 无关** |

| 场景 | f32 状态流量 | i8 状态+缓冲 | 降幅 | 含权重(741 MB)的总流量 |
|---|---|---|---|---|
| prefill-32（1 chunk） | 2304 MiB | 47.3 MiB | **48.7×** | 3.19 GB → **0.83 GB（−74%）** |
| prefill-8 | 576 MiB | 38.3 MiB | 12.2× | — |
| decode T=1（每 token） | 72 MiB | 36.4 MiB | **2.0×** | 0.85 GB → 0.81 GB（−4%） |

三条结论：

1. **i8 读的收益与 chunk 长度成正比**（比值 = 2T：T=32 时 48.7×，T=1 时只有 2×）。
   因为 f32 是"每 token 重读"，i8 是"每 chunk 只读写一次"。
2. **prefill 时状态流量从 72% 的占比掉到 5.7%**，权重流（741 MB）变成绝对主项（98%）；
   而 **decode 时权重流本来就占 96%**，所以 i8 读在 decode 上只省 4%——
   decode 要省带宽只能靠**批处理**或权重驻留，靠状态量化没用。
3. 代价是**指令数 +6.1%**（prefill-32 9.90M → 10.51M）。折算"每指令搬运的字节"：
   f32 323 B/instr → i8 79 B/instr，即 i8 版对带宽的依赖降到 1/4。
   在带宽受限的机器上是 4× 收益，在发射受限的机器上略亏。

（激活/mix/KV/卷积历史的流量两条路径相同，所以不进入"变化"的统计；量级 ~40 MB/chunk，
 相比 777 MB 权重可以忽略。）

### 7.1.2 逐位置 / 跨 chunk 的误差趋势（新测，回答"i8 读会不会累积"）

`--positions N` 模式：用 N 个确定性 id 的 prompt，把 **每个位置（行）** 的残差流单独
对拍（只挑 L0/L11/L23 三层，行跨步 34000，够 32 行）。

| 测试 | f32 读 | i8 读 |
|---|---|---|
| L0 逐位置相对误差，chunk 1（token 0–31） | 1e-5 … 3.3e-2（部分位置 ~1e-5） | 2.8e-2 … 7.8e-2 |
| L0 逐位置相对误差，chunk 2（token 32–63） | 0 … 3.0e-2（多数 ~2e-5） | 2.7e-2 … 5.2e-2 |
| 趋势 | **不随位置增长** | **不随位置增长，跨 chunk 也不增长** |
| S(L0) 与参考的差，32 / 64 token | 3.0e-4 / 2.2e-4 | —（逐层 5.2e-2 见上表） |
| L11 / L23 | ~1.0（见下） | ~1.0（数值几乎相同） |

两个结论：

1. **i8 状态读的误差是"平台"，不累积**：既不在 chunk 内随位置涨，也不跨 chunk 涨
   （chunk 2 与 chunk 1 同量级）。这与 2609.04098 的机制解释（delta 规则沿当前 key
   方向覆写、噪声被压成平台）一致。
2. **L11/L23 的逐层相对误差不能当误差指标**：参考自身对 1e-7 扰动在 L4 就放大到
   9.3e-3、L23 到 7.6e-2，所以任何实现差异到 L11 之后都会变成"两条不同的轨迹"，
   rel ≈ 1.0；而 f32 与 i8 在那几层的数值几乎一样，说明这 1.0 是**模型混沌**造成的，
   不是量化造成的。真正有意义的指标只有**输出分布**（i8: corr 0.9935 / argmax 一致 /
   top-10 9/10；f32: 0.9959 / 一致 / 9/10）。

另外修掉一个对拍脚本的 bug：跨 chunk 时转储只保留**最后一个 chunk** 的行，早先版本拿
设备 token 32–63 去比参考 token 0–31，会得出"rel ≈ 2.0"的假结论（已按尾部对齐）。

仍未覆盖：生成循环（16+ token）、>64 token（`kMaxSeq = 64`）、以及"对官方 BF16 而非
W8A8 参考"的保真度。

### 7.1.3 权重预分块（已接线）：tile-major，`[N/32][K/32][32][32]`

行主序下**一个 32x32 i8 tile 是 32 段 32 B、段间相距 K 字节**，在 64 B burst 的 DDR
上只有半个 burst。改成 tile-major（每 tile 连续 1 KiB）后：

| | 行主序 | 预分块 |
|---|---|---|
| 一个 1 KiB tile | 32 段 × 32 B（半 burst） | **16 × 64 B 满 burst** |
| 描述符 | strides=(K,1)，k 方向按列自增走 32 B | strides=(32,1)，k 方向**按行自增**走 1024 B |
| 静态指令 | 53,954 | 54,158（+204，8 个 tile 各多配一次 strides） |
| prefill-5 动态 | 4,356,825 | 4,357,227（**+402 = +0.009%**） |
| 权重流量/每 chunk | 741 MiB × 2（半 burst）≈ 1.48 GB | 非 embedding 464 MiB 满 burst + lm_head 243 MiB 半 burst ≈ 0.95 GB |
| prefill-32 总流量 | ≈ 1.5 GB + 47 MiB 状态 | **≈ 0.95 GB**（降 ~37%） |

接线要点（踩过的坑）：

* **N 必须是 32 的倍数**（一个 tile 装 32 个输出通道）。GDN 的 b/a 投影是 **N=16**，
  凑不满一个 tile → 这两个保持行主序、仍走 `gemm_nt`。**第一次接错就是漏了这条**
  （内核按 tiled 读、数据是行主序）→ β/α 全错，corr 0.9345、argmax 11751。
  修好后 corr 回到 0.9935、argmax 279、mean|Δ| 0.2534。
* **embedding/lm_head 不能预分块**：查表要“一行 1 KB 连续”，和 GEMM 要的 32x32 tile
  互斥。现在保留行主序（`LM` 是 `EMB` 的别名），那 243 MiB 仍有半 burst 浪费；
  要吃满得再存一份 tiled 的 lm_head（blob +243 MiB）。
* 转换器 `copy_w` 的置换**逐字节验证过**；kernel 用最小用例
  （`check_gemm_tiled_sim.py`，A 5x1024、B 6144x1024 tiled，即**真实 qkv 形状**）
  对拍 numpy：`gemm_nt` 与 `gemm_nt_tiled` 均 **0/30720 元素不同**。

验证与遗留：预分块后**残差流 X、GDN 状态 S、KV cache 与行主序版本逐位相同**（逐层
计算完全一致）。但 **logits 有 8/248320 个元素不同**（max|Δ| 4.48），差异来自权重区
之前的一个 4 字节 scratch 单元（偏移 512936，落在 embedding 表范围内，两个版本写入
的值不同）。这是**遗留问题**（疑似 scratch 区与 embedding 表重叠，与预分块无关——
行主序版本也会写这个地址），查清前不能声称“预分块与行主序逐位等价”。

### 7.2 外积（秩 1 写）在 f32 里做

`upd.add(u, byrow)` + `upd.mul(k, bycol)` 就是 `u ⊗ k`，4 条指令/列块，
比 i8 外积（4×(mma+acvt+2 乘+加) = 20 条）少，且少一次 u 的量化误差。
与决策 6.3 的字面写法（i8×i8→i32）不同；数值上更准、指令更少。

### 7.3 q/k 的量化粒度

用 **per-(token, head)** 的 absmax（128 维一个 scale），不是 per-token。
设备端 `quant_heads` 与参考一致；q 的 `1/sqrt(d_k)` 折进 dequant 标量
（编译期常量乘进 scale 向量，避免运行期标量浮点乘法）。

## 8. 未完成 / 下一步

| 项 | 状态 |
|---|---|
| M0 环境（重建 sim_bridge + 0.6B 基线） | ✅ |
| M1 单算子对齐（embedding/rmsnorm/GEMM/lm_head） | ✅ 逐位（embedding 0，GEMM 1.4e-7） |
| M2 全注意力层（hd256/GQA/输出门/部分 RoPE） | ✅ 1e-7 级 |
| M3 单层 GDN（短卷积/L2norm/递推/门控） | ✅ 2e-7 级 |
| M4 24 层 + 权重转换全链路 | ✅ 端到端跑通，argmax/top-10 一致 |
| M5 decode 循环 + 采样 + tokenizer | ⚠️ 部分：**贪心 decode + tokenizer 串联已做**（`tools/chat_qwen35.py`，10 token 与参考逐个相同）；**没做**温度/top-p 采样 |
| M6 视觉塔 + MTP | ❌ 未做（权重已下载，`model.visual.*` 与 `mtp.*` 在 safetensors 里） |
| 性能优化 | ❌ 未做：GDN 常驻化（§7.1）、chunked delta rule、权重预分块 |
| 长序列 / decode 验证 | ✅ decode 10 步与参考逐个相同；32 token prefill 数值对拍（argmax/top-10）。跨 chunk（>32 token prompt，max_seq=64）的卷积历史/KV 增长仍未数值对拍 |

## 9. 回归（0.6B 基线，改共享文件后必对）

本轮**没有修改 `qwen3_ops/` 的任何文件**（共享头是拷过来的），实测仍为：

| 项 | 值 |
|---|---|
| 静态指令数 | **40,178** |
| guest 二进制 | **160,732 B** |
| prefill 32 动态指令数 | **2,319,041** |
| logits | 与旧输出 **max\|Δ\| = 0**（逐位） |

## 10. 复现命令

```bash
# 1) 模拟器桥（若 npucc 又丢了）
python3 npucc/build_sim.py

# 2) 权重（HF 主站不通，走 hf-mirror）
python3 tile_api_llvm_ds/qwen35_ops/tools/download_qwen35.py --dst qwen3.5-0.8b-bf16

# 3) 转换 + 编译 + 对拍
cd tile_api_llvm_ds/qwen35_ops
make pqw header blob guest
make check

# 4) 逐层对拍（需要 -DQ35_DUMP 构建）
make build/qwen35d.bin    # 见 Makefile 的 dump 目标
python3 tools/check_qwen35.py --stages --guest build/qwen35d.bin

# 5) 指令数
make stats
../../npucc/build/sim_bridge build/qwen35.bin build/qwen35_data32.bin /tmp/o.bin 0 4000000000 64
```

## 10.5 平坦模式（layout = flat）

`layout = flat` 是矩阵寄存器的附属布局：一个 Acc（1024×f32）正好是 hidden 的一整行，
flat 指令把它当作一条一维向量，**内部仍按 32 个元素一拍**执行。只有 FLAT 指令族接受 flat
操作数，其余指令一律要求 `row_major`（ISA/sim 双向校验）。

| 指令 | 语义 | 在算子里的用途 |
|---|---|---|
| `flat.reduce.{sum,sumsq,max,min,absmax}` | 整条向量一个标量（每 32 元素一棵树 + 顺序合并） | rmsnorm 的整行 sumsq |
| `flat.reduce.argmax` | 整条向量一个 `(index, value)` | 词表 argmax |
| `flat.mulx` / `flat.mulv` | 整条向量 × 标量 / 逐元素乘 | ×1/rms、×weight |

**tile 层 API**（`include/tile.h`，不新增类型）：`Acc::set_layout_flat()` /
`set_layout_row_major()` / `reduce_flat_sumsq(Vec32&)` / `reduce_flat_argmax(Vec32&, Vec32&)` /
`mul_flat_scalar(const Vec32&)` / `mul_flat(const Acc&)`。算子见 `kernels/flat.h`。

实测（本机，功能模拟器按指令条数计）：

| 用例 | row_major | flat | 说明 |
|---|---|---|---|
| rmsnorm 32×1024（prefill） | 1113 | 790 | flat 丢掉了 32 行摊薄，只快 1.4× |
| rmsnorm 1×1024（decode） | 1115 | **39** | 28.6×；一行一个 Acc，省掉 32 块循环与 31 次 vadd |
| 词表 argmax（248320） | ~35 万/步 | — | 平铺后 243 次 flat.reduce.argmax |
| 生成 5+10 token 总指令 | 42,673,902 | **39,279,804** | 只换 argmax 就省 8%，生成 token 逐个不变 |

两条路径**逐位相同**（rmsnorm 对拍 0/32768 个 bit 不同）：规约顺序都是"每 32 元素一棵树 + 顺序合并"。
所以 flat 不需要改参考，也不会引入新的量化决策翻转。

## 11. 已知陷阱（本轮踩过的，都留在代码注释里）

1. **loop-carried acc + `row_absmax` 静默归零**（§7.1）——最花时间的一个。
2. **sigmoid 写成 x/(1+e^-x)** = silu，会让输出门带符号错误（`attn256.h` 有注释）。
3. **层偏移漏加** → L0 对、其余层全错（`main.cpp` 的 `lwb/lnb/lfb`）。
4. **mma 目的 acc 的 dtype** 被上一轮 `to_fp32` 改成 f32 → 必须重设 I32。
5. **新建 Acc 的 zone 是 0×0** → `fill/div` 静默不干活。
6. **q 的 1/√128 只能折进 scale**：写成运行期标量 `ss.mul(extra)` 实测不生效
   （改成编译期常量 `ss.mul(kQScaleDk)` 才对）。
7. **本机 numpy 2.3.5 + OpenBLAS 0.3.30 在小矩阵上返回 NaN/垃圾**
   （`[15][15]@[15][256]` 实测 100 次里 32 次非有限）。参考实现里 attention 的
   两个 matmul 必须用 `np.einsum`，否则参考本身会算错（这个坑一开始被误判成
   "模型结构错了"）。
8. **调试转储**要放在正确的位置：残差流要在**整层结束时**转储（放在注意力之后
   会漏掉 MLP），MIX 要在 **l2norm 之前**转储（l2norm 原地改写 q/k 区）。
