# 调试转储（`-DQ35_DUMP`）的槽位约定

`main.cpp` 里所有 `dump_f32/dump_i8` 都写进 blob 的 `kDUMP` 区（4 MB），
槽位以 **float 个数** 为单位。`check_qwen35.py --stages` 按同一套槽位读回。

| 槽位 | 内容 | 何时写 |
|---|---|---|
| 0 / 512 | beta / alpha（[rows][16]） | GDN 层 0，量化前 |
| 1024 | 卷积前的 qkv（`buf` 的第 3 行起） | GDN 层 0，卷积后 |
| 1024 + 32×6144 | 卷积+silu 之后、**l2norm 之前**的 MIX | 同上 |
| 200000 + t×32 | 递推的 pred（32 个 d_v） | 每 token，head0/行块0 |
| 300000 + t×32 | 递推的 u | 同上 |
| 500000 / 520000 / 540000 | layer0 的 gate / up / swiglu 输出 | 层 0 的 MLP |
| 560000 / 570000 | layer0 的 down 输出 / MLP 输入的 h2 | 同上 |
| 300 万字节起（= 槽 800000） | 递推里 (head0, 行块0) 的 S，逐 token 32×128 | 每 token |
| 600000 / 630000 / 650000 / 660000 | layer3 的 qg / 紧凑 qf / kf / vf | 全注意力层 3 |
| 670000 / 690000 | layer3 的 attention 输出（乘门前 / 乘门后） | 同上 |
| 900000 + L×6000 | **每层结束**（MLP 加完）的残差流 | 全部 24 层 |

坑：
* 残差流必须在**整层结束**转储（放在注意力之后会漏掉 MLP，看起来像"MLP 没生效"）。
* MIX 必须在 **l2norm 之前**转储（l2norm 原地改写 q/k 区，之后 dump 出来的是归一化后的值）。
* 转储版**不跑 lm_head**，否则 logits 区会被覆盖（`kDumpMode`）。
