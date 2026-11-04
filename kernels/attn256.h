#ifndef YAI_QWEN35_OPS_ATTN256_H
#define YAI_QWEN35_OPS_ATTN256_H

// ============================================================================
// Qwen3.5 全注意力层（6 层）：head_dim=256 + GQA 4:1 + 输出门 + 部分 RoPE
//
// 与 qwen3_ops/attention_gqa.h 的 attention_head_hd128_blocks 是同一套算法
// （KV 按 32 行分块 + online softmax），差别只在 head_dim 128→256：
//   * S = Σ_{d<256} q8_d·k8_dᵀ 变成 8 个 K 块（i32 精确累加）；
//   * 输出 O 从 4 个 32 宽块变成 **8 个** 32 宽块 → r0..r7 八个常驻 acc。
//
// 寄存器预算（硬上限 16 acc / 16 tile / 32 vec32）：
//   acc: r0..r7 (8) + s (1) + part (1) = 10 ≤ 16
//   tile: qt / kt / vt / pt = 4 ≤ 16
//   vec32: qsv / ksv / vsv / maximum / den / nmm / corr / pmax = 8
// 为什么不能"q/k/v 各 8 个 tile 全塞进来"：8*3 = 24 > 16，会直接
// `ran out of registers`。所以 q/k 的 8 个 K 块是**循环里逐块载入**的（一个 qt
// 反复用），只有输出方向需要 8 个常驻 acc。
//
// 输出门：官方是 `attn_output * sigmoid(gate)`，gate 来自 q_proj 每头**后 256 列**
// （`view(*shape, -1, 2*head_dim).chunk(2, -1)`）。所以 gate 在 2048 宽的输出里
// 是"每 512 列里后 256 列"，必须按头切片处理（见 mul_sigmoid_rows 的调用处）。
// ============================================================================
#include "tile.h"

#include "config.h"
#include "io.h"
#include "matrix.h"

namespace qwen35_ops {

// ---------------------------------------------------------------------------
// 输出门：dst[t][c] *= sigmoid(gate[t][c])（两个视图形状相同、跨步可不同）
//
// sigmoid(x) = 1 / (1 + exp(-x))，用 acc 的 4 步实现（与 numpy 参考同式子）：
//   e = x * (-1) → e.exp() → e += 1 → 1 / e
// 需要两个 acc：一个放 x（要被除），一个放分母。没有 acc 拷贝指令，所以分母
// 用第三个 acc 从内存再读一次 gate（一个 aload 比想办法拷贝便宜）。
// ---------------------------------------------------------------------------
YAI_OP_INLINE void mul_sigmoid_rows(const MatrixView &dst, const MatrixView &gate) {
  Acc g{};      // gate 值（最终变成 1/(1+e^-x)）
  Acc e{};      // 分母
  Acc d{};      // dst
  config_acc(g, gate);
  config_acc(e, gate);
  config_acc(d, dst);
  for (std::uint32_t m = 0; m < dst.rows; m += kBlock) {
    const std::uint8_t nr = block_zone(dst.rows, m);
    for (std::uint32_t n = 0; n < dst.cols; n += kBlock) {
      const std::uint8_t nc = block_zone(dst.cols, n);
      const std::uint32_t og = m * static_cast<std::uint32_t>(gate.row_bytes) +
                               n * static_cast<std::uint32_t>(gate.col_bytes);
      const std::uint32_t od = m * static_cast<std::uint32_t>(dst.row_bytes) +
                               n * static_cast<std::uint32_t>(dst.col_bytes);
      set_zone(e, nr, nc);
      load_acc_at(e, gate.data + og);
      e.mul(-1.0f);
      e.exp();
      e.add(1.0f);                    // e = 1 + e^-gate
      set_zone(g, nr, nc);
      g.fill(1.0f);
      g.div(e);                       // g = 1/(1+e^-gate) = sigmoid(gate)
      // 注意：不能写成 gate/(1+e^-gate) —— 那是 silu 不是 sigmoid（会带符号错误）
      set_zone(d, nr, nc);
      load_acc_at(d, dst.data + od);
      d.mul(g);
      store_acc_at(d, dst.data + od);
    }
  }
}

// ---------------------------------------------------------------------------
// 一个 32 宽输出块：res += (P·V_d) * pmax
// （与 0.6B 的实现相同：V 的量化 scale 已经折进 P，这里只乘 P 的行 scale）
// ---------------------------------------------------------------------------
YAI_ALWAYS_INLINE void pv_block256(Acc &res, Tile &pt, Tile &vt, Acc &part,
                                   const MatrixView &v8, std::uint32_t j,
                                   std::uint32_t d, const Vec32 &pmax) {
  load_tile_at(vt, v8.data + j * static_cast<std::uint32_t>(v8.row_bytes) +
                       d * static_cast<std::uint32_t>(v8.col_bytes));
  part.set_dtype(AccDType::I32);      // 上一轮 to_fp32 会把它改成 f32
  pt.matmul(vt, part, /*zero=*/true, /*lt=*/false, /*rt=*/false);
  part.to_fp32();
  part.mul(pmax, true);               // 反量化
  res.add(part);
}

// ---------------------------------------------------------------------------
// 单个 (q 头, kv 头) 的 KV 分块 attention（head_dim = 256）
//
//   q8 : [q_rows][256]   k8/v8 : [kv_len][256]（通常直接是 KV cache 的前 kv_len 行）
//   out: [q_rows][256] f32
//   pos0 = 本 q 块第一行的绝对位置（decode 时 = kv_len-1）
//
// 每个 (head, q 块) 维护 4 类状态：maximum / den / r0..r7。
// 每个 KV 块 j：S_j → ×q scale ×k scale ×qk_scale → mask → running max 修正
// → P=exp(S-nmm) → 折 V scale → 量化 P → 8 个输出块各一次 P·V。
// ---------------------------------------------------------------------------
YAI_OP_INLINE void attention_head_hd256_blocks(
    const MatrixView &q8, const MatrixView &k8, const MatrixView &v8,
    const MatrixView &q_scale, const MatrixView &k_scale,
    const MatrixView &v_scale, const MatrixView &out, std::uint32_t q_rows,
    std::uint32_t kv_len, std::int32_t pos0, bool causal, float qk_scale) {
  const std::uint8_t nr = static_cast<std::uint8_t>(q_rows);
  constexpr std::uint8_t ZB = static_cast<std::uint8_t>(kBlock);

  // ---- 0) 状态初始化 ----
  Vec32 qsv{};
  load_vec_n(qsv, q_scale, 0, nr);
  Vec32 maximum{};
  init_f32_vec(maximum, nr);
  maximum.fill(-3.402823466e+38F);
  Vec32 den{};
  init_f32_vec(den, nr);
  den.fill(0.0f);
  Acc r0{}, r1{}, r2{}, r3{}, r4{}, r5{}, r6{}, r7{};
  init_f32_acc_zero(r0, nr, ZB);
  init_f32_acc_zero(r1, nr, ZB);
  init_f32_acc_zero(r2, nr, ZB);
  init_f32_acc_zero(r3, nr, ZB);
  init_f32_acc_zero(r4, nr, ZB);
  init_f32_acc_zero(r5, nr, ZB);
  init_f32_acc_zero(r6, nr, ZB);
  init_f32_acc_zero(r7, nr, ZB);

  // ---- 1) 描述符外提（stride/dtype 与块无关）----
  Tile qt{};
  config_tile(qt, q8);
  set_zone(qt, nr, ZB);
  Tile kt{};
  config_tile(kt, k8);
  Tile vt{};
  config_tile(vt, v8);
  Vec32 ksv{};
  config_vec(ksv, k_scale);
  Vec32 vsv{};
  config_vec(vsv, v_scale);
  Acc part{};
  part.set_strides(q8.row_bytes, q8.col_bytes);
  set_zone(part, nr, ZB);
  Acc s{};
  Tile pt{};
  pt.set_dtype(TileDType::I8);

  // 因果时整块被 mask 的 KV 块直接跳过（贡献恰好为 0）
  std::uint32_t kv_end = kv_len;
  if (causal) {
    const std::uint32_t need = static_cast<std::uint32_t>(pos0) + q_rows;
    if (need < kv_end) kv_end = need;
  }
  for (std::uint32_t j = 0; j < kv_end; j += kBlock) {
    const std::uint8_t nc = block_zone(kv_len, j);

    // 1a) S = q8·k8ᵀ：8 个 K 块在 i32 里累加
    s.set_dtype(AccDType::I32);
    set_zone(s, nr, nc);
    set_zone(kt, nc, ZB);
    #pragma clang loop YAI_LOOP_UNROLL
    for (std::uint32_t d = 0; d < static_cast<std::uint32_t>(kHeadDim); d += kBlock) {
      load_tile_at(qt, q8.data + d * static_cast<std::uint32_t>(q8.col_bytes));
      load_tile_at(kt, k8.data + j * static_cast<std::uint32_t>(k8.row_bytes) +
                           d * static_cast<std::uint32_t>(k8.col_bytes));
      qt.matmul(kt, s, /*zero=*/d == 0, /*lt=*/false, /*rt=*/true);
    }

    // 1b) 还原量化 scale + qk_scale
    s.to_fp32();
    s.mul(qsv, true);
    set_len(ksv, nc);
    load_vec_at(ksv, k_scale.data + j * static_cast<std::uint32_t>(k_scale.row_bytes));
    s.mul(ksv, false);
    s.mul(qk_scale);

    // 1c) 因果 mask（偏移 pos0 - j）
    if (causal) {
      s.mask(pos0 - static_cast<std::int32_t>(j));
    }

    // 1d) running max + correction
    Vec32 nmm = s.row_max();
    nmm.max(maximum);
    Vec32 corr = maximum;
    corr.sub(nmm);
    corr.exp();
    den.mul(corr);
    r0.mul(corr, true);
    r1.mul(corr, true);
    r2.mul(corr, true);
    r3.mul(corr, true);
    r4.mul(corr, true);
    r5.mul(corr, true);
    r6.mul(corr, true);
    r7.mul(corr, true);
    s.sub(nmm, true);
    s.exp();
    s.row_sum(den);                     // 分母累加必须在乘 V scale 之前
    maximum = nmm;

    // 1e) 折 V 的 scale → 量化 P
    set_len(vsv, nc);
    load_vec_at(vsv, v_scale.data + j * static_cast<std::uint32_t>(v_scale.row_bytes));
    s.mul(vsv, false);
    Vec32 pmax = s.row_absmax();
    pmax.mul(kInv127);
    set_zone(pt, nr, nc);
    s.quant_ext(pmax, pt);

    // 1f) 8 个输出块累加
    set_zone(vt, nc, ZB);
    pv_block256(r0, pt, vt, part, v8, j, 0 * kBlock, pmax);
    pv_block256(r1, pt, vt, part, v8, j, 1 * kBlock, pmax);
    pv_block256(r2, pt, vt, part, v8, j, 2 * kBlock, pmax);
    pv_block256(r3, pt, vt, part, v8, j, 3 * kBlock, pmax);
    pv_block256(r4, pt, vt, part, v8, j, 4 * kBlock, pmax);
    pv_block256(r5, pt, vt, part, v8, j, 5 * kBlock, pmax);
    pv_block256(r6, pt, vt, part, v8, j, 6 * kBlock, pmax);
    pv_block256(r7, pt, vt, part, v8, j, 7 * kBlock, pmax);
  }

  // ---- 2) 最后才除分母 ----
  r0.div(den, true);
  store_acc(r0, out, 0, 0 * kBlock);
  r1.div(den, true);
  store_acc(r1, out, 0, 1 * kBlock);
  r2.div(den, true);
  store_acc(r2, out, 0, 2 * kBlock);
  r3.div(den, true);
  store_acc(r3, out, 0, 3 * kBlock);
  r4.div(den, true);
  store_acc(r4, out, 0, 4 * kBlock);
  r5.div(den, true);
  store_acc(r5, out, 0, 5 * kBlock);
  r6.div(den, true);
  store_acc(r6, out, 0, 6 * kBlock);
  r7.div(den, true);
  store_acc(r7, out, 0, 7 * kBlock);
}

// ---------------------------------------------------------------------------
// GQA driver：q 头 h 用 kv 头 h/4；q 按 32 行分块，因果偏移 pos0 = kv_len - q_rows
// （prefill 和 decode 都成立）。scale 是 per-token（整行一个）。
// ---------------------------------------------------------------------------
YAI_OP_INLINE void attention_heads_gqa256(const MatrixView &q8,
                                          const MatrixView &k8,
                                          const MatrixView &v8,
                                          const MatrixView &q_scale,
                                          const MatrixView &k_scale,
                                          const MatrixView &v_scale,
                                          const MatrixView &out, int heads,
                                          int kv_heads, int head_dim,
                                          bool causal, float qk_scale) {
  const int group = heads / kv_heads;
  const std::uint32_t q_rows = q8.rows;
  const std::uint32_t kv_len = k8.rows;
  const std::int32_t pos0 =
      causal ? static_cast<std::int32_t>(kv_len) - static_cast<std::int32_t>(q_rows) : 0;
  for (int h = 0; h < heads; ++h) {
    const int kv = h / group;
    const std::uint32_t qc = static_cast<std::uint32_t>(h * head_dim);
    const std::uint32_t kvc = static_cast<std::uint32_t>(kv * head_dim);
    const std::uint32_t hd = static_cast<std::uint32_t>(head_dim);
    for (std::uint32_t qi = 0; qi < q_rows; qi += kBlock) {
      const std::uint8_t nr = block_zone(q_rows, qi);
      attention_head_hd256_blocks(
          sub_view(q8, qi, qc, nr, hd), sub_view(k8, 0, kvc, kv_len, hd),
          sub_view(v8, 0, kvc, kv_len, hd),
          sub_view(q_scale, qi, 0, nr, 1), k_scale, v_scale,
          sub_view(out, qi, qc, nr, hd), nr, kv_len,
          causal ? pos0 + static_cast<std::int32_t>(qi) : 0, causal, qk_scale);
    }
  }
}

} // namespace qwen35_ops

#endif
