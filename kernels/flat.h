#ifndef YAI_QWEN35_OPS_FLAT_H
#define YAI_QWEN35_OPS_FLAT_H

// ============================================================================
// 平坦模式（layout = flat）的算子
//
// 一个 Acc 装 1024 个 f32，正好是 hidden 的一整行。flat 指令把它当作一条
// 一维向量，按 32 个元素一拍做规约/逐元素，因此：
//   * 整行 sumsq / absmax / argmax 是**一条指令**（内部 32 拍），
//     而 row_major 路径要按 32 列分块，再做 31 次 vadd 合并；
//   * 规约顺序是"每 32 元素一棵树，再顺序合并"，与 row_major 路径逐位相同。
//
// 只有 flat 白名单里的指令能在 flat 配置下执行；搬运/逐元素仍然用
// row_major，所以每次切换要两条 cfg.setx（layout=1 / layout=0）。
// ============================================================================
#include "tile.h"

#include "config.h"
#include "io.h"
#include "matrix.h"

namespace qwen35_ops {

// 整行（1024 = 32x32）rmsnorm：load → flat sumsq → ×1/rms → ×weight → store。
// 权重按 32x32 常驻一个 Acc，跨行复用。
YAI_OP_INLINE void rmsnorm_rows_flat(const MatrixView &x, const MatrixView &w,
                                     const MatrixView &out, float inv) {
  Vec32 ss{};
  ss.set_dtype(Vec32DType::F32);
  ss.set_stride(4);
  set_len(ss, 1);

  Acc wa{};
  wa.set_data(w.data);
  wa.set_strides(128, 4);   // 32x32 连续
  wa.set_dtype(AccDType::F32);
  wa.set_valid_zone(32, 32);
  wa.load();
  wa.set_layout_flat();

  for (std::uint32_t m = 0; m < x.rows; ++m) {
    Acc a{};
    a.set_data(x.data + m * static_cast<std::uint32_t>(x.row_bytes));
    a.set_strides(128, 4);
    a.set_dtype(AccDType::F32);
    a.set_valid_zone(32, 32);
    a.set_layout_row_major();   // 上一轮可能停在同一物理寄存器的 flat 上
    a.load();
    a.set_layout_flat();
    a.reduce_flat_sumsq(ss);
    ss.mul(inv);
    ss.add(kEps);
    ss.rsqrt();
    a.mul_flat_scalar(ss);
    a.mul_flat(wa);
    a.set_layout_row_major();
    a.set_data(out.data + m * static_cast<std::uint32_t>(out.row_bytes));
    a.set_strides(128, 4);
    a.store();
  }
}


// 单 token 的 embedding + rmsnorm 融合（decode 路径）。
//
// 一个 token 的词向量就是 embedding 表里**连续的一行**（1024 个 i8 = 1 KB），
// 所以不需要 gather：把描述符的 base 指到 emb + id*1024、row_stride=32、
// col_stride=1，一次 tload 就是整条 1024 向量。随后 widen/acvt/×scale 得到
// 残差流 x，再在同一个 Acc 上做 flat rmsnorm 得到 h。
YAI_OP_INLINE void embedding_rmsnorm_row_flat(const MatrixView &table,
                                              const MatrixView &scale,
                                              std::uint32_t id,
                                              const MatrixView &w,
                                              const MatrixView &out_x,
                                              const MatrixView &out_h,
                                              float inv) {
  // 词表里连续的一行：1024 个 i8
  Tile t{};
  t.set_data(table.data + id * 1024);
  t.set_strides(32, 1);
  t.set_dtype(TileDType::I8);
  t.set_valid_zone(32, 32);
  t.load();

  Acc a{};
  a.set_strides(128, 4);
  a.set_dtype(AccDType::I32);   // widen 的目的必须是 i32
  a.set_valid_zone(32, 32);
  a.set_layout_row_major();
  t.widen(a);
  a.to_fp32();                  // 之后 Acc 的配置是 f32
  const std::uint32_t scale_bits =
      *reinterpret_cast<const std::uint32_t *>(scale.data + id * 4);
  a.mul(__builtin_bit_cast(float, scale_bits));

  // 残差流起点 x
  a.set_data(out_x.data);
  a.set_strides(128, 4);
  a.store();

  // 整行 rmsnorm：weight 常驻一个 Acc
  Acc wa{};
  wa.set_data(w.data);
  wa.set_strides(128, 4);
  wa.set_dtype(AccDType::F32);
  wa.set_valid_zone(32, 32);
  wa.set_layout_row_major();
  wa.load();
  wa.set_layout_flat();

  Vec32 ss{};
  ss.set_dtype(Vec32DType::F32);
  ss.set_stride(4);
  set_len(ss, 1);

  a.set_layout_flat();
  a.reduce_flat_sumsq(ss);
  ss.mul(inv);
  ss.add(kEps);
  ss.rsqrt();
  a.mul_flat_scalar(ss);
  a.mul_flat(wa);
  a.set_layout_row_major();
  a.set_data(out_h.data);
  a.set_strides(128, 4);
  a.store();
  // 布局是"配置寄存器"的状态：离开函数前必须还原，否则后续复用同一物理
  // 寄存器的 aload/astore 会带着 flat 布局、报 invalid_configuration。
  wa.set_layout_row_major();
}


// prefill 版本：把描述符里**与 token 无关**的字段（row_stride / col_stride /
// rows / cols / dtype / weight 常驻）全部提到循环外，每 token 只更新 base 与
// scale，并且只在 reduce 前后各切一次 layout。这样每 token 的固定开销从
// ~15 条降到 ~4 条，prefill 也能比 gather 路径（41 条/token）更省。
// 不内联：这个循环体比较大，内联进 forward_chunk 会拖累整个函数的寄存器
// 分配（实测 decode 每步多 2.8 万条）。每个 chunk 只调用一次，call 开销可忽略。
__attribute__((noinline)) void embedding_rmsnorm_rows_flat(const MatrixView &table,
                                               const MatrixView &scale,
                                               const MatrixView &ids,
                                               const MatrixView &w,
                                               const MatrixView &out_x,
                                               const MatrixView &out_h,
                                               float inv) {
  Tile t{};
  t.set_strides(32, 1);
  t.set_dtype(TileDType::I8);
  t.set_valid_zone(32, 32);

  Acc a{};
  a.set_strides(128, 4);
  a.set_valid_zone(32, 32);
  a.set_dtype(AccDType::I32);

  Vec32 ss{};
  ss.set_dtype(Vec32DType::F32);
  ss.set_stride(4);
  set_len(ss, 1);

  Acc wa{};
  wa.set_data(w.data);
  wa.set_strides(128, 4);
  wa.set_dtype(AccDType::F32);
  wa.set_valid_zone(32, 32);
  wa.set_layout_row_major();
  wa.load();
  wa.set_layout_flat();

  for (std::uint32_t row = 0; row < ids.rows; ++row) {
    const std::uint32_t id =
        *reinterpret_cast<const std::uint32_t *>(ids.data + row * 4);
    const std::uint32_t scale_bits =
        *reinterpret_cast<const std::uint32_t *>(scale.data + id * 4);

    t.set_data(table.data + id * 1024);   // 每 token 只改 base
    t.load();
    a.set_dtype(AccDType::I32);
    a.set_layout_row_major();             // 上一次迭代末尾已还原，这里保证首轮
    t.widen(a);
    a.to_fp32();
    a.mul(__builtin_bit_cast(float, scale_bits));

    a.set_data(out_x.data + row * 4096);  // 残差流
    a.store();

    a.set_layout_flat();
    a.reduce_flat_sumsq(ss);
    ss.mul(inv);
    ss.add(kEps);
    ss.rsqrt();
    a.mul_flat_scalar(ss);
    a.mul_flat(wa);
    a.set_layout_row_major();
    a.set_data(out_h.data + row * 4096);
    a.store();
  }
  wa.set_layout_row_major();
}

// 整行 argmax：每个 1024（32x32 连续）块用一条 flat.reduce.argmax，
// 再与累计最优比较。返回全局下标。
YAI_ALWAYS_INLINE std::uint32_t argmax_flat(const MatrixView &logits,
                                            const MatrixView &argf,
                                            const MatrixView &argi) {
  Vec32 mx{};
  mx.set_dtype(Vec32DType::F32);
  mx.set_stride(4);
  set_len(mx, 1);
  Vec32 ix{};
  ix.set_dtype(Vec32DType::U32);
  ix.set_stride(4);
  set_len(ix, 1);

  std::uint32_t best = 0;
  std::uint32_t best_key = 0;
  bool first = true;
  for (std::uint32_t n = 0; n < logits.cols; n += 1024) {
    const std::uint32_t rem = logits.cols - n;
    const std::uint32_t len = rem >= 1024 ? 1024 : rem;   // 32 的倍数
    Acc a{};
    a.set_data(logits.data + n * 4);
    a.set_strides(128, 4);
    a.set_dtype(AccDType::F32);
    a.set_valid_zone(static_cast<std::uint8_t>(len / 32), 32);
    a.set_layout_row_major();
    a.load();
    a.set_layout_flat();
    a.reduce_flat_argmax(ix, mx);
    a.set_layout_row_major();
    store_vec(mx, argf, 0);
    store_vec(ix, argi, 0);
    const std::uint32_t bits = *reinterpret_cast<const std::uint32_t *>(argf.data);
    const std::uint32_t idx = *reinterpret_cast<const std::uint32_t *>(argi.data);
    const std::uint32_t key = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    if (first || key > best_key) {
      best_key = key;
      best = n + idx;
      first = false;
    }
  }
  return best;
}

} // namespace qwen35_ops

#endif
