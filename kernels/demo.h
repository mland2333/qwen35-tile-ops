#ifndef YAI_QWEN35_OPS_DEMO_H
#define YAI_QWEN35_OPS_DEMO_H

#include "tile.h"

#include "config.h"
#include "matrix.h"

namespace qwen35_ops {

// ----------------------------------------------------------------------------
// 三个"从视图造寄存器"的小函数（通常放在 matrix.h 里，这里为了演示放在一起）。
// 它们只是把 MatrixView 的字段喂给上一节的全字段构造函数：
//   stride  ← v.row_bytes / v.col_bytes
//   zone    ← v.rows / v.cols（构造函数会截到 32）
//   base    ← v.data
//   dtype   ← v.dtype（Acc 默认 f32，做累加时显式传 i32）
// ----------------------------------------------------------------------------
YAI_ALWAYS_INLINE Tile make_tile(const MatrixView &v) {
  return Tile(v.data, v.row_bytes, v.col_bytes, v.rows, v.cols,
              static_cast<TileDType>(v.dtype));
}

YAI_ALWAYS_INLINE Acc make_acc(const MatrixView &v, AccDType dtype = AccDType::F32) {
  return Acc(v.data, v.row_bytes, v.col_bytes, v.rows, v.cols, dtype);
}

YAI_ALWAYS_INLINE Vec32 make_vec32(const MatrixView &v) {
  return Vec32(v.data, v.col_bytes, v.rows, static_cast<Vec32DType>(v.dtype));
}

// ----------------------------------------------------------------------------
// 算子：out[M][N] = (x[M][K] · W[N][K]ᵀ) * s[M]
//   任意 M/N/K（不必是 32 的倍数）：按 32×32 分块，尾块用 block_zone 收窄 zone。
//   描述符与块位置无关，所以 stride/dtype 只在最外层配一次，循环里只 set_data。
// ----------------------------------------------------------------------------
YAI_OP_INLINE void demo_mac(const MatrixView &x, const MatrixView &w,
                            const MatrixView &s, const MatrixView &out) {
  Tile xt(x.data, x.row_bytes, x.col_bytes, 32, 32, TileDType::I8);    // 左：x 的块
  Tile wt(w.data, w.row_bytes, w.col_bytes, 32, 32, TileDType::I8);    // 右：W 的块
  Acc a(out.data, out.row_bytes, out.col_bytes, 32, 32, AccDType::I32);  // 累加器
  Vec32 v(s.data, s.col_bytes, 32, Vec32DType::F32);                   // 每行 scale

  for (std::uint32_t m = 0; m < out.rows; m += 32) {
    const std::uint8_t rows = block_zone(out.rows, m);        // 尾块只有 rows 行
    v.set_valid_zone(rows);
    v.set_data(s.data + m * s.row_bytes);
    v.load();
    for (std::uint32_t n = 0; n < out.cols; n += 32) {
      const std::uint8_t cols = block_zone(out.cols, n);
      a.set_dtype(AccDType::I32);                             // 每块重新 i32 累加
      a.set_valid_zone(rows, cols);
      a.fill(0.0f);                                           // 清零（mma 的 zero=0 常累加）
      for (std::uint32_t k = 0; k < x.cols; k += 32) {        // K 方向累加
        const std::uint8_t kc = block_zone(x.cols, k);
        xt.set_valid_zone(rows, kc);
        xt.set_data(x.data + m * x.row_bytes + k * x.col_bytes);
        xt.load();
        wt.set_valid_zone(cols, kc);
        wt.set_data(w.data + n * w.row_bytes + k * w.col_bytes);
        wt.load();
        // zero/lt/rt 必须是编译期常量，所以这里恒为「累加」
        xt.matmul(wt, a, /*zero=*/false, /*lt=*/false, /*rt=*/true);
      }
      a.to_fp32();                                            // i32 → f32
      a.mul(v, /*col=*/true);                                 // 每行 scale 广播
      a.set_data(out.data + m * out.row_bytes + n * out.col_bytes);
      a.store();
    }
  }
  a.set_layout_row_major();                                   // 复位 layout（约定）
}

} // namespace qwen35_ops

#endif
