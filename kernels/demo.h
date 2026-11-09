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
//
// 三个针对硬件流水线的写法：
//   1) K 方向双缓冲：x/w 各两个 tile，循环里"预取下一块 + 算当前块"，
//      两个操作数在不同寄存器上 → 记分牌允许 LD 与 MMA 重叠（没有 WAR）；
//   2) 第一块用 mma(zero=1) 顺带清零，不用单独的 fill(0)（少一次跨队列依赖 +
//      一次 4 KB 的 acc 写）；
//   3) 尾块剥离：满 32 的块只在外层配一次 valid_zone，循环里不再写配置寄存器。
// ----------------------------------------------------------------------------
YAI_OP_INLINE void demo_mac(const MatrixView &x, const MatrixView &w,
                            const MatrixView &s, const MatrixView &out) {
  Tile xt0(x.data, x.row_bytes, x.col_bytes, 32, 32, TileDType::I8);   // x 的双缓冲
  Tile xt1(x.data, x.row_bytes, x.col_bytes, 32, 32, TileDType::I8);
  Tile wt0(w.data, w.row_bytes, w.col_bytes, 32, 32, TileDType::I8);   // w 的双缓冲
  Tile wt1(w.data, w.row_bytes, w.col_bytes, 32, 32, TileDType::I8);
  Acc a(out.data, out.row_bytes, out.col_bytes, 32, 32, AccDType::I32);
  Acc ao(out.data, out.row_bytes, out.col_bytes, 32, 32, AccDType::F32);  // f32 输出
  Vec32 v(s.data, s.col_bytes, 32, Vec32DType::F32);                     // vRow = 每行 scale
  Vec32 ones{};                                                          // vCol = 1
  ones.set_dtype(Vec32DType::F32);
  ones.set_stride(4);
  ones.set_valid_zone(32);
  ones.fill(1.0f);

  const std::uint32_t kfull = x.cols & ~31u;        // 满 32 的部分
  const std::uint32_t ktail = x.cols - kfull;       // 尾块元素数

  for (std::uint32_t m = 0; m < out.rows; m += 32) {
    const std::uint8_t rows = block_zone(out.rows, m);
    v.set_valid_zone(rows);
    v.set_data(s.data + m * s.row_bytes);
    v.load();
    for (std::uint32_t n = 0; n < out.cols; n += 32) {
      const std::uint8_t cols = block_zone(out.cols, n);
      a.set_dtype(AccDType::I32);          // 每个输出块重新 i32 累加（上一次 to_fp32 改成了 f32）
      a.set_valid_zone(rows, cols);        // acc 的 zone 跟着输出块（尾块收窄）
      // 满块：zone 只在每块开始配一次（K 方向全部是 32）
      xt0.set_valid_zone(rows, 32);
      xt1.set_valid_zone(rows, 32);
      wt0.set_valid_zone(cols, 32);
      wt1.set_valid_zone(cols, 32);

      if (kfull >= 64) {
        xt0.set_data(x.data + m * x.row_bytes);
        wt0.set_data(w.data + n * w.row_bytes);
        xt0.load();
        wt0.load();
        xt1.set_data(x.data + m * x.row_bytes + 32 * x.col_bytes);
        wt1.set_data(w.data + n * w.row_bytes + 32 * w.col_bytes);
        xt1.load();
        wt1.load();
        xt0.matmul(wt0, a, /*zero=*/true, false, true);   // 第 0 块（顺带清零）
        bool pending1 = true;                            // 下一块在 buf1 里
        for (std::uint32_t k = 64; k < kfull; k += 32) {
          if (pending1) {
            xt0.set_data(x.data + m * x.row_bytes + k * x.col_bytes);
            wt0.set_data(w.data + n * w.row_bytes + k * w.col_bytes);
            xt0.load();
            wt0.load();
            xt1.matmul(wt1, a, false, false, true);      // 与上面的 load 并行
            pending1 = false;
          } else {
            xt1.set_data(x.data + m * x.row_bytes + k * x.col_bytes);
            wt1.set_data(w.data + n * w.row_bytes + k * w.col_bytes);
            xt1.load();
            wt1.load();
            xt0.matmul(wt0, a, false, false, true);
            pending1 = true;
          }
        }
        if (pending1) {                                  // 收尾：最后预取的那块
          xt1.matmul(wt1, a, false, false, true);
        } else {
          xt0.matmul(wt0, a, false, false, true);
        }
      } else if (kfull == 32) {
        xt0.set_data(x.data + m * x.row_bytes);
        wt0.set_data(w.data + n * w.row_bytes);
        xt0.load();
        wt0.load();
        xt0.matmul(wt0, a, /*zero=*/true, false, true);
      }

      if (ktail != 0) {                                  // 尾块：单独一次，zone 收窄
        if (kfull == 0) a.fill(0.0f);                    // K < 32：没有第一块，手动清零
        xt0.set_valid_zone(rows, static_cast<std::uint8_t>(ktail));
        wt0.set_valid_zone(cols, static_cast<std::uint8_t>(ktail));
        xt0.set_data(x.data + m * x.row_bytes + kfull * x.col_bytes);
        wt0.set_data(w.data + n * w.row_bytes + kfull * w.col_bytes);
        xt0.load();
        wt0.load();
        xt0.matmul(wt0, a, /*zero=*/false, false, true);   // 唯一调用点（flag 必须常量）
      }

      ones.set_valid_zone(cols);                          // vCol.len = cols
      ao.set_valid_zone(rows, cols);
      adequant(ao, a, ones, v);                           // ao = f32(a) * 1 * v[i]
      ao.set_data(out.data + m * out.row_bytes + n * out.col_bytes);
      ao.store();
    }
  }
}

} // namespace qwen35_ops

#endif
