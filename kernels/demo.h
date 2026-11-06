#ifndef YAI_QWEN35_OPS_DEMO_H
#define YAI_QWEN35_OPS_DEMO_H

#include "tile.h"

#include "config.h"
#include "matrix.h"

namespace qwen35_ops {

// 最小算子：out = (x · Wᵀ) * s
//   x [32][32] i8，W [32][32] i8，s [32] f32（每行 scale），out [32][32] f32
// 全部用结构体的"全字段构造函数"：strides / dtype / zone / base 一次配好。
YAI_OP_INLINE void demo_mac(const MatrixView &x, const MatrixView &w,
                            const MatrixView &s, const MatrixView &out) {
  Tile xt = make_tile(x);                 // 左操作数：i8 + 视图的 stride/zone
  xt.load();
  Tile wt = make_tile(w);                 // 右操作数（逻辑转置）
  wt.load();

  Acc a = make_acc(out, AccDType::I32);   // 累加器：i32，strides 直接用 out 的
  xt.matmul(wt, a, /*zero=*/true, /*lt=*/false, /*rt=*/true);

  Vec32 v = make_vec32(s);                // 每行 scale
  v.load();

  a.to_fp32();                            // i32 → f32
  a.mul(v, /*col=*/true);                 // 向量 → 矩阵：按行广播
  a.store();                              // stride 已经是 out 的，直接写

  a.set_layout_row_major();               // 用完复位 layout（约定）
}

} // namespace qwen35_ops

#endif
