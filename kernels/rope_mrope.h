#ifndef YAI_QWEN35_OPS_ROPE_MROPE_H
#define YAI_QWEN35_OPS_ROPE_MROPE_H

// ============================================================================
// 部分旋转 RoPE（25% = 64 维）+ M-RoPE（t/h/w 三段 [11,11,10]）
//
// 官方的两件事：
//  1. **部分旋转**：head_dim=256，只转前 rot_dim = 256*0.25 = 64 维，后 192 维原样。
//  2. **M-RoPE**：32 个旋转对按 mrope_section=[11,11,10] 分给 (t, h, w) 三个坐标，
//     并且 `mrope_interleaved=true` 时按 i%3 交错取坐标（见 modeling 里的
//     `recomposition_frequencies`：h 取 idx = slice(1, 33, 3)，w 取 slice(2, 30, 3)）。
//
// 为什么这里只需要一张表
// --------------------
// **纯文本**输入的 position_ids 是 arange 复制成 4 行（text + t/h/w 全相同），
// 于是"交错取 t/h/w"与"全部取同一个位置"结果完全一样 —— M-RoPE 退化成普通的
// 1D 部分旋转 RoPE。所以 cos/sin 表就是 [pos][32] 的 1D 表（32 个旋转对）。
// 将来接视觉塔时，需要把"位置三元组"先摊平成每 token 的 32 个角度（一张
// [max_seq][32] 的表仍然够用），kernel 这边不用改 —— 接口保持不变。
//
// 与 rope_gqa.h 的关系：旋转本体完全复用 rope_head_hd（rotate-half、按 32 列
// 分块、按 32 行分块）；这里只负责"只转前 64 维"这件事：把 head 视图的前 64 列
// 切出来传进去。half = 64/2 = 32 正好是一个 Vec32，所以 cos/sin 一次载入即可。
// ============================================================================
#include "tile.h"

#include "config.h"
#include "io.h"
#include "matrix.h"
#include "rope_gqa.h"

namespace qwen35_ops {

// head  : [rows][256] f32（单个头的切片，行跨步继承父视图）
// cos_t : [rows][32] f32（已经按绝对位置切好行）
// 原地修改 head 的前 64 列。
YAI_OP_INLINE void rope_partial_mrope(const MatrixView &head,
                                      const MatrixView &cos_t,
                                      const MatrixView &sin_t) {
  // 只转前 kRotDim 维：切一个 64 宽的视图，rope_head_hd 里 half = 32 → 1 个块
  const MatrixView rot = sub_view(head, 0, 0, head.rows,
                                  static_cast<std::uint32_t>(kRotDim));
  rope_head_hd(rot, cos_t, sin_t);
}

} // namespace qwen35_ops

#endif
