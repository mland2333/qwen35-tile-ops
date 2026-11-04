#ifndef YAI_QWEN35_OPS_QWEN35_H
#define YAI_QWEN35_OPS_QWEN35_H

// ============================================================================
// Qwen3.5-0.8B 文本塔算子库的汇总头
//
// 分层规则（与 qwen3_ops 相同，整个库只有这一条硬约束）：
//   编排层（src/main.cpp）只碰标量 + MatrixView；
//   Tile/Acc/Vec32/Vec8 这些寄存器值只在算子内部出现，绝不跨函数边界。
//
// 文件一览（按一次前向的数据流排序）：
//   matrix.h        MatrixView + sub_view + block_zone（大矩阵描述符，纯标量）
//   io.h            MatrixView 的 32x32 块 <-> Acc/Tile/Vec32
//   rmsnorm.h       rmsnorm_rows：行 RMSNorm（两遍：先 sumsq 再缩放）
//   quant.h         quant_rows：per-token int8 量化
//   gemm.h          gemm_nt：int8 GEMM（K 块 i32 累加 + 双 scale epilogue）
//   embedding.h     embedding_rows：词表 gather + widen + 反量化
//   rope_gqa.h      rope_head_hd：任意 head_dim 的 rotate-half，按 32 列分块
//   rope_mrope.h    rope_partial_mrope：部分旋转（前 64 维）+ M-RoPE 说明
//   attn256.h       全注意力（hd=256、GQA 4:1、输出门、KV 分块 online softmax）
//   gdn.h           GDN：短卷积 + L2norm + delta 递推（行块常驻）+ 门控 norm
//   mlp.h           swiglu + add_rows
// ============================================================================
#include "attn256.h"
#include "config.h"
#include "embedding.h"
#include "gdn.h"
#include "gemm.h"
#include "io.h"
#include "matrix.h"
#include "mlp.h"
#include "quant.h"
#include "rmsnorm.h"
#include "rope_gqa.h"
#include "rope_mrope.h"

#endif
