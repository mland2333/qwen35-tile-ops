#ifndef YAI_QWEN35_OPS_CONFIG_H
#define YAI_QWEN35_OPS_CONFIG_H

// ============================================================================
// Qwen3.5-0.8B 算子库的编译期开关 + 数值常量
//
// 与 qwen3_ops/config.h 的差别只有常量部分：多了 Qwen3.5 的 head_dim=256、
// 部分旋转 rot=64、GDN 的头数/维度。开关和 dtype 码完全一致。
// ============================================================================

// 算子级内联：内联后 MatrixView 的维度变成编译期常量，块循环能完全展开。
#ifndef YAI_OP_INLINE
#define YAI_OP_INLINE YAI_ALWAYS_INLINE
#endif

// 块循环展开策略（与 0.6B 相同的取舍：展开只影响代码大小，与数值无关）
#ifndef YAI_LOOP_UNROLL
#define YAI_LOOP_UNROLL unroll(full)
#endif

namespace qwen35_ops {

// 硬件 tile 的边长：一个 Tile/Acc 是 32x32，Vec32 是 32 个元素。
inline constexpr int kBlock = 32;

// ---- Qwen3.5 的形状常量（只放这里，kernel 里不再出现魔法数字）----
inline constexpr int kHeadDim = 256;      // 全注意力的 head_dim
inline constexpr int kRotDim = 64;        // 部分旋转：256 * 0.25
inline constexpr int kHalfRot = 32;       // 旋转对个数（rot_dim/2）
inline constexpr int kHeads = 8;          // Q 头
inline constexpr int kKvHeads = 2;        // KV 头（GQA 4:1）
inline constexpr int kGqaGroup = 4;
inline constexpr int kGdnHeads = 16;      // GDN 的 key/value 头
inline constexpr int kDv = 128;           // GDN value 头维度
inline constexpr int kDk = 128;           // GDN key 头维度
inline constexpr int kConvK = 4;          // 短卷积核长

// RMSNorm 的 eps（HF rms_norm_eps=1e-6）。注意它加在"平方均值"上。
inline constexpr float kEps = 1e-6f;

// per-token 量化的 scale 系数：scale = absmax / 127（乘法形式，与 numpy 参考
// 的 absmax * f32(1/127) 一致；用常量避免设备端触发软浮点除法）。
inline constexpr float kInv127 = 1.0f / 127.0f;

// attention 的 QK 缩放：head_dim=256 → 1/16（编译期常量，设备端没有 sqrt）
inline constexpr float kInvSqrt256 = 0.0625f;
inline constexpr float kInvSqrt128 = 0.08838834764831845f;

// GDN 里 q 的 1/sqrt(d_k) 缩放：官方 torch_recurrent_gated_delta_rule 里有
// （`query = query / query.shape[-1] ** 0.5`），qwen35_dense_demo 漏了——以官方为准。
inline constexpr float kQScaleDk = 0.08838834764831845f;   // 1/sqrt(128)

// MatrixView::dtype 的取值，必须与 ISA 的 dtype 字段一致：i8=0,u8=1,i32=2,u32=3,f32=4
inline constexpr std::uint8_t kDtI8 = 0;
inline constexpr std::uint8_t kDtU8 = 1;
inline constexpr std::uint8_t kDtI32 = 2;
inline constexpr std::uint8_t kDtU32 = 3;
inline constexpr std::uint8_t kDtF32 = 4;

} // namespace qwen35_ops

#endif
