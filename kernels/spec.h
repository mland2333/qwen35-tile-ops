#ifndef YAI_QWEN35_OPS_SPEC_H
#define YAI_QWEN35_OPS_SPEC_H

// ============================================================================
// 投机解码（prompt lookup）需要的两个原语
//
//   1) 草稿：在已经生成的 ids 里找与"当前后缀"最长的匹配，把匹配之后的
//      token 抄过来当候选。零模型成本，适合重复性文本 / thinking 链。
//   2) 状态检查点：验证块会推进 GDN 状态（S + 卷积历史，共 19.3 MiB，在
//      blob 里是连续的一段 [kS_BASE, kKV_BASE)），部分接受时要回滚到验证前。
//      用 32x32 f32 的 4 KB 块搬，来回约 2x4932 条 load/store。
//
// KV cache 不用检查点：它按位置索引，回滚只需要把位置指针退回去，数据下次
// 覆盖即可。
// ============================================================================
#include "tile.h"

#include "config.h"
#include "io.h"
#include "matrix.h"
#include "model_qwen35.h"

namespace qwen35_ops {

// GDN 状态 + 卷积历史在 blob 里是连续的一段
inline constexpr std::uint32_t kStateBytes =
    static_cast<std::uint32_t>(mb::kKV_BASE - mb::kS_BASE);
// 状态槽（验证前的一份 S + 卷积历史）
inline constexpr std::uint32_t kCkptBytes = kStateBytes;

// ---- 输入检查点：只存"递推的输入"，部分接受时用它重放递推 ----
//   每 (层, 行)：mix 整行（24 KB，f32）+ q8/k8（各 2 KB，i8）+ qscr/kscr/betas/alphas（各 64 B）
//   外加卷积历史的 3 行（73,728 B）。总量：18 层 x 8 行 x ~27 KB ≈ 3.9 MB（vs 状态 144 MiB）。
inline constexpr std::uint32_t kInRows = 8;   // kDraftMax + 1              // 1+n 最多 8 行
inline constexpr std::uint32_t kInMixBytes = static_cast<std::uint32_t>(mb::kConvDim) * 4;
inline constexpr std::uint32_t kInQ8Bytes = static_cast<std::uint32_t>(mb::kKvw);
inline constexpr std::uint32_t kInK8Bytes = kInQ8Bytes;
inline constexpr std::uint32_t kInScrBytes = static_cast<std::uint32_t>(mb::kGdnHeads) * 4;
// 每层按字段分块存（同字段的行连续），这样重放时可以直接用原始 layout 的视图
inline constexpr std::uint32_t kInMixArea = kInRows * kInMixBytes;     // 8*24576
inline constexpr std::uint32_t kInQ8Area = kInRows * kInQ8Bytes;       // 8*2048
inline constexpr std::uint32_t kInScrArea = kInRows * kInScrBytes;     // 8*64
inline constexpr std::uint32_t kInOffQ8 = kInMixArea;
inline constexpr std::uint32_t kInOffK8 = kInOffQ8 + kInQ8Area;
inline constexpr std::uint32_t kInOffQScr = kInOffK8 + kInQ8Area;
inline constexpr std::uint32_t kInOffKScr = kInOffQScr + kInScrArea;
inline constexpr std::uint32_t kInOffBeta = kInOffKScr + kInScrArea;
inline constexpr std::uint32_t kInOffAlpha = kInOffBeta + kInScrArea;
inline constexpr std::uint32_t kInLayerBytes = kInOffAlpha + kInScrArea;
inline constexpr std::uint32_t kInBytes = 18u * kInLayerBytes;
inline constexpr std::uint32_t kHistRowBytes = 3u * static_cast<std::uint32_t>(mb::kConvDim) * 4;
inline constexpr std::uint32_t kHistBufBytes = 18u * kInRows * kHistRowBytes;

// scratch 布局：状态槽 | 历史行缓冲 | 输入缓冲 | 全行 logits
YAI_OP_INLINE std::byte *state_slot_ptr(std::byte *s) { return s; }
YAI_OP_INLINE std::byte *histbuf_ptr(std::byte *s) { return s + kStateBytes; }
YAI_OP_INLINE std::byte *inbuf_ptr(std::byte *s) { return histbuf_ptr(s) + kHistBufBytes; }
YAI_OP_INLINE std::byte *logits_all_ptr(std::byte *s) { return inbuf_ptr(s) + kInBytes; }
YAI_OP_INLINE std::byte *in_layer_ptr(std::byte *s, int layer) {
  return inbuf_ptr(s) + static_cast<std::uint32_t>(layer) * kInLayerBytes;
}
YAI_OP_INLINE std::byte *hist_row_ptr(std::byte *s, int layer, std::uint32_t row) {
  return histbuf_ptr(s) + (static_cast<std::uint32_t>(layer) * kInRows + row) * kHistRowBytes;
}

// i8 字节块拷贝（32x32 tile，bytes 是 1024 的倍数）
YAI_OP_INLINE void copy_bytes_i8(std::byte *dst, std::byte *src, std::uint32_t bytes) {
  Tile t{};
  t.set_strides(32, 1);
  t.set_dtype(TileDType::I8);
  t.set_valid_zone(32, 32);
  for (std::uint32_t off = 0; off < bytes; off += 1024) {
    t.set_data(src + off);
    t.load();
    t.set_data(dst + off);
    t.store();
  }
}

// 小标量块拷贝（16 个 f32 的 scale/gate 行）
YAI_OP_INLINE void copy_vec16(std::byte *dst, std::byte *src) {
  Vec32 v{};
  v.set_dtype(Vec32DType::F32);
  v.set_stride(4);
  set_len(v, static_cast<std::uint8_t>(mb::kGdnHeads));
  load_vec_at(v, src);
  store_vec_at(v, dst);
}

inline constexpr std::uint32_t kLogitsAllBytes =
    static_cast<std::uint32_t>(mb::kSeq) * static_cast<std::uint32_t>(mb::kVocab) * 4;

// 4 KB（32x32 f32）一块的拷贝：bytes 必须是 4096 的倍数。
YAI_OP_INLINE void copy_state_4k(std::byte *dst, std::byte *src, std::uint32_t bytes) {
  Acc a{};
  a.set_strides(128, 4);
  a.set_dtype(AccDType::F32);
  a.set_valid_zone(32, 32);
  for (std::uint32_t off = 0; off < bytes; off += 4096) {
    a.set_data(src + off);
    a.load();
    a.set_data(dst + off);
    a.store();
  }
}

YAI_OP_INLINE void save_state(const std::byte *blob, std::byte *ckpt) {
  copy_state_4k(ckpt, const_cast<std::byte *>(blob) + mb::kS_BASE, kStateBytes);
}

YAI_OP_INLINE void restore_state(const std::byte *blob, const std::byte *ckpt) {
  copy_state_4k(const_cast<std::byte *>(blob) + mb::kS_BASE, const_cast<std::byte *>(ckpt),
                kStateBytes);
}

// 在 ids[0..pos) 里找最长后缀匹配，返回抄来的候选个数（<= maxk）。
// 匹配不能与后缀本身重叠；匹配之后不足 1 个 token 时退回更短的后缀。
YAI_OP_INLINE std::uint32_t lookup_draft(const std::uint32_t *ids, std::uint32_t pos,
                                         std::uint32_t maxk, std::uint32_t *out) {
  // 后缀要**包含 ids[pos]**（当前已生成、还没被处理的那个 token）：匹配段结束在
  // 某个较早的位置 e，那么 e+1 起的 token 就是 pos+1 起的候选。
  for (std::uint32_t len = pos + 1; len >= 1; --len) {
    const std::uint32_t suffix = pos + 1 - len;
    for (std::uint32_t s = 0; s + len <= pos; ++s) {
      bool ok = true;
      for (std::uint32_t i = 0; i < len; ++i) {
        if (ids[s + i] != ids[suffix + i]) {
          ok = false;
          break;
        }
      }
      if (!ok) continue;
      std::uint32_t n = 0;
      while (n < maxk && s + len + n <= pos) {
        out[n] = ids[s + len + n];
        ++n;
      }
      if (n != 0) return n;
      break;   // 匹配后面没东西可抄，换个更短的后缀
    }
  }
  return 0;
}

YAI_OP_INLINE std::byte *ckpt_ptr(std::byte *scratch) { return state_slot_ptr(scratch); }

} // namespace qwen35_ops

#endif
