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
// 检查点 + 全行 logits 的 scratch（放在 blob 之后的空闲 RAM 里）
inline constexpr std::uint32_t kCkptBytes = kStateBytes;
inline constexpr std::uint32_t kLogitsAllBytes =
    static_cast<std::uint32_t>(mb::kSeq) * static_cast<std::uint32_t>(mb::kVocab) * 4;

// 4 KB（32x32 f32）一块的拷贝：bytes 必须是 4096 的倍数。
YAI_OP_INLINE void copy_state_4k(std::byte *dst, std::byte *src, std::uint32_t bytes) {
  Acc a{};
  a.set_strides(128, 4);
  a.set_dtype(AccDType::F32);
  a.set_valid_zone(32, 32);
  a.set_layout_row_major();
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

// scratch 布局：检查点 | 全行 logits
YAI_OP_INLINE std::byte *ckpt_ptr(std::byte *scratch) { return scratch; }
YAI_OP_INLINE std::byte *logits_all_ptr(std::byte *scratch) { return scratch + kCkptBytes; }

} // namespace qwen35_ops

#endif
