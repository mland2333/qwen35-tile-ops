// ============================================================================
// Qwen3.5-0.8B 文本塔：一次前向（prefill 任意长度 + 可选 decode）
//
// 24 层 = 18 层 GDN（线性注意力）+ 6 层全注意力（full_attention_interval = 4，
// 即第 3/7/11/15/19/23 层）。两种层共享同一个 MLP 与残差骨架。
//
// 控制块 CTRL = [n_prompt, n_gen, max_seq, mode]：
//   mode=0：内置生成循环（prefill 按 kSeq 分块 + 贪心 argmax）
//   mode=1：单步——只处理绝对位置 CTRL[0] 的 1 个 token（宿主自己采样用）
//
// 权重/状态/scratch 的偏移全部来自自动生成的 config/model_qwen35.h
// （tools/convert_pqwf35.py 产出），这里不写任何硬编码地址。
// ============================================================================
#include "../kernels/attn256.h"
#include "../kernels/embedding.h"
#include "../kernels/qwen35.h"
#include "model_qwen35.h"

namespace qwen35_ops {

#ifdef Q35_DUMP
// ---- 调试转储：把中间张量拷进 LOGITS 区（993 KB，只在最后才用）----
// 目的是逐阶段与 numpy 参考对拍：embedding / 门控 / 卷积 / q8k8 / 递推 / o_proj。
YAI_ALWAYS_INLINE void dump_f32(std::byte *logits, int slot, const MatrixView &src) {
  // dst 是紧凑的 [rows][cols] f32：行跨步必须是 cols*4（不是 4！）
  const MatrixView dst{logits + static_cast<long>(slot) * 4,
                       static_cast<std::int32_t>(src.cols) * 4, 4, src.rows,
                       src.cols, kDtF32};
  for (std::uint32_t m = 0; m < src.rows; m += kBlock) {
    for (std::uint32_t n = 0; n < src.cols; n += kBlock) {
      Acc t{};
      load_acc(t, src, m, n);
      store_acc(t, dst, m, n);
    }
  }
}
YAI_ALWAYS_INLINE void dump_i8(std::byte *logits, int slot, const MatrixView &src) {
  const MatrixView dst{logits + static_cast<long>(slot),
                       static_cast<std::int32_t>(src.cols), 1, src.rows,
                       src.cols, kDtI8};
  for (std::uint32_t m = 0; m < src.rows; m += kBlock) {
    for (std::uint32_t n = 0; n < src.cols; n += kBlock) {
      Tile t{};
      load_tile(t, src, m, n);
      store_tile(t, dst, m, n);
    }
  }
}
#endif

// 调试标记（只写一个 int32 到 kARGF；正式跑时被 argmax 覆盖也不影响）
YAI_ALWAYS_INLINE void mark(std::byte *blob, std::int32_t v) {
  *reinterpret_cast<volatile std::int32_t *>(blob + mb::kARGF) = v;
}


// ---- 模型里所有矩阵的视图（地址/形状/跨步），按块传给前向 ----
struct ModelViews {
  std::byte *blob;
  MatrixView ids, x, emb, emb_s, lm, lm_s, nf, cos_t, sin_t;
  MatrixView h2, h8, hs;
  MatrixView qkv, mix, q8, k8, qscr, kscr, o32, z, betas, alphas;
  MatrixView qg, qf, kr, vr, attn, at8, as, out;
  MatrixView lg, lu, lsw, ls8, lss;
  MatrixView logits, argf, argi;
};

// ---------------------------------------------------------------------------
// GDN 层（一层）
//   h    : 本层输入（rmsnorm 之后的 f32 激活）
//   h8/hs: h 的 per-token 量化
//   x    : 残差流（本层输出要加回这里）
// ---------------------------------------------------------------------------
YAI_ALWAYS_INLINE void gdn_layer(const ModelViews &w, std::byte *lw, std::byte *ln,
                                 const MatrixView &h8, const MatrixView &hs,
                                 const MatrixView &x, std::uint32_t rows,
                                 int layer) {
  constexpr int H = mb::kHidden;
  constexpr int CD = mb::kConvDim;

  // GDN 层的序号（每 4 层里 3 个 GDN）：层内权重/状态/历史都按它索引
  const int gi = layer - layer / 4;
  // 注意：GW_*/GN_* 是"第 0 层的绝对偏移"，每层的增量必须在这里加，
  // 漏加会让所有层都用第 0 层的权重（L0 对、其余全错）。
  const long lwb = static_cast<long>(gi) * mb::kLGW_STRIDE;
  const long lnb = static_cast<long>(gi) * mb::kLGN_STRIDE;
  const MatrixView wqkv{w.blob + lwb + mb::kGW_QKV, H, 1, CD, H, kDtI8};
  const MatrixView wqkv_s{w.blob + lwb + mb::kGW_QKV_S, 4, 4, CD, 1, kDtF32};
  const MatrixView wz{w.blob + lwb + mb::kGW_Z, H, 1, mb::kKvw, H, kDtI8};
  const MatrixView wz_s{w.blob + lwb + mb::kGW_Z_S, 4, 4, mb::kKvw, 1, kDtF32};
  const MatrixView wb{w.blob + lwb + mb::kGW_BOUT, H, 1, mb::kGdnHeads, H, kDtI8};
  const MatrixView wb_s{w.blob + lwb + mb::kGW_BOUT_S, 4, 4, mb::kGdnHeads, 1, kDtF32};
  const MatrixView wa{w.blob + lwb + mb::kGW_AOUT, H, 1, mb::kGdnHeads, H, kDtI8};
  const MatrixView wa_s{w.blob + lwb + mb::kGW_AOUT_S, 4, 4, mb::kGdnHeads, 1, kDtF32};
  const MatrixView wo{w.blob + lwb + mb::kGW_OUT, mb::kKvw, 1, H, mb::kKvw, kDtI8};
  const MatrixView wo_s{w.blob + lwb + mb::kGW_OUT_S, 4, 4, H, 1, kDtF32};
  // ---- f32 小张量 ----
  const MatrixView dtb{w.blob + lnb + mb::kGN_DTB, 4, 4, mb::kGdnHeads, 1, kDtF32};
  const MatrixView aexp{w.blob + lnb + mb::kGN_AEXP, 4, 4, mb::kGdnHeads, 1, kDtF32};
  const MatrixView normw{w.blob + lnb + mb::kGN_NORMW, 4, 4, mb::kDv, 1, kDtF32};
  const MatrixView convw{w.blob + lnb + mb::kGN_CONVW, mb::kConvK * 4, 4, CD, mb::kConvK, kDtF32};

  std::byte *state = w.blob + mb::kS_BASE + gi * mb::kS_LAYER;
  std::byte *hist = w.blob + mb::kHIST_BASE + gi * mb::kHIST_LAYER;

  const MatrixView sview{state, mb::kDk * 4, 4,
                         static_cast<std::uint32_t>(mb::kGdnHeads * mb::kDv),
                         static_cast<std::uint32_t>(mb::kDk), kDtF32};
  // 短卷积的 staging 缓冲：共享的 w.qkv（[3+SEQ][6144]）。
  // 前 3 行（kConvK-1）放上一块留下的历史（在 blob 的 HIST 区，每层 3 行）。
  MatrixView buf = sub_view(w.qkv, 0, 0, 3 + rows, CD);
  const MatrixView histv{hist, CD * 4, 4, 3, CD, kDtF32};
  copy_rows_f32(buf, 0, histv, 0, 3);
  const MatrixView mix = sub_view(w.mix, 0, 0, rows, CD);
  const MatrixView qk_in = sub_view(w.mix, 0, 0, rows, CD);

  // ---- 1) 两个 16 维门控投影 + 门控标量 ----
  mark(w.blob, 101);   // MARK gdn:gates
  const MatrixView braw = sub_view(w.qscr, 0, 0, rows, mb::kGdnHeads);
  const MatrixView araw = sub_view(w.kscr, 0, 0, rows, mb::kGdnHeads);
  gemm_nt(h8, wb, wb_s, hs, braw);
  gemm_nt(h8, wa, wa_s, hs, araw);
  gdn_gates(braw, araw, dtb, aexp, sub_view(w.betas, 0, 0, rows, mb::kGdnHeads),
            sub_view(w.alphas, 0, 0, rows, mb::kGdnHeads));

  // ---- 2) qkv 投影直接写进卷积缓冲的第 3 行起（前 3 行是历史）----
  mark(w.blob, 102);   // MARK gdn:qkv
  gemm_nt_tiled(h8, wqkv, wqkv_s, hs, sub_view(buf, 3, 0, rows, CD));
  // ---- 3) 输出门投影 ----
  mark(w.blob, 103);   // MARK gdn:z
  gemm_nt_tiled(h8, wz, wz_s, hs, sub_view(w.z, 0, 0, rows, mb::kKvw));

  // ---- 4) 短卷积 + silu ----
  mark(w.blob, 104);   // MARK gdn:conv
  gdn_conv_silu(buf, convw, mix);
#ifdef Q35_DUMP
  if (layer == 0) {
    std::byte *dg = w.blob + mb::kDUMP;
    constexpr int NS = mb::kSeq;
    dump_f32(dg, 1024, sub_view(buf, 3, 0, rows, CD));                  // 卷积前的 qkv
    dump_f32(dg, 1024 + NS * CD, sub_view(w.mix, 0, 0, rows, CD));      // 卷积后 mix
  }
#endif

  // ---- 5) 把最后 3 行写回历史（供下一块用）----
  copy_rows_f32(histv, 0, buf, rows, 3);

  // ---- 6) q/k 的 L2 归一化（沿 128 维）----
  mark(w.blob, 105);   // MARK gdn:l2
  // q 的 1/sqrt(d_k)：i8 路径折进 dequant 标量（所以这里只做 l2norm），
  // f32 路径没有反量化、必须在这里就乘上去。
  if constexpr (kGdnI8Read) {
    l2norm_heads<false>(qk_in, 0, mb::kGdnHeads);
  } else {
    l2norm_heads<true>(qk_in, 0, mb::kGdnHeads);
  }
  l2norm_heads<false>(qk_in, static_cast<std::uint32_t>(mb::kKvw), mb::kGdnHeads);  // k

#ifdef Q35_DUMP
  if (layer == 0) {
    dump_f32(w.blob + mb::kDUMP, 230000, sub_view(w.mix, 0, 0, rows, CD));  // l2norm 之后
  }
#endif
  mark(w.blob, 106);   // MARK gdn:quant
  // per-(token,head) int8 量化（q 的 1/sqrt(d_k) 折进 dequant 标量）
  // 只有 i8 读路径才需要；f32 读路径直接读归一化后的 f32，省掉这两遍
  if constexpr (kGdnI8Read) {
    quant_heads<true>(qk_in, 0, mb::kGdnHeads, sub_view(w.q8, 0, 0, rows, mb::kKvw),
                      sub_view(w.qscr, 0, 0, rows, mb::kGdnHeads));
    quant_heads<false>(qk_in, static_cast<std::uint32_t>(mb::kKvw), mb::kGdnHeads,
                       sub_view(w.k8, 0, 0, rows, mb::kKvw),
                       sub_view(w.kscr, 0, 0, rows, mb::kGdnHeads));
  }

#ifdef Q35_DUMP
  if (layer == 0) {
    std::byte *dg = w.blob + mb::kDUMP;
    constexpr int NS = mb::kSeq;
    dump_f32(dg, 0, sub_view(w.betas, 0, 0, rows, mb::kGdnHeads));         // beta
    dump_f32(dg, 512, sub_view(w.alphas, 0, 0, rows, mb::kGdnHeads));      // alpha
    (void)NS;
  }
#endif
  // ---- 8) delta 递推 ----
  mark(w.blob, 107);   // MARK gdn:rec
  if constexpr (kGdnI8Read) {
    gdn_recurrence_i8(sub_view(w.q8, 0, 0, rows, mb::kKvw),
                      sub_view(w.k8, 0, 0, rows, mb::kKvw),
                      sub_view(w.mix, 0, 0, rows, CD),
                      sub_view(w.qscr, 0, 0, rows, mb::kGdnHeads),
                      sub_view(w.kscr, 0, 0, rows, mb::kGdnHeads),
                      sub_view(w.betas, 0, 0, rows, mb::kGdnHeads),
                      sub_view(w.alphas, 0, 0, rows, mb::kGdnHeads), sview,
                      sub_view(w.o32, 0, 0, rows, mb::kKvw));
  } else {
    gdn_recurrence_f32(sub_view(w.mix, 0, 0, rows, CD),
                       sub_view(w.betas, 0, 0, rows, mb::kGdnHeads),
                       sub_view(w.alphas, 0, 0, rows, mb::kGdnHeads), sview,
                       sub_view(w.o32, 0, 0, rows, mb::kKvw));
  }

#ifdef Q35_DUMP
  if (layer == 0) {
    std::byte *dg = w.blob + mb::kDUMP;
    constexpr int NS = mb::kSeq;
    dump_f32(dg, 1024 + 2 * NS * CD + 1024 + 2 * NS * mb::kKvw,
             sub_view(w.o32, 0, 0, rows, mb::kKvw));                        // 递推输出
    dump_f32(dg, 1024 + 2 * NS * CD + 1024 + 3 * NS * mb::kKvw,
             sub_view(w.z, 0, 0, rows, mb::kKvw));                          // 输出门
  }
#endif
  // ---- 9) gated RMSNorm（per head over 128）+ silu(z) ----
  mark(w.blob, 108);   // MARK gdn:gnorm
  gdn_gated_norm(sub_view(w.o32, 0, 0, rows, mb::kKvw), normw,
                 sub_view(w.z, 0, 0, rows, mb::kKvw));

#ifdef Q35_DUMP
  if (layer == 0) {
    std::byte *dg = w.blob + mb::kDUMP;
    constexpr int NS = mb::kSeq;
    dump_f32(dg, 1024 + 2 * NS * CD + 1024 + 4 * NS * mb::kKvw,
             sub_view(w.o32, 0, 0, rows, mb::kKvw));                        // gated norm 之后
  }
#endif
  // ---- 10) 量化 + o_proj ----
  mark(w.blob, 109);   // MARK gdn:oproj
  quant_rows(sub_view(w.o32, 0, 0, rows, mb::kKvw), sub_view(w.at8, 0, 0, rows, mb::kKvw),
             sub_view(w.as, 0, 0, rows, 1));
  gemm_nt_tiled(sub_view(w.at8, 0, 0, rows, mb::kKvw), wo, wo_s,
          sub_view(w.as, 0, 0, rows, 1), sub_view(w.out, 0, 0, rows, H));
}

// ---------------------------------------------------------------------------
// 全注意力层（一层）：q(+gate) / k / v → qk-norm → 部分 RoPE → GQA → 输出门
// ---------------------------------------------------------------------------
YAI_ALWAYS_INLINE void full_layer(const ModelViews &w, std::byte *lw, std::byte *ln,
                                  const MatrixView &h8, const MatrixView &hs,
                                  const MatrixView &x, std::uint32_t rows,
                                  std::uint32_t pos, int layer) {
  constexpr int H = mb::kHidden;
  constexpr int QW = mb::kQW;
  constexpr int KW = mb::kKW;
  constexpr int HD = mb::kHeadDim;

  const int fi = layer / 4;
  const long lfb = static_cast<long>(fi) * mb::kLFW_STRIDE;
  const long lfnb = static_cast<long>(fi) * mb::kLFN_STRIDE;
  const MatrixView wqg{w.blob + lfb + mb::kFW_QG, H, 1, 2 * QW, H, kDtI8};
  const MatrixView wqg_s{w.blob + lfb + mb::kFW_QG_S, 4, 4, 2 * QW, 1, kDtF32};
  const MatrixView wk{w.blob + lfb + mb::kFW_K, H, 1, KW, H, kDtI8};
  const MatrixView wk_s{w.blob + lfb + mb::kFW_K_S, 4, 4, KW, 1, kDtF32};
  const MatrixView wv{w.blob + lfb + mb::kFW_V, H, 1, KW, H, kDtI8};
  const MatrixView wv_s{w.blob + lfb + mb::kFW_V_S, 4, 4, KW, 1, kDtF32};
  const MatrixView wo{w.blob + lfb + mb::kFW_O, QW, 1, H, QW, kDtI8};
  const MatrixView wo_s{w.blob + lfb + mb::kFW_O_S, 4, 4, H, 1, kDtF32};

  const MatrixView qn{w.blob + lfnb + mb::kFN_QN, 4, 4, HD, 1, kDtF32};
  const MatrixView kn{w.blob + lfnb + mb::kFN_KN, 4, 4, HD, 1, kDtF32};

  std::byte *kv = w.blob + mb::kKV_BASE + fi * mb::kKV_STRIDE;
  const MatrixView kcache{kv, KW, 1, static_cast<std::uint32_t>(mb::kMaxSeq), KW, kDtI8};
  const MatrixView kcscale{kv + mb::kKV_SOFF, 4, 4,
                           static_cast<std::uint32_t>(mb::kMaxSeq), 1, kDtF32};
  const MatrixView vcache{kv + mb::kKV_VOFF, KW, 1,
                          static_cast<std::uint32_t>(mb::kMaxSeq), KW, kDtI8};
  const MatrixView vcscale{kv + mb::kKV_VOFF + mb::kKV_SOFF, 4, 4,
                           static_cast<std::uint32_t>(mb::kMaxSeq), 1, kDtF32};

  const MatrixView cos_p = sub_view(w.cos_t, pos, 0, rows,
                                    static_cast<std::uint32_t>(mb::kHalf));
  const MatrixView sin_p = sub_view(w.sin_t, pos, 0, rows,
                                    static_cast<std::uint32_t>(mb::kHalf));

  // ---- 1) 三个投影 ----
  mark(w.blob, 110);   // MARK full:proj
  const MatrixView qg = sub_view(w.qg, 0, 0, rows, 2 * QW);
  const MatrixView qf = sub_view(w.qf, 0, 0, rows, QW);
  const MatrixView kr = sub_view(w.kr, 0, 0, rows, KW);
  const MatrixView vr = sub_view(w.vr, 0, 0, rows, KW);
  gemm_nt_tiled(h8, wqg, wqg_s, hs, qg);
  gemm_nt_tiled(h8, wk, wk_s, hs, kr);
  gemm_nt_tiled(h8, wv, wv_s, hs, vr);

  // ---- 2) k/v：per-head qk-norm + 部分 RoPE（原地） ----
  mark(w.blob, 111);   // MARK full:krope
  for (int kvh = 0; kvh < mb::kKvHeads; ++kvh) {
    const std::uint32_t c0 = static_cast<std::uint32_t>(kvh * HD);
    MatrixView kh = sub_view(kr, 0, c0, rows, HD);
    rmsnorm_rows(kh, kn, kh, 1.0f / static_cast<float>(HD));
    rope_partial_mrope(kh, cos_p, sin_p);
  }
  // ---- 3) q：per-head qk-norm + 部分 RoPE，再紧凑拷贝到 qf ----
  mark(w.blob, 112);   // MARK full:qrope
  for (int hh = 0; hh < mb::kHeads; ++hh) {
    const std::uint32_t c0 = static_cast<std::uint32_t>(hh * 2 * HD);   // 每头 512 列
    const std::uint32_t q0 = static_cast<std::uint32_t>(hh * HD);
    MatrixView qh = sub_view(qg, 0, c0, rows, HD);
    rmsnorm_rows(qh, qn, qh, 1.0f / static_cast<float>(HD));
    rope_partial_mrope(qh, cos_p, sin_p);
    // 拷贝到紧凑 qf（[rows][8*256]）
    for (std::uint32_t m = 0; m < rows; m += kBlock) {
      for (std::uint32_t c = 0; c < static_cast<std::uint32_t>(HD); c += kBlock) {
        Acc t{};
        load_acc(t, qh, m, c);
        store_acc(t, qf, m, q0 + c);
      }
    }
  }

#ifdef Q35_DUMP
  if (layer == 3) {
    std::byte *dg = w.blob + mb::kDUMP;
    dump_f32(dg, 600000, sub_view(w.qg, 0, 0, rows, 2 * QW));   // q_proj 原始输出
    dump_f32(dg, 630000, sub_view(w.qf, 0, 0, rows, QW));       // 紧凑 q（norm+rope 后）
    dump_f32(dg, 650000, sub_view(w.kr, 0, 0, rows, KW));       // k（norm+rope 后）
    dump_f32(dg, 660000, sub_view(w.vr, 0, 0, rows, KW));       // v
  }
#endif
  // ---- 4) 把本块 k/v 量化写进 cache ----
  mark(w.blob, 113);   // MARK full:kvq
  quant_rows(kr, sub_view(kcache, pos, 0, rows, KW), sub_view(kcscale, pos, 0, rows, 1));
  quant_rows(vr, sub_view(vcache, pos, 0, rows, KW), sub_view(vcscale, pos, 0, rows, 1));

  // ---- 5) q 的 per-token 量化 + attention ----
  mark(w.blob, 114);   // MARK full:attn
  const MatrixView q8 = sub_view(w.q8, 0, 0, rows, QW);
  const MatrixView qs = sub_view(w.as, 0, 0, rows, 1);
  quant_rows(qf, q8, qs);
  const std::uint32_t kv_len = pos + rows;
  const MatrixView attn = sub_view(w.attn, 0, 0, rows, QW);
  attention_heads_gqa256(q8, sub_view(kcache, 0, 0, kv_len, KW),
                         sub_view(vcache, 0, 0, kv_len, KW), qs, kcscale, vcscale,
                         attn, mb::kHeads, mb::kKvHeads, HD, /*causal=*/true,
                         kInvSqrt256);

#ifdef Q35_DUMP
  if (layer == 3) {
    dump_f32(w.blob + mb::kDUMP, 670000, sub_view(w.attn, 0, 0, rows, QW));  // attention 输出
  }
#endif
  // ---- 6) 输出门：每头 256 列里 q 之后紧跟 256 列 gate ----
  mark(w.blob, 115);   // MARK full:gate
  for (int hh = 0; hh < mb::kHeads; ++hh) {
    const std::uint32_t o0 = static_cast<std::uint32_t>(hh * HD);
    const std::uint32_t g0 = static_cast<std::uint32_t>(hh * 2 * HD + HD);
    mul_sigmoid_rows(sub_view(attn, 0, o0, rows, HD), sub_view(qg, 0, g0, rows, HD));
  }

#ifdef Q35_DUMP
  if (layer == 3) {
    dump_f32(w.blob + mb::kDUMP, 690000, sub_view(w.attn, 0, 0, rows, QW));  // 乘门之后
  }
#endif
  // ---- 7) o_proj ----
  mark(w.blob, 116);   // MARK full:oproj
  quant_rows(attn, sub_view(w.at8, 0, 0, rows, QW), sub_view(w.as, 0, 0, rows, 1));
  gemm_nt_tiled(sub_view(w.at8, 0, 0, rows, QW), wo, wo_s,
          sub_view(w.as, 0, 0, rows, 1), sub_view(w.out, 0, 0, rows, H));
}

// ---------------------------------------------------------------------------
// 一块 token（rows <= kSeq）的完整前向。pos 是本块第一个 token 的绝对位置。
// ---------------------------------------------------------------------------
YAI_ALWAYS_INLINE void forward_chunk(const ModelViews &w, std::uint32_t pos,
                                     std::uint32_t rows, bool want_logits) {
  constexpr int H = mb::kHidden;
  constexpr float kInvH = 1.0f / static_cast<float>(mb::kHidden);

  const MatrixView x = sub_view(w.x, 0, 0, rows, H);
  const MatrixView h2 = sub_view(w.h2, 0, 0, rows, H);
  const MatrixView h8 = sub_view(w.h8, 0, 0, rows, H);
  const MatrixView hs = sub_view(w.hs, 0, 0, rows, 1);
  const MatrixView out = sub_view(w.out, 0, 0, rows, H);
  const MatrixView ids_p = sub_view(w.ids, pos, 0, rows, 1);
  const MatrixView lg = sub_view(w.lg, 0, 0, rows, mb::kInter);
  const MatrixView lu = sub_view(w.lu, 0, 0, rows, mb::kInter);
  const MatrixView lsw = sub_view(w.lsw, 0, 0, rows, mb::kInter);
  const MatrixView ls8 = sub_view(w.ls8, 0, 0, rows, mb::kInter);
  const MatrixView lss = sub_view(w.lss, 0, 0, rows, 1);

  // ---- embedding（残差流的起点）----
  mark(w.blob, 10);
  embedding_rows(w.emb, w.emb_s, ids_p, x);
  mark(w.blob, 11);
#ifdef Q35_DUMP
  dump_f32(w.blob + mb::kDUMP,
           1024 + 2 * mb::kSeq * mb::kConvDim + 1024 + 5 * mb::kSeq * mb::kKvw,
           sub_view(w.x, 0, 0, rows, H));                                   // embedding
#endif

  for (int L = 0; L < mb::kLayers; ++L) {
    const bool full = (L % 4) == 3;
    std::byte *lw;
    std::byte *ln;
    if (full) {
      const int fi = L / 4;
      lw = w.blob + mb::kLFW_BASE + fi * mb::kLFW_STRIDE;
      ln = w.blob + mb::kLFN_BASE + fi * mb::kLFN_STRIDE;
    } else {
      const int gi = L - L / 4;
      lw = w.blob + mb::kLGW_BASE + gi * mb::kLGW_STRIDE;
      ln = w.blob + mb::kLGN_BASE + gi * mb::kLGN_STRIDE;
    }
    const MatrixView n1{ln, 4, 4, H, 1, kDtF32};
    const MatrixView n2{ln + 4 * H, 4, 4, H, 1, kDtF32};

    // ---- 注意力子层 ----
    mark(w.blob, 20 + L * 4);
    rmsnorm_rows(x, n1, h2, kInvH);
    quant_rows(h2, h8, hs);
    if (full) {
      full_layer(w, lw, ln, h8, hs, x, rows, pos, L);
    } else {
      gdn_layer(w, lw, ln, h8, hs, x, rows, L);
    }
    add_rows(x, out);

    // ---- MLP 子层（两种层相同）----
    mark(w.blob, 22 + L * 4);
    const long mlp_off = full ? static_cast<long>(L / 4) * mb::kLFW_STRIDE
                              : static_cast<long>(L - L / 4) * mb::kLGW_STRIDE;
    const MatrixView wg{w.blob + mlp_off + (full ? mb::kFW_G : mb::kGW_G), H, 1, mb::kInter, H, kDtI8};
    const MatrixView wg_s{w.blob + mlp_off + (full ? mb::kFW_G_S : mb::kGW_G_S), 4, 4, mb::kInter, 1, kDtF32};
    const MatrixView wu{w.blob + mlp_off + (full ? mb::kFW_U : mb::kGW_U), H, 1, mb::kInter, H, kDtI8};
    const MatrixView wu_s{w.blob + mlp_off + (full ? mb::kFW_U_S : mb::kGW_U_S), 4, 4, mb::kInter, 1, kDtF32};
    const MatrixView wd{w.blob + mlp_off + (full ? mb::kFW_D : mb::kGW_D), mb::kInter, 1, H, mb::kInter, kDtI8};
    const MatrixView wd_s{w.blob + mlp_off + (full ? mb::kFW_D_S : mb::kGW_D_S), 4, 4, H, 1, kDtF32};
    rmsnorm_rows(x, n2, h2, kInvH);
    quant_rows(h2, h8, hs);
    gemm_nt_tiled(h8, wg, wg_s, hs, lg);
    gemm_nt_tiled(h8, wu, wu_s, hs, lu);
    swiglu(lg, lu, lsw);
    quant_rows(lsw, ls8, lss);
    gemm_nt_tiled(ls8, wd, wd_s, lss, out);
#ifdef Q35_DUMP
    if (L == 0) {
      std::byte *dg = w.blob + mb::kDUMP;
      dump_f32(dg, 500000, sub_view(w.lg, 0, 0, rows, mb::kInter));
      dump_f32(dg, 520000, sub_view(w.lu, 0, 0, rows, mb::kInter));
      dump_f32(dg, 540000, sub_view(w.lsw, 0, 0, rows, mb::kInter));
      dump_f32(dg, 560000, sub_view(w.out, 0, 0, rows, H));
      dump_f32(dg, 570000, sub_view(w.h2, 0, 0, rows, H));
    }
#endif
    add_rows(x, out);
#ifdef Q35_DUMP
    // 每层结束后的残差流（用于"第一处发散在哪一层"）
    // 逐位置（= 逐行）残差流：只挑 3 层，行跨步够 32 行（用于"误差随位置涨不涨"的测试）
    if (L == 0) {
      dump_f32(w.blob + mb::kDUMP, 800000 + 0 * 34000, sub_view(w.x, 0, 0, rows, H));
    } else if (L == 11) {
      dump_f32(w.blob + mb::kDUMP, 800000 + 1 * 34000, sub_view(w.x, 0, 0, rows, H));
    } else if (L == 23) {
      dump_f32(w.blob + mb::kDUMP, 800000 + 2 * 34000, sub_view(w.x, 0, 0, rows, H));
    }
#endif
  }

  if (want_logits) {
    // 只算最后一行的 final-norm + lm_head（省掉整块的 2048×1024 + 254 MB 权重读）
    const std::uint32_t last = rows - 1;
    const MatrixView xl = sub_view(w.x, last, 0, 1, H);
    const MatrixView hl = sub_view(w.h2, 0, 0, 1, H);
    const MatrixView hl8 = sub_view(w.h8, 0, 0, 1, H);
    const MatrixView hls = sub_view(w.hs, 0, 0, 1, 1);
    rmsnorm_rows(xl, w.nf, hl, kInvH);
    quant_rows(hl, hl8, hls);
    gemm_nt(hl8, w.lm, w.lm_s, hls, sub_view(w.logits, 0, 0, 1, mb::kVocab));
  }
}

// 贪心 argmax（对 logits 第 0 行）。跨 32 列块比较时把 f32 位模式映射成单调整型
// key：-march=rv64im 没有 F 扩展，标量浮点比较会拉 libgcc 的软浮点 helper。
YAI_ALWAYS_INLINE std::uint32_t argmax_logits(const ModelViews &w) {
  const MatrixView logits1 = sub_view(w.logits, 0, 0, 1, mb::kVocab);
  std::uint32_t best = 0;
  std::uint32_t best_key = 0;
  bool first = true;
  for (std::uint32_t n = 0; n < mb::kVocab; n += kBlock) {
    Acc a{};
    load_acc(a, logits1, 0, n);
    Vec32 mx = a.row_max();
    Vec32 ix = a.row_argmax();
    store_vec(mx, w.argf, 0);
    store_vec(ix, w.argi, 0);
    const std::uint32_t bits = *reinterpret_cast<const std::uint32_t *>(w.argf.data);
    const std::uint32_t idx = *reinterpret_cast<const std::uint32_t *>(w.argi.data);
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

int main() {
  std::uintptr_t base;
  asm volatile("mv %0, x31" : "=r"(base));
  auto *blob = reinterpret_cast<std::byte *>(base);
  using namespace qwen35_ops;
  constexpr int H = mb::kHidden;
  constexpr std::uint32_t N = mb::kSeq;

  ModelViews w{};
  w.blob = blob;
  w.ids = MatrixView{blob + mb::kIDS, 4, 4, mb::kMaxSeq, 1, kDtI32};
  w.x = MatrixView{blob + mb::kX, H * 4, 4, N, H, kDtF32};
  w.emb = MatrixView{blob + mb::kEMB, H, 1, mb::kVocab, H, kDtI8};
  w.emb_s = MatrixView{blob + mb::kEMB_S, 4, 4, mb::kVocab, 1, kDtF32};
  w.lm = MatrixView{blob + mb::kLM, H, 1, mb::kVocab, H, kDtI8};
  w.lm_s = MatrixView{blob + mb::kLM_S, 4, 4, mb::kVocab, 1, kDtF32};
  w.nf = MatrixView{blob + mb::kNF, 4, 4, H, 1, kDtF32};
  w.cos_t = MatrixView{blob + mb::kCOS, mb::kHalf * 4, 4, mb::kMaxSeq, mb::kHalf, kDtF32};
  w.sin_t = MatrixView{blob + mb::kSIN, mb::kHalf * 4, 4, mb::kMaxSeq, mb::kHalf, kDtF32};
  w.h2 = MatrixView{blob + mb::kH2, H * 4, 4, N, H, kDtF32};
  w.h8 = MatrixView{blob + mb::kH8, H, 1, N, H, kDtI8};
  w.hs = MatrixView{blob + mb::kHS, 4, 4, N, 1, kDtF32};
  w.qkv = MatrixView{blob + mb::kQKV, mb::kConvDim * 4, 4, 3 + N, mb::kConvDim, kDtF32};
  w.mix = MatrixView{blob + mb::kMIX, mb::kConvDim * 4, 4, N, mb::kConvDim, kDtF32};
  w.q8 = MatrixView{blob + mb::kQ8, mb::kKvw, 1, N, mb::kKvw, kDtI8};
  w.k8 = MatrixView{blob + mb::kK8, mb::kKvw, 1, N, mb::kKvw, kDtI8};
  w.qscr = MatrixView{blob + mb::kQSCR, 16 * 4, 4, N, 16, kDtF32};
  w.kscr = MatrixView{blob + mb::kKSCR, 16 * 4, 4, N, 16, kDtF32};
  w.o32 = MatrixView{blob + mb::kO32, mb::kKvw * 4, 4, N, mb::kKvw, kDtF32};
  w.z = MatrixView{blob + mb::kZ, mb::kKvw * 4, 4, N, mb::kKvw, kDtF32};
  w.betas = MatrixView{blob + mb::kBETAS, 16 * 4, 4, N, 16, kDtF32};
  w.alphas = MatrixView{blob + mb::kALPHAS, 16 * 4, 4, N, 16, kDtF32};
  w.qg = MatrixView{blob + mb::kQG, 2 * mb::kQW * 4, 4, N, 2 * mb::kQW, kDtF32};
  w.qf = MatrixView{blob + mb::kQF, mb::kQW * 4, 4, N, mb::kQW, kDtF32};
  w.kr = MatrixView{blob + mb::kKR, mb::kKW * 4, 4, N, mb::kKW, kDtF32};
  w.vr = MatrixView{blob + mb::kVR, mb::kKW * 4, 4, N, mb::kKW, kDtF32};
  w.attn = MatrixView{blob + mb::kATTN, mb::kQW * 4, 4, N, mb::kQW, kDtF32};
  w.at8 = MatrixView{blob + mb::kAT8, mb::kQW, 1, N, mb::kQW, kDtI8};
  w.as = MatrixView{blob + mb::kAS, 4, 4, N, 1, kDtF32};
  w.out = MatrixView{blob + mb::kOUT, H * 4, 4, N, H, kDtF32};
  w.lg = MatrixView{blob + mb::kLG, mb::kInter * 4, 4, N, mb::kInter, kDtF32};
  w.lu = MatrixView{blob + mb::kLU, mb::kInter * 4, 4, N, mb::kInter, kDtF32};
  w.lsw = MatrixView{blob + mb::kLSW, mb::kInter * 4, 4, N, mb::kInter, kDtF32};
  w.ls8 = MatrixView{blob + mb::kLS8, mb::kInter, 1, N, mb::kInter, kDtI8};
  w.lss = MatrixView{blob + mb::kLSS, 4, 4, N, 1, kDtF32};
  w.logits = MatrixView{blob + mb::kLOGITS, mb::kVocab * 4, 4, 1, mb::kVocab, kDtF32};
  w.argf = MatrixView{blob + mb::kARGF, 4, 4, 1, 1, kDtF32};
  w.argi = MatrixView{blob + mb::kARGI, 4, 4, 1, 1, kDtI32};

  auto *ctrl = reinterpret_cast<volatile std::int32_t *>(blob + mb::kCTRL);
  const std::uint32_t n_prompt = static_cast<std::uint32_t>(ctrl[0]);
  const std::uint32_t n_gen = static_cast<std::uint32_t>(ctrl[1]);
  const std::uint32_t mode = static_cast<std::uint32_t>(ctrl[3]);
  if (mode == 1) {                       // 单步：宿主侧采样用
    forward_chunk(w, n_prompt, 1, /*want_logits=*/true);
    return 0;
  }
#ifdef Q35_DUMP
  // 转储模式：不跑 lm_head，否则 logits 会把 LOGITS 区里的中间量全部覆盖
  constexpr bool kDumpMode = true;
#else
  constexpr bool kDumpMode = false;
#endif
  const std::uint32_t total = n_prompt + n_gen;
  std::uint32_t pos = 0;
  while (pos < total) {
    const std::uint32_t rows =
        (pos < n_prompt) ? ((n_prompt - pos < N) ? (n_prompt - pos) : N) : 1;
    const bool want_logits = (pos + rows >= n_prompt) && !kDumpMode;
    forward_chunk(w, pos, rows, want_logits);
    pos += rows;
    if (pos >= n_prompt && pos < total) {
      const std::uint32_t id = argmax_logits(w);
      *reinterpret_cast<volatile std::uint32_t *>(w.ids.data + pos * 4) = id;
    }
  }
  return 0;
}
