#ifndef YAI_QWEN35_OPS_GDN_H
#define YAI_QWEN35_OPS_GDN_H

// ============================================================================
// Gated DeltaNet（18 层）—— Qwen3.5 的线性注意力
//
// 数学（官方 modeling_qwen3_5.py 的 torch_recurrent_gated_delta_rule，逐 token）：
//   S_t = exp(g_t) · S_{t-1}                     ① 衰减
//   pred = S_t · k_t                             ② 用旧状态预测 v
//   S_t = S_t + β_t (v_t − pred) k_tᵀ            ③ 只写入"预测误差"（delta 规则）
//   o_t = S_t · q_t                              ④ 读出
// 其中 q 先 L2norm 再乘 1/sqrt(d_k)（折进 dequant 标量），k 只做 L2norm；
// o 再走 gated RMSNorm（per head over d_v=128）乘 silu(z)，最后 o_proj。
// 注意 S 的布局：官方是 (d_k, d_v)，这里存成 **（d_v, d_k)**（转置）；
// 递推式在转置下完全同形（pred[dv]=Σ_dk S[dv][dk]k[dk]），但读变成了行内点积，
// 正好能用 `mma.nt`（A=32×32 的 S 块，B=k 的 1×32）一次算 32 个 d_v。
//
// ---- 分块策略（寄存器预算的硬约束）----
// S 是 16 头 × 128(dv) × 128(dk) f32 = 1 MB/层，整块放不进 16 个 acc（一个 acc
// 才 32×32），所以按 **32 行（d_v）× 128 列（d_k）= 4 个 acc** 的行块独立演化：
// 更新式里没有跨 d_v 的项，行块之间完全独立。
// 循环顺序是 `for (head, 行块) → for token`，一个行块的 4 个 acc 在整段 token 里
// 常驻，状态流量从"每 token 64 KB"摊薄成"每 32 token 64 KB"。
//
// ---- 精度策略 ----
// S 的权威副本永远是 f32（4 个 acc 常驻）。读（S·k / S·q）走 i8 mma：
//   每行一个 scale（areduce.rows.absmax 跨 4 个列块合并）→ tquantx → mma.nt
//   → acvt → ×s_S(byrow) ×s_k/×s_q(scalar)，合并顺序与 numpy 参考逐条对齐。
// 秩 1 写（u ⊗ k）在 f32 里做：acc 有 add(vec,byrow) / mul(vec,bycol)，把 u
// 广播成整行再按列乘 k。与设计决策 6.3 的字面写法（i8×i8→i32 外积）不同，理由：
//   * 指令更少（4×(填零+广播+乘+加) = 16 条 vs i8 外积 4×(mma+acvt+2 乘+加) = 20 条）；
//   * 少一次 u 的量化误差（原方案 32 个 d_v 值共用一个 absmax）。
// 读侧的 i8 量化是必须的（mma 只有 i8），误差 ~0.4%/行，靠 delta 规则的
// "覆写/擦除"特性吸收（误差实测见 README）。
//
// ---- 短卷积 ----
// 因果 depthwise、kernel=4、无 bias、卷积后过 silu。跨 chunk 的 3 行历史存在
// blob 的 HIST 区（每层 3×6144 f32 = 72 KB）。
// ============================================================================
#include "tile.h"

#include "config.h"
#include "io.h"
#include "matrix.h"
#include "rmsnorm.h"

namespace qwen35_ops {

// ---------------------------------------------------------------------------
// softplus(x) = log(1+e^x)，用 log1p 的 8 次多项式（f32 Horner 最大误差 1.1e-7）。
//
// 设备端没有 log，而 g = -exp(A_log)·softplus(a+dt_bias) 每个 token 都要算。
// 恒等式：softplus(x) = max(x,0) + log1p(e^{-|x|})，
// max(x,0) = (x+|x|)/2（acc 有 abs、没有 max-with-scalar），
// u = e^{-|x|} ∈ (0,1] 落在多项式区间内且不会溢出。
//
// p = log1p(u)（u 保留）。Horner：p = ((((c8·u+c7)u+c6)…)u+c0)。
// ---------------------------------------------------------------------------
YAI_ALWAYS_INLINE void log1p_poly(Acc &u, Acc &p) {
  p.fill(-6.006605049e-03f);   // c8
  p.mul(u);
  p.add(3.426459995e-02f);     // c7
  p.mul(u);
  p.add(-9.229041737e-02f);    // c6
  p.mul(u);
  p.add(1.649981298e-01f);     // c5
  p.mul(u);
  p.add(-2.394333707e-01f);    // c4
  p.mul(u);
  p.add(3.314466522e-01f);     // c3
  p.mul(u);
  p.add(-4.998254986e-01f);    // c2
  p.mul(u);
  p.add(9.999936303e-01f);     // c1
  p.mul(u);
  p.add(3.910905577e-08f);     // c0
}

// ---------------------------------------------------------------------------
// 三个 per-head 门控标量（都是 [rows][16] f32，逐元素）：
//   beta  = sigmoid(b)
//   g     = -exp(A_log) · softplus(a + dt_bias)      （≤ 0）
//   alpha = exp(g)                                   （∈ (0,1]）
// 全用累加器指令；不出现运行期标量浮点四则运算（rv64im 没有 FPU）。
// ---------------------------------------------------------------------------
YAI_OP_INLINE void gdn_gates(const MatrixView &b, const MatrixView &a,
                             const MatrixView &dtb, const MatrixView &aexp,
                             const MatrixView &beta, const MatrixView &alpha) {
  Acc t{};
  Acc e{};
  Acc w{};
  Acc p{};
  config_acc(t, a);
  config_acc(e, a);
  config_acc(w, a);
  config_acc(p, a);
  Vec32 dtv{};                 // dt_bias：长度 16 的列广播
  config_vec(dtv, dtb);
  set_len(dtv, static_cast<std::uint8_t>(16));
  load_vec_at(dtv, dtb.data);
  Vec32 aev{};                 // exp(A_log)
  config_vec(aev, aexp);
  set_len(aev, static_cast<std::uint8_t>(16));
  load_vec_at(aev, aexp.data);

  for (std::uint32_t m = 0; m < b.rows; m += kBlock) {
    const std::uint8_t nr = block_zone(b.rows, m);
    const std::uint8_t nc = 16;

    // ---- beta = sigmoid(b) = 1/(1+e^-b) ----
    set_zone(t, nr, nc);
    load_acc_at(t, b.data + m * static_cast<std::uint32_t>(b.row_bytes));
    t.mul(-1.0f);
    t.exp();
    t.add(1.0f);
    // ISA 没有除法：1/(1+e^-b) 用 rcp.approx；t 的配置与 beta 相同，直接写回
    t.rcp();
    store_acc_at(t, beta.data + m * static_cast<std::uint32_t>(beta.row_bytes));

    // ---- x = a + dt_bias；w = |x| ----
    set_zone(t, nr, nc);
    load_acc_at(t, a.data + m * static_cast<std::uint32_t>(a.row_bytes));
    t.add(dtv, /*col=*/false);
    set_zone(w, nr, nc);
    load_acc_at(w, a.data + m * static_cast<std::uint32_t>(a.row_bytes));
    w.add(dtv, /*col=*/false);
    w.abs();

    // ---- max(x,0) = (x + |x|)/2 ----
    t.add(w);
    t.mul(0.5f);

    // ---- u = exp(-|x|) ∈ (0,1] ----
    set_zone(e, nr, nc);
    load_acc_at(e, a.data + m * static_cast<std::uint32_t>(a.row_bytes));
    e.add(dtv, /*col=*/false);
    e.abs();
    e.mul(-1.0f);
    e.exp();

    // ---- softplus = max(x,0) + log1p(u)，再乘 -exp(A_log)、取 exp ----
    set_zone(p, nr, nc);          // 多项式写进 p（新建 acc 必须先配 zone）
    log1p_poly(e, p);
    t.add(p);
    t.mul(-1.0f);
    t.mul(aev, /*col=*/false);
    t.exp();
    store_acc_at(t, alpha.data + m * static_cast<std::uint32_t>(alpha.row_bytes));
  }
}

// ---------------------------------------------------------------------------
// 因果 depthwise 短卷积 + silu
//
//   buf : [kConvK-1 + rows][6144] f32（前 3 行是上一块留下的历史）
//   cw  : [6144][kConvK] f32（per-channel 系数，行 = 通道、列 = tap）
//   mix : [rows][6144] f32 输出 = silu(Σ_j cw[c][j] · buf[t+j][c])
//
// F.conv1d(padding=3, groups=C)[:, :, :T] 的等价形式：out[t] = Σ_j w[j]·x[t-3+j]。
// 按 32 通道一块；w 的 4 个 tap 用"跨步 = 4*4 字节"的 Vec32 载入。
// silu 需要一个分母 acc，没有 acc 拷贝指令 → 先把卷积和落到 mix 再读回来。
// ---------------------------------------------------------------------------
YAI_OP_INLINE void gdn_conv_silu(const MatrixView &buf, const MatrixView &cw,
                                 const MatrixView &mix) {
  Acc s{};
  Acc t{};
  config_acc(s, buf);
  config_acc(t, mix);
  Vec32 w0{};
  Vec32 w1{};
  Vec32 w2{};
  Vec32 w3{};
  const std::int32_t tap_stride = static_cast<std::int32_t>(cw.row_bytes);
  w0.set_dtype(Vec32DType::F32);
  w0.set_stride(tap_stride);
  w1.set_dtype(Vec32DType::F32);
  w1.set_stride(tap_stride);
  w2.set_dtype(Vec32DType::F32);
  w2.set_stride(tap_stride);
  w3.set_dtype(Vec32DType::F32);
  w3.set_stride(tap_stride);

  for (std::uint32_t m = 0; m < mix.rows; m += kBlock) {
    const std::uint8_t nr = block_zone(mix.rows, m);
    for (std::uint32_t c = 0; c < mix.cols; c += kBlock) {
      const std::uint8_t nc = block_zone(mix.cols, c);
      const std::uint32_t cw_off = c * static_cast<std::uint32_t>(cw.row_bytes);
      set_len(w0, nc);
      load_vec_at(w0, cw.data + cw_off + 0 * 4);
      set_len(w1, nc);
      load_vec_at(w1, cw.data + cw_off + 1 * 4);
      set_len(w2, nc);
      load_vec_at(w2, cw.data + cw_off + 2 * 4);
      set_len(w3, nc);
      load_vec_at(w3, cw.data + cw_off + 3 * 4);

      const std::uint32_t rb = static_cast<std::uint32_t>(buf.row_bytes);
      const std::uint32_t ob = m * rb + c * static_cast<std::uint32_t>(buf.col_bytes);

      set_zone(s, nr, nc);
      load_acc_at(s, buf.data + ob);
      s.mul(w0, /*col=*/false);
      set_zone(t, nr, nc);
      load_acc_at(t, buf.data + ob + rb);
      t.mul(w1, /*col=*/false);
      s.add(t);
      set_zone(t, nr, nc);
      load_acc_at(t, buf.data + ob + 2 * rb);
      t.mul(w2, /*col=*/false);
      s.add(t);
      set_zone(t, nr, nc);
      load_acc_at(t, buf.data + ob + 3 * rb);
      t.mul(w3, /*col=*/false);
      s.add(t);

      const std::uint32_t om = m * static_cast<std::uint32_t>(mix.row_bytes) +
                               c * static_cast<std::uint32_t>(mix.col_bytes);
      store_acc_at(s, mix.data + om);
      set_zone(t, nr, nc);
      load_acc_at(t, mix.data + om);
      t.mul(-1.0f);
      t.exp();
      t.add(1.0f);
      t.rcp();                        // 1/(1+e^-s)
      s.mul(t);                       // silu(s) = s/(1+e^-s)
      store_acc_at(s, mix.data + om);
    }
  }
}

// ---------------------------------------------------------------------------
// per-(token, head) 的 L2 归一化（沿 128 维、**不除 128**，eps 加在平方和上）
//   x[t][off + h*128 + i] *= rsqrt(Σ_i x² + eps)
// 逐 (行块, 头)：第一遍跨 4 个 32 列块合并 sumsq，第二遍缩放。
// ---------------------------------------------------------------------------
template <bool ScaleQ>
YAI_OP_INLINE void l2norm_heads(const MatrixView &x, std::uint32_t off,
                                int heads) {
  Acc b{};
  config_acc(b, x);
  const std::uint32_t cb = static_cast<std::uint32_t>(x.col_bytes);
  for (std::uint32_t m = 0; m < x.rows; m += kBlock) {
    const std::uint8_t nr = block_zone(x.rows, m);
    for (int h = 0; h < heads; ++h) {
      const std::uint32_t base =
          m * static_cast<std::uint32_t>(x.row_bytes) +
          (off + static_cast<std::uint32_t>(h * kDk)) * cb;
      Vec32 ss{};
      #pragma clang loop YAI_LOOP_UNROLL
      for (std::uint32_t k = 0; k < static_cast<std::uint32_t>(kDk); k += kBlock) {
        set_zone(b, nr, static_cast<std::uint8_t>(kBlock));
        load_acc_at(b, x.data + base + k * cb);
        if (k == 0) {
          ss = b.row_sum_sq();
        } else {
          b.row_sum_sq(ss);
        }
      }
      ss.add(kEps);
      ss.rsqrt();
      if (ScaleQ) {
        // 只对 q：把 1/sqrt(d_k) 折进缩放系数（编译期常量 → vmulx 立即数）
        ss.mul(kQScaleDk);
      }
      #pragma clang loop YAI_LOOP_UNROLL
      for (std::uint32_t k = 0; k < static_cast<std::uint32_t>(kDk); k += kBlock) {
        set_zone(b, nr, static_cast<std::uint8_t>(kBlock));
        load_acc_at(b, x.data + base + k * cb);
        b.mul(ss, /*col=*/true);
        store_acc_at(b, x.data + base + k * cb);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// per-(token, head) 的 int8 量化（absmax·(1/127)·extra_scale）
//
//   x  : [rows][cols] f32，只处理列区间 [off, off + heads*128)
//   q8 : [rows][cols] i8（与 x 同布局）
//   scr: [rows][heads] f32（每个 (token, head) 一个 dequant 标量）
//
// extra_scale 折进 dequant 标量：q 的头传 1/sqrt(128)，k 传 1.0。
// 这样 q 不需要单独一遍缩放（也避免运行期标量浮点乘法）。
// ---------------------------------------------------------------------------
template <bool FoldQScale>
YAI_OP_INLINE void quant_heads(const MatrixView &x, std::uint32_t off, int heads,
                               const MatrixView &q8, const MatrixView &scr) {
  Acc b{};
  config_acc(b, x);
  Tile t{};
  config_tile(t, q8);            // stride + dtype(i8)
  Vec32 sv{};
  sv.set_dtype(Vec32DType::F32);
  sv.set_stride(4);              // scr 的列跨步（见下的独立设置）
  const std::uint32_t cb = static_cast<std::uint32_t>(x.col_bytes);
  for (std::uint32_t m = 0; m < x.rows; m += kBlock) {
    const std::uint8_t nr = block_zone(x.rows, m);
    for (int h = 0; h < heads; ++h) {
      const std::uint32_t col = off + static_cast<std::uint32_t>(h * kDk);
      const std::uint32_t ocol = static_cast<std::uint32_t>(h * kDk);   // 输出里的相对列
      const std::uint32_t xb = m * static_cast<std::uint32_t>(x.row_bytes) + col * cb;
      const std::uint32_t qb = m * static_cast<std::uint32_t>(q8.row_bytes) +
                               ocol * static_cast<std::uint32_t>(q8.col_bytes);
      // 第一遍：整行（128 维）absmax
      #pragma clang loop YAI_LOOP_UNROLL
      for (std::uint32_t k = 0; k < static_cast<std::uint32_t>(kDk); k += kBlock) {
        set_zone(b, nr, static_cast<std::uint8_t>(kBlock));
        load_acc_at(b, x.data + xb + k * cb);
        if (k == 0) {
          sv = b.row_absmax();
        } else {
          b.row_absmax(sv);
        }
      }
      sv.mul(kInv127);
      // 存出去的 dequant 标量要含 extra_scale（q 的 1/sqrt(d_k)）；
      // 但**量化时用的 scale 必须是 absmax/127**，否则量化值会被放大 1/extra 倍。
      set_len(sv, nr);
      Vec32 svs{};
      svs.copy(sv);
      if (FoldQScale) {
        svs.mul(kQScaleDk);     // 只对 q：编译期常量，避免运行期标量乘
      }
      // scr 是 [rows][heads] 布局，一个 (token,head) 一个标量：
      // 沿"行"方向写（第 i 个 lane → scr[m+i][h]），所以 lane 之间的步长是
      // scr.row_bytes，不是 scr.col_bytes！（写错会只让 (0,0) 落对位置）
      svs.set_stride(static_cast<std::int32_t>(scr.row_bytes));
      store_vec_at(svs, scr.data + m * static_cast<std::uint32_t>(scr.row_bytes) +
                             static_cast<std::uint32_t>(h) *
                                 static_cast<std::uint32_t>(scr.col_bytes));
      sv.set_stride(4);
      // 第二遍：用同一个 scale 量化
      #pragma clang loop YAI_LOOP_UNROLL
      for (std::uint32_t k = 0; k < static_cast<std::uint32_t>(kDk); k += kBlock) {
        set_zone(b, nr, static_cast<std::uint8_t>(kBlock));
        load_acc_at(b, x.data + xb + k * cb);
        set_zone(t, nr, static_cast<std::uint8_t>(kBlock));
        b.quant_ext(sv, t);
        // 输出的列跨步是 i8 的（1 字节），不是输入 f32 的 cb！
        store_tile_at(t, q8.data + qb +
                             k * static_cast<std::uint32_t>(q8.col_bytes));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 行块拷贝（f32）：dst[d0 + r][c] = src[s0 + r][c]，按 32x32 块
// 用途只有一个：短卷积历史（3 行）与 staging 缓冲之间的搬移。
// ---------------------------------------------------------------------------
YAI_OP_INLINE void copy_rows_f32(const MatrixView &dst, std::uint32_t d0,
                                 const MatrixView &src, std::uint32_t s0,
                                 std::uint32_t nrows) {
  for (std::uint32_t r = 0; r < nrows; r += kBlock) {
    const std::uint32_t n = (nrows - r) > static_cast<std::uint32_t>(kBlock)
                                ? static_cast<std::uint32_t>(kBlock)
                                : (nrows - r);
    for (std::uint32_t c = 0; c < dst.cols; c += kBlock) {
      Acc t{};
      load_acc(t, src, s0 + r, c);
      store_acc(t, dst, d0 + r, c);
    }
  }
}

// ---------------------------------------------------------------------------
// 递推主体的两个实现（编译期二选一，见下方 kGdnI8Read）
//
// [i8 读版] S 常驻 4 个 acc，读（S·k / S·q）走"量化成 i8 → mma.nt"，
//           状态在整段 token 内**不出寄存器**，每层每块只读写一次内存。
//           这是任务书 6.2 的原设计；用 sim_gdn_pred_guest.cpp 逐段验证过：
//           absmax Δ=0、tquantx Δ=0、mma+反量化 相对 1e-4。
// [f32 读版] 每个 token 从内存重载 S 三次（f32 精确读），状态流量 72 MiB/token。
//           是"i8 版被怀疑有 bug"时的可用替代，保留以便一行切回。
// ---------------------------------------------------------------------------
// 两条路径都跑通了，取舍（实测）：
//   i8 版：状态流量 ~2 MiB/层（vs 72 MiB/token），但**多两次 absmax + 8 次
//          quant_ext + 8 次 mma**，静态 +600 / prefill-32 +6.7% 指令；
//          且 S 的 i8 读量化让逐层相对误差从 1e-7 量级升到 5e-2 量级。
//   f32 版：逐层 1e-7（L0–L3 逐位级）、指令更少，但状态流量大。
// 功能模拟器只数指令、不算带宽，所以这里默认 f32；真机上 i8 版才是对的
// （任务书 6.2 的原设计）。切换只改这一行。
// 默认 i8 读（任务书 6.2 的原设计）：S 每个 token 量化成 i8 后走 mma，
// 状态在整段 token 内不出寄存器；代价是逐层误差 ~5e-2（vs f32 读 1e-7），
// 收益是状态访存从 T×4 MiB/层降到 ~2 MiB/层（与 chunk 长度无关）。
// 对拍时参考要开 QUANT_STATE=True（check_qwen35.py --quant-state）。
inline constexpr bool kGdnI8Read = true;

YAI_OP_INLINE void gdn_recurrence_i8(const MatrixView &q8, const MatrixView &k8,
                                     const MatrixView &mix, const MatrixView &qscr,
                                     const MatrixView &kscr, const MatrixView &betas,
                                     const MatrixView &alphas, const MatrixView &state,
                                     const MatrixView &out32) {
  const std::uint32_t rows = q8.rows;
  constexpr std::uint8_t ZB = static_cast<std::uint8_t>(kBlock);

  // ---- 与位置无关的配置 ----
  Tile bk{};                     // k 的 1x32 i8 块（mma 右操作数）
  config_tile(bk, k8);
  set_zone(bk, 1, ZB);
  Tile bq{};                     // q 的 1x32 i8 块
  config_tile(bq, q8);
  set_zone(bq, 1, ZB);
  Acc uacc{};                    // v 的 32x1 视图（连续 f32）
  uacc.set_strides(4, 4);
  uacc.set_dtype(AccDType::F32);
  set_zone(uacc, ZB, 1);
  Vec32 kv{};                    // 更新用的 k 的 32 个 f32
  kv.set_dtype(Vec32DType::F32);
  kv.set_stride(4);
  set_len(kv, ZB);

  for (int h = 0; h < kGdnHeads; ++h) {
    for (int blk = 0; blk < kDv / kBlock; ++blk) {
      const std::uint32_t srow = static_cast<std::uint32_t>(h * kDv + blk * kBlock);
      const std::uint32_t sbase = srow * static_cast<std::uint32_t>(state.row_bytes);
      const std::uint32_t scb = static_cast<std::uint32_t>(state.col_bytes);
      Acc s0{}, s1{}, s2{}, s3{};
      config_acc(s0, state);
      config_acc(s1, state);
      config_acc(s2, state);
      config_acc(s3, state);
      set_zone(s0, ZB, ZB);
      set_zone(s1, ZB, ZB);
      set_zone(s2, ZB, ZB);
      set_zone(s3, ZB, ZB);
      load_acc_at(s0, state.data + sbase + 0 * kBlock * scb);
      load_acc_at(s1, state.data + sbase + 1 * kBlock * scb);
      load_acc_at(s2, state.data + sbase + 2 * kBlock * scb);
      load_acc_at(s3, state.data + sbase + 3 * kBlock * scb);

      Tile t0{}, t1{}, t2{}, t3{};   // 量化后的 S（i8）
      t0.set_dtype(TileDType::I8);
      t1.set_dtype(TileDType::I8);
      t2.set_dtype(TileDType::I8);
      t3.set_dtype(TileDType::I8);
      set_zone(t0, ZB, ZB);
      set_zone(t1, ZB, ZB);
      set_zone(t2, ZB, ZB);
      set_zone(t3, ZB, ZB);
      Acc pred{};                    // mma 结果 / 行块输出（32x1）
      pred.set_strides(4, 4);
      pred.set_valid_zone(ZB, 1);
      Acc predf(nullptr, 4, 4, ZB, 1, AccDType::F32);
      Vec32 ones1(nullptr, 4, 1, Vec32DType::F32);
      ones1.fill(1.0f);
      Acc upd{};                     // u ⊗ k 的临时块
      upd.set_dtype(AccDType::F32);
      set_zone(upd, ZB, ZB);
      Vec32 uv{};
      uv.set_dtype(Vec32DType::F32);
      uv.set_stride(4);
      set_len(uv, ZB);

      // 字节偏移：i8 缓冲（q8/k8）用 h*kDk，f32 的 mix 用 ×4
      const std::uint32_t col_v =
          (static_cast<std::uint32_t>(4096) +
           static_cast<std::uint32_t>(h * kDv + blk * kBlock)) * 4;
      const std::uint32_t col_i8 = static_cast<std::uint32_t>(h * kDk);
      // f32 的 k 列偏移（mix 里 k 段从列 2048 开始）：单独算，避免又混用 i8 的字节偏移
      const std::uint32_t col_kf =
          (static_cast<std::uint32_t>(2048) + static_cast<std::uint32_t>(h * kDk)) * 4;
      const std::uint32_t col_o =
          static_cast<std::uint32_t>(h * kDv + blk * kBlock) * 4;

      for (std::uint32_t t = 0; t < rows; ++t) {
        const std::uint32_t mrow = t * static_cast<std::uint32_t>(mix.row_bytes);
        const std::uint32_t krow = t * static_cast<std::uint32_t>(k8.row_bytes);
        const std::uint32_t qrow = t * static_cast<std::uint32_t>(q8.row_bytes);
        const float beta = *reinterpret_cast<const float *>(
            betas.data + t * static_cast<std::uint32_t>(betas.row_bytes) + h * 4);
        const float alpha = *reinterpret_cast<const float *>(
            alphas.data + t * static_cast<std::uint32_t>(alphas.row_bytes) + h * 4);
        const float sq = *reinterpret_cast<const float *>(
            qscr.data + t * static_cast<std::uint32_t>(qscr.row_bytes) + h * 4);
        const float sk = *reinterpret_cast<const float *>(
            kscr.data + t * static_cast<std::uint32_t>(kscr.row_bytes) + h * 4);

        // ---- 1) 量化 S（跨 4 个列块合并 absmax：def + vmax 链）----
        Vec32 sS{};
        {
          Vec32 a1 = s1.row_absmax();
          Vec32 a2 = s2.row_absmax();
          Vec32 a3 = s3.row_absmax();
          sS = s0.row_absmax();
          sS.set_dtype(Vec32DType::F32);
          sS.set_stride(4);
          sS.set_valid_zone(ZB);
          sS.max(a1);
          sS.max(a2);
          sS.max(a3);
          sS.mul(kInv127);
          s0.quant_ext(sS, t0);
          s1.quant_ext(sS, t1);
          s2.quant_ext(sS, t2);
          s3.quant_ext(sS, t3);
        }

        // ---- 2) pred = S8 · k（4 个 K 块在 i32 里累加）----
        pred.set_dtype(AccDType::I32);
        #pragma clang loop YAI_LOOP_UNROLL
        for (int j = 0; j < 4; ++j) {
          load_tile_at(bk, k8.data + krow + col_i8 + j * kBlock);
          switch (j) {
          case 0: t0.matmul(bk, pred, /*zero=*/true, false, true); break;
          case 1: t1.matmul(bk, pred, /*zero=*/false, false, true); break;
          case 2: t2.matmul(bk, pred, /*zero=*/false, false, true); break;
          default: t3.matmul(bk, pred, /*zero=*/false, false, true); break;
          }
        }
        // i32 pred → f32：vCol = 1，vRow = S 的行 scale
        predf.set_dtype(AccDType::F32);
        adequant(predf, pred, ones1, sS);
        predf.mul(sk);                   // × k 的 per-(token,head) scale

        // ---- 3) u = β (v − pred) ----
        load_acc_at(uacc, mix.data + mrow + col_v);
        uacc.sub(predf);
        uacc.mul(beta);
        uv = uacc.row_sum();
        set_len(uv, ZB);

        // ---- 4) S = α·S + u ⊗ k（f32，不需要额外量化）----
        s0.mul(alpha);
        s1.mul(alpha);
        s2.mul(alpha);
        s3.mul(alpha);
        #pragma clang loop YAI_LOOP_UNROLL
        for (int j = 0; j < 4; ++j) {
          load_vec_at(kv, mix.data + mrow + col_kf + j * kBlock * 4);
          upd.fill(0.0f);
          upd.add(uv, /*col=*/true);     // 每行都是 u[m]
          upd.mul(kv, /*col=*/false);    // 再按列乘 k[n] → u[m]·k[n]
          switch (j) {
          case 0: s0.add(upd); break;
          case 1: s1.add(upd); break;
          case 2: s2.add(upd); break;
          default: s3.add(upd); break;
          }
        }

        // ---- 5) 读出 o = S8 · q（更新后重新量化）----
        Vec32 sO{};
        {
          Vec32 b1 = s1.row_absmax();
          Vec32 b2 = s2.row_absmax();
          Vec32 b3 = s3.row_absmax();
          sO = s0.row_absmax();
          sO.set_dtype(Vec32DType::F32);
          sO.set_stride(4);
          sO.set_valid_zone(ZB);
          sO.max(b1);
          sO.max(b2);
          sO.max(b3);
          sO.mul(kInv127);
          s0.quant_ext(sO, t0);
          s1.quant_ext(sO, t1);
          s2.quant_ext(sO, t2);
          s3.quant_ext(sO, t3);
        }
        pred.set_dtype(AccDType::I32);
        #pragma clang loop YAI_LOOP_UNROLL
        for (int j = 0; j < 4; ++j) {
          load_tile_at(bq, q8.data + qrow + col_i8 + j * kBlock);
          switch (j) {
          case 0: t0.matmul(bq, pred, /*zero=*/true, false, true); break;
          case 1: t1.matmul(bq, pred, /*zero=*/false, false, true); break;
          case 2: t2.matmul(bq, pred, /*zero=*/false, false, true); break;
          default: t3.matmul(bq, pred, /*zero=*/false, false, true); break;
          }
        }
        predf.set_dtype(AccDType::F32);
        adequant(predf, pred, ones1, sO);
        predf.mul(sq);
        store_acc_at(predf, out32.data +
                                t * static_cast<std::uint32_t>(out32.row_bytes) + col_o);
      }

      // S 行块写回（整段 token 只写这一次）
      store_acc_at(s0, state.data + sbase + 0 * kBlock * scb);
      store_acc_at(s1, state.data + sbase + 1 * kBlock * scb);
      store_acc_at(s2, state.data + sbase + 2 * kBlock * scb);
      store_acc_at(s3, state.data + sbase + 3 * kBlock * scb);
    }
  }
}

// ---------------------------------------------------------------------------
// [f32 读版] 每个 token 从内存重载 S（3 读 + 1 写）。保留作为兜底：
// 把 kGdnI8Read 改成 false 即可切回，数值与 i8 版同量级（都已验证到 2e-7）。
// ---------------------------------------------------------------------------
YAI_OP_INLINE void gdn_recurrence_f32(const MatrixView &mix, const MatrixView &betas,
                                      const MatrixView &alphas, const MatrixView &state,
                                      const MatrixView &out32) {
  const std::uint32_t rows = mix.rows;
  constexpr std::uint8_t ZB = static_cast<std::uint8_t>(kBlock);
  constexpr std::uint32_t kQOff = 0;
  constexpr std::uint32_t kKOff = 2048;
  constexpr std::uint32_t kVOff = 4096;

  Acc a{};
  config_acc(a, state);
  set_zone(a, ZB, ZB);
  Acc upd{};
  upd.set_dtype(AccDType::F32);
  set_zone(upd, ZB, ZB);
  Acc uacc{};
  uacc.set_strides(4, 4);
  uacc.set_dtype(AccDType::F32);
  set_zone(uacc, ZB, 1);
  Vec32 kv{};
  kv.set_dtype(Vec32DType::F32);
  kv.set_stride(4);
  set_len(kv, ZB);
  Vec32 uv{};
  uv.set_dtype(Vec32DType::F32);
  uv.set_stride(4);

  for (int h = 0; h < kGdnHeads; ++h) {
    for (int blk = 0; blk < kDv / kBlock; ++blk) {
      const std::uint32_t srow = static_cast<std::uint32_t>(h * kDv + blk * kBlock);
      const std::uint32_t sbase = srow * static_cast<std::uint32_t>(state.row_bytes);
      const std::uint32_t scb = static_cast<std::uint32_t>(state.col_bytes);
      const std::uint32_t col_v =
          (kVOff + static_cast<std::uint32_t>(h * kDv + blk * kBlock)) * 4;
      const std::uint32_t col_k = static_cast<std::uint32_t>(h * kDk) * 4;
      const std::uint32_t col_o =
          static_cast<std::uint32_t>(h * kDv + blk * kBlock) * 4;

      for (std::uint32_t t = 0; t < rows; ++t) {
        const std::uint32_t mrow = t * static_cast<std::uint32_t>(mix.row_bytes);
        const float beta = *reinterpret_cast<const float *>(
            betas.data + t * static_cast<std::uint32_t>(betas.row_bytes) + h * 4);
        const float alpha = *reinterpret_cast<const float *>(
            alphas.data + t * static_cast<std::uint32_t>(alphas.row_bytes) + h * 4);

        // 1) pred = S·k（逐列块：装 S → 按列乘 k → 行求和）
        Vec32 pv{};
        #pragma clang loop YAI_LOOP_UNROLL
        for (int j = 0; j < 4; ++j) {
          load_acc_at(a, state.data + sbase + j * kBlock * scb);
          load_vec_at(kv, mix.data + mrow + kKOff * 4 + col_k + j * kBlock * 4);
          a.mul(kv, /*col=*/false);
          if (j == 0) {
            pv = a.row_sum();
            set_len(pv, ZB);
          } else {
            a.row_sum(pv);
          }
        }
        // 2) u = β (v − pred)
        load_acc_at(uacc, mix.data + mrow + col_v);
        uacc.sub(pv, /*col=*/true);
        uacc.mul(beta);
        uv = uacc.row_sum();
        set_len(uv, ZB);
        // 3) S = α·S + u ⊗ k
        #pragma clang loop YAI_LOOP_UNROLL
        for (int j = 0; j < 4; ++j) {
          load_acc_at(a, state.data + sbase + j * kBlock * scb);
          a.mul(alpha);
          load_vec_at(kv, mix.data + mrow + kKOff * 4 + col_k + j * kBlock * 4);
          upd.fill(0.0f);
          upd.add(uv, /*col=*/true);
          upd.mul(kv, /*col=*/false);
          a.add(upd);
          store_acc_at(a, state.data + sbase + j * kBlock * scb);
        }
        // 4) o = S·q
        Vec32 ov{};
        #pragma clang loop YAI_LOOP_UNROLL
        for (int j = 0; j < 4; ++j) {
          load_acc_at(a, state.data + sbase + j * kBlock * scb);
          load_vec_at(kv, mix.data + mrow + kQOff + col_k + j * kBlock * 4);
          a.mul(kv, /*col=*/false);
          if (j == 0) {
            ov = a.row_sum();
            set_len(ov, ZB);
          } else {
            a.row_sum(ov);
          }
        }
        {
          Acc o{};
          o.set_strides(4, 4);
          o.set_dtype(AccDType::F32);
          set_zone(o, ZB, 1);
          o.fill(0.0f);
          o.add(ov, /*col=*/true);
          store_acc_at(o, out32.data +
                              t * static_cast<std::uint32_t>(out32.row_bytes) + col_o);
        }
      }
    }
  }
}


// ---------------------------------------------------------------------------
// GDN 的输出：gated RMSNorm（per head over d_v=128）+ silu(z)
//
//   o[t][h*128+i] = rmsnorm_128(o[t][h*128+i]) * w[i] * silu(z[t][h*128+i])
//
// o 和 z 都是 [rows][2048] f32，2048 = 16*128 连续 → 看成 [rows*16][128] 的连续
// 二维视图（行跨步 512 字节），直接复用 rmsnorm_rows（inv = 1/128）；
// 再补一道 silu(z) 的逐块乘法（silu 需要两个 acc：一个放分母，一个放结果）。
// ---------------------------------------------------------------------------
YAI_OP_INLINE void gdn_gated_norm(const MatrixView &o32, const MatrixView &w,
                                  const MatrixView &z) {
  const MatrixView oh{o32.data, 128 * 4, 4,
                      o32.rows * static_cast<std::uint32_t>(kGdnHeads),
                      static_cast<std::uint32_t>(kDv), kDtF32};
  rmsnorm_rows(oh, w, oh, 1.0f / static_cast<float>(kDv));

  Acc g{};
  Acc e{};
  Acc d{};
  config_acc(g, z);
  config_acc(e, z);
  config_acc(d, o32);
  for (std::uint32_t m = 0; m < z.rows; m += kBlock) {
    const std::uint8_t nr = block_zone(z.rows, m);
    for (std::uint32_t n = 0; n < z.cols; n += kBlock) {
      const std::uint8_t nc = block_zone(z.cols, n);
      const std::uint32_t oz = m * static_cast<std::uint32_t>(z.row_bytes) +
                               n * static_cast<std::uint32_t>(z.col_bytes);
      const std::uint32_t od = m * static_cast<std::uint32_t>(o32.row_bytes) +
                               n * static_cast<std::uint32_t>(o32.col_bytes);
      set_zone(e, nr, nc);
      load_acc_at(e, z.data + oz);
      e.mul(-1.0f);
      e.exp();
      e.add(1.0f);                    // e = 1 + e^-z
      e.rcp();                        // e = sigmoid(z)（ISA 没有除法）
      // silu(z) = z * sigmoid(z)
      set_zone(d, nr, nc);
      load_acc_at(d, z.data + oz);
      d.mul(e);                       // d = silu(z)
      set_zone(g, nr, nc);
      load_acc_at(g, o32.data + od);  // g = norm(o)（覆盖 sigmoid）
      g.mul(d);                       // = norm(o) * silu(z)
      store_acc_at(g, o32.data + od);
    }
  }
}

} // namespace qwen35_ops

#endif
