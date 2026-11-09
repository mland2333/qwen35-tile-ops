// ============================================================================
// gemm_nt：int8 GEMM（右操作数逻辑转置）+ 双向 scale 的 epilogue
//
//   out[m][n] = ( Σ_k A[m][k] * B[n][k] ) * a_scale[m] * b_scale[n]
//
// 结构：m 块（A 的行）→ n 组（每组 G 个输出列块）→ k 块（INT32 累加）
//
// 性能要点（均为实测）:
//   1) 描述符外提：stride/dtype/zone 与块位置无关，只在生命周期起点写一次；
//      内层每次载入只写 base_addr（见 io.h 文件头）。
//   2) n 方向寄存器分块：不同输出列块只共享 A，一次 A 载入喂 G 个 mma。
//      G = 8：1 A tile + 8 B tile = 9 个 XTT（上限 16），8 个 acc（上限 16）。
//      注意 G 不是越大越好：更大的 G 让组内临时值更多、RA 更难排，实测有最优值。
//   3) 组内的 tile/acc 都在**组内创建**（活区间短），这样 RA 能把寄存器复用起来。
//
// 为什么 K 要累加完才转换：mma 是 i8×i8→i32，K 方向全程在 i32 里累加，只在最后做一次
// acvt + 两次 scale —— 与 host 参考逐位一致（中间转 f32 会多一次舍入）。
// ============================================================================
#ifndef YAI_QWEN35_OPS_GEMM_H
#define YAI_QWEN35_OPS_GEMM_H

#include "tile.h"
#include "config.h"
#include "io.h"
#include "matrix.h"

namespace qwen35_ops {

// 一组里放几个输出列块（XTT/XTA 各 16 个的硬限制：G + 1 <= 16）。
inline constexpr std::uint32_t kGemmNGroup = 8;

// 单个输出列块的完整 K 累加 + epilogue：尾块（不足一组）与 K 不是 32 倍数时
// 走这条兜底路径，所以它自己负责全部 zone/stride 配置。
YAI_ALWAYS_INLINE void gemm_one_n_block(const MatrixView &a, const MatrixView &b,
                                        const MatrixView &b_scale,
                                        const Vec32 &as, const MatrixView &out,
                                        Tile &ta, Tile &tb, std::uint32_t m,
                                        std::uint32_t n) {
  const std::uint8_t rows = block_zone(a.rows, m);
  const std::uint8_t cols = block_zone(b.rows, n);
  Acc o{};
  init_i32_acc(o, rows, cols);
  set_zone(ta, rows, static_cast<std::uint8_t>(kBlock));
  set_zone(tb, cols, static_cast<std::uint8_t>(kBlock));
  load_tile_at(ta, a.data + m * static_cast<std::uint32_t>(a.row_bytes));
  load_tile_at(tb, b.data + n * static_cast<std::uint32_t>(b.row_bytes));
  ta.matmul(tb, o, /*zero=*/true, /*lt=*/false, /*rt=*/true);
  for (std::uint32_t k = kBlock; k < a.cols; k += kBlock) {
    load_tile_at(ta, a.data + m * static_cast<std::uint32_t>(a.row_bytes) +
                         k * static_cast<std::uint32_t>(a.col_bytes));
    load_tile_at(tb, b.data + n * static_cast<std::uint32_t>(b.row_bytes) +
                         k * static_cast<std::uint32_t>(b.col_bytes));
    ta.matmul(tb, o, /*zero=*/false, /*lt=*/false, /*rt=*/true);
  }
  Vec32 ws{};
  load_vec_n(ws, b_scale, n, block_zone(b.rows, n));
  Acc of{};
  of.set_strides(out.row_bytes, out.col_bytes);
  of.set_dtype(AccDType::F32);
  set_zone(of, rows, cols);
  of.set_dtype(AccDType::F32);
  adequant(of, o, ws, as);          // of = f32(o) * ws[j] * as[i]
  store_acc_at(of, out.data + m * static_cast<std::uint32_t>(out.row_bytes) +
                       n * static_cast<std::uint32_t>(out.col_bytes));
}

// 快路径的输出块 epilogue。
YAI_ALWAYS_INLINE void gemm_epilogue_block(const Vec32 &as, Vec32 &ws,
                                           Acc &o, const MatrixView &b_scale,
                                           const MatrixView &out,
                                           std::uint32_t m, std::uint32_t n,
                                           std::uint8_t rows) {
  ws.load();                            // 描述符自增（len*stride）指向下一个 n 块
  Acc of(out.data, out.row_bytes, out.col_bytes, rows, static_cast<std::uint32_t>(kBlock),
         AccDType::F32);
  of.set_dtype(AccDType::F32);
  adequant(of, o, ws, as);
  store_acc_at(of, out.data + m * static_cast<std::uint32_t>(out.row_bytes) +
                       n * static_cast<std::uint32_t>(out.col_bytes));
}

YAI_OP_INLINE void gemm_nt(const MatrixView &a, const MatrixView &b,
                               const MatrixView &b_scale,
                               const MatrixView &a_scale,
                               const MatrixView &out) {
  constexpr std::uint32_t G = kGemmNGroup;
  constexpr std::uint8_t ZB = static_cast<std::uint8_t>(kBlock);

  Tile ta{};
  config_tile(ta, a);
  // 描述符自增：按列自增的步长是 cols*col_stride（= 块的字节宽度），
  // 正好是下一个 k 块；于是内层每次只要 tload，不必再写 base_addr / 算地址。
  ta.set_flags(2);
  // A tile 的 zone 只依赖 m 的行块，col 恒为 32 → 在 m 循环里设一次
  // ws（b_scale 向量）：dtype/len/stride 全程不变，一次性配好
  Vec32 ws{};
  config_vec(ws, b_scale);
  set_len(ws, ZB);
  // 向量描述符自增：每次 vload 后 base += len*stride（= 下一个 n 块的 32 个 scale）
  ws.set_flags(1);
  // B 的 G 个 tile 在函数级创建：zone 全程不变，组内只重写一次 base_addr，
  // 之后靠描述符自增前进（flags=2 = 按列自增一整个块的宽度）。
  Tile t0{};
  config_tile(t0, b);
  set_zone(t0, ZB, ZB);
  t0.set_flags(2);
  Tile t1{};
  config_tile(t1, b);
  set_zone(t1, ZB, ZB);
  t1.set_flags(2);
  Tile t2{};
  config_tile(t2, b);
  set_zone(t2, ZB, ZB);
  t2.set_flags(2);
  Tile t3{};
  config_tile(t3, b);
  set_zone(t3, ZB, ZB);
  t3.set_flags(2);
  Tile t4{};
  config_tile(t4, b);
  set_zone(t4, ZB, ZB);
  t4.set_flags(2);
  Tile t5{};
  config_tile(t5, b);
  set_zone(t5, ZB, ZB);
  t5.set_flags(2);
  Tile t6{};
  config_tile(t6, b);
  set_zone(t6, ZB, ZB);
  t6.set_flags(2);
  Tile t7{};
  config_tile(t7, b);
  set_zone(t7, ZB, ZB);
  t7.set_flags(2);

  const std::uint32_t ab = static_cast<std::uint32_t>(a.row_bytes);
  const std::uint32_t ac = static_cast<std::uint32_t>(a.col_bytes);
  const std::uint32_t bb = static_cast<std::uint32_t>(b.row_bytes);
  const std::uint32_t bc = static_cast<std::uint32_t>(b.col_bytes);

  for (std::uint32_t m = 0; m < a.rows; m += kBlock) {
    const std::uint8_t rows = block_zone(a.rows, m);
    Vec32 as{};
    load_vec_n(as, a_scale, m, rows);
    set_zone(ta, rows, ZB);
    ws.set_data(b_scale.data);          // n 从 0 开始，之后靠自增
    std::uint32_t n = 0;
    // ---- 快路径：K 是 32 的倍数，且这一组 G 个输出块都满 32 列 ----
    // （zone 因此在整个 k 循环里不变，可以提到循环外）
    if (a.cols % kBlock == 0) {
      const std::uint32_t am = m * ab;
      for (; n + G * kBlock <= b.rows; n += G * kBlock) {
        // 组内临时值：活区间只到本组结束
        Acc o0{};
        init_i32_acc(o0, rows, ZB);
        Acc o1{};
        init_i32_acc(o1, rows, ZB);
        Acc o2{};
        init_i32_acc(o2, rows, ZB);
        Acc o3{};
        init_i32_acc(o3, rows, ZB);
        Acc o4{};
        init_i32_acc(o4, rows, ZB);
        Acc o5{};
        init_i32_acc(o5, rows, ZB);
        Acc o6{};
        init_i32_acc(o6, rows, ZB);
        Acc o7{};
        init_i32_acc(o7, rows, ZB);
        o0.set_strides(out.row_bytes, out.col_bytes);   // store 用，与位置无关
        o1.set_strides(out.row_bytes, out.col_bytes);   // store 用，与位置无关
        o2.set_strides(out.row_bytes, out.col_bytes);   // store 用，与位置无关
        o3.set_strides(out.row_bytes, out.col_bytes);   // store 用，与位置无关
        o4.set_strides(out.row_bytes, out.col_bytes);   // store 用，与位置无关
        o5.set_strides(out.row_bytes, out.col_bytes);   // store 用，与位置无关
        o6.set_strides(out.row_bytes, out.col_bytes);   // store 用，与位置无关
        o7.set_strides(out.row_bytes, out.col_bytes);   // store 用，与位置无关
        const std::uint32_t bn = n * bb;
        ta.set_data(a.data + am);            // 本组起点写一次
        ta.load();
        t0.set_data(b.data + bn);
        t0.load();
        t1.set_data(b.data + bn + 1 * kBlock * bb);
        t1.load();
        t2.set_data(b.data + bn + 2 * kBlock * bb);
        t2.load();
        t3.set_data(b.data + bn + 3 * kBlock * bb);
        t3.load();
        t4.set_data(b.data + bn + 4 * kBlock * bb);
        t4.load();
        t5.set_data(b.data + bn + 5 * kBlock * bb);
        t5.load();
        t6.set_data(b.data + bn + 6 * kBlock * bb);
        t6.load();
        t7.set_data(b.data + bn + 7 * kBlock * bb);
        t7.load();
        ta.matmul(t0, o0, /*zero=*/true, false, true);
        ta.matmul(t1, o1, /*zero=*/true, false, true);
        ta.matmul(t2, o2, /*zero=*/true, false, true);
        ta.matmul(t3, o3, /*zero=*/true, false, true);
        ta.matmul(t4, o4, /*zero=*/true, false, true);
        ta.matmul(t5, o5, /*zero=*/true, false, true);
        ta.matmul(t6, o6, /*zero=*/true, false, true);
        ta.matmul(t7, o7, /*zero=*/true, false, true);
        // 其余 K 块：描述符自增已经指到下一块，这里只需要 tload
        for (std::uint32_t k = kBlock; k < a.cols; k += kBlock) {
          ta.load();
          t0.load();
          t1.load();
          t2.load();
          t3.load();
          t4.load();
          t5.load();
          t6.load();
          t7.load();
          ta.matmul(t0, o0, /*zero=*/false, false, true);
          ta.matmul(t1, o1, /*zero=*/false, false, true);
          ta.matmul(t2, o2, /*zero=*/false, false, true);
          ta.matmul(t3, o3, /*zero=*/false, false, true);
          ta.matmul(t4, o4, /*zero=*/false, false, true);
          ta.matmul(t5, o5, /*zero=*/false, false, true);
          ta.matmul(t6, o6, /*zero=*/false, false, true);
          ta.matmul(t7, o7, /*zero=*/false, false, true);
        }
        gemm_epilogue_block(as, ws, o0, b_scale, out, m, n, rows);
        gemm_epilogue_block(as, ws, o1, b_scale, out, m, n + 1 * kBlock, rows);
        gemm_epilogue_block(as, ws, o2, b_scale, out, m, n + 2 * kBlock, rows);
        gemm_epilogue_block(as, ws, o3, b_scale, out, m, n + 3 * kBlock, rows);
        gemm_epilogue_block(as, ws, o4, b_scale, out, m, n + 4 * kBlock, rows);
        gemm_epilogue_block(as, ws, o5, b_scale, out, m, n + 5 * kBlock, rows);
        gemm_epilogue_block(as, ws, o6, b_scale, out, m, n + 6 * kBlock, rows);
        gemm_epilogue_block(as, ws, o7, b_scale, out, m, n + 7 * kBlock, rows);
      }
    }

    // ---- 兜底路径：不足一组的尾块、以及 K 不是 32 倍数的情形 ----
    Tile tb{};
    config_tile(tb, b);
    for (; n < b.rows; n += kBlock)
      gemm_one_n_block(a, b, b_scale, as, out, ta, tb, m, n);
  }
}


// ============================================================================
// gemm_nt_tiled：权重**预分块**（tile-major）版本的同一个 GEMM
//
//   weights[n][k] 在 blob 里按 [N/32][K/32][32][32] 排布 → 每个 32x32 i8 tile
//   连续 1 KiB。地址 = ((n/32)*(K/32) + (k/32)) * 1024。
//
// 为什么要这样（行主序的问题）：行主序时一个 tile 是"32 段 32 B、段间相距 K 字节"，
// 在 64 B burst 的 DDR 上每次只用半个 burst；预分块后一个 tile 是 16 个满 burst，
// 而且 32 行的数据落在连续的 1 KiB 里，DRAM 页局部性也从"32 个远地址"变成"1 个"。
//
// 关键实现点：k 方向相邻 tile 的距离恰好 = 32 * row_stride（= 32*32 = 1024 B），
// 所以把 tile 描述符设成 strides=(32,1) + **按行自增**（flags=1）后，内层的
// tload 会自动走到下一个 k 块 —— 和行主序版本一样，内层不需要重算地址。
// 数值上与本文件的 gemm_nt **逐位相同**：最小用例（A 32x128、B 256x128 tiled、
// N=256 走 G=8 快路径）实测 0/8192 个元素不同（check_gemm_tiled_sim.py）。
// 注意：N 必须是 32 的倍数（一个 tile 装 32 个输出通道）；GDN 的 b/a 投影是
// N=16，凑不满一个 tile，那两个仍然走行主序的 gemm_nt。
// ============================================================================
YAI_OP_INLINE void gemm_nt_tiled(const MatrixView &a, const MatrixView &wt,
                                 const MatrixView &b_scale,
                                 const MatrixView &a_scale,
                                 const MatrixView &out) {
  constexpr std::uint32_t G = kGemmNGroup;
  constexpr std::uint8_t ZB = static_cast<std::uint8_t>(kBlock);
  const std::uint32_t nblk = wt.cols / kBlock;          // K 方向块数
  const std::uint32_t nbstride = nblk * kBlock * kBlock; // 相邻 n 块的距离（= K*32 B）

  Tile ta{};
  config_tile(ta, a);
  ta.set_flags(2);
  Vec32 ws{};
  config_vec(ws, b_scale);
  set_len(ws, ZB);
  ws.set_flags(1);

  // B 的 8 个 tile：strides=(32,1)（连续 1 KiB），按行自增 1024 B
  Tile t0{}; config_tile(t0, wt); t0.set_strides(kBlock, 1); set_zone(t0, ZB, ZB); t0.set_flags(1);
  Tile t1{}; config_tile(t1, wt); t1.set_strides(kBlock, 1); set_zone(t1, ZB, ZB); t1.set_flags(1);
  Tile t2{}; config_tile(t2, wt); t2.set_strides(kBlock, 1); set_zone(t2, ZB, ZB); t2.set_flags(1);
  Tile t3{}; config_tile(t3, wt); t3.set_strides(kBlock, 1); set_zone(t3, ZB, ZB); t3.set_flags(1);
  Tile t4{}; config_tile(t4, wt); t4.set_strides(kBlock, 1); set_zone(t4, ZB, ZB); t4.set_flags(1);
  Tile t5{}; config_tile(t5, wt); t5.set_strides(kBlock, 1); set_zone(t5, ZB, ZB); t5.set_flags(1);
  Tile t6{}; config_tile(t6, wt); t6.set_strides(kBlock, 1); set_zone(t6, ZB, ZB); t6.set_flags(1);
  Tile t7{}; config_tile(t7, wt); t7.set_strides(kBlock, 1); set_zone(t7, ZB, ZB); t7.set_flags(1);

  const std::uint32_t ab = static_cast<std::uint32_t>(a.row_bytes);
  for (std::uint32_t m = 0; m < a.rows; m += kBlock) {
    const std::uint8_t rows = block_zone(a.rows, m);
    Vec32 as{};
    load_vec_n(as, a_scale, m, rows);
    set_zone(ta, rows, ZB);
    ws.set_data(b_scale.data);
    std::uint32_t n = 0;
    if (a.cols % kBlock == 0) {
      const std::uint32_t am = m * ab;
      for (; n + G * kBlock <= wt.rows; n += G * kBlock) {
        Acc o0{}; init_i32_acc(o0, rows, ZB);
        Acc o1{}; init_i32_acc(o1, rows, ZB);
        Acc o2{}; init_i32_acc(o2, rows, ZB);
        Acc o3{}; init_i32_acc(o3, rows, ZB);
        Acc o4{}; init_i32_acc(o4, rows, ZB);
        Acc o5{}; init_i32_acc(o5, rows, ZB);
        Acc o6{}; init_i32_acc(o6, rows, ZB);
        Acc o7{}; init_i32_acc(o7, rows, ZB);
        o0.set_strides(out.row_bytes, out.col_bytes);
        o1.set_strides(out.row_bytes, out.col_bytes);
        o2.set_strides(out.row_bytes, out.col_bytes);
        o3.set_strides(out.row_bytes, out.col_bytes);
        o4.set_strides(out.row_bytes, out.col_bytes);
        o5.set_strides(out.row_bytes, out.col_bytes);
        o6.set_strides(out.row_bytes, out.col_bytes);
        o7.set_strides(out.row_bytes, out.col_bytes);
        const std::uint32_t bn = (n / kBlock) * nbstride;
        ta.set_data(a.data + am);
        ta.load();
        t0.set_data(wt.data + bn + 0 * nbstride); t0.load();
        t1.set_data(wt.data + bn + 1 * nbstride); t1.load();
        t2.set_data(wt.data + bn + 2 * nbstride); t2.load();
        t3.set_data(wt.data + bn + 3 * nbstride); t3.load();
        t4.set_data(wt.data + bn + 4 * nbstride); t4.load();
        t5.set_data(wt.data + bn + 5 * nbstride); t5.load();
        t6.set_data(wt.data + bn + 6 * nbstride); t6.load();
        t7.set_data(wt.data + bn + 7 * nbstride); t7.load();
        ta.matmul(t0, o0, true, false, true);
        ta.matmul(t1, o1, true, false, true);
        ta.matmul(t2, o2, true, false, true);
        ta.matmul(t3, o3, true, false, true);
        ta.matmul(t4, o4, true, false, true);
        ta.matmul(t5, o5, true, false, true);
        ta.matmul(t6, o6, true, false, true);
        ta.matmul(t7, o7, true, false, true);
        for (std::uint32_t k = kBlock; k < a.cols; k += kBlock) {
          // tile-major 下 k 方向相邻 tile 距离恰好 rows*row_stride = 32*32 = 1024 B，
          // 所以描述符"按行自增"（flags=1）会自动走到下一个 k 块，内层只需 tload。
          ta.load();
          t0.load(); t1.load(); t2.load(); t3.load();
          t4.load(); t5.load(); t6.load(); t7.load();
          ta.matmul(t0, o0, false, false, true);
          ta.matmul(t1, o1, false, false, true);
          ta.matmul(t2, o2, false, false, true);
          ta.matmul(t3, o3, false, false, true);
          ta.matmul(t4, o4, false, false, true);
          ta.matmul(t5, o5, false, false, true);
          ta.matmul(t6, o6, false, false, true);
          ta.matmul(t7, o7, false, false, true);
        }
        gemm_epilogue_block(as, ws, o0, b_scale, out, m, n, rows);
        gemm_epilogue_block(as, ws, o1, b_scale, out, m, n + 1 * kBlock, rows);
        gemm_epilogue_block(as, ws, o2, b_scale, out, m, n + 2 * kBlock, rows);
        gemm_epilogue_block(as, ws, o3, b_scale, out, m, n + 3 * kBlock, rows);
        gemm_epilogue_block(as, ws, o4, b_scale, out, m, n + 4 * kBlock, rows);
        gemm_epilogue_block(as, ws, o5, b_scale, out, m, n + 5 * kBlock, rows);
        gemm_epilogue_block(as, ws, o6, b_scale, out, m, n + 6 * kBlock, rows);
        gemm_epilogue_block(as, ws, o7, b_scale, out, m, n + 7 * kBlock, rows);
      }
    }
    // 尾块（N 不是 32 的倍数时才会走到；我们所有预分块权重的 N 都是 32 的倍数）
    if (n < wt.rows) {
      Tile ta2{};
      config_tile(ta2, a);
      ta2.set_flags(0);
      Tile tb2{};
      config_tile(tb2, wt);
      tb2.set_strides(kBlock, 1);
      tb2.set_flags(0);
      for (; n < wt.rows; n += kBlock) {
        const std::uint8_t cols = block_zone(wt.rows, n);
        Acc o{};
        init_i32_acc(o, rows, cols);
        o.set_strides(out.row_bytes, out.col_bytes);
        set_zone(ta2, rows, ZB);
        set_zone(tb2, cols, ZB);
        const std::uint32_t nb = (n / kBlock) * nbstride;
        // 尾块走显式寻址（flags=0）：tile-major 下第 k 块的偏移就是 k*kBlock 字节。
        // 注意 mma 的 zero 标志必须是编译期常量，所以首块/其余要分开写。
        ta2.set_data(a.data + m * ab);
        tb2.set_data(wt.data + nb);
        ta2.load();
        tb2.load();
        ta2.matmul(tb2, o, /*zero=*/true, false, true);
        for (std::uint32_t k = kBlock; k < a.cols; k += kBlock) {
          ta2.set_data(a.data + m * ab + k * static_cast<std::uint32_t>(a.col_bytes));
          tb2.set_data(wt.data + nb + k * kBlock);
          ta2.load();
          tb2.load();
          ta2.matmul(tb2, o, /*zero=*/false, false, true);
        }
        Vec32 w2{};
        load_vec_n(w2, b_scale, n, cols);
        Acc of(out.data, out.row_bytes, out.col_bytes, rows, cols, AccDType::F32);
        of.set_dtype(AccDType::F32);
        adequant(of, o, w2, as);
        store_acc_at(of, out.data + m * static_cast<std::uint32_t>(out.row_bytes) +
                             n * static_cast<std::uint32_t>(out.col_bytes));
      }
    }
  }
}

} // namespace qwen35_ops

#endif
