#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在模拟器上跑 Qwen3.5 guest，并与 numpy 参考对拍。

两种模式：
  * 默认：端到端 logits 比对（相关性 / 最大偏差 / top-k 重合 / argmax）
  * --stages：配合 -DQ35_DUMP 构建的 guest，逐层比对残差流（能定位"第一处发散"）

参考实现有两档（--ref）：
  quant   : 与设备一致的 W8A8（权重 per-channel int8 + 激活 per-token int8），
            GDN 的 S 走 f32（与设备相同）——这是**主对拍口径**
  f32     : 权重反量化、不量化激活（看量化总代价）

用法：
  python3 tools/check_qwen35.py --guest build/qwen35.bin --blob build/qwen35_data.bin
  python3 tools/check_qwen35.py --stages --guest build/qwen35d.bin --blob build/qwen35_data.bin
"""
from __future__ import annotations

import argparse
import math
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # qwen35_ops/
sys.path.insert(0, str(ROOT / "tools"))
import ref_qwen35 as R                                    # noqa: E402

MODEL_DIR = Path(os.environ.get(
    "QWEN35_MODEL", ROOT.parent.parent / "qwen3.5-0.8b-bf16"))
NPUCC = Path(os.environ.get("NPUCC", ROOT.parent.parent / "npucc"))
BRIDGE = NPUCC / "build" / "sim_bridge"
PAD = 0x80000


def hdr_consts(path: Path) -> dict[str, int]:
    txt = path.read_text()
    return {m.group(1): int(m.group(2))
            for m in re.finditer(r"inline constexpr (?:long|int) k(\w+) = (\d+);", txt)}


def run_guest(guest: Path, blob: Path, out: Path, steps: str, extra_mib: str) -> str:
    if not BRIDGE.is_file():
        subprocess.run([sys.executable, str(NPUCC / "build_sim.py")], check=True)
    data = blob.read_bytes()
    if len(data) < PAD:
        data += b"\x00" * (PAD - len(data))
    tmp = ROOT / "build" / "check_in.bin"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(data)
    r = subprocess.run([str(BRIDGE), str(guest), str(tmp), str(out), "0", steps,
                        extra_mib], capture_output=True, text=True)
    sys.stdout.write("  sim: " + r.stdout.strip() + "\n")
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit("guest 运行失败")
    return r.stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guest", default="build/qwen35.bin")
    ap.add_argument("--blob", default="build/qwen35_data.bin")
    ap.add_argument("--header", default="config/model_qwen35.h")
    ap.add_argument("--steps", default="4000000000")
    ap.add_argument("--extra-mib", default="64")
    ap.add_argument("--ref", default="quant", choices=("quant", "f32"))
    ap.add_argument("--stages", action="store_true",
                    help="逐层比对残差流（需要 -DQ35_DUMP 构建的 guest）")
    ap.add_argument("--positions", type=int, default=0,
                    help="用 N 个 token 的 prompt，按位置（行）比对 3 个层的残差流；"
                         "需要 -DQ35_DUMP 构建，N<=32 时为单 chunk")
    a = ap.parse_args()

    guest = Path(a.guest) if Path(a.guest).is_absolute() else ROOT / a.guest
    blob = Path(a.blob) if Path(a.blob).is_absolute() else ROOT / a.blob
    c = hdr_consts(Path(a.header) if Path(a.header).is_absolute() else ROOT / a.header)
    out = ROOT / "build" / "check_out.bin"

    print(f"guest {guest.name} ({guest.stat().st_size} B), blob {blob.name} "
          f"({blob.stat().st_size} B), 参考 {a.ref}")
    pos_ids = None
    if a.positions:
        pos_ids = [(i * 7919 + 13) % c["Vocab"] for i in range(a.positions)]
        data = bytearray(blob.read_bytes())
        struct.pack_into(f"<{len(pos_ids)}i", data, c["IDS"], *pos_ids)
        struct.pack_into("<i", data, c["CTRL"], len(pos_ids))
        patched = ROOT / "build" / "check_blob_pos.bin"
        patched.write_bytes(data)
        blob = patched
        print(f"  prompt 换成 {len(pos_ids)} 个确定性 id（首位 {pos_ids[0]}）")
    run_guest(guest, blob, out, a.steps, a.extra_mib)
    b = out.read_bytes()

    # blob 里的 token id 就是这次跑的输入
    ids = list(struct.unpack_from(f"<{c['MaxSeq']}i", b, c["IDS"]))
    n = struct.unpack_from("<i", b, c["CTRL"])[0]
    ids = ids[:n]
    print(f"  输入 {n} 个 token: {ids[:8]}{'...' if n > 8 else ''}")

    m = R.load(MODEL_DIR, quant=(a.ref == "quant"))
    m.reset()
    st = {}
    lg = m.forward(ids, st)
    ref = lg[-1]

    if a.positions:
        print(f"\n逐位置（行）相对误差，prompt = {a.positions} token:")
        for li, L in enumerate((0, 11, 23)):
            # 转储只保留**最后一个 chunk** 的行；参考要按同样的尾部对齐，
            # 否则跨 chunk 时会拿设备 token 32..63 去比参考 token 0..31（上一次的假 2.0 就这么来的）
            rows = min(n, c["Seq"])
            d = np.frombuffer(b, dtype=np.float32, count=rows * c["Hidden"],
                              offset=c["DUMP"] + (800000 + li * 34000) * 4
                              ).reshape(rows, -1)
            rf = st[f"layer{L}"][n - rows:]
            rels = [float(np.linalg.norm(d[r] - rf[r])
                          / max(np.linalg.norm(rf[r]), 1e-30)) for r in range(rows)]
            n = rows
            head = "  ".join(f"{v:.2e}" for v in rels[:6])
            step = max(1, n // 8)
            trend = "  ".join(f"{rels[r]:.2e}" for r in range(0, n, step))
            print(f"  L{L:2d}: 前6位 {head}")
            print(f"       每 {step} 位 {trend}   末位 {rels[-1]:.2e}   "
                  f"最小 {min(rels):.2e} 最大 {max(rels):.2e}")
        return

    if a.stages:
        print("\n逐层残差流（device vs 参考）:")
        worst = 0.0
        for L in range(c["Layers"]):
            d = np.frombuffer(b, dtype=np.float32, count=n * c["Hidden"],
                              offset=c["DUMP"] + (900000 + L * 6000) * 4).reshape(n, -1)
            rf = st[f"layer{L}"]
            rel = float(np.linalg.norm(d - rf) / max(np.linalg.norm(rf), 1e-30))
            worst = max(worst, rel)
            print(f"  L{L:2d} rel {rel:.3e}")
        print(f"  最差 {worst:.3e}")

    if a.positions:
        print(f"\n逐位置（行）相对误差，prompt = {a.positions} token:")
        for li, L in enumerate((0, 11, 23)):
            # 转储只保留**最后一个 chunk** 的行；参考要按同样的尾部对齐，
            # 否则跨 chunk 时会拿设备 token 32..63 去比参考 token 0..31（上一次的假 2.0 就这么来的）
            rows = min(n, c["Seq"])
            d = np.frombuffer(b, dtype=np.float32, count=rows * c["Hidden"],
                              offset=c["DUMP"] + (800000 + li * 34000) * 4
                              ).reshape(rows, -1)
            rf = st[f"layer{L}"][n - rows:]
            rels = [float(np.linalg.norm(d[r] - rf[r])
                          / max(np.linalg.norm(rf[r]), 1e-30)) for r in range(rows)]
            n = rows
            head = "  ".join(f"{v:.2e}" for v in rels[:6])
            step = max(1, n // 8)
            trend = "  ".join(f"{rels[r]:.2e}" for r in range(0, n, step))
            print(f"  L{L:2d}: 前6位 {head}")
            print(f"       每 {step} 位 {trend}   末位 {rels[-1]:.2e}   "
                  f"最小 {min(rels):.2e} 最大 {max(rels):.2e}")
        return

    if a.stages:
        # 转储版 guest 不跑 lm_head（否则 logits 区会覆盖转储区），所以没有 logits 可比
        print("\n（转储版不跑 lm_head，跳过 logits 比对；端到端请用 build/qwen35.bin）")
        return

    dev = np.frombuffer(b, dtype=np.float32, count=c["Vocab"], offset=c["LOGITS"])
    d = np.abs(dev - ref)
    corr = float(np.corrcoef(dev.astype(np.float64), ref.astype(np.float64))[0, 1])
    top_dev = np.argsort(dev)[-10:][::-1]
    top_ref = np.argsort(ref)[-10:][::-1]
    print("\nlogits 比对（最后一行）:")
    print(f"  max|Δ| {float(d.max()):.4g}   mean|Δ| {float(d.mean()):.4g}   "
          f"corr {corr:.6f}")
    print(f"  argmax dev {int(dev.argmax())} / ref {int(ref.argmax())}   "
          f"top10 重合 {len(set(top_dev.tolist()) & set(top_ref.tolist()))}/10")
    # 分布层面的量化误差指标
    print(f"  |Δ| 中位数 {float(np.median(d)):.4g}   "
          f"超 1.0 的 token 数 {int((d > 1.0).sum())}/{c['Vocab']}")


if __name__ == "__main__":
    main()
