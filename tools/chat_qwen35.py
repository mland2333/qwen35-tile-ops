#!/usr/bin/env python3
"""Qwen3.5-0.8B 文本进 / 文本出（设备端贪心，单轮 ChatML）。

    python3 tools/chat_qwen35.py "你好，请用一句话介绍你自己。" --max-new 16

流程：tokenizer → ChatML 模板（含 assistant 前缀与 <think>）→ patch blob 的
ids/CTRL 并清零状态区（S/卷积历史/KV/scratch）→ 跑 guest → 读回生成的 id → 解码。
"""
from __future__ import annotations

import argparse
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from tokenizer import Tokenizer  # noqa: E402

EOS = {248044, 248045, 248046}  # <|im_end|> 与 Qwen3.5 的 eos 组


def read_layout(path: Path) -> dict[str, int]:
    txt = path.read_text()
    return {m.group(1): int(m.group(2))
            for m in re.finditer(r"inline constexpr (?:long|int) k(\w+) = (\d+);", txt)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="你好，请用一句话介绍你自己。")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--header", default=str(ROOT / "config" / "model_qwen35.h"))
    ap.add_argument("--blob", default=str(ROOT / "build" / "qwen35_data.bin"))
    ap.add_argument("--guest", default=str(ROOT / "build" / "qwen35.bin"))
    ap.add_argument("--bridge", default=str(ROOT / "build" / "sim_bridge"))
    ap.add_argument("--tokenizer", default=str(ROOT.parent / "qwen3.5-0.8b-bf16" / "tokenizer.json"))
    ap.add_argument("--no-think", action="store_true", help="关闭 thinking 前缀")
    ap.add_argument("--spec", action="store_true", help="开启 prompt-lookup 投机解码")
    args = ap.parse_args()

    c = read_layout(Path(args.header))
    tk = Tokenizer(args.tokenizer)
    prefix = "<|im_start|>user\n" + args.text + "<|im_end|>\n<|im_start|>assistant\n"
    prefix += "<think>\n\n</think>\n\n" if args.no_think else "<think>\n"
    ids = tk.encode(prefix)
    if len(ids) + args.max_new > c["MaxSeq"]:
        print(f"prompt {len(ids)} + {args.max_new} > max_seq {c['MaxSeq']}", file=sys.stderr)
        return 2

    work = ROOT / "build" / "chat_blob.bin"
    shutil.copyfile(args.blob, work)
    data = bytearray(work.read_bytes())
    # 清零状态区（GDN 状态 / 卷积历史 / KV cache / scratch），保留权重
    start, end = c["S_BASE"], c["SCRATCH_END"]
    data[start:end] = bytes(end - start)
    struct.pack_into(f"<{c['MaxSeq']}i", data, c["IDS"], *(ids + [0] * (c["MaxSeq"] - len(ids))))
    struct.pack_into("<4i", data, c["CTRL"], len(ids), args.max_new, c["MaxSeq"],
                     2 if args.spec else 0)
    work.write_bytes(data)

    ids_out = ROOT / "build" / "chat_ids.bin"
    r = subprocess.run([args.bridge, args.guest, str(work), "/dev/null",
                        "--blob-base", "0x81000000",
                        "--dump", f"{ids_out}@0:{4 * c['MaxSeq']}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return 1
    out = struct.unpack_from(f"<{c['MaxSeq']}i", ids_out.read_bytes())
    gen = list(out[len(ids):len(ids) + args.max_new])
    stop = []
    for t in gen:
        if t in EOS:
            break
        stop.append(t)
    print(f"prompt: {len(ids)} tokens, new: {len(gen)}")
    print("--- output ---")
    print(tk.decode(stop))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
