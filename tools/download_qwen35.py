#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载 Qwen/Qwen3.5-0.8B 的权重与分词器（走 hf-mirror，HF 主站在本机不通）。

本机没有 huggingface_hub，也不允许装包，所以直接用 curl 拉
`https://hf-mirror.com/<repo>/resolve/main/<file>`（会 307 到 CDN）。

    python3 tools/download_qwen35.py --dst ../../qwen3.5-0.8b-bf16

注意 safetensors 在仓库里的名字带分片后缀：
`model.safetensors-00001-of-00001.safetensors`（1.75 GB, bf16）。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = "Qwen/Qwen3.5-0.8B"
MIRROR = "https://hf-mirror.com"
FILES = [
    "model.safetensors-00001-of-00001.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="../../qwen3.5-0.8b-bf16")
    ap.add_argument("--repo", default=REPO)
    a = ap.parse_args()
    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        target = dst / f
        if target.is_file() and target.stat().st_size > 0:
            print(f"skip {f} ({target.stat().st_size} B)")
            continue
        url = f"{MIRROR}/{a.repo}/resolve/main/{f}"
        print(f"fetch {f} ...", flush=True)
        r = subprocess.run(["curl", "-sL", "--fail", "-m", "1800", "-o", str(target), url])
        if r.returncode != 0:
            raise SystemExit(f"下载失败: {url}")
        print(f"  {target.stat().st_size} B")
    print("done:", dst)


if __name__ == "__main__":
    main()
