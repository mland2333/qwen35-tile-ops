#!/usr/bin/env python3
"""给 clangd / VS Code 生成 compile_commands.json。

两条 TU：
  * src/main.cpp        —— 设备侧的 guest（riscv64 + 自定义 xaicore 扩展）；
  * tools/sim_bridge.cpp —— 主机侧的桥（host g++）。

clangd 自带的 clang 不认识 -march=rv64im_xaicore，仓库根的 .clangd 会把它换成
-march=rv64im（只影响 IDE，不影响真实构建）。头文件（kernels/*.h）由 clangd
从"包含它的 TU"推断编译命令。
"""

import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
clang = str((root / ".." / "llvm-project" / "build" / "bin" / "clang++").resolve())
sim = str((root / ".." / "Tile-NPU-Sim").resolve())

entries = [
    {
        "directory": str(root),
        "file": str(root / "src" / "main.cpp"),
        "arguments": [
            clang,
            "--target=riscv64-unknown-elf",
            # 真实构建是 -march=rv64im_xaicore；clangd 自带的 clang 不认识这个
            # 扩展，会放弃语义分析，所以 IDE 这里退化成标准 rv64im。
            "-march=rv64im",
            "-mabi=lp64",
            "-ffixed-x30",
            "-ffixed-x31",
            "-std=c++20",
            "-O1",
            "-nostdinc++",
            "-isystem", "test_include",
            "-I", "include",
            "-I", "kernels",
            "-I", ".",
            "-I", "config",
            "-c", "src/main.cpp",
            "-o", "build/main.o",
        ],
    },
    {
        "directory": str(root),
        "file": str(root / "tools" / "sim_bridge.cpp"),
        "arguments": [
            "g++",
            "-std=c++20",
            "-O2",
            "-I", sim + "/include",
            "-I", sim + "/src",
            "-c", "tools/sim_bridge.cpp",
            "-o", "build/sim_bridge_main.o",
        ],
    },
]

out = root / "compile_commands.json"
out.write_text(json.dumps(entries, indent=2) + "\n")
print(f"wrote {out} ({len(entries)} entries)")
