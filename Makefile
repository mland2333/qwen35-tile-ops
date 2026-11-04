# Qwen3.5-0.8B 文本塔（Tile 层 + YAICore 后端）
#
# 依赖：本仓库的 YAI clang（默认 ../build/bin/clang++）、riscv64-unknown-elf 工具链、
# Tile-NPU-Sim 的 guest link.ld、以及我们自己的模拟器桥（npucc/build_sim.py）。
#
# 可用变量覆盖：CLANG / CROSS / LINK_LD / NPUCC / TILE_NPU_SIM / PY

CLANG     ?= ../build/bin/clang++
CROSS     ?= riscv64-unknown-elf-
LINK_LD   ?= $(HOME)/Tile-NPU-Sim/tests/guest/link.ld
NPUCC     ?= ../../npucc
# 系统 python3 没有 numpy；用 DSH 的 bundled python（可用 make PY=... 覆盖）
PY        ?= $(HOME)/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/bin/python3

# x31 = blob 基址、x30 = 栈顶：由 sim_bridge/crt0 约定传入，编译器不得挪用，
# 否则 main 里 asm("mv %0, x31") 可能读到被清零的 x31（0.6B 实测踩过）。
TARGET    := --target=riscv64-unknown-elf -march=rv64im_xaicore -mabi=lp64 \
             -ffixed-x30 -ffixed-x31 \
             -std=c++20 -O1 -nostdinc++ -isystem test_include
INC       := -I include -I kernels -I . -I config
BUILD     := build
KERNELS   := $(wildcard kernels/*.h)
SRC       := ../../qwen3.5-0.8b-bf16
PQW       := $(BUILD)/qwen35-0.8b-int8.pqw
HDR       := config/model_qwen35.h
BLOB      := $(BUILD)/qwen35_data.bin
GUEST     := $(BUILD)/qwen35.bin

.PHONY: all pqw header blob guest check clean stats

all: $(GUEST)

$(BUILD):
	mkdir -p $(BUILD)

# safetensors(bf16) → PQWF 扩展（int8 per-channel + f32 小张量）
$(PQW): tools/quantize_qwen35.py tools/ref_qwen35.py | $(BUILD)
	$(PY) tools/quantize_qwen35.py --src $(SRC) --out $@

pqw: $(PQW)

# PQWF → guest blob + 布局头
$(HDR) $(BLOB): $(PQW) tools/convert_pqwf35.py tools/ref_qwen35.py
	$(PY) tools/convert_pqwf35.py --pqwf $(PQW) --header $(HDR) --blob $(BLOB)

header blob: $(HDR) $(BLOB)

$(BUILD)/main.o: src/main.cpp $(KERNELS) $(HDR) | $(BUILD)
	$(CLANG) $(TARGET) $(INC) -c $< -o $@

$(BUILD)/crt0.o: src/crt0.S | $(BUILD)
	$(CROSS)as -mabi=lp64 $< -o $@

$(BUILD)/qwen35.elf: $(BUILD)/main.o $(BUILD)/crt0.o
	$(CROSS)ld -T $(LINK_LD) $^ -o $@

$(GUEST): $(BUILD)/qwen35.elf
	$(CROSS)objcopy -O binary $< $@
	@echo "guest: $@ ($$(stat -f%z $@) 字节)"

guest: $(GUEST)

# 调试转储版（-DQ35_DUMP）：把每层残差流、layer0/3 的中间张量写进 blob 的 DUMP 区，
# 配合 tools/check_qwen35.py --stages 逐层对拍。注意它跳过 lm_head（否则 logits
# 会把转储区覆盖掉）。
$(BUILD)/qwen35d.bin: src/main.cpp $(KERNELS) $(HDR) | $(BUILD)
	$(CLANG) $(TARGET) $(INC) -DQ35_DUMP -c src/main.cpp -o $(BUILD)/maind.o
	$(CROSS)ld -T $(LINK_LD) $(BUILD)/maind.o $(BUILD)/crt0.o -o $(BUILD)/qwen35d.elf
	$(CROSS)objcopy -O binary $(BUILD)/qwen35d.elf $@
	@echo "dump guest: $@"

dump: $(BUILD)/qwen35d.bin

# 静态指令数（与 0.6B 的 40,178 同一口径：main_big.cpp 编译出的汇编里
# "以 tab 开头的助记符行"）
$(BUILD)/main.s: src/main.cpp $(KERNELS) $(HDR) | $(BUILD)
	$(CLANG) $(TARGET) $(INC) -S $< -o $@

stats: $(BUILD)/main.s
	@echo "静态指令数: $$(grep -cE '^[[:space:]]+[a-z]' $(BUILD)/main.s)"

check: $(GUEST) $(BLOB)
	$(PY) tools/check_qwen35.py --guest $(GUEST) --blob $(BLOB)

clean:
	rm -rf $(BUILD)
