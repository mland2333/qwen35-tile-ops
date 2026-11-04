# 官方实现快照（只作对照，不参与编译/运行）

`modeling_qwen3_5.py` 是 `huggingface/transformers`（main 分支）里
`src/transformers/models/qwen3_5/modeling_qwen3_5.py` 的原文快照，用于校对
`qwen35_dense_demo` 与官方实现的差异（见 `../../README.md` §4）。

本机没有 torch，所以它**不能运行**，只作为"权威语义"的参照：
`torch_recurrent_gated_delta_rule`（q 的 1/√d_k、状态布局 (d_k,d_v)）、
`Qwen3_5RMSNorm`（零中心 1+w）、`Qwen3_5Attention`（q/gate 每头交错）、
`causal_conv1d_fn`（无 bias + silu）、`Qwen3_5TextRotaryEmbedding`
（partial rotary + M-RoPE 的 `recomposition_frequencies`）。

抓取方式：
```bash
curl -sL -o modeling_qwen3_5.py \
  https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
```
