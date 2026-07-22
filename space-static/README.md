---
title: LoRA T5 Dialogue Summarizer
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: mit
short_description: T5 + LoRA summarizer running 100% in your browser
models:
  - lucianoon/t5-small-lora-tweetsumm-onnx
---

# 🧬 LoRA T5 — Dialogue Summarizer (in-browser)

Static Space: the model runs **entirely in the visitor's browser** via
[Transformers.js](https://huggingface.co/docs/transformers.js) — no server,
no API keys, no data leaves the page.

- Model: [`lucianoon/t5-small-lora-tweetsumm-onnx`](https://huggingface.co/lucianoon/t5-small-lora-tweetsumm-onnx)
  (LoRA rank-4 merged into T5-Small, ONNX, INT8, ~90 MB one-time download)
- Training code and rank ablation:
  [github.com/lucianoon/lora-tweetsumm](https://github.com/lucianoon/lora-tweetsumm)
