---
title: LoRA T5 Dialogue Summarizer
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: T5 + LoRA for customer service dialogue summaries
models:
  - google-t5/t5-small
  - lucianoon/t5-small-lora-tweetsumm
---

# 🧬 LoRA T5 — Dialogue Summarizer

Interactive demo of **T5-Small fine-tuned with LoRA** (rank 4, rsLoRA scaling)
for abstractive summarization of customer service dialogues from the
TweetSumm dataset.

- **0.24% trainable parameters** (147K of 60M) — ROUGE-L 0.357
- Adapter loaded from the Hub: [`lucianoon/t5-small-lora-tweetsumm`](https://huggingface.co/lucianoon/t5-small-lora-tweetsumm)
- Training code, rank ablation study and evaluation pipeline:
  [github.com/lucianoon/lora-tweetsumm](https://github.com/lucianoon/lora-tweetsumm)

Paste a customer–agent conversation (or click an example) and get a concise,
actionable summary.
