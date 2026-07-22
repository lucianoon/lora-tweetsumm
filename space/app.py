"""
Hugging Face Space for the LoRA fine-tuned T5 dialogue summarizer.

Self-contained: loads the base T5 model plus the trained LoRA adapter
directly from the Hugging Face Hub, so the Space does not need the full
training repository. If the adapter repo is unavailable, falls back to
the base model with a visible notice.

Configuration via environment variables (Space settings → Variables):
    BASE_MODEL_ID     — base model (default: google-t5/t5-small)
    ADAPTER_MODEL_ID  — LoRA adapter repo (default: lucianoon/t5-small-lora-tweetsumm)
"""

from __future__ import annotations

import logging
import os

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "google-t5/t5-small")
ADAPTER_MODEL_ID = os.getenv("ADAPTER_MODEL_ID", "lucianoon/t5-small-lora-tweetsumm")

PREFIX = "summarize: "
MAX_SRC_LENGTH = 512

# ── Example dialogues ─────────────────────────────────────────
EXAMPLES = [
    [
        "Customer: Hi, I need to change my flight from NYC to LA on Oct 15.\n"
        "Agent: Sure, I can help with that. Let me check available options.\n"
        "Customer: I'd prefer a morning flight if possible.\n"
        "Agent: I found a flight at 8:30 AM with Delta. The fare difference is $120.\n"
        "Customer: That works. Please go ahead and change it.\n"
        "Agent: Done! Your new flight is DL1542 departing JFK at 8:30 AM on Oct 15. "
        "You'll receive a confirmation email shortly."
    ],
    [
        "Customer: My baggage didn't arrive with my flight. Flight AA2901 from Miami.\n"
        "Agent: I'm sorry about that. Let me file a delayed baggage report for you.\n"
        "Customer: It's a black Samsonite suitcase with a red tag.\n"
        "Agent: Got it. I've filed report #BG78432. We'll deliver it to your hotel "
        "within 24 hours. You'll get SMS updates.\n"
        "Customer: What if it doesn't arrive?\n"
        "Agent: If not delivered in 48h, we'll start a claim process. "
        "You can also buy essentials up to $50 and we'll reimburse."
    ],
    [
        "Customer: I want to cancel my reservation for tomorrow. Booking ref XKMT92.\n"
        "Agent: I can see your booking. Please note there's a $75 cancellation fee "
        "since it's less than 24 hours before departure.\n"
        "Customer: That's fine, please cancel it.\n"
        "Agent: Your reservation has been cancelled. The refund minus the $75 fee "
        "will be processed to your original payment method within 5-7 business days.\n"
        "Customer: Thank you."
    ],
]


def load_model() -> tuple:
    """Load tokenizer and model (base + LoRA adapter when available).

    Returns:
        (tokenizer, model, notice) — notice is an HTML string when the
        adapter could not be loaded, else an empty string.
    """
    logger.info("Loading tokenizer and base model: %s", BASE_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID)

    try:
        from peft import PeftModel

        logger.info("Loading LoRA adapter from the Hub: %s", ADAPTER_MODEL_ID)
        model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_ID)
        notice = ""
    except Exception as exc:  # noqa: BLE001 — fallback deliberado
        logger.warning("Could not load adapter %s (%s). Using base T5.", ADAPTER_MODEL_ID, exc)
        model = base_model
        notice = (
            "<div class='model-notice'>⚠️ LoRA adapter "
            f"<code>{ADAPTER_MODEL_ID}</code> could not be loaded — running the "
            "<strong>base T5 model without fine-tuning</strong>.</div>"
        )

    model.eval()
    return tokenizer, model, notice


tokenizer, model, model_notice = load_model()


def summarize(dialogue: str, max_new_tokens: int, num_beams: int) -> str:
    """Generate a summary for a single dialogue (mirrors src/inference.py)."""
    if not dialogue.strip():
        return "⚠️ Please enter a dialogue to summarize."

    inputs = tokenizer(
        PREFIX + dialogue,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SRC_LENGTH,
    )
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            num_beams=int(num_beams),
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


with gr.Blocks(
    title="LoRA T5 — Dialogue Summarizer",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css="""
        .main-header { text-align: center; padding: 1.5rem 0 0.5rem; }
        .main-header h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.2rem; font-weight: 800; margin-bottom: 0.3rem;
        }
        .main-header p { color: #64748b; font-size: 1.05rem; }
        .badge-row {
            display: flex; justify-content: center; gap: 0.5rem;
            margin-bottom: 1rem; flex-wrap: wrap;
        }
        .badge {
            background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 999px;
            padding: 0.25rem 0.75rem; font-size: 0.8rem; color: #475569; font-weight: 500;
        }
        .model-notice {
            max-width: 900px; margin: 0 auto 1rem; padding: 0.75rem 1rem;
            border: 1px solid #f59e0b; background: #fffbeb; color: #92400e;
            border-radius: 8px; font-size: 0.9rem; text-align: center;
        }
        .footer {
            text-align: center; padding: 1rem; color: #94a3b8; font-size: 0.85rem;
            border-top: 1px solid #e2e8f0; margin-top: 1.5rem;
        }
    """,
) as demo:
    gr.HTML("""
        <div class="main-header">
            <h1>🧬 Dialogue Summarizer</h1>
            <p>T5 fine-tuned with LoRA on TweetSumm —
               customer service dialogue summarization</p>
        </div>
        <div class="badge-row">
            <span class="badge">🤖 T5-Small</span>
            <span class="badge">⚡ LoRA r=4</span>
            <span class="badge">📊 0.24% trainable params</span>
            <span class="badge">📈 ROUGE-L 0.357</span>
        </div>
    """)

    if model_notice:
        gr.HTML(model_notice)

    with gr.Row():
        with gr.Column(scale=1):
            dialogue_input = gr.Textbox(
                label="💬 Customer Service Dialogue",
                placeholder="Paste a customer-agent conversation here...",
                lines=10,
                max_lines=20,
            )
            with gr.Accordion("⚙️ Generation Settings", open=False):
                max_tokens_slider = gr.Slider(
                    minimum=16,
                    maximum=128,
                    value=48,
                    step=8,
                    label="Max Tokens",
                    info="Maximum length of the generated summary",
                )
                num_beams_slider = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=1,
                    step=1,
                    label="Beam Search Width",
                    info="Higher = better quality, slower generation",
                )
            summarize_btn = gr.Button("✨ Summarize", variant="primary", size="lg")

        with gr.Column(scale=1):
            summary_output = gr.Textbox(
                label="📝 Generated Summary",
                lines=6,
                interactive=False,
            )
            gr.HTML("""
                <div style="background: linear-gradient(135deg, #f0f4ff, #faf5ff);
                            border-radius: 12px; padding: 1rem 1.25rem;
                            margin-top: 0.75rem; border: 1px solid #e0e7ff;">
                    <p style="margin: 0 0 0.5rem; font-weight: 600; color: #4338ca;">
                        💡 How it works
                    </p>
                    <p style="margin: 0; color: #64748b; font-size: 0.9rem; line-height: 1.5;">
                        This model uses <strong>LoRA</strong> (Low-Rank Adaptation) to fine-tune
                        only 0.24% of T5's parameters on customer service dialogues.
                        The result: a lightweight adapter that turns verbose conversations
                        into concise, actionable summaries.
                    </p>
                </div>
            """)

    gr.Examples(
        examples=EXAMPLES,
        inputs=[dialogue_input],
        label="📋 Example Dialogues (click to try)",
    )

    gr.HTML("""
        <div class="footer">
            Built with 🤗 Transformers, PEFT &amp; Gradio •
            <a href="https://github.com/lucianoon/lora-tweetsumm" target="_blank"
               style="color: #818cf8;">Training code on GitHub</a> •
            <a href="https://arxiv.org/abs/2106.09685" target="_blank"
               style="color: #818cf8;">LoRA paper</a>
        </div>
    """)

    summarize_btn.click(
        fn=summarize,
        inputs=[dialogue_input, max_tokens_slider, num_beams_slider],
        outputs=summary_output,
    )
    dialogue_input.submit(
        fn=summarize,
        inputs=[dialogue_input, max_tokens_slider, num_beams_slider],
        outputs=summary_output,
    )


if __name__ == "__main__":
    demo.launch()
