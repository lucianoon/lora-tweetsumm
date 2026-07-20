#!/usr/bin/env python3
"""
Gradio demo for the LoRA fine-tuned T5 dialogue summarizer.

Launch:
    python -m scripts.demo
    python -m scripts.demo --share   # public link (72h)
    python -m scripts.demo --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import gradio as gr

from src.config import load_config
from src.data import load_tokenizer
from src.inference import summarize
from src.model import build_model, load_trained_model

logger = logging.getLogger(__name__)

# ── Example dialogues for the demo ────────────────────────────
EXAMPLES = [
    # ── English Examples ──
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
    # ── Exemplos em Português ──
    [
        "Cliente: Olá, preciso alterar meu voo de Nova York "
        "para Los Angeles no dia 15 de outubro.\n"
        "Atendente: Claro, posso ajudar com isso. Deixe-me verificar as opções disponíveis.\n"
        "Cliente: Prefiro um voo pela manhã, se possível.\n"
        "Atendente: Encontrei um voo às 8:30 da manhã com a "
        "Delta. A diferença de tarifa é de R$ 120.\n"
        "Cliente: Perfeito. Por favor, prossiga com a alteração.\n"
        "Atendente: Feito! Seu novo voo é o DL1542 partindo de "
        "JFK às 8:30 da manhã em 15 de outubro. "
        "Você receberá um e-mail de confirmação em breve."
    ],
    [
        "Cliente: Minha bagagem não chegou com o meu voo. Voo AA2901 vindo de Miami.\n"
        "Atendente: Sinto muito por isso. Deixe-me registrar um "
        "relatório de bagagem atrasada para você.\n"
        "Cliente: É uma mala Samsonite preta com uma etiqueta vermelha.\n"
        "Atendente: Entendido. Registrei o relatório nº BG78432. Entregaremos em seu hotel "
        "dentro de 24 horas. Você receberá atualizações por SMS.\n"
        "Cliente: E se não chegar?\n"
        "Atendente: Se não for entregue em 48 horas, iniciaremos o processo de reembolso. "
        "Você também pode comprar itens essenciais de até R$ 50 e nós reembolsaremos."
    ],
]


def create_app(
    config_path: str | None = None,
    checkpoint: str | None = None,
    allow_untrained: bool = False,
) -> gr.Blocks:
    """Build the Gradio app with the fine-tuned model loaded."""

    config = load_config(config_path)
    tokenizer = load_tokenizer(config)
    model_notice = ""
    try:
        model = load_trained_model(config, checkpoint)
    except FileNotFoundError:
        if not allow_untrained:
            raise
        logger.warning("No trained adapter found. Launching demo with an untrained LoRA adapter.")
        model = build_model(config)
        model.eval()
        model_notice = (
            "<div class='model-notice'>No trained adapter checkpoint was found. "
            "This demo is running with an untrained LoRA adapter.</div>"
        )

    translator_pt_en = []
    translator_en_pt = []

    def get_pt_en():
        if not translator_pt_en:
            logger.info("Loading translation model (PT -> EN: Helsinki-NLP/opus-mt-ROMANCE-en)...")
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model_name = "Helsinki-NLP/opus-mt-ROMANCE-en"
            tok = AutoTokenizer.from_pretrained(model_name)
            mod = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            translator_pt_en.append((tok, mod))
        return translator_pt_en[0]

    def get_en_pt():
        if not translator_en_pt:
            logger.info("Loading translation model (EN -> PT: Helsinki-NLP/opus-mt-tc-big-en-pt)...")
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"
            tok = AutoTokenizer.from_pretrained(model_name)
            mod = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            translator_en_pt.append((tok, mod))
        return translator_en_pt[0]

    def generate_summary(dialogue: str, max_tokens: int, num_beams: int, translate: bool) -> str:
        """Generate a summary from the input dialogue."""
        if not dialogue.strip():
            return "⚠️ Please enter a dialogue to summarize."

        # Override inference params with slider values
        config.inference.max_new_tokens = max_tokens
        config.inference.num_beams = num_beams

        if translate:
            try:
                logger.info("Translating input from PT to EN...")
                tok, mod = get_pt_en()
                inputs = tok(dialogue, return_tensors="pt", padding=True)
                outputs = mod.generate(**inputs)
                dialogue = tok.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                logger.error(f"Error during PT -> EN translation: {e}")
                return f"⚠️ Translation error (PT -> EN): {e}"

        summary = summarize(model, tokenizer, dialogue, config)

        if translate:
            try:
                logger.info("Translating summary from EN to PT...")
                tok, mod = get_en_pt()
                inputs = tok(summary, return_tensors="pt", padding=True)
                outputs = mod.generate(**inputs)
                summary = tok.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                logger.error(f"Error during EN -> PT translation: {e}")
                return f"⚠️ Translation error (EN -> PT): {e} (English summary: {summary})"

        return summary

    # ── UI ─────────────────────────────────────────────────────
    with gr.Blocks(
        title="LoRA T5 — Dialogue Summarizer",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
            .main-header {
                text-align: center;
                padding: 1.5rem 0 0.5rem;
            }
            .main-header h1 {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.2rem;
                font-weight: 800;
                margin-bottom: 0.3rem;
            }
            .main-header p {
                color: #64748b;
                font-size: 1.05rem;
            }
            .badge-row {
                display: flex;
                justify-content: center;
                gap: 0.5rem;
                margin-bottom: 1rem;
                flex-wrap: wrap;
            }
            .badge {
                background: #f1f5f9;
                border: 1px solid #e2e8f0;
                border-radius: 999px;
                padding: 0.25rem 0.75rem;
                font-size: 0.8rem;
                color: #475569;
                font-weight: 500;
            }
            .model-notice {
                max-width: 900px;
                margin: 0 auto 1rem;
                padding: 0.75rem 1rem;
                border: 1px solid #f59e0b;
                background: #fffbeb;
                color: #92400e;
                border-radius: 8px;
                font-size: 0.9rem;
                text-align: center;
            }
            .footer {
                text-align: center;
                padding: 1rem;
                color: #94a3b8;
                font-size: 0.85rem;
                border-top: 1px solid #e2e8f0;
                margin-top: 1.5rem;
            }
        """,
    ) as app:
        # Header
        gr.HTML("""
            <div class="main-header">
                <h1>🧬 Dialogue Summarizer</h1>
                <p>T5 fine-tuned with LoRA on TweetSumm —
                   customer service dialogue summarization</p>
            </div>
            <div class="badge-row">
                <span class="badge">🤖 T5-Small</span>
                <span class="badge">⚡ LoRA r=4</span>
                <span class="badge">🍎 Apple MPS</span>
                <span class="badge">📊 ~0.5% trainable params</span>
            </div>
        """)

        if model_notice:
            gr.HTML(model_notice)

        with gr.Row():
            # Left: Input
            with gr.Column(scale=1):
                dialogue_input = gr.Textbox(
                    label="💬 Customer Service Dialogue",
                    placeholder="Paste a customer-agent conversation here...",
                    lines=10,
                    max_lines=20,
                )

                translate_checkbox = gr.Checkbox(
                    label="🌐 Traduzir (PT ↔ EN)",
                    value=False,
                    info="Traduz diálogos em português para inglês antes de resumir, "
                    "e o resumo de volta para português.",
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

                summarize_btn = gr.Button(
                    "✨ Summarize",
                    variant="primary",
                    size="lg",
                )

            # Right: Output
            with gr.Column(scale=1):
                summary_output = gr.Textbox(
                    label="📝 Generated Summary",
                    lines=6,
                    interactive=False,
                )

                gr.HTML("""
                    <div style="
                        background: linear-gradient(135deg, #f0f4ff, #faf5ff);
                        border-radius: 12px;
                        padding: 1rem 1.25rem;
                        margin-top: 0.75rem;
                        border: 1px solid #e0e7ff;
                    ">
                        <p style="margin: 0 0 0.5rem; font-weight: 600; color: #4338ca;">
                            💡 How it works
                        </p>
                        <p style="margin: 0; color: #64748b; font-size: 0.9rem; line-height: 1.5;">
                            This model uses <strong>LoRA</strong> (Low-Rank Adaptation) to fine-tune
                            only ~0.5% of T5's parameters on customer service dialogues.
                            The result: a lightweight adapter that turns verbose conversations
                            into concise, actionable summaries.
                        </p>
                    </div>
                """)

        # Examples
        gr.Examples(
            examples=EXAMPLES,
            inputs=[dialogue_input],
            label="📋 Example Dialogues (click to try)",
        )

        # Footer
        gr.HTML("""
            <div class="footer">
                Built with 🤗 Transformers, PEFT & Gradio •
                LoRA fine-tuned on TweetSumm dataset •
                <a href="https://arxiv.org/abs/2106.09685" target="_blank" style="color: #818cf8;">
                    LoRA paper
                </a>
            </div>
        """)

        # Events
        summarize_btn.click(
            fn=generate_summary,
            inputs=[dialogue_input, max_tokens_slider, num_beams_slider, translate_checkbox],
            outputs=summary_output,
        )
        dialogue_input.submit(
            fn=generate_summary,
            inputs=[dialogue_input, max_tokens_slider, num_beams_slider, translate_checkbox],
            outputs=summary_output,
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio demo for LoRA T5 summarizer")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "LoRA adapter checkpoint directory. Defaults to the latest checkpoint under "
            "training.output_dir."
        ),
    )
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="Launch the demo with an untrained LoRA adapter if no checkpoint is available",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(name)-20s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    app = create_app(args.config, args.checkpoint, args.allow_untrained)

    logger.info("Launching Gradio demo on port %d...", args.port)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
