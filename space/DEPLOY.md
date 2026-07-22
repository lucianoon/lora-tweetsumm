# Publicando o demo no Hugging Face Spaces

O Space é autossuficiente: `app.py` carrega o T5 base e o adapter LoRA
diretamente do Hub. São dois uploads — o **adapter** (repo de modelo) e o
**Space** (esta pasta). Total: ~5 minutos depois do login.

## 0. Pré-requisitos

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login   # cole um token com permissão de escrita (hf.co/settings/tokens)
```

## 1. Treinar e publicar o adapter

```bash
# Treina com a config padrão (r=4, ~1 min em M4 / GPU)
python -m scripts.train

# Descubra o checkpoint mais recente
ls checkpoints/t5-lora-tweetsumm/

# Publica o adapter no Hub (o repo é criado automaticamente)
huggingface-cli upload lucianoon/t5-small-lora-tweetsumm \
  checkpoints/t5-lora-tweetsumm/checkpoint-225 . --repo-type model
```

> O nome `lucianoon/t5-small-lora-tweetsumm` é o default esperado pelo
> `app.py`. Se usar outro, defina a variável `ADAPTER_MODEL_ID` nas
> configurações do Space (Settings → Variables).

## 2. Criar e publicar o Space

```bash
huggingface-cli repo create lora-tweetsumm-demo --type space --space_sdk gradio

# A partir da raiz do repositório:
huggingface-cli upload lucianoon/lora-tweetsumm-demo space . --repo-type space
```

O Space builda sozinho (CPU gratuito é suficiente para o T5-Small) e fica em:
`https://huggingface.co/spaces/lucianoon/lora-tweetsumm-demo`

## 3. Depois de no ar

Adicione o link no README principal do repositório GitHub, por exemplo como
badge no topo:

```markdown
[![Open in Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue.svg)](https://huggingface.co/spaces/lucianoon/lora-tweetsumm-demo)
```

## Notas de design

- O `app.py` do Space **não usa** o pacote `src/` — é autossuficiente de
  propósito, para o Space não precisar do repositório de treino inteiro.
  A lógica de geração espelha `src/inference.py` (prefixo `summarize: `,
  truncamento em 512 tokens, `inference_mode`).
- Se o adapter não puder ser carregado, o app cai para o T5 base **com um
  aviso visível na UI** — nunca finge estar fine-tunado.
- A tradução PT↔EN do demo local (`scripts/demo.py`) ficou de fora do Space:
  os dois modelos Helsinki-NLP dobrariam o uso de memória e o tempo de
  build no tier gratuito. O demo local continua tendo o recurso.
