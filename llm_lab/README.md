# llm_lab

`llm_lab` is a small integration layer that unifies:

- Qwen3.5 model config/model/inference components vendored inside `llm_lab/models/qwen35`
- local family abstractions for future training/inference extensions

Qwen3.5 serving in `llm_lab` is self-contained and does not import `transformers_Qwen3_5`.
It supports both:

- `native` backend (local reimplementation)
- `transformers` backend (via local `third_party/transformers`)

Design goals:

- keep model families isolated (own tokenizer + own chat template)
- share orchestration (CLI, family registry, runtime selection)
- make it easy to add new model families later

Current families:

- `qwen35`: inference-focused family backed by `llm_lab` local Qwen3.5 implementation
- `nanochat_style`: local nanochat-style training/inference pipeline (decoupled from third_party)

## Quick usage

List supported families:

`python -m llm_lab.cli.infer --list-families`

Inspect model builders:

`python -m llm_lab.cli.build_model --list-families`

Inspect effective merged config:

`python -m llm_lab.cli.info --mode infer --family-config llm_lab/configs/families/qwen35.example.json`

Describe a family builder:

`python -m llm_lab.cli.build_model --family qwen35 --describe-only`

Build model from family config `model` section:

`python -m llm_lab.cli.build_model --family-config llm_lab/configs/families/qwen35.example.json`

Run qwen inference:

`python -m llm_lab.cli.infer --family qwen35 --weights-dir weights/Qwen3.5-0.8B --prompt "Explain KV cache briefly."`

Run inference from a family config:

`python -m llm_lab.cli.infer --family-config llm_lab/configs/families/qwen35.example.json --prompt "Explain KV cache briefly."`

Run nanochat-style inference:

`python -m llm_lab.cli.infer --family nanochat_style --source sft --prompt "Hello!"`

Launch nanochat-style training:

`python -m llm_lab.cli.train --family nanochat_style -- --depth=12 --num-iterations=200`

Direct local nanochat-style train entrypoint:

`python -m llm_lab.nanochat_style.train --run nanochat_local --depth 12 --num-iterations 200 --device-type cpu`

Launch training using family config defaults:

`python -m llm_lab.cli.train --family-config llm_lab/configs/families/nanochat_style.example.json`

Launch training from JSON config:

`python -m llm_lab.cli.train --config llm_lab/configs/train/nanochat_d12_example.json`

Start local server:

`python -m llm_lab.cli.serve --family qwen35 --weights-dir weights/Qwen3.5-0.8B --host 127.0.0.1 --port 8000`

Start server from a family config:

`python -m llm_lab.cli.serve --family-config llm_lab/configs/families/qwen35.example.json --host 127.0.0.1 --port 8000`

Notes:

- tokenizer and chat template remain family-specific
- chat completion prompt rendering is family-specific
- qwen35 training is not wired yet in this first scaffold
- nanochat_style is fully decoupled from third_party
- example family configs are under `llm_lab/configs/families`
- example train configs are under `llm_lab/configs/train`
- model construction is exposed via per-family `ModelBuilder` adapters
