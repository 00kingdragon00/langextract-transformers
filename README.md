## LangExtract Hugging Face Transformers Provider

A LangExtract provider plugin that integrates Hugging Face Transformers via `transformers.pipeline`.

### Installation

```bash
pip install -e .
```

### Supported model IDs (prefixes)

This provider is selected for model IDs beginning with any of the following:

- `hf:` e.g. `hf:meta-llama/Llama-3-8b`
- `huggingface:` e.g. `huggingface:google/gemma-2-2b-it`
- `hf/` or `huggingface/` also work

Only the prefix is used for routing; it is stripped before loading the model in Transformers.

### Usage with LangExtract

Minimal example:

```python
import langextract as lx

result = lx.extract(
        text="Your document here",
        model_id="hf:distilbert/distilgpt2",
        prompt_description="Extract entities",
        examples=[
                {"input": "John lives in Berlin.", "output": {"extractions": [{"task": "location", "task_attributes": {"city": "Berlin"}}]}},
        ],
)
```

Advanced usage (forwarding generation kwargs to Transformers):

```python
from langextract import factory

config = factory.ModelConfig(
        model_id="hf:meta-llama/Llama-3-8b",
        provider="HfTransformersLanguageModel",
        # Forwarded to transformers.pipeline() during init
        task="text-generation",
        device_map="auto",  # or device=0 (GPU index). If both provided, device is ignored.
        torch_dtype="auto",
)

model = factory.create_model(config)

prompts = ["Return JSON with an 'extractions' array."]
outputs = list(model.infer(prompts, max_new_tokens=512, temperature=0.2, top_p=0.9))
```

### Generation parameters supported

The following kwargs are forwarded to `pipeline(...)(prompt, **kwargs)` when calling `infer(...)`:

- `max_new_tokens`, `min_new_tokens`, `max_length`
- `temperature`, `top_k`, `top_p`, `repetition_penalty`, `no_repeat_ngram_size`
- `do_sample`, `num_beams`, `num_return_sequences`, `return_full_text`
- `eos_token_id`, `pad_token_id`, `batch_size`

Defaults:

- `max_new_tokens`: Set from `LX_HF_MAX_NEW_TOKENS` env var (default `2048`).
- `return_full_text`: Defaults to `False`.

### Environment variables

- `LX_HF_MAX_NEW_TOKENS` — Overrides the default `max_new_tokens` used by the provider when not explicitly set.
    - PowerShell (current session):
        ```powershell
        $env:LX_HF_MAX_NEW_TOKENS = "1024"
        ```
- Standard Hugging Face variables like `HF_HOME` or `TRANSFORMERS_CACHE` can be used to control cache locations (handled by Transformers, not by this plugin).

### Device configuration notes

- You may pass `device` (e.g., `0` for the first CUDA GPU) or `device_map` (e.g., `"auto"`). If both are provided, this provider drops `device` to avoid conflicts with Accelerate.
- Other `transformers.pipeline` kwargs like `torch_dtype` are forwarded unchanged.

### Output schema guardrails

- The provider clips to the first balanced JSON object/array in the model output to reduce `json.loads` errors.
- If the model returns an alternate schema like `{ "extractions": [{"task": ..., "task_attributes": {...}}] }`, the provider normalizes it to LangExtract’s expected schema:
    `{ "extractions": [{"extraction_class": "task", "extraction_text": "...", "attributes": {...}}] }`.
- If parsing fails, a safe fallback `{ "extractions": [] }` is returned so pipelines remain stable.

### Development

1. Install in development mode: `pip install -e .`
2. Run tests: `python test_plugin.py`
3. Build package: `python -m build`
4. Publish to PyPI: `twine upload dist/*`

### License

Apache License 2.0
