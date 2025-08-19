"""Custom provider implementation for HuggingFace models used by LangExtract.

This provider integrates `transformers.pipeline` into LangExtract via the
provider registry. It supports common generation-style tasks and robustly
extracts generated text from various pipeline outputs.

Production notes:
- Avoids noisy stdout prints (uses logging instead).
- Guards against common device/device_map conflicts.
- Returns valid JSON consistently (fallback) to keep LangExtract stable.
"""

from typing import List, Any, Optional
import os
import logging
import json
import langextract as lx
from transformers import pipeline


logger = logging.getLogger(__name__)


@lx.providers.registry.register(r"^(hf|huggingface)(:|/)?", priority=10)
class HfTransformersLanguageModel(lx.inference.BaseLanguageModel):
    """LangExtract provider for HuggingFace models.

    Examples of model_id this provider accepts:
      - hf:meta-llama/Llama-3-8b
      - huggingface:google/gemma-3-4b-it
      - hf/meta-llama/Llama-3-8b
    """

    def __init__(self, model_id: str, **kwargs):
        """Initialize the HuggingFace provider.

        Args:
            model_id: The model identifier (may be prefixed with hf: or huggingface:).
            **kwargs: pipeline/task/device and generation parameters.
        """
        super().__init__()
        raw_kwargs = dict(kwargs)
        # Drop non-pipeline args sometimes forwarded by LangExtract
        raw_kwargs.pop("format_type", None)
        raw_kwargs.pop("max_workers", None)
        # Avoid accelerate conflict when both are provided
        if "device_map" in raw_kwargs and "device" in raw_kwargs:
            raw_kwargs.pop("device", None)
        self.task = raw_kwargs.pop("task", "text-generation")
        self.model_id = self._normalize_model_id(model_id)
        self.generator = pipeline(
            task=self.task,
            model=self.model_id,
            **raw_kwargs,
        )

    def infer(self, batch_prompts: List[str], **kwargs):
        """Run inference on a batch of prompts.

        Accepts extra kwargs from LangExtract but forwards only supported
        generation parameters to the transformers pipeline.
        """
        # Forward only supported generation kwargs
        gen_allowed = {
            "max_new_tokens",
            "min_new_tokens",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "no_repeat_ngram_size",
            "do_sample",
            "num_beams",
            "num_return_sequences",
            "return_full_text",
            "max_length",
            "eos_token_id",
            "pad_token_id",
            "batch_size",
        }
        gen_kwargs = {k: v for k, v in kwargs.items() if k in gen_allowed}
        # Configurable default via env var; keep generous default for extraction tasks
        default_max_new = int(os.getenv("LX_HF_MAX_NEW_TOKENS", "2048"))
        gen_kwargs.setdefault("max_new_tokens", default_max_new)
        gen_kwargs.setdefault("return_full_text", False)

        for prompt in batch_prompts:
            try:
                outputs = self.generator(prompt, **gen_kwargs)
                text = self._extract_text(outputs)
            except Exception as e:
                logger.exception("HF provider generation failed: %s", e)
                yield [
                    lx.inference.ScoredOutput(score=1.0, output=self._fallback_json())
                ]
                continue

            logger.debug("HF Provider raw output: %s", text)
            # Clip to first balanced JSON block to prevent "Extra data" parse errors
            clipped = self._json_clip(text)
            # Normalize schema to what LangExtract expects if needed
            normalized = self._normalize_for_langextract(clipped)
            if clipped != text:
                logger.debug("HF Provider clipped JSON: %s", clipped)
            if normalized and normalized != clipped:
                logger.debug("HF Provider normalized JSON: %s", normalized)

            final_out = normalized if normalized else clipped
            if not self._looks_like_json(final_out):
                # As a production guardrail, return an empty extractions array
                final_out = self._fallback_json()
            yield [lx.inference.ScoredOutput(score=1.0, output=final_out)]

    def _extract_text(self, outputs: Any) -> str:
        """Extract generated text from transformers pipeline outputs robustly."""
        try:
            if isinstance(outputs, list) and outputs:
                first = outputs[0]
                if isinstance(first, dict):
                    return (
                        first.get("generated_text")
                        or first.get("summary_text")
                        or first.get("translation_text")
                        or first.get("text")
                        or str(first)
                    )
                return str(first)
            if isinstance(outputs, dict):
                return (
                    outputs.get("generated_text")
                    or outputs.get("summary_text")
                    or outputs.get("translation_text")
                    or outputs.get("text")
                    or str(outputs)
                )
            return str(outputs)
        except Exception:
            return str(outputs)

    def _json_clip(self, text: str) -> str:
        """Return only the first balanced JSON object/array substring if present.

        Helps LangExtract json.loads avoid "Extra data" when models add prose
        before/after the JSON.
        """
        if not text:
            return text
        s = self._strip_code_fences(str(text).strip())
        # Find first JSON opening
        start = None
        for i, ch in enumerate(s):
            if ch in "[{":
                start = i
                break
        if start is None:
            return s
        stack = []
        for j in range(start, len(s)):
            ch = s[j]
            if ch in "[{":
                stack.append(ch)
            elif ch in "]}":
                if not stack:
                    continue
                top = stack[-1]
                if (top == "[" and ch == "]") or (top == "{" and ch == "}"):
                    stack.pop()
                    if not stack:
                        return s[start : j + 1]
        return s

    @staticmethod
    def _strip_code_fences(s: str) -> str:
        # Remove Markdown code fences if present
        if s.startswith("```"):
            parts = s.split("\n", 1)
            s = parts[1] if len(parts) > 1 else ""
            if s.rstrip().endswith("```"):
                s = s[: s.rfind("```")]
        return s

    def _normalize_for_langextract(self, text: str) -> Optional[str]:
        """Convert model JSON into LangExtract's expected schema if needed.

        Expected: {"extractions": [{"extraction_class": str,
                                      "extraction_text": str,
                                      "attributes": {...}}]}
        Model often returns: {"extractions": [{"task": str,
                                               "task_attributes": {...}}]}
        """
        try:
            obj = json.loads(text)
            if not isinstance(obj, dict):
                return None
            items = obj.get("extractions")
            if not isinstance(items, list) or not items:
                return None
            # Check if already in expected schema
            sample = items[0]
            if isinstance(sample, dict) and {
                "extraction_class",
                "extraction_text",
                "attributes",
            }.issubset(sample.keys()):
                return None
            # Transform task/task_attributes -> extraction schema
            normalized_items = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                extraction_text = (
                    it.get("task") or it.get("extraction_text") or it.get("text") or ""
                )
                attributes = it.get("task_attributes") or it.get("attributes") or {}
                normalized_items.append(
                    {
                        "extraction_class": "task",
                        "extraction_text": extraction_text,
                        "attributes": attributes,
                    }
                )
            normalized = {"extractions": normalized_items}
            return json.dumps(normalized, ensure_ascii=False)
        except Exception:
            return None

    @staticmethod
    def _looks_like_json(s: str) -> bool:
        if not s:
            return False
        t = s.strip()
        if not (
            (t.startswith("{") and t.endswith("}"))
            or (t.startswith("[") and t.endswith("]"))
        ):
            return False
        try:
            json.loads(t)
            return True
        except Exception:
            return False

    @staticmethod
    def _fallback_json() -> str:
        return json.dumps({"extractions": []}, ensure_ascii=False)

    @staticmethod
    def _normalize_model_id(model_id: str) -> str:
        prefixes = ("hf:", "huggingface:", "hf/", "huggingface/")
        for p in prefixes:
            if model_id.startswith(p):
                return model_id[len(p) :]
        return model_id
