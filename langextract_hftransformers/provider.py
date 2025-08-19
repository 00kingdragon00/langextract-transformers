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

    @property
    def requires_fence_output(self) -> bool:
        """HuggingFace JSON mode returns raw JSON without fences."""
        return False

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
                text = outputs[0]["generated_text"]

                # Clip to first balanced JSON block to prevent "Extra data" parse errors
                clipped = self._json_clip(text)
                # Normalize schema to what LangExtract expects if needed
                normalized = self._normalize_for_langextract(clipped)

                final_out = normalized if normalized else clipped
                if not self._looks_like_json(final_out):
                    # As a production guardrail, return an empty extractions array
                    final_out = self._fallback_json()

                yield [lx.inference.ScoredOutput(score=1.0, output=final_out)]
            except Exception as e:
                logger.exception("HF provider generation failed: %s", e)
                yield [
                    lx.inference.ScoredOutput(score=1.0, output=self._fallback_json())
                ]
                continue

    def _json_clip(self, text: str) -> str:
        """Return only the first balanced JSON object/array substring if present.

        Helps LangExtract json.loads avoid "Extra data" when models add prose
        before/after the JSON. Ensures we return a dict, not an array.
        """
        if not text:
            return text
        s = self._strip_code_fences(str(text).strip())

        # Find first JSON opening (prefer object over array)
        start = None
        for i, ch in enumerate(s):
            if ch == "{":  # Prefer objects
                start = i
                break

        # If no object found, look for array
        if start is None:
            for i, ch in enumerate(s):
                if ch == "[":
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
                        json_content = s[start : j + 1]
                        # If we got an array, wrap it in extractions object
                        if json_content.startswith("["):
                            try:
                                import json

                                array_data = json.loads(json_content)
                                return json.dumps(
                                    {"extractions": array_data}, ensure_ascii=False
                                )
                            except:
                                return json_content
                        return json_content
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
        Model often returns: {"task": str, "task_attributes": {...}}
        """
        try:
            obj = json.loads(text)
            if not isinstance(obj, dict):
                return None

            # Check if it's already in the correct format
            if "extractions" in obj and isinstance(obj["extractions"], list):
                return None  # Already correct format

            # Handle single task object format: {"task": "...", "task_attributes": {...}}
            if "task" in obj and "task_attributes" in obj:
                extraction = {
                    "extraction_class": "task",
                    "extraction_text": obj["task"],
                    "attributes": obj["task_attributes"],
                }
                normalized = {"extractions": [extraction]}
                return json.dumps(normalized, ensure_ascii=False)

            # Handle array of tasks: [{"task": "...", "task_attributes": {...}}, ...]
            if isinstance(obj, list):
                normalized_items = []
                for item in obj:
                    if isinstance(item, dict):
                        task_text = (
                            item.get("task")
                            or item.get("extraction_text")
                            or item.get("text")
                            or ""
                        )
                        attributes = (
                            item.get("task_attributes") or item.get("attributes") or {}
                        )
                        normalized_items.append(
                            {
                                "extraction_class": "task",
                                "extraction_text": task_text,
                                "attributes": attributes,
                            }
                        )
                if normalized_items:
                    normalized = {"extractions": normalized_items}
                    return json.dumps(normalized, ensure_ascii=False)

            # Handle old array format in "extractions" key but wrong schema
            items = obj.get("extractions")
            if isinstance(items, list) and items:
                # Check if already in expected schema
                sample = items[0]
                if isinstance(sample, dict) and {
                    "extraction_class",
                    "extraction_text",
                    "attributes",
                }.issubset(sample.keys()):
                    return None  # Already correct

                # Transform task/task_attributes -> extraction schema
                normalized_items = []
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    extraction_text = (
                        it.get("task")
                        or it.get("extraction_text")
                        or it.get("text")
                        or ""
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
