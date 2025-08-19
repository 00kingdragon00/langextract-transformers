"""Provider implementation for HfTransformers."""

import os
import langextract as lx


@lx.providers.registry.register(r'^hftransformers', priority=10)
class HfTransformersLanguageModel(lx.inference.BaseLanguageModel):
    """LangExtract provider for HfTransformers.

    This provider handles model IDs matching: ['^hftransformers']
    """

    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        """Initialize the HfTransformers provider.

        Args:
            model_id: The model identifier.
            api_key: API key for authentication.
            **kwargs: Additional provider-specific parameters.
        """
        super().__init__()
        self.model_id = model_id
        self.api_key = api_key or os.environ.get('HFTRANSFORMERS_API_KEY')

        # self.client = YourClient(api_key=self.api_key)
        self._extra_kwargs = kwargs

    def infer(self, batch_prompts, **kwargs):
        """Run inference on a batch of prompts.

        Args:
            batch_prompts: List of prompts to process.
            **kwargs: Additional inference parameters.

        Yields:
            Lists of ScoredOutput objects, one per prompt.
        """
        for prompt in batch_prompts:
            # result = self.client.generate(prompt, **kwargs)
            result = f"Mock response for: {prompt[:50]}..."
            yield [lx.inference.ScoredOutput(score=1.0, output=result)]
