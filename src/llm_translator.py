"""llm_translator.py — Translate a gloss sequence into a natural-language sentence.

Backends (tried in priority order):

1. **llama.cpp** via ``llama-cpp-python`` — preferred for edge deployment.
2. **HuggingFace transformers** — fallback / development mode.
3. **Echo stub** — returns glosses joined by spaces; used when no model is
   present so the rest of the pipeline can run without any LLM.
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    "Translate the following ASL gloss sequence to fluent English.\n"
    "Glosses: {glosses}\n"
    "Translation:"
)


class LLMTranslator:
    """Translate a list of gloss strings into a natural-language sentence.

    Parameters
    ----------
    backend:
        Inference backend to use.  One of ``"llama_cpp"``,
        ``"transformers"``, or ``"stub"``.  When ``"llama_cpp"`` or
        ``"transformers"`` is requested but the library / model is
        unavailable, the translator silently falls back to the stub.
    model_path:
        Path to the model file.
        - For ``llama_cpp``: path to a ``.gguf`` quantised model.
        - For ``transformers``: HuggingFace model name or local directory.
    max_tokens:
        Maximum number of tokens to generate.
    """

    def __init__(
        self,
        backend: str = "stub",
        model_path: Optional[str] = None,
        max_tokens: int = 64,
    ) -> None:
        self.max_tokens = max_tokens
        self._backend = "stub"
        self._model = None

        if backend == "llama_cpp":
            self._try_init_llama_cpp(model_path)
        elif backend == "transformers":
            self._try_init_transformers(model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, glosses: List[str]) -> str:
        """Translate *glosses* into a natural-language sentence.

        Parameters
        ----------
        glosses:
            Ordered list of gloss label strings, e.g.
            ``["I", "WANT", "COFFEE"]``.

        Returns
        -------
        Translated sentence string.
        """
        if not glosses:
            return ""

        prompt = _PROMPT_TEMPLATE.format(glosses=" ".join(glosses))

        if self._backend == "llama_cpp":
            return self._generate_llama_cpp(prompt)
        if self._backend == "transformers":
            return self._generate_transformers(prompt)
        return self._echo_stub(glosses)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_init_llama_cpp(self, model_path: Optional[str]) -> None:
        """Attempt to load a llama.cpp model."""
        if not model_path:
            return
        try:
            from llama_cpp import Llama  # type: ignore[import]

            self._model = Llama(model_path=model_path, n_ctx=512, verbose=False)
            self._backend = "llama_cpp"
        except Exception as exc:
            logger.warning("llama.cpp model could not be loaded from %r (%s).", model_path, exc)
            self._model = None
        """Attempt to load a HuggingFace transformers model."""
        if not model_path:
            return
        try:
            from transformers import pipeline  # type: ignore[import]

            self._model = pipeline("text-generation", model=model_path)
            self._backend = "transformers"
        except Exception as exc:
            logger.warning(
                "transformers model could not be loaded from %r (%s).", model_path, exc
            )
            self._model = None
        """Run inference with llama.cpp and return the generated text."""
        output = self._model(prompt, max_tokens=self.max_tokens, stop=["\n"])
        return output["choices"][0]["text"].strip()

    def _generate_transformers(self, prompt: str) -> str:
        """Run inference with a HuggingFace pipeline and return the generated text."""
        output = self._model(prompt, max_new_tokens=self.max_tokens)
        return output[0]["generated_text"].replace(prompt, "").strip()

    @staticmethod
    def _echo_stub(glosses: List[str]) -> str:
        """Return glosses joined by spaces (no model required)."""
        return " ".join(glosses)
