"""
llm_translator.py – Local LLM gloss-to-sentence translator.

Converts a sequence of sign-language gloss tokens (e.g. ["I", "WANT", "WATER"])
into a fluent natural-language sentence using a locally-running language model.

Two inference backends are supported:
    "transformers"  – HuggingFace Transformers (default, CPU/GPU)
    "llama_cpp"     – llama.cpp via the llama-cpp-python binding (quantised,
                      suitable for edge/Qualcomm deployment)

A ``StubTranslator`` is also provided for offline testing without any model
weights.
"""

from __future__ import annotations

from typing import List, Optional


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

class _TransformersBackend:
    """HuggingFace Transformers inference backend.

    TODO: load a real seq2seq or causal LM and implement ``generate``.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._tokenizer = None
        self._model = None
        self._load()

    def _load(self) -> None:
        """Load tokenizer and model from *model_path*.

        TODO: replace stub with real load::

            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            self._model.eval()
        """
        # Placeholder: no model loaded
        pass

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        """Return a generated sentence for *prompt*.

        TODO: replace stub with real inference::

            import torch
            inputs = self._tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                ids = self._model.generate(
                    **inputs, max_new_tokens=max_new_tokens
                )
            return self._tokenizer.decode(ids[0], skip_special_tokens=True)
        """
        # Placeholder: echo the gloss tokens as a sentence
        return prompt


class _LlamaCppBackend:
    """llama.cpp inference backend via ``llama-cpp-python``.

    TODO: load a GGUF model and implement ``generate``.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._llm = None
        self._load()

    def _load(self) -> None:
        """Load the GGUF model from *model_path*.

        TODO: replace stub with real load::

            from llama_cpp import Llama
            self._llm = Llama(model_path=self.model_path, n_ctx=512)
        """
        # Placeholder: no model loaded
        pass

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        """Return a generated sentence for *prompt*.

        TODO: replace stub with real inference::

            output = self._llm(prompt, max_tokens=max_new_tokens)
            return output["choices"][0]["text"].strip()
        """
        # Placeholder: echo the gloss tokens as a sentence
        return prompt


class _StubBackend:
    """No-op backend for unit testing without model weights."""

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:  # noqa: ARG002
        return f"[STUB] {prompt}"


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

class LLMTranslator:
    """Translate a gloss sequence into a natural-language sentence.

    Usage::

        translator = LLMTranslator(model_path="models/llm")
        sentence = translator.translate(["I", "WANT", "WATER"])
        # → "I would like some water, please."

    Args:
        model_path: Filesystem path to the local model weights directory
            (Transformers) or GGUF file (llama.cpp).  Pass *None* to use the
            no-op stub backend (useful for testing).
        backend: ``"transformers"`` (default), ``"llama_cpp"``, or ``"stub"``.
        prompt_template: Python format string used to wrap gloss tokens before
            sending to the LLM.  Must contain ``{glosses}``.
        max_new_tokens: Maximum number of tokens the LLM may generate.
    """

    _DEFAULT_PROMPT = (
        "Translate the following sign-language gloss sequence into a fluent "
        "English sentence.\nGlosses: {glosses}\nSentence:"
    )

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = "transformers",
        prompt_template: Optional[str] = None,
        max_new_tokens: int = 64,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self._prompt_template = prompt_template or self._DEFAULT_PROMPT

        if model_path is None or backend == "stub":
            self._backend: _StubBackend | _TransformersBackend | _LlamaCppBackend = (
                _StubBackend()
            )
        elif backend == "transformers":
            self._backend = _TransformersBackend(model_path)
        elif backend == "llama_cpp":
            self._backend = _LlamaCppBackend(model_path)
        else:
            raise ValueError(
                f"Unknown LLM backend '{backend}'. "
                "Choose 'transformers', 'llama_cpp', or 'stub'."
            )

    def translate(self, glosses: List[str]) -> str:
        """Convert *glosses* to a natural-language sentence.

        Args:
            glosses: Ordered list of gloss token strings.

        Returns:
            A fluent English sentence string.
        """
        if not glosses:
            return ""
        gloss_str = " ".join(glosses)
        prompt = self._prompt_template.format(glosses=gloss_str)
        return self._backend.generate(prompt, max_new_tokens=self.max_new_tokens)
