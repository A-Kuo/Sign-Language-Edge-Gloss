"""
src/llm_translator.py
──────────────────────
Stage 3: Raw gloss tokens → Fluid English sentence

Supported backends (auto-detected in priority order):
  1. llama-cpp-python  — fastest on CPU, supports GGUF quantised models
  2. transformers      — standard HuggingFace pipeline (fp16/fp32)
  3. Stub              — returns gloss tokens verbatim (offline fallback)

Recommended model for hackathon
─────────────────────────────────
  Meta-Llama-3.2-3B-Instruct-Q4_K_M.gguf   (~2 GB, ~5 tok/s on laptop CPU)

  Download:
    huggingface-cli download \
      bartowski/Llama-3.2-3B-Instruct-GGUF \
      Meta-Llama-3.2-3B-Instruct-Q4_K_M.gguf \
      --local-dir models/

  Then set LLAMA_GGUF_PATH="models/Meta-Llama-3.2-3B-Instruct-Q4_K_M.gguf"

Quantisation note
──────────────────
Q4_K_M is a 4-bit NormalFloat quantisation (similar to NF4 used in QLoRA).
Each weight w ∈ ℝ is mapped to the nearest element in a 16-value codebook
derived from the normal distribution N(0,1), with per-block scale factors:

  w_q = argmin_{c ∈ C₁₆} |w / s_block − c|

This halves VRAM vs FP16 with <1 % perplexity degradation on most benchmarks —
the ideal tradeoff for edge inference without an NPU available.
"""

from __future__ import annotations

import os
import time

# ─── System prompt — keep it minimal; the model needs zero-shot reliability ─
_SYSTEM_PROMPT = (
    "You are an expert American Sign Language interpreter. "
    "I will give you raw ASL gloss tokens in ALL CAPS. "
    "Convert them into a single, grammatically correct, conversational English sentence. "
    "Output ONLY the translated sentence — no explanation, no punctuation commentary."
)


class LLMTranslator:
    """
    Local LLM wrapper.  Tries llama.cpp first, then HuggingFace transformers.

    Parameters
    ----------
    model_path : str | None
        Path to a GGUF model file.  If None, reads the LLAMA_GGUF_PATH
        environment variable, then falls back to the transformers backend.
    disabled : bool
        When True the translator is a no-op (useful for --no-llm flag).
    max_new_tokens : int
        Generation budget.  ASL sentences rarely exceed 30 tokens.
    """

    def __init__(
        self,
        model_path: str | None = None,
        disabled:   bool       = False,
        max_new_tokens: int    = 60,
    ):
        self.disabled       = disabled
        self.max_new_tokens = max_new_tokens
        self._backend       = None
        self._backend_type  = "stub"

        if disabled:
            print("[LLMTranslator] Disabled — gloss passthrough mode.")
            return

        gguf_path = model_path or os.getenv("LLAMA_GGUF_PATH")
        if gguf_path and os.path.isfile(gguf_path):
            self._load_llamacpp(gguf_path)
        else:
            self._load_transformers()

    # ──────────────────────────────────────────────────────────────────────
    def _load_llamacpp(self, path: str) -> None:
        try:
            from llama_cpp import Llama   # type: ignore
            print(f"[LLMTranslator] Loading GGUF via llama-cpp: {path}")
            t0 = time.time()
            self._backend = Llama(
                model_path   = path,
                n_ctx        = 512,
                n_threads    = os.cpu_count() or 4,
                verbose      = False,
            )
            print(f"[LLMTranslator] llama.cpp ready ({time.time()-t0:.1f}s)")
            self._backend_type = "llamacpp"
        except ImportError:
            print("[LLMTranslator] llama-cpp-python not installed — falling back.")
            self._load_transformers()

    def _load_transformers(self) -> None:
        hf_model = os.getenv(
            "LLAMA_HF_MODEL",
            "meta-llama/Llama-3.2-3B-Instruct"
        )
        try:
            from transformers import pipeline as hf_pipeline   # type: ignore
            import torch
            print(f"[LLMTranslator] Loading HuggingFace model: {hf_model}")
            print("                (This may take a minute on first run …)")
            t0     = time.time()
            device = 0 if torch.cuda.is_available() else -1
            self._backend = hf_pipeline(
                "text-generation",
                model     = hf_model,
                device    = device,
                torch_dtype = torch.float16 if device == 0 else torch.float32,
            )
            print(f"[LLMTranslator] HuggingFace pipeline ready ({time.time()-t0:.1f}s)")
            self._backend_type = "transformers"
        except Exception as e:
            print(f"[LLMTranslator] Could not load any LLM ({e}).")
            print("                Running in stub mode — glosses returned as-is.")

    # ──────────────────────────────────────────────────────────────────────
    def translate(self, glosses: list[str]) -> str:
        """
        Convert a list of gloss tokens to a natural English sentence.

        Parameters
        ----------
        glosses : list[str]   e.g. ["ME", "STORE", "GO", "PAST"]

        Returns
        -------
        str  e.g. "I went to the store."
        """
        if self.disabled or self._backend_type == "stub":
            return " ".join(glosses)

        gloss_str = " ".join(glosses)
        prompt    = f"Gloss: {gloss_str}\nEnglish:"

        try:
            if self._backend_type == "llamacpp":
                return self._infer_llamacpp(prompt)
            elif self._backend_type == "transformers":
                return self._infer_transformers(prompt)
        except Exception as e:
            print(f"[LLMTranslator] Inference error: {e}")
            return gloss_str

        return gloss_str

    # ──────────────────────────────────────────────────────────────────────
    def _infer_llamacpp(self, user_prompt: str) -> str:
        """
        Uses the llama.cpp ChatML / Llama-3 message format.
        The model receives: [system] + [user: gloss] → [assistant: English]
        """
        messages = [
            {"role": "system",  "content": _SYSTEM_PROMPT},
            {"role": "user",    "content": user_prompt},
        ]
        response = self._backend.create_chat_completion(
            messages        = messages,
            max_tokens      = self.max_new_tokens,
            temperature     = 0.2,      # low temp → deterministic translation
            repeat_penalty  = 1.1,
        )
        text = response["choices"][0]["message"]["content"].strip()
        return text

    def _infer_transformers(self, user_prompt: str) -> str:
        full_prompt = f"<|system|>\n{_SYSTEM_PROMPT}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
        out = self._backend(
            full_prompt,
            max_new_tokens  = self.max_new_tokens,
            do_sample       = True,
            temperature     = 0.2,
            pad_token_id    = self._backend.tokenizer.eos_token_id,
        )
        generated = out[0]["generated_text"]
        # Strip the prompt prefix
        if "<|assistant|>" in generated:
            generated = generated.split("<|assistant|>")[-1]
        return generated.strip()
