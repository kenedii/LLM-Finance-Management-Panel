# llm_provider.py
"""
LLM provider factory.
Supports: openai, deepseek (OpenAI-compatible), grok (xAI), gemini (google),
anthropic, and local PyTorch transformer models.
"""

import os
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------------------------------------
# Optional imports: safely wrapped
# -------------------------------------------------------------
try:
    from openai import OpenAI
except:
    OpenAI = None

try:
    from xai_sdk import Client as XAIClient
except:
    XAIClient = None

try:
    from anthropic import Anthropic
except:
    Anthropic = None

try:
    from google.generativeai import Client as GoogleAIClient
except:
    GoogleAIClient = None


# -------------------------------------------------------------
#  LOCAL PYTORCH MODEL WRAPPER
# -------------------------------------------------------------
class LocalPyTorchLLM:
    def __init__(self, model_path: str):
        if not os.path.isdir(model_path) and not os.path.isfile(model_path):
            raise RuntimeError(f"Local model path not found: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def chat(self, messages: List[Dict[str, str]], max_tokens=256) -> str:
        """
        messages = [{ "role": "user", "content": "..." }]
        Only the last message content is used for prompting.
        """
        prompt = messages[-1]["content"]

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


# -------------------------------------------------------------
# PROVIDER FACTORY
# -------------------------------------------------------------
class LLMProvider:
    @staticmethod
    def get(provider: str) -> Any:
        p = (provider or "openai").strip().lower()

        # ---------------------------------------------------------
        # OPENAI (including DeepSeek-compatible mode if api_base set)
        # ---------------------------------------------------------
        if p == "openai":
            if OpenAI is None:
                raise RuntimeError("openai package not installed. pip install openai")

            api_key = os.getenv("OPENAI_API_KEY")
            base = os.getenv("OPENAI_API_BASE") or None

            return OpenAI(api_key=api_key, base_url=base)

        # ---------------------------------------------------------
        # DEEPSEEK (OpenAI-compatible endpoint only)
        # ---------------------------------------------------------
        if p == "deepseek":
            if OpenAI is None:
                raise RuntimeError("openai package required to call DeepSeek")

            ds_key = os.getenv("DEEPSEEK_API_KEY")
            ds_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")

            return OpenAI(
                api_key=ds_key,
                base_url=ds_base
            )

        # ---------------------------------------------------------
        # GROK (xAI)
        # ---------------------------------------------------------
        if p in ("grok", "xai"):
            key = os.getenv("XAI_API_KEY")
            if XAIClient is None:
                raise RuntimeError("xai_sdk not installed. pip install xai-sdk")
            return XAIClient(api_key=key)

        # ---------------------------------------------------------
        # ANTHROPIC (Claude)
        # ---------------------------------------------------------
        if p in ("anthropic", "claude"):
            key = os.getenv("ANTHROPIC_API_KEY")
            if Anthropic is None:
                raise RuntimeError("anthropic package not installed. pip install anthropic")
            return Anthropic(api_key=key)

        # ---------------------------------------------------------
        # GEMINI (Google)
        # ---------------------------------------------------------
        if p in ("gemini", "google"):
            key = os.getenv("GOOGLE_API_KEY")
            if GoogleAIClient is None:
                raise RuntimeError("google-generativeai not installed. pip install google-generativeai")
            return GoogleAIClient(api_key=key)

        # ---------------------------------------------------------
        # LOCAL PYTORCH TRANSFORMERS MODEL
        # ---------------------------------------------------------
        if p in ("local", "pytorch", "hf", "transformers"):
            model_path = os.getenv("LOCAL_MODEL_PATH")
            if not model_path:
                raise RuntimeError("LOCAL_MODEL_PATH not set in .env")
            return LocalPyTorchLLM(model_path)

        # ---------------------------------------------------------
        # Unknown provider
        # ---------------------------------------------------------
        raise ValueError(f"Unsupported provider: {provider}")
