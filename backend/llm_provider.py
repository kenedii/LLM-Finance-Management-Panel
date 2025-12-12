# llm_provider.py
"""
LLM provider factory.
Supports: openai, deepseek (native or OpenAI-compatible), grok (xAI), gemini (google),
anthropic, and a local llama-cpp client.
"""

import os
from typing import Any

# Optional: import SDKs if installed. Wrap in try/except to avoid import-time failure.
try:
    import openai  # openai sdk can be used to call OpenAI-compatible endpoints (and DeepSeek if base set)
except Exception:
    openai = None

try:
    # xAI SDK (Grok)
    # Official project appears as `xai_sdk` in docs / github; some community wrappers exist too.
    from xai_sdk import Client as XAIClient
except Exception:
    XAIClient = None

# DeepSeek: some projects publish deepseek or deepseek-sdk
try:
    import deepseek  # optional native DeepSeek SDK
except Exception:
    deepseek = None

# Anthropic (optional)
try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None

# Google generative (optional)
try:
    from google.generativeai import Client as GoogleAIClient
except Exception:
    GoogleAIClient = None

# Local LLM via llama-cpp-python
try:
    from llama_cpp import Llama
except Exception:
    Llama = None


class LLMProvider:
    @staticmethod
    def get(provider: str) -> Any:
        p = (provider or "openai").strip().lower()

        if p == "openai":
            if openai is None:
                raise RuntimeError("openai package not installed. pip install openai")
            api_key = os.getenv("OPENAI_API_KEY")
            base = os.getenv("OPENAI_API_BASE") or None
            # Example minimal wrapper returning a function to call chat completions
            def openai_client():
                openai.api_key = api_key
                if base:
                    openai.api_base = base
                return openai
            return openai_client()

        if p == "deepseek":
            # Prefer native DeepSeek SDK if installed, otherwise use OpenAI-compatible endpoint
            ds_key = os.getenv("DEEPSEEK_API_KEY")
            ds_base = os.getenv("DEEPSEEK_API_BASE") or "https://api.deepseek.com"
            if deepseek is not None:
                # Example: deepseek.DeepSeekClient(api_key=...)  <-- adapt to actual SDK interface
                return deepseek.DeepSeekClient(api_key=ds_key)  # replace with correct init if different
            else:
                # fallback to OpenAI-compatible usage via openai SDK
                if openai is None:
                    raise RuntimeError("openai package required as fallback for DeepSeek. pip install openai")
                openai.api_key = ds_key
                openai.api_base = ds_base
                return openai

        if p == "grok" or p == "xai":
            # xAI Grok by default expects XAI_API_KEY env var. The docs show a Client in xai_sdk.
            xai_key = os.getenv("XAI_API_KEY") or os.getenv("XAI_API_KEY".upper())
            if XAIClient is None:
                raise RuntimeError("xai_sdk not installed. pip install xai-sdk (or xai-grok-sdk)")
            # Example usage (adapt to exact SDK):
            client = XAIClient(api_key=xai_key)
            return client

        if p in ("anthropic", "claude"):
            key = os.getenv("ANTHROPIC_API_KEY")
            if Anthropic is None:
                raise RuntimeError("anthropic package not installed. pip install anthropic")
            client = Anthropic(api_key=key)
            return client

        if p in ("gemini", "google"):
            key = os.getenv("GOOGLE_API_KEY")
            if GoogleAIClient is None:
                raise RuntimeError("google generativeai package not installed. pip install google-generativeai")
            client = GoogleAIClient(api_key=key)
            return client

        if p in ("local", "llama"):
            model_path = os.getenv("LOCAL_MODEL_PATH")
            if Llama is None:
                raise RuntimeError("llama-cpp-python not installed. pip install llama-cpp-python")
            if not model_path:
                raise RuntimeError("LOCAL_MODEL_PATH not set in environment")
            return Llama(model_path=model_path)

        raise ValueError(f"Unsupported provider: {provider}")
