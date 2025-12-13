# llm_provider.py
"""
LLM provider factory.
Supports: openai, deepseek (OpenAI-compatible), grok (xAI), gemini (google),
anthropic, and local PyTorch transformer models.
"""

import os
from typing import Any, Dict, List, Optional

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
    import google.generativeai as genai
except:
    genai = None


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
#  PROVIDER WRAPPERS (Unified chat interface)
# -------------------------------------------------------------
class OpenAIWrapper:
    def __init__(self, client: Any):
        self.client = client

    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, max_tokens: int = 512, temperature: float = 0.7) -> str:
        # Convert to OpenAI style messages and call chat.completions
        m = messages
        if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
            model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            comp = self.client.chat.completions.create(model=model_name, messages=m, temperature=temperature)
            return comp.choices[0].message.content if comp and getattr(comp, "choices", None) else ""
        raise RuntimeError("OpenAI client missing chat.completions interface")


class DeepSeekWrapper(OpenAIWrapper):
    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, max_tokens: int = 512, temperature: float = 0.7) -> str:
        m = messages
        if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
            model_name = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            comp = self.client.chat.completions.create(model=model_name, messages=m, temperature=temperature)
            return comp.choices[0].message.content if comp and getattr(comp, "choices", None) else ""
        raise RuntimeError("DeepSeek client missing chat.completions interface")


class AnthropicWrapper:
    def __init__(self, client: Any):
        self.client = client

    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, max_tokens: int = 512, temperature: float = 0.7) -> str:
        mdl = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-202410")
        # Anthropics expects messages list; convert if needed
        try:
            resp = self.client.messages.create(
                model=mdl,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages]
            )
            # Response content is a list of blocks
            blocks = getattr(resp, "content", [])
            if isinstance(blocks, list) and blocks:
                # Combine any text blocks
                texts = []
                for b in blocks:
                    t = getattr(b, "text", None) or (b.get("text") if isinstance(b, dict) else None)
                    if t:
                        texts.append(t)
                return "\n".join(texts)
            return getattr(resp, "output_text", "")
        except Exception as e:
            return f"Anthropic error: {e}"


class GeminiWrapper:
    def __init__(self, api_key: str):
        if genai is None:
            raise RuntimeError("google-generativeai not installed. pip install google-generativeai")
        genai.configure(api_key=api_key)

    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, max_tokens: int = 512, temperature: float = 0.7) -> str:
        mdl = model or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        try:
            model_obj = genai.GenerativeModel(mdl)
            # Concatenate messages into a single prompt (Gemini supports multi-turn, but keep simple)
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            resp = model_obj.generate_content(prompt, generation_config={"temperature": temperature, "max_output_tokens": max_tokens})
            # Extract text
            return getattr(resp, "text", "")
        except Exception as e:
            return f"Gemini error: {e}"


class XAIWrapper:
    def __init__(self, client: Any):
        self.client = client

    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, max_tokens: int = 512, temperature: float = 0.7) -> str:
        mdl = model or os.getenv("XAI_MODEL", "grok-beta")
        try:
            # Attempt a generic interface; if SDK differs, catch and report error
            # Many SDKs use an OpenAI-like interface; try fallbacks
            if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
                comp = self.client.chat.completions.create(model=mdl, messages=messages, temperature=temperature)
                return comp.choices[0].message.content if comp and getattr(comp, "choices", None) else ""
            # Fallback: try a hypothetical messages.create
            if hasattr(self.client, "messages") and hasattr(self.client.messages, "create"):
                resp = self.client.messages.create(model=mdl, messages=messages, max_tokens=max_tokens, temperature=temperature)
                return getattr(resp, "text", "")
            return "xAI client does not expose a known chat interface."
        except Exception as e:
            return f"xAI error: {e}"


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
            client = OpenAI(api_key=api_key, base_url=base)
            return OpenAIWrapper(client)

        # ---------------------------------------------------------
        # DEEPSEEK (OpenAI-compatible endpoint only)
        # ---------------------------------------------------------
        if p == "deepseek":
            if OpenAI is None:
                raise RuntimeError("openai package required to call DeepSeek")
            ds_key = os.getenv("DEEPSEEK_API_KEY")
            ds_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
            client = OpenAI(api_key=ds_key, base_url=ds_base)
            return DeepSeekWrapper(client)

        # ---------------------------------------------------------
        # GROK (xAI)
        # ---------------------------------------------------------
        if p in ("grok", "xai"):
            key = os.getenv("XAI_API_KEY")
            if XAIClient is None:
                raise RuntimeError("xai_sdk not installed. pip install xai-sdk")
            client = XAIClient(api_key=key)
            return XAIWrapper(client)

        # ---------------------------------------------------------
        # ANTHROPIC (Claude)
        # ---------------------------------------------------------
        if p in ("anthropic", "claude"):
            key = os.getenv("ANTHROPIC_API_KEY")
            if Anthropic is None:
                raise RuntimeError("anthropic package not installed. pip install anthropic")
            client = Anthropic(api_key=key)
            return AnthropicWrapper(client)

        # ---------------------------------------------------------
        # GEMINI (Google)
        # ---------------------------------------------------------
        if p in ("gemini", "google"):
            key = os.getenv("GOOGLE_API_KEY")
            if genai is None:
                raise RuntimeError("google-generativeai not installed. pip install google-generativeai")
            return GeminiWrapper(api_key=key)

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
