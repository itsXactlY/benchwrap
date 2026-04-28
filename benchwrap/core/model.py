"""
Model backends for benchwrap.
One interface: generate(prompt) -> Prediction. Any model, any provider.
"""

from abc import ABC, abstractmethod
from typing import Optional
import json
import time
import urllib.request
import urllib.error

from benchwrap.core.types import Prompt, Prediction


class ModelBackend(ABC):
    """Base class for all model backends.
    
    The entire contract: given a Prompt, return a Prediction.
    No hidden prompt manipulation. No answer injection.
    What goes in is what the adapter formatted. What comes out is what the model generated.
    """

    @abstractmethod
    def generate(self, prompt: Prompt, **kwargs) -> Prediction:
        """Generate a response for the given prompt.
        
        Args:
            prompt: Formatted prompt from the adapter
            **kwargs: Backend-specific options (temperature, max_tokens, etc.)
        
        Returns:
            Prediction with raw text + metadata.
        """
        ...

    def generate_batch(
        self, prompts: list[Prompt], **kwargs
    ) -> list[Prediction]:
        """Generate responses for multiple prompts.
        
        Default: sequential. Backends can override for parallel/batch inference.
        """
        return [self.generate(p, **kwargs) for p in prompts]

    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g. 'ollama', 'openai', 'hf')"""
        ...

    @abstractmethod
    def model_id(self) -> str:
        """Model identifier (e.g. 'openhermes:7b-v2.5', 'gpt-4')"""
        ...

    def diagnose(self) -> dict:
        """Check if this backend is available and working."""
        return {"backend": self.name(), "model": self.model_id(), "status": "unknown"}


class OllamaBackend(ModelBackend):
    """Ollama local inference via its native API.
    
    Uses Ollama's /api/chat endpoint (NOT OpenAI-compatible /v1).
    This avoids issues with reasoning models returning empty content
    via the OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        model: str = "openhermes:7b-v2.5",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        num_predict: int = 512,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.num_predict = num_predict

    def name(self) -> str:
        return "ollama"

    def model_id(self) -> str:
        return self.model

    def generate(self, prompt: Prompt, **kwargs) -> Prediction:
        temperature = kwargs.get("temperature", self.temperature)
        num_predict = kwargs.get("max_tokens", self.num_predict)

        # Build messages for Ollama
        messages = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        messages.extend(prompt.messages)

        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())
        except urllib.error.URLError as e:
            return Prediction(
                text=f"[ERROR: {e}]",
                model=self.model,
                backend=self.name(),
                latency_ms=(time.time() - start) * 1000,
            )

        latency = (time.time() - start) * 1000
        text = result.get("message", {}).get("content", "")

        return Prediction(
            text=text,
            model=self.model,
            backend=self.name(),
            latency_ms=latency,
            tokens_in=result.get("prompt_eval_count", 0),
            tokens_out=result.get("eval_count", 0),
            raw_response=result,
        )

    def diagnose(self) -> dict:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            available = self.model in models
            return {
                "backend": self.name(),
                "model": self.model,
                "status": "ok" if available else "model_not_found",
                "available_models": models,
            }
        except Exception as e:
            return {
                "backend": self.name(),
                "model": self.model,
                "status": "unreachable",
                "error": str(e),
            }


class OpenAICompatBackend(ModelBackend):
    """OpenAI-compatible API backend.
    
    Works with: OpenAI, NVIDIA NIM, Together AI, Anyscale, vLLM, etc.
    Uses the /v1/chat/completions endpoint.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: str = "sk-placeholder",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def name(self) -> str:
        if "nvidia" in self.base_url:
            return "nim"
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            return "local-api"
        return "openai-compat"

    def model_id(self) -> str:
        return self.model

    def generate(self, prompt: Prompt, **kwargs) -> Prediction:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        messages = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        messages.extend(prompt.messages)

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())
        except urllib.error.URLError as e:
            return Prediction(
                text=f"[ERROR: {e}]",
                model=self.model,
                backend=self.name(),
                latency_ms=(time.time() - start) * 1000,
            )

        latency = (time.time() - start) * 1000
        text = ""
        tokens_in = 0
        tokens_out = 0

        if "choices" in result and result["choices"]:
            text = result["choices"][0].get("message", {}).get("content", "")
        usage = result.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)

        return Prediction(
            text=text,
            model=self.model,
            backend=self.name(),
            latency_ms=latency,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            raw_response=result,
        )

    def diagnose(self) -> dict:
        try:
            req = urllib.request.Request(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                json.loads(resp.read())
            return {
                "backend": self.name(),
                "model": self.model,
                "status": "ok",
            }
        except Exception as e:
            return {
                "backend": self.name(),
                "model": self.model,
                "status": "error",
                "error": str(e),
            }


class AnthropicCompatBackend(ModelBackend):
    """Anthropic-Messages-compatible backend.

    Works with: api.anthropic.com, MiniMax's /anthropic surface, and any
    other server that speaks /v1/messages. Differs from OpenAI-compat in:
      - endpoint path:  /v1/messages
      - auth header:    x-api-key + anthropic-version
      - request body:   system is top-level (not a message); messages
                        otherwise mirror OpenAI's role/content shape
      - response shape: content[].text (block list) instead of
                        choices[].message.content
    """

    def __init__(
        self,
        model: str,
        api_key: str = "",
        base_url: str = "https://api.anthropic.com",
        anthropic_version: str = "2023-06-01",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.anthropic_version = anthropic_version
        self.temperature = temperature
        self.max_tokens = max_tokens

    def name(self) -> str:
        if "minimax" in self.base_url:
            return "minimax"
        return "anthropic-compat"

    def model_id(self) -> str:
        return self.model

    def generate(self, prompt: Prompt, **kwargs) -> Prediction:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        body = {
            "model": self.model,
            "messages": list(prompt.messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if prompt.system:
            body["system"] = prompt.system

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/messages",
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": self.anthropic_version,
            },
            method="POST",
        )

        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as he:
            try:
                err_body = he.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            return Prediction(
                text=f"[ERROR HTTP {he.code}: {err_body[:300]}]",
                model=self.model,
                backend=self.name(),
                latency_ms=(time.time() - start) * 1000,
            )
        except urllib.error.URLError as e:
            return Prediction(
                text=f"[ERROR: {e}]",
                model=self.model,
                backend=self.name(),
                latency_ms=(time.time() - start) * 1000,
            )

        latency = (time.time() - start) * 1000

        # Anthropic returns content as a list of blocks; concat all text blocks.
        text = ""
        for block in result.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
        usage = result.get("usage", {}) or {}
        return Prediction(
            text=text,
            model=self.model,
            backend=self.name(),
            latency_ms=latency,
            tokens_in=usage.get("input_tokens", 0),
            tokens_out=usage.get("output_tokens", 0),
            raw_response=result,
        )

    def diagnose(self) -> dict:
        # /v1/messages doesn't expose a list endpoint on every provider; do a
        # cheap probe with max_tokens=1 instead.
        try:
            self.generate(
                Prompt(system=None,
                       messages=[{"role": "user", "content": "ping"}],
                       raw_text="ping"),
                max_tokens=1,
            )
            return {"backend": self.name(), "model": self.model, "status": "ok"}
        except Exception as e:
            return {"backend": self.name(), "model": self.model,
                    "status": "error", "error": str(e)}


def parse_backend(spec: str) -> ModelBackend:
    """Parse a backend specification string into a ModelBackend.
    
    Formats:
        ollama:model_name                    -> OllamaBackend
        ollama:model_name@host               -> OllamaBackend with custom host
        openai:model_name                    -> OpenAICompatBackend (OpenAI)
        nim:model_name                       -> OpenAICompatBackend (NVIDIA NIM)
        api:model_name@base_url              -> OpenAICompatBackend (custom URL)
        api:model_name@base_url#key          -> OpenAICompatBackend with API key
    
    Examples:
        ollama:openhermes:7b-v2.5
        ollama:qwen2.5:7b-instruct-q8_0@http://192.168.0.2:11434
        nim:meta/llama-3.3-70b-instruct
        openai:gpt-4
        api:my-model@http://localhost:8000/v1
    """
    parts = spec.split(":", 1)
    backend_type = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    if backend_type in ("minimax", "minimax-cn", "anthropic"):
        # Anthropic-Messages transport. Format: minimax:MODEL[@URL][#KEY]
        model = rest
        base_url = None
        api_key = ""
        if "@" in rest:
            model, url_part = rest.split("@", 1)
            if "#" in url_part:
                base_url, api_key = url_part.split("#", 1)
            else:
                base_url = url_part
        elif "#" in rest:
            model, api_key = rest.split("#", 1)

        import os
        if backend_type == "minimax":
            base_url = base_url or "https://api.minimax.io/anthropic"
            api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        elif backend_type == "minimax-cn":
            base_url = base_url or "https://api.minimaxi.com/anthropic"
            api_key = api_key or os.environ.get("MINIMAX_CN_API_KEY", "")
        else:  # anthropic
            base_url = base_url or "https://api.anthropic.com"
            api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

        return AnthropicCompatBackend(
            model=model, api_key=api_key, base_url=base_url,
        )

    if backend_type == "ollama":
        # ollama:model or ollama:model@host
        if "@" in rest:
            model, host = rest.rsplit("@", 1)
            return OllamaBackend(model=model, host=host)
        return OllamaBackend(model=rest)

    elif backend_type in ("openai", "nim", "api"):
        # Parse model@url#key format
        model = rest
        base_url = "https://api.openai.com/v1"
        api_key = "sk-placeholder"

        if "@" in rest:
            model, url_part = rest.split("@", 1)
            if "#" in url_part:
                base_url, api_key = url_part.split("#", 1)
            else:
                base_url = url_part

        if backend_type == "nim":
            base_url = "https://integrate.api.nvidia.com/v1"
            # Try to load NIM key from env
            import os
            api_key = os.environ.get("NIM_API_KEY", api_key)

        elif backend_type == "openai":
            import os
            api_key = os.environ.get("OPENAI_API_KEY", api_key)

        return OpenAICompatBackend(
            model=model, api_key=api_key, base_url=base_url
        )

    raise ValueError(
        f"Unknown backend type '{backend_type}'. "
        f"Supported: ollama, openai, nim, api"
    )
