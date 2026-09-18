"""Minimal chat-completions client for a configured Sara/Qwen endpoint."""
import os
import hashlib
import math
import json
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

load_dotenv()

class SaraClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 120,
                 runtime_identity: dict | None = None):
        parsed=urlparse(base_url)
        if parsed.scheme not in {"http","https"} or not parsed.netloc or not model:
            raise ValueError("Provide a valid Sara/Qwen base URL and model")
        if not math.isfinite(timeout) or timeout<=0:
            raise ValueError("Sara timeout must be positive and finite")
        self.base_url, self.model, self.api_key, self.timeout=base_url.rstrip("/"),model,api_key,timeout
        self.runtime_identity=runtime_identity

    @property
    def identity(self):
        result={"model":self.model,"endpoint_sha256":hashlib.sha256(self.base_url.encode()).hexdigest(),
                "temperature":0,"seed_requested":True,"max_tokens":3024}
        if self.runtime_identity is not None:
            result["runtime"]=self.runtime_identity
        return result

    @classmethod
    def from_environment(cls):
        base,model=os.environ.get("SARA_BASE_URL"),os.environ.get("SARA_MODEL")
        if not base or not model:
            raise ValueError("Set SARA_BASE_URL and SARA_MODEL before running the agentic method")
        runtime=None
        if path:=os.environ.get("SARA_RUNTIME_MANIFEST"):
            receipt=json.loads(Path(path).read_text())
            keys=("server","source_revision","model_sha256","api_alias","cuda_toolkit",
                  "cuda_architecture","context_tokens","thinking","thinking_token_budget")
            runtime={key:receipt[key] for key in keys}
            if runtime["api_alias"]!=model:
                raise ValueError("Runtime manifest model alias differs from SARA_MODEL")
        return cls(base,model,os.environ.get("SARA_API_KEY"),
                   timeout=float(os.environ.get("SARA_TIMEOUT_SECONDS","120")),runtime_identity=runtime)

    def complete(self,messages,tools,seed):
        headers={"Content-Type":"application/json"}
        if self.api_key:
            headers["Authorization"]=f"Bearer {self.api_key}"
        response=requests.post(self.base_url+"/chat/completions",headers=headers,timeout=self.timeout,json={
            "model":self.model,"messages":messages,"tools":tools,"tool_choice":"required",
            "temperature":0,"seed":seed,"parallel_tool_calls":True,"max_tokens":3024})
        # Avoid writing arbitrary server response bodies or request headers to traces.
        if not response.ok:
            raise RuntimeError(f"Sara endpoint returned HTTP {response.status_code}")
        payload=response.json()
        if not payload.get("choices"):
            raise RuntimeError("Sara endpoint returned no choices")
        message=payload["choices"][0]["message"]
        # Only retain standard visible content/tool calls, not provider-specific fields.
        clean={"role":"assistant","content":message.get("content"),"tool_calls":message.get("tool_calls",[])}
        return clean,payload.get("usage")
