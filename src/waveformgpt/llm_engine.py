"""
LLM-Powered Natural Language Engine for WaveformGPT.

Supports multiple LLM backends:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Ollama (local models like Llama, Mistral)
- OpenRouter (multiple providers)
"""

import os
import json
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class LLMResponse:
    """Response from LLM query."""
    answer: str
    reasoning: Optional[str] = None
    code: Optional[str] = None  # Generated Python code for complex queries
    confidence: float = 1.0
    tokens_used: int = 0


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def query(self, prompt: str, system_prompt: str) -> LLMResponse:
        """Send query to LLM and get response."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available."""
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI GPT backend."""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client
    
    def query(self, prompt: str, system_prompt: str) -> LLMResponse:
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0
        
        # Try to parse structured response
        return self._parse_response(content, tokens)
    
    def _parse_response(self, content: str, tokens: int) -> LLMResponse:
        # Try to extract JSON from response
        try:
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
                data = json.loads(json_str)
                return LLMResponse(
                    answer=data.get("answer", content),
                    reasoning=data.get("reasoning"),
                    code=data.get("code"),
                    confidence=data.get("confidence", 0.9),
                    tokens_used=tokens
                )
        except Exception:
            pass
        
        # Try to parse as raw JSON
        try:
            # Look for JSON object in the response
            import re
            json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return LLMResponse(
                    answer=data.get("answer", content),
                    reasoning=data.get("reasoning"),
                    code=data.get("code"),
                    confidence=data.get("confidence", 0.9),
                    tokens_used=tokens
                )
        except Exception:
            pass
        
        # Return as plain text
        return LLMResponse(answer=content, tokens_used=tokens)


class AnthropicBackend(LLMBackend):
    """Anthropic Claude backend."""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        return self._client
    
    def query(self, prompt: str, system_prompt: str) -> LLMResponse:
        client = self._get_client()
        
        response = client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        
        return self._parse_response(content, tokens)
    
    def _parse_response(self, content: str, tokens: int) -> LLMResponse:
        try:
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
                data = json.loads(json_str)
                return LLMResponse(
                    answer=data.get("answer", content),
                    reasoning=data.get("reasoning"),
                    code=data.get("code"),
                    confidence=data.get("confidence", 0.9),
                    tokens_used=tokens
                )
        except:
            pass
        
        return LLMResponse(answer=content, tokens_used=tokens)


class OllamaBackend(LLMBackend):
    """Ollama local LLM backend."""
    
    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
    
    def is_available(self) -> bool:
        try:
            import requests
            response = requests.get(f"{self.host}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def query(self, prompt: str, system_prompt: str) -> LLMResponse:
        import requests
        
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1
                }
            },
            timeout=120
        )
        
        data = response.json()
        content = data.get("response", "")
        
        return self._parse_response(content, 0)
    
    def _parse_response(self, content: str, tokens: int) -> LLMResponse:
        try:
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
                data = json.loads(json_str)
                return LLMResponse(
                    answer=data.get("answer", content),
                    reasoning=data.get("reasoning"),
                    code=data.get("code"),
                    confidence=data.get("confidence", 0.9),
                    tokens_used=tokens
                )
        except:
            pass
        
        return LLMResponse(answer=content, tokens_used=tokens)


class WaveformLLM:
    """
    LLM-powered waveform query engine.
    
    This engine can answer arbitrary natural language questions about waveforms
    by understanding the context and generating appropriate analysis code.
    """
    
    SYSTEM_PROMPT = '''You are WaveformGPT, an expert digital hardware verification assistant.
You analyze VCD (Value Change Dump) waveform files from digital simulations.

You have access to a waveform database with these signals:
{signals}

Time range: {time_start} to {time_end} {time_unit}

Here is sample data from the waveform (time, value pairs):
{sample_data}

When answering questions:
1. Be precise about timing (use exact time values from the data)
2. For digital signals: 0=low, 1=high, x=unknown, z=high-impedance
3. Analyze the actual data provided to give specific answers
4. Be concise but informative

If you need to compute something, show your calculation.
Answer directly and clearly.'''

    def __init__(self, backend: Optional[LLMBackend] = None):
        """
        Initialize with a specific backend or auto-detect.
        
        Args:
            backend: LLM backend to use. If None, auto-detects available backends.
        """
        self.backend = backend or self._auto_detect_backend()
        self._waveform_context = {}
    
    def _auto_detect_backend(self) -> LLMBackend:
        """Auto-detect available LLM backend."""
        # Try OpenAI first
        openai = OpenAIBackend()
        if openai.is_available():
            return openai
        
        # Try Anthropic
        anthropic = AnthropicBackend()
        if anthropic.is_available():
            return anthropic
        
        # Try local Ollama
        ollama = OllamaBackend()
        if ollama.is_available():
            return ollama
        
        raise RuntimeError(
            "No LLM backend available. Set one of:\n"
            "  - OPENAI_API_KEY environment variable\n"
            "  - ANTHROPIC_API_KEY environment variable\n"
            "  - Run Ollama locally (ollama serve)"
        )
    
    def set_waveform_context(self, 
                              signals: List[str],
                              time_range: tuple,
                              time_unit: str = "ns",
                              sample_data: Optional[Dict[str, List]] = None):
        """
        Set the waveform context for queries.
        
        Args:
            signals: List of signal names
            time_range: (start_time, end_time) tuple
            time_unit: Time unit (ns, ps, etc.)
            sample_data: Optional sample of signal data for context
        """
        self._waveform_context = {
            "signals": signals,
            "time_start": time_range[0],
            "time_end": time_range[1],
            "time_unit": time_unit,
            "sample_data": sample_data or {}
        }
    
    def query(self, question: str, 
              signal_data_provider: Optional[Callable] = None) -> LLMResponse:
        """
        Answer a natural language question about the waveform.
        
        Args:
            question: Natural language question
            signal_data_provider: Optional function to get signal data for code execution
            
        Returns:
            LLMResponse with answer and optional code
        """
        # Build sample data string
        sample = self._waveform_context.get("sample_data", {})
        sample_str = ""
        if sample:
            for sig, data in list(sample.items()):
                # Format nicely
                data_preview = data[:20] if len(data) > 20 else data
                sample_str += f"  {sig}: {data_preview}\n"
        else:
            sample_str = "  (no data available)"
        
        # Build system prompt with context
        system_prompt = self.SYSTEM_PROMPT.format(
            signals=", ".join(self._waveform_context.get("signals", [])),
            time_start=self._waveform_context.get("time_start", 0),
            time_end=self._waveform_context.get("time_end", 1000),
            time_unit=self._waveform_context.get("time_unit", "ns"),
            sample_data=sample_str
        )
        
        # Query LLM
        response = self.backend.query(question, system_prompt)
        
        return response
    
    def _execute_code(self, code: str, data_provider: Callable) -> Any:
        """Safely execute generated code with sandboxed globals."""
        # Create sandboxed environment
        sandbox = {
            "get_signal_values": lambda s: data_provider(s),
            # Add safe builtins
            "len": len,
            "max": max,
            "min": min,
            "sum": sum,
            "abs": abs,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "range": range,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
        }
        
        # Execute and capture result
        exec_globals = {"__builtins__": {}}
        exec_globals.update(sandbox)
        
        # Wrap code to capture result
        wrapped = f"__result__ = None\n{code}\n"
        exec(wrapped, exec_globals)
        
        return exec_globals.get("__result__")


def get_llm_backend(provider: str = "auto", **kwargs) -> LLMBackend:
    """
    Get an LLM backend by provider name.
    
    Args:
        provider: One of 'openai', 'anthropic', 'ollama', or 'auto'
        **kwargs: Provider-specific arguments (api_key, model, host, etc.)
    
    Returns:
        Configured LLM backend
    """
    if provider == "auto":
        return WaveformLLM()._auto_detect_backend()
    elif provider == "openai":
        return OpenAIBackend(**kwargs)
    elif provider == "anthropic":
        return AnthropicBackend(**kwargs)
    elif provider == "ollama":
        return OllamaBackend(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
