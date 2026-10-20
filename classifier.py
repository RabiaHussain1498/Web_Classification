"""
Classification Engine supporting local Ollama models and OpenAI-compatible endpoints.
Includes robust JSON parsing, prompt formatting, and structured output validation.
"""

import os
import json
import re
import requests
from typing import Dict, Any, Optional
from prompt_config import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES

class FinancialClassifier:
    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2.5:7b",
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 250
    ):
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if self.provider == "mock":
            self.endpoint = "mock"
            self.chat_url = "mock"
        elif self.provider == "ollama":
            self.endpoint = endpoint or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            if not self.endpoint.endswith("/api/chat") and not self.endpoint.endswith("/api/generate"):
                self.chat_url = f"{self.endpoint.rstrip('/')}/api/chat"
            else:
                self.chat_url = self.endpoint
        else:
            self.endpoint = endpoint or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.chat_url = f"{self.endpoint.rstrip('/')}/chat/completions"
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def _clean_json(self, raw_output: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from LLM response safely."""
        text = raw_output.strip()
        # Remove markdown fenced code blocks if present
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                text = match.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try finding the first '{' and last '}'
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end+1])
                except json.JSONDecodeError:
                    pass
        return None

    def classify_text(self, input_text: str, timeout: int = 30) -> Dict[str, Any]:
        """Send formatted text to LLM and return structured classification."""
        if self.provider == "mock":
            # Rule-based heuristic simulation for pipeline testing
            lower = input_text.lower()
            if any(k in lower for k in ["atm network", "defi", "crypto", "exchange and clearing", "cross-border b2b payments platform"]):
                model = "B - Product-led"
                build = "Own"
            elif any(k in lower for k in ["software vendor", "apis", "developer"]):
                model = "C - Software vendor"
                build = "Own"
            elif any(k in lower for k in ["wealth management", "financial advisor", "advisory", "planning"]):
                model = "D - Capital & advisory-led"
                build = "None"
            elif any(k in lower for k in ["consulting", "bpo", "outsourced"]):
                model = "E - Tech & professional services"
                build = "None"
            else:
                model = "A - Services-led"
                build = "Vendor"
                
            return {
                "success": True,
                "data": {
                    "business_model": model,
                    "what_it_sells": "Financial product" if model in ["A - Services-led", "B - Product-led"] else "Capital & advisory",
                    "build": build,
                    "institution_type": "Traditional Bank" if model == "A - Services-led" else "Crypto & Digital Assets",
                    "sector_group": "Banking",
                    "confidence": "High",
                    "build_evidence": f"Mock simulated inference: {build} architecture detected.",
                    "what_they_do": "Financial company operations."
                },
                "raw": "{}"
            }

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        # Add 1-2 few-shot examples for grounding
        for ex in FEW_SHOT_EXAMPLES[:2]:
            messages.append({"role": "user", "content": ex["input"]})
            messages.append({"role": "assistant", "content": json.dumps(ex["output"])})
            
        messages.append({"role": "user", "content": input_text})

        if self.provider == "ollama":
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "stop": ["```", "User:", "Assistant:"]
                }
            }
            try:
                res = requests.post(self.chat_url, json=payload, timeout=timeout)
                if res.status_code == 200:
                    data = res.json()
                    raw_msg = data.get("message", {}).get("content", "")
                    parsed = self._clean_json(raw_msg)
                    if parsed:
                        return {"success": True, "data": parsed, "raw": raw_msg}
                    return {"success": False, "error": "Invalid JSON returned", "raw": raw_msg}
                return {"success": False, "error": f"Ollama HTTP {res.status_code}: {res.text}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        else:
            # OpenAI / vLLM / LiteLLM compatible
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"}
            }
            try:
                res = requests.post(self.chat_url, headers=headers, json=payload, timeout=timeout)
                if res.status_code == 200:
                    data = res.json()
                    raw_msg = data["choices"][0]["message"]["content"]
                    parsed = self._clean_json(raw_msg)
                    if parsed:
                        return {"success": True, "data": parsed, "raw": raw_msg}
                    return {"success": False, "error": "Invalid JSON returned", "raw": raw_msg}
                return {"success": False, "error": f"API HTTP {res.status_code}: {res.text}"}
            except Exception as e:
                return {"success": False, "error": str(e)}
