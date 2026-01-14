# enhanced_llm_utils.py
import time
import logging
import re
import json
import requests  # Diperlukan untuk Ollama backup
from typing import Optional, Dict, Any

# Import library Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    GOOGLE_SDK_AVAILABLE = True
except ImportError:
    GOOGLE_SDK_AVAILABLE = False

from app.config import Config

class EnhancedLLMClient:
    """
    Enhanced LLM Client yang mendukung arsitektur Hybrid:
    1. Primary: Google Gemini 2.5 Flash (via SDK)
    2. Backup: Ollama Phi-3 Mini (via HTTP Request)
    """
    
    def __init__(self, provider: str = "gemini"):
        self.logger = logging.getLogger(__name__)
        self.provider = provider.lower()
        
        self.logger.info(f"Initializing EnhancedLLMClient with provider: {self.provider.upper()}")

        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "ollama":
            self._init_ollama()
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'gemini' or 'ollama'.")

    def _init_gemini(self):
        """Setup untuk Google Gemini"""
        if not GOOGLE_SDK_AVAILABLE:
            raise ImportError("Library 'google-genai' belum terinstall. Jalankan: pip install google-genai")
        
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY tidak ditemukan di konfigurasi.")

        self.model_name = Config.GEMINI_MODEL
        try:
            self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        except Exception as e:
            self.logger.error(f"Gagal inisialisasi Gemini Client: {e}")
            raise

    def _init_ollama(self):
        """Setup untuk Ollama Local"""
        self.base_url = Config.OLLAMA_URL  # e.g., http://localhost:11434/api/chat
        self.model_name = Config.OLLAMA_MODEL
        self.timeout = Config.OLLAMA_TIMEOUT

    def call_llm(self, 
                 prompt: str, 
                 temperature: float = 0.1, 
                 max_retries: int = 1, 
                 use_streaming: bool = False) -> Optional[str]:
        """
        Router utama untuk memanggil LLM berdasarkan provider yang aktif.
        """
        if self.provider == "gemini":
            return self._call_gemini(prompt, temperature, max_retries, use_streaming)
        elif self.provider == "ollama":
            return self._call_ollama(prompt, temperature, max_retries, use_streaming)
        return None

    # =========================================================================
    # IMPLEMENTASI GEMINI (PRIMARY)
    # =========================================================================
    def _call_gemini(self, prompt: str, temperature: float, max_retries: int, use_streaming: bool) -> Optional[str]:
        # Konfigurasi Generasi
        # PERBAIKAN: Menghapus 'thinking_config' yang menyebabkan error pada SDK versi tertentu
        config = types.GenerateContentConfig(
            temperature=temperature,
        )
        
        # Force JSON output untuk Query Planner (temperature rendah)
        if temperature < 0.2 and not use_streaming:
            config.response_mime_type = "application/json"

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                if use_streaming:
                    response_stream = self.client.models.generate_content_stream(
                        model=self.model_name,
                        contents=prompt,
                        config=config
                    )
                    # Menggabungkan chunk
                    full_text = "".join(chunk.text for chunk in response_stream if chunk.text)
                else:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=config
                    )
                    full_text = response.text
                
                execution_time = time.time() - start_time
                self.logger.info(f"Gemini request completed in {execution_time:.2f}s")
                return self._clean_response(full_text)
                        
            except APIError as e:
                self.logger.warning(f"Gemini API Error (Attempt {attempt+1}): {e}")
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Gemini Unexpected Error: {e}")
                return None
        return None

    # =========================================================================
    # IMPLEMENTASI OLLAMA (BACKUP)
    # =========================================================================
    def _call_ollama(self, prompt: str, temperature: float, max_retries: int, use_streaming: bool) -> Optional[str]:
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": use_streaming,
            "options": {
                "temperature": temperature,
                "num_ctx": 4096 # Konteks window standar
            }
        }

        # Force JSON mode untuk Ollama jika temperature rendah
        if temperature < 0.2 and not use_streaming:
            payload["format"] = "json"

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                if use_streaming:
                    # Handling Streaming Ollama
                    response = requests.post(self.base_url, json=payload, headers=headers, stream=True, timeout=self.timeout)
                    response.raise_for_status()
                    
                    full_text = ""
                    for line in response.iter_lines():
                        if line:
                            try:
                                json_chunk = json.loads(line.decode('utf-8'))
                                chunk_content = json_chunk.get("message", {}).get("content", "")
                                full_text += chunk_content
                                if json_chunk.get("done"):
                                    break
                            except json.JSONDecodeError:
                                continue
                else:
                    # Handling Non-Streaming Ollama
                    response = requests.post(self.base_url, json=payload, headers=headers, timeout=self.timeout)
                    response.raise_for_status()
                    result = response.json()
                    full_text = result.get("message", {}).get("content", "")

                execution_time = time.time() - start_time
                self.logger.info(f"Ollama request completed in {execution_time:.2f}s")
                return self._clean_response(full_text)

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Ollama Connection Error (Attempt {attempt+1}): {e}")
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Ollama Unexpected Error: {e}")
                return None
        return None

    # =========================================================================
    # UTILITIES
    # =========================================================================
    def _clean_response(self, text: str) -> str:
        """Membersihkan respons dari karakter yang tidak diinginkan."""
        if not text:
            return ""
        
        # Hapus escape characters umum
        clean_text = text.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
        
        # Hapus token spesial (biasanya muncul di model lokal/Ollama)
        clean_text = re.sub(r'<\|im_sep\|>|<\|im_end\|>|<\|im_start\|>', '', clean_text)
        
        # Hapus blok markdown ```json atau ```html wrapper
        clean_text = re.sub(r'^```[a-zA-Z]*\n', '', clean_text)
        clean_text = re.sub(r'\n```$', '', clean_text)
        
        return clean_text.strip()
    
    def check_connection(self) -> bool:
        """Memeriksa koneksi ke provider yang aktif."""
        if self.provider == "gemini":
            try:
                self.client.models.generate_content(
                    model=self.model_name,
                    contents="Ping",
                    config=types.GenerateContentConfig(temperature=0.0)
                )
                return True
            except Exception as e:
                self.logger.warning(f"Gemini Check Failed: {e}")
                return False
        
        elif self.provider == "ollama":
            try:
                # Cek endpoint tags untuk melihat apakah service up
                tags_url = self.base_url.replace("/api/chat", "/api/tags")
                resp = requests.get(tags_url, timeout=5)
                return resp.status_code == 200
            except Exception as e:
                self.logger.warning(f"Ollama Check Failed: {e}")
                return False
        
        return False