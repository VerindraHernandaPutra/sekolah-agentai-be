# main.py
import os
import sys
import logging
import uvicorn

# 1. Tambahkan direktori 'app' ke sys.path agar modul bisa diimport dengan benar
# Ini penting jika Anda menjalankan 'python main.py' dari root
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, 'app')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

try:
    # Import konfigurasi dan utilitas
    from app.config import Config
    from app.utils import setup_logging
    
    # Setup logging terpusat sebelum import API
    # Ini agar log dari uvicorn dan aplikasi memiliki format yang sama
    setup_logging(level=logging.INFO)
    logger = logging.getLogger("main")

    # Import aplikasi FastAPI
    from app.api import app

    if __name__ == "__main__":
        # Banner Startup agar terlihat profesional di terminal
        print("\n" + "="*60)
        print(f"🚀  SIDOARJO SCHOOL CHATBOT (AgentAI) v2.1")
        print("="*60)
        print(f"🌍  Environment : {'DEBUG' if Config.API_DEBUG else 'PRODUCTION'}")
        print(f"📡  Server URL  : http://{Config.API_HOST}:{Config.API_PORT}")
        print(f"📄  Docs URL    : http://{Config.API_HOST}:{Config.API_PORT}/docs")
        print("-" * 60)
        print(f"🤖  Primary LLM : {Config.GEMINI_MODEL} (Cloud)")
        print(f"🔄  Backup LLM  : {Config.OLLAMA_MODEL} (Local)")
        print("="*60 + "\n")
        
        # Menjalankan Uvicorn Server
        uvicorn.run(
            "app.api:app",       # String import path (lebih baik untuk reload)
            host=Config.API_HOST,
            port=Config.API_PORT,
            reload=Config.API_DEBUG, # Auto-reload jika ada perubahan kode (Hanya mode Debug)
            log_level="info"
        )

except ImportError as e:
    print(f"\n❌ CRITICAL ERROR: Gagal mengimport modul.")
    print(f"Detail: {e}")
    print("\nTips Perbaikan:")
    print("1. Pastikan struktur folder benar: 'main.py' sejajar dengan folder 'app/'.")
    print("2. Pastikan virtual environment aktif (venv).")
    print("3. Jalankan: pip install -r requirements.txt")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")
    sys.exit(1)