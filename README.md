# 🏫 AgentAI Backend - School Search RAG System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

**AgentAI** adalah sistem backend cerdas berbasis **RAG (Retrieval-Augmented Generation)** yang dirancang untuk membantu tim sales mencari data sekolah secara presisi menggunakan bahasa alami.

Sistem ini menggunakan arsitektur **Hybrid LLM** (Google Gemini + Ollama) dan dilengkapi sistem autentikasi database yang aman.

---

## 🚀 Fitur Unggulan

- 🧠 **Hybrid Intelligence:** Otomatis beralih ke **Ollama (Lokal)** jika **Google Gemini** bermasalah.
- 🎯 **Smart Query Planner:** Akurasi 100% untuk filter numerik (contoh: "Siswa > 500").
- 🔍 **Hybrid Search:** Menggabungkan pencarian vektor (makna) dan metadata (filter).
- 🛡️ **Secure Auth:** Sistem Registrasi & Login menggunakan JWT dan Database SQLite.

---

## 📦 Persiapan Awal (Prasyarat)

Sebelum memulai, pastikan komputer Anda sudah terinstal:

1.  **[Python 3.10](https://www.python.org/downloads/)** (Wajib versi 3.10 agar kompatibel).
2.  **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** (Wajib nyala).
3.  **Git** (Untuk clone project).

---

## ⚙️ Langkah Instalasi (Wajib Berurutan)

Ikuti langkah ini satu per satu agar sistem berjalan lancar.

### 1. Setup Environment Python

Buka terminal (CMD/PowerShell) di folder `sekolah-agentai`, lalu jalankan:

```cmd
# 1. Buat Virtual Environment
python -m venv venv

# 2. Aktifkan Virtual Environment (Windows)
.\venv\Scripts\activate

# 3. Install Library yang Dibutuhkan
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Nyalakan Layanan Pendukung (Docker)

Sistem ini butuh "Otak" (Ollama) dan "Ingatan" (Qdrant) yang berjalan di Docker.

**A. Siapkan Folder Penyimpanan Data (Agar tidak hilang saat restart):**

```cmd
mkdir D:\install\ollama_models
mkdir C:\qdrant_storage
```

_(Sesuaikan path folder jika drive D tidak ada di laptop Anda)_

**B. Jalankan Database Qdrant:**

```cmd
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v "C:\qdrant_storage:/qdrant/storage" qdrant/qdrant
```

**C. Jalankan AI Lokal (Ollama):**

```cmd
docker run -d --gpus all --name ollama -v "D:\install\ollama_models:/root/.ollama" -p 11434:11434 -e OLLAMA_KEEP_ALIVE=9999h ollama/ollama
```

_(Hapus `--gpus all` jika laptop tidak punya GPU NVIDIA)_

**D. Download Model Cerdas (Hanya perlu sekali):**

```cmd
docker exec -it ollama ollama pull phi3:mini
```

### 3. Masukkan Data Sekolah (Import Data)

Database Qdrant masih kosong saat pertama kali dijalankan. Kita perlu mengisinya dengan data CSV.

Pastikan terminal masih di folder `sekolah-agentai` dan `(venv)` aktif:

```cmd
python import/import.py
```

_Tunggu hingga proses selesai 100%._

### 4. Konfigurasi Kunci Rahasia (`.env`)

Buat file baru bernama `.env` di dalam folder `sekolah-agentai`, lalu isi dengan teks berikut:

```ini
# --- Database ---
QDRANT_HOST=localhost
QDRANT_PORT=6333

# --- Primary LLM (Google Gemini) ---
# Dapatkan API Key Gratis di: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
GEMINI_API_KEY="ISI_API_KEY_GEMINI_ANDA_DISINI"
GEMINI_MODEL="gemini-2.5-flash"

# --- Backup LLM (Ollama Local) ---
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="phi3:mini"

# --- API Settings ---
API_HOST="0.0.0.0"
API_PORT=8000
API_DEBUG=True

# --- Authentication ---
SECRET_KEY="rahasia-super-aman-ganti-di-production"
ACCESS_TOKEN_EXPIRE_MINUTES=300
```

---

## ▶️ Menjalankan Backend

Setelah semua langkah di atas selesai, jalankan perintah ini:

```cmd
python main.py
```

**Tanda Berhasil:**
Anda akan melihat log seperti ini:

```text
INFO:     ✅ Database tables created/verified (users.db).
INFO:     🚀 Initializing AgentAI system components...
INFO:     ✅ Connected to Qdrant...
INFO:     ✅ SUKSES: Terhubung ke Google Gemini (Primary).
INFO:     Uvicorn running on [http://0.0.0.0:8000](http://0.0.0.0:8000)
```

**Biarkan terminal ini tetap terbuka selama menggunakan aplikasi.**

---

## 📚 Dokumentasi API

Anda bisa mengetes Backend tanpa Frontend melalui halaman dokumentasi interaktif:
👉 **http://localhost:8000/docs**

### Endpoint Penting:

1.  **`POST /register`**: Mendaftarkan user baru.
2.  **`POST /token`**: Login user.
3.  **`POST /query`**: Bertanya ke AI.

---

## 📂 Struktur Folder Project

```text
/
├── app/
│   ├── api.py               # Otak Utama (Routes & Logic)
│   ├── auth.py              # Sistem Keamanan (JWT)
│   ├── config.py            # Pengaturan
│   ├── database.py          # Koneksi Database User (SQLite)
│   ├── models_user.py       # Struktur Tabel User
│   └── ... (Modul AI lainnya)
├── import/
│   └── import.py            # Script upload data ke Qdrant
├── users.db                 # File Database User (Otomatis dibuat)
├── .env                     # File Konfigurasi Rahasia
├── main.py                  # Tombol Start
└── requirements.txt         # Daftar Library
```

---

## 🧪 Pengujian (Testing)

Untuk memastikan fitur _failover_ berfungsi:

1.  Jalankan aplikasi normal -\> Pastikan log: `Running with: Google Gemini`.
2.  Ubah `GEMINI_API_KEY` di `.env` menjadi salah.
3.  Restart aplikasi -\> Pastikan log berubah menjadi: `Running with: Ollama (Local)`.
4.  API `/query` harus tetap bisa menjawab pertanyaan dengan benar.

---

## 🔄 Cara Menjalankan Setelah Setup

Jika Anda sudah pernah melakukan setup di atas dan komputer baru saja dinyalakan ulang, ikuti langkah praktis ini:

1.  **Buka Docker Desktop**: Pastikan aplikasi Docker Desktop sudah berjalan.
2.  **Bangunkan Container**: Buka terminal, jalankan perintah ini untuk menghidupkan kembali Qdrant & Ollama:

```cmd
docker start qdrant ollama
```

3.  **Jalankan Backend**:

```cmd
# Masuk folder project
cd sekolah-agentai

# Aktifkan venv (Jika belum aktif)
.\venv\Scripts\activate

# Jalankan Aplikasi
python main.py
```

**Tanda Berhasil:**
Anda akan melihat log seperti ini:

```text
INFO:     ✅ Database tables created/verified (users.db).
INFO:     🚀 Initializing AgentAI system components...
INFO:     ✅ Connected to Qdrant...
INFO:     ✅ SUKSES: Terhubung ke Google Gemini (Primary).
INFO:     Uvicorn running on [http://0.0.0.0:8000](http://0.0.0.0:8000)
```

---

## 📜 Lisensi

Project ini dibuat untuk keperluan Kerja Praktik (KP) di PT. Telkom Indonesia.

```

```
