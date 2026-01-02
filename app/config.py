import os
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration for Sidoarjo School Chatbot (AgentAI).
    Centralized settings for Database, LLMs (Gemini & Ollama), and API.
    """
    
    # =================================================================
    # 1. DATABASE CONFIGURATION (Qdrant)
    # =================================================================
    COLLECTION_NAME = "sekolah_baru"
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_TIMEOUT = 30.0
    
    # Model Embedding
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # =================================================================
    # 2. LLM CONFIGURATION (HYBRID)
    # =================================================================
    # PRIMARY: Google Gemini
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # BACKUP: Ollama
    _OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_URL = f"{_OLLAMA_BASE}/api/chat"
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
    OLLAMA_TIMEOUT = 60.0 
    
    # Legacy / Disabled Providers
    TELKOM_AI_URL = None
    TELKOM_AI_API_KEY = None
    TELKOM_AI_MODEL = None
    TELKOM_AI_TIMEOUT = 0

    # =================================================================
    # 3. API & SECURITY CONFIGURATION
    # =================================================================
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_DEBUG = str(os.getenv("API_DEBUG", "False")).lower() in ("true", "1", "t")
    
    # Authentication
    API_USERNAME = os.getenv("API_USERNAME", "admin")
    API_PASSWORD = os.getenv("API_PASSWORD", "securepassword123")
    SECRET_KEY = os.getenv("SECRET_KEY", "default-dev-secret-key")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 300

    # =================================================================
    # 4. LLM PARAMETERS
    # =================================================================
    LLM_TEMPERATURES = {
        "query_planning": 0.1,
        "response_generation": 0.3,
        "html_generation": 0.1
    }

    MAX_CONVERSATION_HISTORY = 5
    CACHE_SIZE = 1000
    DEFAULT_LIMIT = 20
    MAX_LIMIT = 1000

    HTML_RESPONSE_CONFIG = {
        "enable_styling": True,
        "responsive_design": True,
        "table_max_width": "100%",
        "primary_color": "#2c5aa0",
        "secondary_color": "#f8f9fa",
        "border_color": "#dee2e6",
        "text_color": "#212529",
        "success_color": "#28a745",
        "warning_color": "#ffc107",
        "danger_color": "#dc3545"
    }

    # =================================================================
    # 5. DOMAIN KNOWLEDGE (DATA RULES)
    # =================================================================

    NUMERIC_FIELDS = ['pd', 'ptk', 'jml_lab', 'jml_perpus', 'jml_rk', 'rombel', 'jumlah_kirim', 'pegawai']

    VALID_FIELDS = {
        "npsn", "nama", "sekolah_id", "sekolah_id_enkrip", "bentukPendidikan", 
        "bentukPendidikanGroup", "jalurPendidikan", "jenjangPendidikan", 
        "jenisPendidikan", "status_sekolah", "kementerianPembina", "namaProvinsi", 
        "kodeProvinsi", "namaKabupaten", "kodeKabupaten", "namaKecamatan", 
        "kodeKecamatan", "namaDesa", "kodeWilayah", "alamatJalan", "rt", "rw", 
        "lintang", "bujur", "nomorTelepon", "email", "akreditasi", "namaYayasan", 
        "npyp", "jumlah_kirim", "ptk", "pegawai", "pd", "rombel", "jml_rk", 
        "jml_lab", "jml_perpus", "sinkron_terakhir", "kepala_sekolah", "operator"
    }
    
    FIELD_DISPLAY_NAMES = {
        'npsn': 'NPSN', 'nama': 'Nama Sekolah', 'sekolah_id': 'ID Sistem',
        'bentukPendidikan': 'Jenjang', 'status_sekolah': 'Status',
        'alamatJalan': 'Alamat', 'namaDesa': 'Desa', 'namaKecamatan': 'Kecamatan',
        'nomorTelepon': 'No. Telp', 'email': 'Email', 'akreditasi': 'Akreditasi',
        'pd': 'Jml Siswa', 'ptk': 'Jml Guru', 'pegawai': 'Jml Staf',
        'jml_rk': 'R.Kelas', 'jml_lab': 'Laboratorium', 'jml_perpus': 'Perpustakaan',
        'kepala_sekolah': 'Kepala Sekolah', 'operator': 'Operator Dapodik',
        'sinkron_terakhir': 'Update Terakhir'
    }
    
    FIELD_MAPPING = {
        # Siswa
        'siswa': 'pd', 'murid': 'pd', 'anak': 'pd', 'peserta didik': 'pd', 
        'total siswa': 'pd', 'jumlah siswa': 'pd', 'populasi': 'pd',
        
        # Guru & SDM
        'guru': 'ptk', 'pengajar': 'ptk', 'pendidik': 'ptk', 'jumlah guru': 'ptk',
        'pegawai': 'pegawai', 'staf': 'pegawai', 'karyawan': 'pegawai',
        'kepala sekolah': 'kepala_sekolah', 'kepsek': 'kepala_sekolah',
        'operator': 'operator', 'admin': 'operator',
        
        # Fasilitas
        'kelas': 'jml_rk', 'ruang kelas': 'jml_rk',
        'lab': 'jml_lab', 'laboratorium': 'jml_lab',
        'perpus': 'jml_perpus', 'perpustakaan': 'jml_perpus',
        'rombel': 'rombel', 'rombongan belajar': 'rombel',
        
        # Lokasi
        'alamat': 'alamatJalan', 'jalan': 'alamatJalan', 'lokasi': 'alamatJalan',
        'desa': 'namaDesa', 'kelurahan': 'namaDesa',
        'kecamatan': 'namaKecamatan', 'wilayah': 'namaKecamatan',
        'kabupaten': 'namaKabupaten', 'kota': 'namaKabupaten',
        'provinsi': 'namaProvinsi',
        
        # Kontak
        'telepon': 'nomorTelepon', 'hp': 'nomorTelepon', 'wa': 'nomorTelepon',
        'email': 'email',
        
        # Status
        'status': 'status_sekolah', 'negeri': 'status_sekolah', 'swasta': 'status_sekolah',
        'akreditasi': 'akreditasi', 'nilai': 'akreditasi',
        'jenjang': 'bentukPendidikan', 'sd': 'bentukPendidikan', 'smp': 'bentukPendidikan', 
        'sma': 'bentukPendidikan', 'smk': 'bentukPendidikan'
    }
    
    VALUE_MAPPING = {
        "status_sekolah": {
            "negeri": "Negeri", "public": "Negeri",
            "swasta": "Swasta", "private": "Swasta"
        },
        "akreditasi": {
            "a": "A", "unggul": "A",
            "b": "B", "baik": "B",
            "c": "C", "cukup": "C",
            "belum": "TT", "tidak": "TT"
        },
        "bentukPendidikan": {
            "sd": "SD", "smp": "SMP", "sma": "SMA", "smk": "SMK", 
            "tk": "TK", "paud": "PAUD", "kb": "KB", "slb": "SLB"
        },
        "namaKecamatan": {
            re.compile(r"balong\s*bendo", re.IGNORECASE): "KEC. BALONG BENDO",
            re.compile(r"buduran", re.IGNORECASE): "KEC. BUDURAN",
            re.compile(r"candi", re.IGNORECASE): "KEC. CANDI",
            re.compile(r"gedangan", re.IGNORECASE): "KEC. GEDANGAN",
            re.compile(r"jabon", re.IGNORECASE): "KEC. JABON",
            re.compile(r"krembung", re.IGNORECASE): "KEC. KREMBUNG",
            re.compile(r"krian", re.IGNORECASE): "KEC. KRIAN",
            re.compile(r"prambon", re.IGNORECASE): "KEC. PRAMBON",
            re.compile(r"porong", re.IGNORECASE): "KEC. PORONG",
            re.compile(r"sedati", re.IGNORECASE): "KEC. SEDATI",
            re.compile(r"sidoarjo", re.IGNORECASE): "KEC. SIDOARJO",
            re.compile(r"sukodono", re.IGNORECASE): "KEC. SUKODONO",
            re.compile(r"taman", re.IGNORECASE): "KEC. TAMAN",
            re.compile(r"tanggulangin", re.IGNORECASE): "KEC. TANGGULANGIN",
            re.compile(r"tarik", re.IGNORECASE): "KEC. TARIK",
            re.compile(r"tulangan", re.IGNORECASE): "KEC. TULANGAN",
            re.compile(r"waru", re.IGNORECASE): "KEC. WARU",
            re.compile(r"wonoayu", re.IGNORECASE): "KEC. WONOAYU"
        }
    }
    
    QUERY_INTENTS = {
        "get_info": ["info", "detail", "profil", "data", "tentang", "gimana", "lengkap", "rincian"],
        "search_school": ["cari", "daftar", "list", "temukan", "tampilkan", "mana saja", "yang mana"],
        "get_principal": ["kepala", "kepsek", "pimpinan", "direktur"],
        "count_query": ["berapa", "jumlah", "hitung", "total", "banyaknya"],
        "get_contact": ["kontak", "hubungi", "telpon", "email", "nomor", "alamat"]
    }
    
    COMPREHENSIVE_FIELDS = {
        "get_info": [
            "nama", "npsn", "bentukPendidikan", "status_sekolah", "akreditasi",
            "namaKecamatan", "namaKabupaten", "pd", "ptk", "kepala_sekolah",
            "alamatJalan", "nomorTelepon", "email", "jml_lab", "jml_perpus", "jumlah_kirim"
        ],
        "search_school": [
            "nama", "npsn", "bentukPendidikan", "status_sekolah", "akreditasi",
            "namaKecamatan", "pd", "ptk"
        ]
    }

    # Regex Patterns untuk deteksi nama sekolah
    SCHOOL_NAME_PATTERNS = [
        r'\b(SD|SMP|SMA|SMK|TK|PAUD|MI|MTs|MA)\s+(NEGERI|SWASTA)?\s*(\d+|\w+(?:\s+\w+)*)',
        r'\b(SD|SMP|SMA|SMK|TK|PAUD)\s+([A-Z][A-Z0-9\s\-\.]+)',
        r'\b(SMAN|SMPN|SDN)\s+(\d+)?\s*([A-Z\s]+)'
    ]

    # Mapping Operator Matematika
    OPERATOR_MAPPING = {
        'lebih dari': '>', 'di atas': '>', 'melebihi': '>', 
        'kurang dari': '<', 'di bawah': '<', 'kurang': '<',
        'minimal': '>=', 'paling sedikit': '>=', 'setidaknya': '>=',
        'maksimal': '<=', 'paling banyak': '<=', 
        'tepat': '==', 'sama dengan': '==', 'persis': '=='
    }

# --- EXPORT MODULE LEVEL VARIABLES ---
# Ini penting agar file lain (seperti auth.py) bisa melakukan import langsung
# Contoh: from app.config import SECRET_KEY
SECRET_KEY = Config.SECRET_KEY
ALGORITHM = Config.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = Config.ACCESS_TOKEN_EXPIRE_MINUTES