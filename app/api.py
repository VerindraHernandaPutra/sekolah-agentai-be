import os
import sys
import time
import logging
import json
from datetime import timedelta
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

# --- IMPORT FASTAPI & SECURITY ---
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel

# Tambahkan path root project ke sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # --- IMPORT KONFIGURASI & AI ---
    # Gunakan 'app.' prefix agar konsisten dengan models_user.py
    from app.config import Config
    from app.enhanced_database import initialize_qdrant, initialize_embedder
    from app.enhanced_llm_utils import EnhancedLLMClient 
    from app.enhanced_query_executor import EnhancedQueryExecutor
    from app.enhanced_response_generator import EnhancedResponseGenerator
    
    # --- IMPORT DATABASE & AUTHENTICATION (FIXED IMPORTS) ---
    # PENTING: Gunakan 'app.database' bukan 'database' agar Metadata Base sinkron
    from app.database import get_db, engine, Base
    from app.models_user import User
    from app.auth import (
        get_current_user_bearer, 
        create_access_token, 
        authenticate_user, 
        get_password_hash,
        ACCESS_TOKEN_EXPIRE_MINUTES
    )
    from app.models import QueryRequest, QueryResponse, HealthResponse, Token

except ImportError as e:
    print(f"Import error in api.py: {e}")
    # Fallback untuk debugging path jika import app. gagal
    try:
        from config import Config
        from database import get_db, engine, Base
        from models_user import User
        from auth import get_current_user_bearer, create_access_token, authenticate_user, get_password_hash, ACCESS_TOKEN_EXPIRE_MINUTES
        from models import QueryRequest, QueryResponse, HealthResponse, Token
        # ... import lainnya manual jika perlu
    except ImportError:
        raise e

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
qdrant_client = None
embedder = None
llm_client = None
executor = None
generator = None
active_llm_provider = "Unknown"
active_llm_model = "Unknown"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle Manager:
    1. Membuat tabel database user jika belum ada.
    2. Inisialisasi koneksi AI (Qdrant, LLM).
    """
    global qdrant_client, embedder, llm_client, executor, generator
    global active_llm_provider, active_llm_model
    
    logger.info("🚀 Initializing AgentAI system components...")
    
    # --- 1. INISIALISASI TABEL DATABASE ---
    try:
        # Ini sekarang menggunakan Base yang SAMA dengan User model
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created/verified (users.db).")
    except Exception as e:
        logger.error(f"❌ Failed to create database tables: {e}")

    # --- 2. INISIALISASI AI COMPONENT ---
    try:
        qdrant_client = initialize_qdrant()
        embedder = initialize_embedder()
        
        try:
            logger.info(f"🔌 Connecting to PRIMARY LLM: Gemini ({Config.GEMINI_MODEL})...")
            primary_client = EnhancedLLMClient(provider="gemini")
            
            if primary_client.check_connection():
                llm_client = primary_client
                active_llm_provider = "Google Gemini"
                active_llm_model = Config.GEMINI_MODEL
                logger.info("✅ SUKSES: Terhubung ke Google Gemini (Primary).")
            else:
                raise ConnectionError("Gemini check_connection returned False")
                
        except Exception as e:
            logger.warning(f"⚠️ GAGAL menghubungkan ke Gemini: {e}")
            logger.info(f"🔌 Connecting to BACKUP LLM: Ollama ({Config.OLLAMA_MODEL})...")
            
            try:
                backup_client = EnhancedLLMClient(provider="ollama")
                if backup_client.check_connection():
                    llm_client = backup_client
                    active_llm_provider = "Ollama (Local)"
                    active_llm_model = Config.OLLAMA_MODEL
                    logger.info("✅ SUKSES: Terhubung ke Ollama Local (Backup).")
                else:
                    raise ConnectionError("Ollama check_connection returned False")
            except Exception as backup_e:
                logger.error(f"❌ FATAL ERROR: Gagal menghubungkan ke Primary DAN Backup LLM.")
                llm_client = None

        if llm_client:
            executor = EnhancedQueryExecutor(qdrant_client, embedder, llm_client)
            generator = EnhancedResponseGenerator(llm_client)
            logger.info(f"🚀 System READY. Running with: {active_llm_provider}")
        else:
            logger.critical("System started in DEGRADED mode (No LLM).")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        # Jangan raise error agar aplikasi tetap jalan minimal untuk DB
    
    yield
    
    logger.info("Shutting down application")

app = FastAPI(
    title="Sidoarjo School Chatbot API (AgentAI)",
    description="RAG-based Chatbot API with Database Authentication.",
    version="2.2.0",
    lifespan=lifespan
)

# CORS configuration
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserCreate(BaseModel):
    username: str
    password: str

# ==========================================
# ENDPOINTS
# ==========================================

@app.post("/register", status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Username already registered"
        )
    
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password)
    
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return {"message": "User created successfully", "username": new_user.username}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, 
        expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")

@app.get("/")
async def root():
    return {
        "message": "AgentAI Backend Service is Running", 
        "version": "2.2.0",
        "auth_type": "database_sqlite",
        "active_llm": active_llm_provider,
        "status": "ready" if llm_client else "degraded"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    llm_healthy = False
    if llm_client and hasattr(llm_client, 'check_connection'):
        try:
            llm_healthy = llm_client.check_connection()
        except:
            llm_healthy = False

    status_map = {
        "qdrant": "healthy" if qdrant_client else "unhealthy",
        "embedder": "healthy" if embedder else "unhealthy",
        "llm_connection": "healthy" if llm_healthy else "unhealthy",
        "active_provider": active_llm_provider
    }
    
    overall = "healthy" if all(v == "healthy" for v in status_map.values()) else "unhealthy"
    return HealthResponse(status=overall, components=status_map)

@app.post("/query", response_model=QueryResponse)
async def process_query(
    query_request: QueryRequest, 
    username: str = Depends(get_current_user_bearer)
):
    start_time = time.time()
    
    if not llm_client or not executor or not generator:
        return QueryResponse(
            response={
                "type": "error", 
                "narrative": "Layanan AI sedang tidak tersedia.", 
                "data": []
            },
            results_count=0, execution_time=0, intent="error", confidence=0.0, response_type="json"
        )
    
    try:
        logger.info(f"Processing query from user {username}: {query_request.query}")
        results, plan = executor.execute_query(query_request.query)
        json_response_str = generator.generate_response(query_request.query, results, plan)
        
        try:
            response_obj = json.loads(json_response_str)
        except json.JSONDecodeError:
            response_obj = {"type": "text", "narrative": json_response_str, "data": [], "meta": {}}
        
        return QueryResponse(
            response=response_obj,
            results_count=len(results),
            execution_time=time.time() - start_time,
            intent=plan.intent,
            confidence=plan.confidence,
            filters_applied=plan.filters,
            response_type="json"
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        error_response = {
            "type": "error",
            "narrative": f"Error: {str(e)}",
            "data": [],
            "meta": {"provider": active_llm_provider}
        }
        return QueryResponse(
            response=error_response,
            results_count=0,
            execution_time=time.time() - start_time,
            intent="error",
            confidence=0.0,
            response_type="json"
        )

@app.get("/sample-queries")
async def sample_queries():
    return {
        "search_school_samples": [
            "Sekolah negeri dengan akreditasi A di Kecamatan Waru",
            "SMA swasta yang punya lebih dari 500 siswa"
        ]
    }