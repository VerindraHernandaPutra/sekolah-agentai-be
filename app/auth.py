from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

# Import konfigurasi dari config.py yang sudah diekspor
from app.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from app.database import get_db
from app.models_user import User

# Setup Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Fungsi Helper Password ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# --- Fungsi Helper Token ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- Logic Login Utama (Database) ---
def authenticate_user(db: Session, username: str, password: str):
    # Cari user di database
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False
    # Cek password hash
    if not verify_password(password, user.hashed_password):
        return False
    return user

# --- Middleware: Cek Token di setiap request ---
# FUNGSI INI DIGANTI NAMANYA AGAR SESUAI DENGAN API.PY
async def get_current_user_bearer(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # Pastikan user masih ada di DB
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user.username

# --- Alias untuk Kompatibilitas ---
# Jika ada kode lain yang memanggil get_current_user_basic, kita arahkan ke sini juga sementara
def get_current_user_basic(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    return get_current_user_bearer(token, db)