# models.py
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None

class QueryResponse(BaseModel):
    # Mengubah tipe response menjadi Any agar bisa menampung JSON Object
    # Struktur: { "type": "list", "narrative": "...", "data": [...] }
    response: Any 
    
    results_count: int
    execution_time: float
    intent: str
    confidence: float
    filters_applied: Optional[Dict[str, Any]] = None
    
    # Response type sekarang defaultnya "json"
    response_type: str = "json"

class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]

class Token(BaseModel):
    access_token: str
    token_type: str