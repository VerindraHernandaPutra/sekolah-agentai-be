# enhanced_database.py
import logging
from config import Config
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

def initialize_qdrant(host: str = None, port: int = None) -> QdrantClient:
    """
    Enhanced Qdrant initialization with better error handling.
    Bertanggung jawab untuk koneksi ke Vector Database.
    """
    # Gunakan nilai dari parameter atau fallback ke Config
    host = host or Config.QDRANT_HOST
    port = port or Config.QDRANT_PORT
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Initializing Qdrant connection at {host}:{port}...")
        
        # Initialize client
        qdrant = QdrantClient(
            host=host,
            port=port,
            timeout=Config.QDRANT_TIMEOUT,
            prefer_grpc=True  # Use gRPC for better performance if available
        )
        
        # Test connection by fetching collections
        collections = qdrant.get_collections()
        collection_names = [c.name for c in collections.collections]
        logger.info(f"✅ Connected to Qdrant. Available collections: {collection_names}")
        
        # Check if our target collection exists
        if Config.COLLECTION_NAME not in collection_names:
            logger.warning(f"⚠️ Collection '{Config.COLLECTION_NAME}' not found! You may need to run data ingestion.")
        else:
            logger.info(f"✅ Collection '{Config.COLLECTION_NAME}' is ready.")
        
        return qdrant
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to Qdrant: {e}")
        raise RuntimeError(f"Qdrant connection failed. Ensure Qdrant server is running at {host}:{port}")

def initialize_embedder(model_name: str = None) -> SentenceTransformer:
    """
    Enhanced embedder initialization.
    Memuat model Sentence Transformer ke memori untuk vektorisasi query.
    """
    model_name = model_name or Config.EMBEDDING_MODEL
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Loading embedding model: {model_name}...")
        embedder = SentenceTransformer(model_name)
        
        # Quick test to ensure model is working
        test_text = "test embedding"
        test_embedding = embedder.encode(test_text, show_progress_bar=False)
        
        # Log vector dimension (penting untuk debugging jika tidak cocok dengan Qdrant config)
        logger.info(f"✅ Embedding model loaded. Vector dimension: {len(test_embedding)}")
        
        return embedder
        
    except Exception as e:
        logger.error(f"❌ Failed to load embedding model: {e}")
        raise RuntimeError(f"Embedding model initialization failed: {e}")

def create_collection_if_needed(qdrant: QdrantClient, vector_size: int = 384):
    """
    Utility function to create collection if it doesn't exist.
    Biasanya dipanggil saat script ingestion/setup awal.
    """
    logger = logging.getLogger(__name__)
    
    try:
        collections = qdrant.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if Config.COLLECTION_NAME not in collection_names:
            logger.info(f"Creating collection '{Config.COLLECTION_NAME}' with size {vector_size}...")
            qdrant.create_collection(
                collection_name=Config.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"✅ Created collection '{Config.COLLECTION_NAME}' successfully")
        else:
            logger.info(f"ℹ️ Collection '{Config.COLLECTION_NAME}' already exists")
            
    except Exception as e:
        logger.warning(f"⚠️ Collection creation check failed: {e}")