import os
import sys
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# === 1. Konfigurasi dasar ===
COLLECTION_NAME = "sekolah_indonesia"  # ganti sesuai kebutuhan
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384  # Dimensi model MiniLM

# Kolom yang rawan diubah jadi angka tapi harus tetap string
FORCE_STRING_COLS = [
    "rt", "rw", "nomorTelepon", "kodeProvinsi", "kodeKabupaten",
    "kodeKecamatan", "kodeWilayah", "npyp"
]

# === 2. Lokasi file CSV gabungan ===
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "..", "combain", "combined.csv")

if not os.path.exists(csv_path):
    print(f"[ERROR] File gabungan tidak ditemukan: {csv_path}")
    print("Jalankan terlebih dahulu: python combain/combain.py")
    sys.exit(1)

print(f"[INFO] Membaca data dari: {csv_path}")

# === 3. Baca CSV dengan perlakuan khusus untuk kolom tertentu ===
df = pd.read_csv(csv_path, dtype={col: str for col in FORCE_STRING_COLS})
df.columns = df.columns.str.strip()

# Pastikan kolom-kolom yang dipaksa string tidak hilang leading zero atau jadi float
for col in FORCE_STRING_COLS:
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str).str.strip()
        df[col] = df[col].str.replace(r"\.0$", "", regex=True)

print(f"[INFO] Total baris: {len(df)}")
print(f"[INFO] Kolom: {list(df.columns)}")

# === 4. Load model embedding ===
print(f"[INFO] Memuat model embedding: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)

# === 5. Koneksi ke Qdrant ===
print("[INFO] Menghubungkan ke Qdrant...")
client = QdrantClient(host="localhost", port=6333)

# Hapus koleksi lama jika sudah ada, lalu buat baru
if client.collection_exists(COLLECTION_NAME):
    print(f"[WARNING] Koleksi '{COLLECTION_NAME}' sudah ada, akan dihapus ulang.")
    client.delete_collection(collection_name=COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)

print(f"[INFO] Koleksi '{COLLECTION_NAME}' berhasil dibuat!")

# === 6. Konversi setiap baris menjadi vektor dan payload ===
points = []
print("[INFO] Membuat embedding dan payload...")
for idx, row in df.iterrows():
    combined_text = " | ".join([str(val) for val in row.values if pd.notna(val)])
    vector = model.encode(combined_text).tolist()
    payload = row.to_dict()

    points.append(PointStruct(
        id=idx,
        vector=vector,
        payload=payload
    ))

# === 7. Upsert ke Qdrant ===
print(f"[INFO] Mengimpor {len(points)} data ke Qdrant...")
client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"[SUCCESS] {len(points)} data berhasil diimpor ke koleksi '{COLLECTION_NAME}'!")
