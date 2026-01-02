# enhanced_response_generator.py
import time
import logging
import json
from typing import List, Dict, Any, Optional
from config import Config
from enhanced_query_planner import QueryPlan
from enhanced_llm_utils import EnhancedLLMClient

class EnhancedResponseGenerator:
    """
    Modern Response Generator.
    Output: JSON String berisi 'narrative' (LLM) dan 'data' (Raw DB).
    Frontend bertanggung jawab merender Tabel berdasarkan 'data'.
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or EnhancedLLMClient(provider="gemini")
        self.logger = logging.getLogger(__name__)
        
    def generate_response(self, user_query: str, schools: List[Dict], plan: QueryPlan) -> str:
        """
        Orchestrates the response generation.
        Returns a JSON string containing:
        {
            "type": "list" | "detail" | "none",
            "narrative": "Penjelasan natural language dari LLM...",
            "data": [ ... list data sekolah asli ... ],
            "meta": { ... info tambahan ... }
        }
        """
        start_time = time.time()
        
        # 1. Handle No Results
        if not schools:
            return self._generate_no_results(user_query, plan)
        
        # 2. Tentukan Tipe Tampilan & Siapkan Data
        try:
            # Bersihkan data untuk frontend (hapus field internal vector, score, dll jika perlu)
            cleaned_data = self._clean_data_for_frontend(schools)
            
            if plan.intent == "get_info":
                # --- SINGLE PROFILE MODE ---
                response_type = "detail"
                # Ambil 1 data teratas
                target_school = cleaned_data[0]
                # Minta LLM buat narasi profil
                narrative = self._generate_narrative_profile(user_query, target_school)
                final_data = target_school
                
            else:
                # --- LIST / TABLE MODE ---
                response_type = "list"
                # Minta LLM buat ringkasan/insight dari data (misal: "Ditemukan 5 SMA Negeri...")
                # Kita kirim subset data ke LLM agar tidak boros token, tapi kirim semua data ke frontend
                narrative = self._generate_narrative_list(user_query, schools, plan)
                final_data = cleaned_data
            
            # 3. Construct Final Response Object
            response_object = {
                "type": response_type,
                "narrative": narrative, # Teks dari AI
                "data": final_data,     # Data murni dari DB (Akurat 100%)
                "meta": {
                    "count": len(schools),
                    "filters": plan.filters
                }
            }
            
            execution_time = time.time() - start_time
            self.logger.info(f"Response generated in {execution_time:.2f}s (Type: {response_type})")
            
            # Return as string untuk dikirim via API
            return json.dumps(response_object)
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return self._generate_fallback_error(str(e))

    def _generate_narrative_list(self, user_query: str, schools: List[Dict], plan: QueryPlan) -> str:
        """
        Menggunakan LLM hanya untuk membuat ringkasan/insight eksekutif.
        Tidak menyuruh LLM menulis ulang data tabel.
        """
        try:
            # Hitung statistik sederhana untuk membantu LLM
            total = len(schools)
            negeri = sum(1 for s in schools if s.get('status_sekolah') == 'Negeri')
            swasta = sum(1 for s in schools if s.get('status_sekolah') == 'Swasta')
            
            # Ambil 3-5 nama sekolah teratas sebagai contoh
            top_names = [s.get('nama') for s in schools[:3]]
            top_names_str = ", ".join(top_names)
            
            # Filter context string
            filter_str = ", ".join([f"{k}: {v}" for k,v in plan.filters.items()])

            prompt = f"""
Kamu adalah Asisten Sales Cerdas untuk PT Telkom.
Tugas: Berikan ringkasan eksekutif singkat (maksimal 3 kalimat) dalam Bahasa Indonesia yang natural untuk menjawab user.

User Query: "{user_query}"
Data Ditemukan: {total} sekolah.
Statistik: {negeri} Negeri, {swasta} Swasta.
Top Result: {top_names_str}, dll.
Filter Aktif: {filter_str}

Panduan:
1. Langsung jawab intinya. Contoh: "Berikut adalah 20 sekolah negeri di Sidoarjo yang Anda cari. Sebagian besar berlokasi di..."
2. Jangan sebutkan "Berikut adalah JSON" atau hal teknis.
3. Berikan insight singkat jika ada (misal: "Sekolah-sekolah ini memiliki potensi tinggi...").
4. Jangan buat list/bullet points, cukup paragraf pendek. Data detail sudah ada di tabel.

Jawab:
"""
            narrative = self.llm_client.call_llm(
                prompt, 
                temperature=0.3, 
                use_streaming=True # Streaming oke untuk narasi pendek
            )
            
            return narrative.strip() if narrative else f"Berikut adalah {total} sekolah yang sesuai dengan pencarian Anda."

        except Exception as e:
            self.logger.warning(f"LLM narrative failed: {e}")
            return f"Ditemukan {len(schools)} sekolah yang sesuai dengan kriteria pencarian."

    def _generate_narrative_profile(self, user_query: str, school: Dict) -> str:
        """
        Membuat narasi profil sekolah.
        """
        try:
            # Flatten data penting untuk konteks LLM
            context_str = f"""
            Nama: {school.get('nama')}
            NPSN: {school.get('npsn')}
            Alamat: {school.get('alamatJalan')}, {school.get('namaKecamatan')}
            Status: {school.get('status_sekolah')}
            Siswa: {school.get('pd')}
            Guru: {school.get('ptk')}
            """

            prompt = f"""
Kamu adalah Asisten Sales Telkom.
Tugas: Jelaskan profil singkat sekolah ini kepada sales dalam 2-3 kalimat persuasif.

Data Sekolah:
{context_str}

Panduan:
1. Highlight potensi sekolah (jumlah siswa besar = potensi internet besar).
2. Gunakan bahasa profesional dan natural.
3. Jangan mengulang semua data spesifik karena user sudah melihat datanya di layar.

Jawab:
"""
            narrative = self.llm_client.call_llm(prompt, temperature=0.4, use_streaming=True)
            return narrative.strip() if narrative else f"Berikut adalah profil detail dari {school.get('nama')}."

        except Exception:
            return f"Berikut adalah data detail untuk {school.get('nama')}."

    def _clean_data_for_frontend(self, schools: List[Dict]) -> List[Dict]:
        """
        Membersihkan data untuk frontend.
        Memastikan semua field ada, mengubah None menjadi string kosong agar aman di JS.
        """
        cleaned_list = []
        # Field internal Qdrant yang tidak perlu dikirim
        exclude_fields = {'embedding_vector', 'vector', '_id', '_score'} 
        
        for s in schools:
            new_s = s.copy()
            
            # 1. Hapus field internal
            for field in exclude_fields:
                new_s.pop(field, None)
            
            # 2. Format Koordinat (Gabungkan jika terpisah)
            if 'lintang' in new_s and 'bujur' in new_s:
                new_s['koordinat'] = f"{new_s['lintang']}, {new_s['bujur']}"
            
            # 3. Pastikan field Valid ada, jika null isi string kosong (bukan "-")
            # Biarkan Frontend yang memutuskan mau menampilkan "-" atau "Tidak Tersedia"
            for field in Config.VALID_FIELDS:
                if field not in new_s or new_s[field] is None:
                    new_s[field] = ""
                # Konversi angka ke string agar konsisten (opsional, tapi aman untuk FE)
                elif isinstance(new_s[field], (int, float)):
                     # Jangan ubah 0 jadi kosong, biarkan 0
                    new_s[field] = new_s[field]
            
            cleaned_list.append(new_s)
        return cleaned_list

    def _generate_no_results(self, query: str, plan: QueryPlan) -> str:
        """Return JSON structure for empty state"""
        response = {
            "type": "none",
            "narrative": f"Mohon maaf, saya tidak menemukan data sekolah yang sesuai dengan kriteria: '{query}'.",
            "data": [],
            "meta": {
                "suggestion": "Coba kurangi filter atau gunakan kata kunci yang lebih umum (misal: cari nama kecamatan saja)."
            }
        }
        return json.dumps(response)

    def _generate_fallback_error(self, error_msg: str) -> str:
        """Return JSON structure for error state"""
        response = {
            "type": "error",
            "narrative": "Terjadi kesalahan saat memproses data.",
            "data": [],
            "meta": {"error": error_msg}
        }
        return json.dumps(response)