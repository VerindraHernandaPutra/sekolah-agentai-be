# enhanced_response_generator.py
import time
import logging
import json
from typing import List, Dict, Any, Optional
from app.config import Config
from app.enhanced_query_planner import QueryPlan
from app.enhanced_llm_utils import EnhancedLLMClient

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
        
        # 1. Handle Special Intents (No Data Needed)
        if plan.intent in ["greeting", "unknown_intent", "out_of_scope_location", "irrelevant_query"]:
            return self._generate_special_response(plan)

        # 2. Handle No Results
        if not schools:
            return self._generate_no_results(user_query, plan)
        
        # 3. Tentukan Tipe Tampilan & Siapkan Data
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
You are a helpful School Database Assistant.
Task: Provide a neutral, factual summary (max 2 sentences) in Indonesian based ONLY on the provided data.

User Query: "{user_query}"
Data Found: {total} schools.
Stats: {negeri} Public (Negeri), {swasta} Private (Swasta).
Top Results: {top_names_str}, etc.
Active Filters: {filter_str}

Guidelines:
1. Be direct and consistent. Example: "Berikut adalah {total} sekolah yang sesuai dengan pencarian Anda."
2. Do NOT add roles, opinions, or insights about "sales", "internet potential", or "Telkom".
3. Do NOT mention specific locations unless they are in the Active Filters.
4. Keep it professional and concise.

Answer:
"""
            narrative = self.llm_client.call_llm(
                prompt, 
                temperature=Config.LLM_TEMPERATURES["response_generation"],
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
You are a helpful School Database Assistant.
Task: Provide a neutral, factual summary (max 2 sentences) in Indonesian based ONLY on the provided data.

Data Sekolah:
{context_str}

Panduan:
1. Provide a brief overview of the school.
2. Use professional and neutral language.
3. Do NOT make up facts or mention "sales potential".

Jawab:
"""
            narrative = self.llm_client.call_llm(prompt, temperature=Config.LLM_TEMPERATURES["response_generation"], use_streaming=True)
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

    def _generate_special_response(self, plan: QueryPlan) -> str:
        """Handle greetings and refusals"""
        narrative = ""

        if plan.intent == "greeting":
            narrative = "Halo! Saya adalah Asisten Database Sekolah Sidoarjo. Saya bisa membantu Anda mencari informasi tentang SD, SMP, SMA, dan SMK di wilayah Sidoarjo. Silakan tanya saya tentang lokasi, status, atau jumlah siswa sekolah!"

        elif plan.intent == "out_of_scope_location":
            loc = plan.filters.get("location", "luar Sidoarjo")
            narrative = f"Mohon maaf, cakupan data saya saat ini HANYA terbatas pada sekolah-sekolah di Kabupaten Sidoarjo. Saya tidak memiliki data sekolah di {loc.title()}."

        elif plan.intent == "unknown_intent":
            keyword = plan.filters.get("unsupported", "tersebut")
            narrative = f"Mohon maaf, database saya hanya mencakup data pokok pendidikan (Dapodik) seperti profil, alamat, jumlah siswa/guru, dan fasilitas dasar. Saya TIDAK memiliki data detail mengenai {keyword}."

        elif plan.intent == "irrelevant_query":
            narrative = "Mohon maaf, saya tidak mengerti permintaan Anda. Saya adalah asisten data sekolah Sidoarjo. Silakan tanya tentang daftar sekolah, lokasi, statistik siswa/guru, atau profil sekolah tertentu."

        response = {
            "type": "none",
            "narrative": narrative,
            "data": [],
            "meta": {"intent": plan.intent}
        }
        return json.dumps(response)

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