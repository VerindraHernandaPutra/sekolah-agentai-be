# enhanced_query_planner.py
import json
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from app.config import Config
from app.utils import (
    normalize_query,
    extract_location_entities,
    extract_numbers_with_context,
    extract_school_name,
    extract_filters_from_query,
    extract_limit_from_query,
    extract_sort_preference
)
from app.enhanced_llm_utils import EnhancedLLMClient

@dataclass
class QueryPlan:
    """Structured query plan object"""
    intent: str
    routing: str  # 'structured', 'semantic', 'hybrid'
    filters: Dict[str, Any]
    text_query: Optional[str]
    fields: List[str]
    limit: int
    sort: Optional[List[Dict[str, str]]]
    confidence: float

class EnhancedQueryPlanner:
    """
    Enhanced query planner that uses Hybrid LLM (Gemini/Ollama) to understand user intent.
    """
    
    def __init__(self, llm_client=None):
        # Jika llm_client belum ada, inisialisasi default (Gemini)
        self.llm_client = llm_client or EnhancedLLMClient(provider="gemini")
        self.cache = {}
        self.logger = logging.getLogger(__name__)
    
    def plan_query(self, user_query: str) -> QueryPlan:
        """
        Main planning pipeline:
        1. Special Intent Check (Greeting, Unsupported, Out of Scope) -> Instant Reply
        2. Check Cache
        3. Get Info Check (Specific School Name) -> Fast Path
        4. Strong Rule-Based Check (Regex Priority) -> Robust Path for Ollama
        5. LLM Analysis (Deep understanding for ambiguous queries) -> Smart Path
        6. Fallback Rule-Based -> Safety Net
        """
        # 0. SPECIAL INTENTS (Pre-computation)
        special_plan = self._detect_special_intents(user_query)
        if special_plan:
            return special_plan

        # 0.5 RELEVANCE CHECK
        # If query is gibberish/irrelevant, stop here.
        if not self._is_query_relevant(user_query):
             return QueryPlan(intent="irrelevant_query", routing="none", filters={}, text_query=user_query, fields=[], limit=0, sort=None, confidence=1.0)

        cache_key = normalize_query(user_query)
        if cache_key in self.cache:
            self.logger.info("Using cached query plan")
            return self.cache[cache_key]
        
        # 1. FAST PATH: Cek apakah ini pencarian nama sekolah spesifik?
        # Jika user mengetik nama sekolah panjang, biasanya mereka ingin detail sekolah itu.
        if self._detect_get_info_intent(user_query):
            plan = self._create_get_info_plan(user_query)
            self.cache[cache_key] = plan
            return plan
        
        # 2. ROBUST PATH: Cek Regex DULUAN.
        # Ini SANGAT PENTING untuk Ollama. Ollama (Phi-3) sering gagal menangkap filter numerik 
        # (seperti "< 500 siswa"). Regex Python jauh lebih akurat untuk ini.
        filters_regex = extract_filters_from_query(user_query)
        self.logger.info(f"Regex extracted: {filters_regex}")
        
        # Kriteria "Strong Filters": Jika ada filter numerik (pd/ptk) atau npsn, percayai Regex.
        # Atau jika regex menemukan lebih dari 1 filter (misal: lokasi + status).
        has_numeric = any(k in filters_regex for k in ['pd', 'ptk', 'jml_lab', 'npsn'])
        is_strong_regex = has_numeric or len(filters_regex) >= 2
        
        if is_strong_regex:
             self.logger.info(f"🎯 Strong Rule-based plan used (High Confidence): {filters_regex}")
             plan = QueryPlan(
                intent="search_school",
                routing="structured",
                filters=filters_regex,
                text_query=user_query,
                fields=Config.COMPREHENSIVE_FIELDS["search_school"],
                limit=20,
                sort=None,
                confidence=0.9
             )
             self.cache[cache_key] = plan
             return plan

        # 3. SMART PATH: Coba gunakan LLM untuk analisis mendalam
        # Gunakan LLM hanya jika query ambigu (tidak tertangkap regex dengan kuat)
        if self.llm_client and self.llm_client.check_connection():
            try:
                self.logger.info("Attempting LLM-based query planning...")
                plan = self._create_llm_plan(user_query)
                
                # Validasi hasil LLM: Jika confidence tinggi, gunakan.
                if plan and plan.confidence > 0.6:
                    self.logger.info(f"LLM Plan Created: {plan.intent} | Filters: {len(plan.filters)}")
                    self.cache[cache_key] = plan
                    return plan
                else:
                    self.logger.warning("LLM plan creation failed or low confidence. Falling back to rules.")
            except Exception as e:
                self.logger.error(f"LLM planning failed: {e}")
        
        # 4. SAFETY NET: Fallback ke Rule-Based (apapun yang didapat)
        self.logger.info("Falling back to rule-based planning")
        plan = self._create_rule_based_plan(user_query)
        self.cache[cache_key] = plan
        return plan

    def _detect_get_info_intent(self, query: str) -> bool:
        """Detect if user wants details about a specific school"""
        query_lower = query.lower()
        
        # 1. Cek kata kunci filter/pencarian (Jika ada, BUKAN get_info)
        # Jika ada kata "kurang dari", "lebih dari", "yang memiliki", itu pasti search
        # Added "dan", "atau" to catch list queries like "SMA dan SMK"
        search_indicators = [
            "kurang dari", "lebih dari", "yang punya", "yang memiliki",
            "dengan akreditasi", "cari", "daftar", "list", "top", "tampilkan",
            "sekolah yang", "sekolah dimana",
            "jumlah", "tabel", "apa saja", "ada sekolah"
        ]
        if any(ind in query_lower for ind in search_indicators):
            return False

        # 2. Keywords check (e.g., "profil sman 1")
        if any(p in query_lower for p in Config.QUERY_INTENTS["get_info"]):
            return True
        
        # 3. Specific school name check 
        school_name = extract_school_name(query)

        # Nama sekolah valid biasanya tidak sepanjang query itu sendiri (kecuali query sangat pendek)
        if school_name and len(school_name) > 3:
            # CHECK: If query contains explicit conjunctions, it's likely a list, not a single school info
            if " dan " in query_lower or " atau " in query_lower:
                return False

            # Jika panjang nama sekolah > 80% panjang query, mungkin itu memang nama sekolah
            # Tapi jika query panjang dan nama sekolah yang terdeteksi juga panjang banget, curigai itu kalimat
            # Updated threshold to 8 words to be safe
            if len(query.split()) > 8:
                return False
            return True
            
        return False

    def _create_get_info_plan(self, query: str) -> QueryPlan:
        """Create plan for specific school detail"""
        school_name = extract_school_name(query)
        filters = {}
        
        if school_name:
            # Use name_contains for partial matching to be more user-friendly
            # (e.g. "telkom" should match "SMK TELKOM SIDOARJO")
            filters["name_contains"] = school_name
            confidence = 0.95
            # USE HYBRID routing because 'name_contains' is not a valid Qdrant structured filter field.
            # We need vector search to find candidates, then manual filtering will enforce the name match.
            routing = "hybrid"
        else:
            # Fallback semantic search if name not perfectly extracted
            confidence = 0.6
            routing = "semantic"
            
        return QueryPlan(
            intent="get_info",
            routing=routing,
            filters=filters,
            text_query=query, # Keep original text for semantic search fallback
            fields=Config.COMPREHENSIVE_FIELDS["get_info"],
            limit=1,
            sort=None,
            confidence=confidence
        )
    
    def _create_llm_plan(self, user_query: str) -> Optional[QueryPlan]:
        """Generate QueryPlan using LLM"""
        try:
            prompt = self._create_comprehensive_prompt(user_query)
            
            # Gunakan mode non-streaming untuk JSON output agar integritas data terjaga
            llm_response = self.llm_client.call_llm(
                prompt, 
                temperature=Config.LLM_TEMPERATURES["query_planning"],
                use_streaming=False
            )
            
            if llm_response:
                return self._parse_llm_plan(llm_response, user_query)
                
        except Exception as e:
            self.logger.error(f"Error in _create_llm_plan: {e}")
        
        return None
    
    def _create_comprehensive_prompt(self, query: str) -> str:
        """
        Constructs the System Prompt.
        Injeksi FIELD_MAPPING dari Config agar LLM paham sinonim bahasa Indonesia.
        """
        
        # 1. Generate Mapping List (untuk konteks LLM)
        mapping_context = []
        seen_fields = set()
        for term, field in Config.FIELD_MAPPING.items():
            if field not in seen_fields:
                mapping_context.append(f"- '{term}' (dan variasinya) -> database field: '{field}'")
                seen_fields.add(field)
            else:
                # Add synonym example
                mapping_context.append(f"- '{term}' -> '{field}'")
        
        mapping_str = "\n".join(mapping_context[:100]) # Limit to top 100 mappings
        
        # 2. Schema Info
        schema_str = "\n".join([f"- {f}" for f in list(Config.VALID_FIELDS)[:20]]) 

        return f"""
You are an expert Query Planner for an Indonesian School Database System.
Your task is to analyze the User Query and convert it into a structured JSON Query Plan.

USER QUERY: "{query}"

### KNOWLEDGE BASE (SYNONYMS)
Use this mapping to understand user terms:
{mapping_str}

### RULES
1. **Filters**: Extract specific criteria.
   - Status: "Negeri" OR "Swasta". HANDLE NEGATION: "bukan negeri" -> "Swasta".
   - Akreditasi: "A", "B", "C".
   - Location: Map to "namaKecamatan" with "KEC. " prefix (e.g., "Candi" -> "KEC. CANDI").
   - Numeric: Use operators for "pd" (siswa), "ptk" (guru), etc. 
     Ex: "di atas 500 murid" -> {{"pd": {{"op": ">", "value": 500}}}}
   - Multiple Values: If multiple items are requested (e.g., "SMA dan SMK"), return a list: ["SMA", "SMK"].
   - Existential: "ada lab", "punya perpustakaan" -> filter for > 0.
   - Partial Name: If user says "nama [X]" (e.g. "nama telkom"), add filter: {{"name_contains": "Telkom"}}.
2. **Intent**:
   - "search_school": List/filter schools (plural, comparison, list).
   - "get_info": Detail of one specific school (singular name).
   - "count_query": Counting statistics (e.g. "berapa jumlah...").
   - "ranking_query": Best/Top schools (e.g. "sekolah terbaik", "unggulan", "favorit").
3. **Routing**:
   - "structured": If query has clear filters (status, loc, etc).
   - "semantic": If query is vague (e.g., "sekolah favorit").
   - "hybrid": If query has both.

### OUTPUT FORMAT (JSON ONLY)
{{
    "intent": "search_school",
    "routing": "hybrid",
    "filters": {{
        "status_sekolah": "Negeri",
        "namaKecamatan": "KEC. WARU",
        "bentukPendidikan": ["SMA", "SMK"]
    }},
    "text_query": "sekolah bagus",
    "fields": ["nama", "npsn", "alamatJalan"],
    "limit": 20,
    "confidence": 0.9
}}

Respond ONLY with valid JSON. No markdown code blocks.
"""
    
    def _parse_llm_plan(self, response: str, user_query: str) -> Optional[QueryPlan]:
        """Safely parse JSON from LLM response"""
        try:
            # 1. Clean Markdown wrappers (```json ... ```)
            cleaned = response.replace("```json", "").replace("```", "").strip()
            
            # 2. Find JSON boundaries
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            if start == -1 or end == -1:
                raise ValueError("No JSON object found")
                
            json_str = cleaned[start:end+1]
            data = json.loads(json_str)
            
            # 3. Validate & Sanitize Filters
            raw_filters = data.get("filters", {})
            clean_filters = self._validate_filters(raw_filters)
            
            # 4. Construct Plan
            return QueryPlan(
                intent=data.get("intent", "search_school"),
                routing=data.get("routing", "hybrid"),
                filters=clean_filters,
                text_query=data.get("text_query", user_query),
                fields=data.get("fields", ["nama", "npsn"]),
                limit=min(data.get("limit", 20), Config.MAX_LIMIT),
                sort=data.get("sort"),
                confidence=float(data.get("confidence", 0.5))
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM plan: {e}. Response was: {response[:100]}...")
            return None

    def _validate_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates keys against Config.VALID_FIELDS and fixes common LLM mistakes.
        """
        validated = {}
        for k, v in filters.items():
            # Direct valid field
            if k in Config.VALID_FIELDS or k in Config.NUMERIC_FIELDS:
                validated[k] = v
            
            # Special keys
            elif k == "name_contains":
                validated[k] = v

            # Common Mistakes correction (LLM terkadang halusinasi nama field)
            elif k.lower() == "kecamatan": validated["namaKecamatan"] = v
            elif k.lower() == "kabupaten": validated["namaKabupaten"] = v
            elif k.lower() == "provinsi": validated["namaProvinsi"] = v
            elif k.lower() == "status": validated["status_sekolah"] = v
            elif k.lower() == "jumlah_siswa": validated["pd"] = v
            elif k.lower() == "jumlah_guru": validated["ptk"] = v
            elif k.lower() == "nama_partial": validated["name_contains"] = v # Alias handling
            
        return validated

    def _create_rule_based_plan(self, user_query: str) -> QueryPlan:
        """Fallback mechanism using regex extraction from utils"""
        # Menggunakan fungsi regex yang kuat dari utils.py
        filters = extract_filters_from_query(user_query)
        
        # Extract limit if exists
        requested_limit = extract_limit_from_query(user_query)
        limit = requested_limit if requested_limit else 20
        # Ensure limit doesn't exceed max config
        limit = min(limit, Config.MAX_LIMIT)

        # Extract sort preference
        sort_pref = extract_sort_preference(user_query)

        # Basic intent detection
        query_lower = user_query.lower()
        intent = "search_school"

        # Check for specific intents
        if any(x in query_lower for x in ["info", "detail", "profil"]):
            intent = "get_info"
        elif any(x in query_lower for x in ["terbaik", "bagus", "unggulan", "favorit", "top", "rank"]):
            intent = "ranking_query"
        elif any(x in query_lower for x in ["berapa", "jumlah", "total", "hitung"]):
            intent = "count_query"
        
        # Basic routing logic
        if len(filters) > 0:
            routing = "structured"
        else:
            routing = "semantic"
            
        return QueryPlan(
            intent=intent,
            routing=routing,
            filters=filters,
            text_query=user_query,
            fields=Config.COMPREHENSIVE_FIELDS.get(intent, ["nama", "npsn"]),
            limit=limit,
            sort=sort_pref,
            confidence=0.5
        )

    def _detect_special_intents(self, user_query: str) -> Optional[QueryPlan]:
        """Detect edge cases: greetings, out of scope, unsupported features"""
        q = user_query.lower()

        # 1. Greetings
        # Exact match or starts with greeting (e.g. "Halo admin")
        clean_q = re.sub(r'[^\w\s]', '', q) # remove punct
        tokens = clean_q.split()
        if any(g in tokens for g in Config.GREETING_KEYWORDS):
             return QueryPlan(intent="greeting", routing="none", filters={}, text_query=user_query, fields=[], limit=0, sort=None, confidence=1.0)

        # 2. Unsupported Features (Biaya, Ekskul, Jurusan specific)
        for keyword in Config.UNSUPPORTED_FEATURES:
            if keyword in q:
                # Special check to ensure it's not a valid search context?
                # For now, strict rejection for these keywords as per requirement.
                return QueryPlan(intent="unknown_intent", routing="none", filters={"unsupported": keyword}, text_query=user_query, fields=[], limit=0, sort=None, confidence=1.0)

        # 3. Out of Scope Locations
        for loc in Config.OUT_OF_SCOPE_LOCATIONS:
            if loc in q:
                return QueryPlan(intent="out_of_scope_location", routing="none", filters={"location": loc}, text_query=user_query, fields=[], limit=0, sort=None, confidence=1.0)

        return None

    def _is_query_relevant(self, query: str) -> bool:
        """
        Check if query contains at least one known domain keyword.
        Constructs a whitelist from Config dynamically.
        """
        q = query.lower()

        # 1. Gather all relevant terms
        relevant_terms = set()

        # Core domain terms
        relevant_terms.update(["sekolah", "pendidikan", "murid", "guru", "siswa", "kelas", "data", "info", "cari", "list", "daftar", "tampilkan", "berapa", "jumlah"])

        # Field Mapping keys (synonyms)
        relevant_terms.update(k.lower() for k in Config.FIELD_MAPPING.keys())

        # Value Mapping values (flattened)
        for category in Config.VALUE_MAPPING.values():
            for k, v in category.items():
                if isinstance(k, re.Pattern): continue # Skip regex keys
                relevant_terms.add(k.lower())
                relevant_terms.add(v.lower())

        # Query Intents (keywords)
        for intents in Config.QUERY_INTENTS.values():
            relevant_terms.update(i.lower() for i in intents)

        # Unsupported & Out of Scope (because if user asks about them, it IS relevant to the domain, just not supported)
        relevant_terms.update(k.lower() for k in Config.UNSUPPORTED_FEATURES)
        relevant_terms.update(k.lower() for k in Config.OUT_OF_SCOPE_LOCATIONS)

        # Greeting (handled by special intent, but safe to include)
        relevant_terms.update(k.lower() for k in Config.GREETING_KEYWORDS)

        # 2. Check overlap
        # Tokenize simply
        tokens = set(re.findall(r'\w+', q))

        # Allow if any token matches a relevant term
        # Use regex boundary check for short terms to avoid false positives (e.g. "sd" in "asdf")
        # For multi-word terms, allow substring match if length > 3

        for term in relevant_terms:
            if len(term) < 4:
                if re.search(rf"\b{re.escape(term)}\b", q):
                    self.logger.info(f"Query '{q}' is relevant due to term (boundary): '{term}'")
                    return True
            else:
                if term in q:
                    self.logger.info(f"Query '{q}' is relevant due to term (substring): '{term}'")
                    return True

        self.logger.info(f"Query '{q}' is IRRELEVANT")
        return False