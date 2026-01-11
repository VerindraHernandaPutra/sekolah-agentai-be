# utils.py
import logging
import re
import requests
import json
import unicodedata
from typing import Dict, Any, Optional, List, Tuple
from qdrant_client import models
from app.config import Config

def setup_logging(level=logging.INFO):
    """Enhanced logging setup"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("rag_system.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    # Suppress verbose logs from external libraries
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

def check_ollama_connection(url: str = None) -> bool:
    """Enhanced Ollama connection check"""
    if url is None:
        # Convert chat endpoint to tags endpoint for lightweight check
        # Config.OLLAMA_URL biasanya .../api/chat, kita ubah ke .../api/tags
        base_url = Config.OLLAMA_URL.replace("/api/chat", "")
        url = f"{base_url}/api/tags"
    
    try:
        # Short timeout for health check
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except Exception as e:
        logging.warning(f"Ollama connection check failed: {e}")
        return False

def normalize_text(text: str) -> str:
    """Comprehensive text normalization"""
    if not text:
        return ""
    
    # Unicode normalization - handle non-breaking spaces
    text = text.replace('\xa0', ' ').replace('\u00a0', ' ')
    text = unicodedata.normalize('NFKD', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Convert to lowercase for comparison
    return text.lower()

def normalize_query(query: str) -> str:
    """Enhanced query normalization for caching keys"""
    normalized = normalize_text(query)
    
    # Remove common stop words that don't change semantic meaning for retrieval
    stop_words = {
        'yang', 'dengan', 'di', 'ke', 'dari', 'untuk', 'pada', 'dalam', 
        'ingin', 'tolong', 'cari', 'carikan', 'tampilkan', 'list', 'daftar'
    }
    
    words = [w for w in normalized.split() if w not in stop_words]
    return ' '.join(words)

def extract_npsn(text: str) -> Optional[str]:
    """
    Extract NPSN (Nomor Pokok Sekolah Nasional) specifically.
    NPSN is an 8-digit unique identifier.
    """
    text = normalize_text(text)
    
    # Pattern 1: Explicit mention (e.g., "npsn 20234567")
    explicit_match = re.search(r'npsn\s*:?\s*(\d{8})', text)
    if explicit_match:
        return explicit_match.group(1)
        
    # Pattern 2: Implicit 8-digit number (excluding phone numbers starting with 08)
    # Looks for 8 digits that are standalone words
    implicit_matches = re.findall(r'\b(?!08)(\d{8})\b', text)
    if implicit_matches:
        return implicit_matches[0]
        
    return None

def extract_numbers_with_context(text: str) -> List[Tuple[int, str, str]]:
    """
    Extract numbers with proper context mapping based on Config.
    Returns list of (number, field_name, operator)
    """
    text_clean = normalize_text(text)
    # Remove quoted sections to avoid false positives in school names
    text_clean = re.sub(r'"[^"]*"|\'[^\']*\'', '', text_clean)
    
    results = []
    
    # 1. Build dynamic patterns from Config.FIELD_MAPPING for numeric fields
    field_keywords = {}
    for keyword, field in Config.FIELD_MAPPING.items():
        if field in Config.NUMERIC_FIELDS:
            if field not in field_keywords:
                field_keywords[field] = []
            field_keywords[field].append(re.escape(keyword))

    # 2. Define Operator Regex Patterns
    # Maps regex group to operator symbol. Order matters (longer matches first).
    op_patterns = [
        (r'(?:lebih\s+dari|di\s+atas|melebihi|lebih\s+besar|banyak\s+dari)', '>'),
        (r'(?:kurang\s+dari|di\s+bawah|lebih\s+kecil|tidak\s+sampai|sedikit\s+dari)', '<'),
        (r'(?:minimal|paling\s+sedikit|setidaknya|gak\s+kurang|tidak\s+kurang)', '>='),
        (r'(?:maksimal|paling\s+banyak|mentok|gak\s+lebih|tidak\s+lebih)', '<='),
        (r'(?:tepat|persis|sama\s+dengan|jumlahnya|sebesar)', '=='),
    ]

    # 3. Scan for each numeric field type
    for field, keywords in field_keywords.items():
        if not keywords: continue
        
        # Join keywords with OR (|) e.g., (siswa|murid|anak)
        kw_regex = '|'.join(keywords)
        
        # Pattern A: Operator + Number + Keyword (e.g., "lebih dari 500 siswa")
        for op_regex, op_symbol in op_patterns:
            # Regex: Operator space Number space Keyword
            pattern_a = f"(?:{op_regex})\\s*(\\d+)\\s*(?:{kw_regex})"
            matches = re.finditer(pattern_a, text_clean)
            for m in matches:
                results.append((int(m.group(1)), field, op_symbol))
                
        # Pattern B: Keyword + Operator + Number (e.g., "jumlah siswa minimal 500")
        for op_regex, op_symbol in op_patterns:
             pattern_b = f"(?:{kw_regex})\\s*(?:yang)?\\s*(?:{op_regex})\\s*(\\d+)"
             matches_b = re.finditer(pattern_b, text_clean)
             for m in matches_b:
                 results.append((int(m.group(1)), field, op_symbol))

        # Pattern C: Keyword + Number (Implicit '>=') (e.g., "500 siswa", "siswa 500")
        # Only if not captured by A or B. We do a simpler pass.
        # Default assumption: "500 siswa" usually implies searching for schools around that size, 
        # but '>=' is a safe default for filtering.
        pattern_c1 = f"(\\d+)\\s*(?:{kw_regex})"
        pattern_c2 = f"(?:{kw_regex})\\s*(\\d+)"
        
        for p in [pattern_c1, pattern_c2]:
            matches_c = re.finditer(p, text_clean)
            for m in matches_c:
                val = int(m.group(1))
                # Avoid duplicates: check if this value/field combo already exists
                if not any(r[0] == val and r[1] == field for r in results):
                    results.append((val, field, '>='))

    return results

def extract_location_entities(text: str) -> Dict[str, str]:
    """
    Enhanced location extraction using Regex patterns from Config.
    Maps informal location names to standardized Database format.
    """
    locations = {}
    text_lower = normalize_text(text)
    
    # 1. Province Detection
    if any(w in text_lower for w in ['jawa timur', 'jatim', 'east java']):
        locations['namaProvinsi'] = 'PROV. JAWA TIMUR'
    
    # 2. Regency Detection
    if 'sidoarjo' in text_lower:
        locations['namaKabupaten'] = 'KAB. SIDOARJO'
    
    # 3. District (Kecamatan) Detection using Compiled Regex from Config
    # This iterates through the VALUE_MAPPING which maps patterns to standard names (e.g. "KEC. CANDI")
    kec_mapping = Config.VALUE_MAPPING.get("namaKecamatan", {})
    
    for pattern, standard_name in kec_mapping.items():
        if pattern.search(text_lower):
            locations['namaKecamatan'] = standard_name
            # We assume one district per query usually implies filtering by that district
            break 
            
    # 4. Village (Desa) Detection - Basic heuristic
    # Looks for "Desa X" or "Kelurahan Y"
    desa_match = re.search(r'(?:desa|kelurahan)\s+([a-z\s]+)', text_lower)
    if desa_match:
        # Clean up the captured group
        potential_desa = desa_match.group(1).strip()
        # Exclude stop words if they got caught
        potential_desa = re.sub(r'\b(di|ke|yang)\b', '', potential_desa).strip()
        if len(potential_desa) > 3:
            locations['namaDesa'] = potential_desa.upper()

    return locations

def extract_school_name(text: str) -> Optional[str]:
    """
    Enhanced school name extraction using patterns from Config.
    Prioritizes quoted strings and known school prefixes.
    Also supports explicit "nama [word]" pattern for strict filtering.
    Avoids false positives for list queries (e.g., "SMA dan SMK").
    """
    text_clean = normalize_text(text)
    
    # 1. Try quoted names first (highest priority)
    quoted = re.search(r'"([^"]+)"', text_clean)
    if quoted:
        return quoted.group(1).upper()

    # 2. Try Explicit "nama [X]" pattern (e.g. "nama telkom")
    # This allows users to force a name filter even if the word isn't a school prefix
    nama_match = re.search(r'\bnama\s+([a-zA-Z0-9]+)(?:\s|$)', text_clean, re.IGNORECASE)
    if nama_match:
        extracted = nama_match.group(1).upper()
        # Avoid stop words or conjunctions being picked up as names
        if extracted not in ["SEKOLAH", "YANG", "DI", "DENGAN", "DAN", "ATAU"]:
            return extracted
        
    # 3. Try Patterns from Config
    for pattern_str in Config.SCHOOL_NAME_PATTERNS:
        match = re.search(pattern_str, text_clean, re.IGNORECASE)
        if match:
            extracted = match.group(0).upper()

            # Additional Check: Avoid matching conjunctions that imply a list query
            # If the extracted name contains " DAN " or " ATAU ", it's likely a list
            if " DAN " in extracted or " ATAU " in extracted:
                continue

            return extracted
            
    return None

def build_qdrant_filter(filters: Dict[str, Any]) -> Optional[models.Filter]:
    """
    Converts a Python dictionary filter into a Qdrant Filter object.
    Supports nested operator logic (e.g., {'pd': {'op': '>', 'value': 500}})
    """
    if not filters:
        return None
    
    conditions = []
    logger = logging.getLogger(__name__)
    
    for key, val in filters.items():
        # Safety check: only process valid fields
        if key not in Config.VALID_FIELDS and key not in Config.NUMERIC_FIELDS:
            continue
            
        # Ignore virtual fields for Qdrant filtering
        if key == "name_contains":
            continue

        try:
            # Case A: Complex Numeric Filter (Dict with 'op' and 'value')
            if isinstance(val, dict) and 'op' in val and 'value' in val:
                op, v = val['op'], val['value']
                r = None
                
                if op == '>': r = models.Range(gt=v)
                elif op == '>=': r = models.Range(gte=v)
                elif op == '<': r = models.Range(lt=v)
                elif op == '<=': r = models.Range(lte=v)
                
                if r:
                    conditions.append(models.FieldCondition(key=key, range=r))
                else:
                    # Handle '==' or unknown op as Exact Match
                    conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=v)))
            
            # Case B: List of values (OR condition for a field)
            elif isinstance(val, list):
                should_conditions = [
                    models.FieldCondition(key=key, match=models.MatchValue(value=v))
                    for v in val
                ]
                conditions.append(models.Filter(should=should_conditions))
                
            # Case C: Simple Value (String/Int/Float) -> Exact Match
            else:
                # Special handling for fields that might need full-text match vs keyword match
                # In Qdrant, 'match' is for keyword/exact.
                
                # For location/status/ids, we usually want exact match if normalized
                if key in ['namaKecamatan', 'namaKabupaten', 'status_sekolah', 'akreditasi', 'bentukPendidikan', 'npsn']:
                     conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=val)))
                else:
                    # For names/addresses, exact match might be too strict, but for structured filtering
                    # it's safer. Semantic search handles the fuzzy part.
                    # Here we assume exact match for filters generated by planner.
                    conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=val)))

        except Exception as e:
            logger.warning(f"Failed to build Qdrant filter for {key}={val}: {e}")
            continue
                
    if not conditions:
        return None
        
    return models.Filter(must=conditions)

def extract_filters_from_query(query: str) -> Dict[str, Any]:
    """
    Robust Rule-Based Filter Extraction.
    Acts as a reliable fallback or pre-processor for the LLM.
    """
    filters = {}
    query_lower = normalize_text(query)
    
    # 1. NPSN Check (Highest Priority) - Identifier Unik
    npsn = extract_npsn(query)
    if npsn:
        filters['npsn'] = npsn
        # If NPSN is found, usually other filters are irrelevant for finding the specific school
        return filters 
        
    # 2. Status Sekolah (Negeri/Swasta) with Negation Support
    is_negeri = 'negeri' in query_lower
    is_swasta = 'swasta' in query_lower

    # Check for negation patterns before the keyword
    negation_negeri = re.search(r'\b(bukan|selain|non)\s+negeri', query_lower)
    negation_swasta = re.search(r'\b(bukan|selain|non)\s+swasta', query_lower)

    if negation_negeri:
        filters['status_sekolah'] = 'Swasta'
    elif negation_swasta:
        filters['status_sekolah'] = 'Negeri'
    elif is_negeri:
        filters['status_sekolah'] = 'Negeri'
    elif is_swasta:
        filters['status_sekolah'] = 'Swasta'
    
    # 3. Akreditasi
    # Regex to catch "akreditasi A", "nilai A", or just "A" if context supports it
    if re.search(r'\b(akreditasi|nilai|grade|peringkat)\s*:?\s*a\b', query_lower): filters['akreditasi'] = 'A'
    elif re.search(r'\b(akreditasi|nilai|grade|peringkat)\s*:?\s*b\b', query_lower): filters['akreditasi'] = 'B'
    elif re.search(r'\b(akreditasi|nilai|grade|peringkat)\s*:?\s*c\b', query_lower): filters['akreditasi'] = 'C'
    
    # 4. Bentuk Pendidikan (Jenjang) - Support Multiple Values
    # Iterate through mapping in Config
    found_forms = set()
    for key, val in Config.VALUE_MAPPING['bentukPendidikan'].items():
        # Ensure whole word match to avoid partials (e.g. "smp" matching inside "smpg")
        # Using regex boundary \b
        if re.search(rf"\b{re.escape(key)}\b", query_lower):
            found_forms.add(val)

    if found_forms:
        found_list = list(found_forms)
        if len(found_list) == 1:
            filters['bentukPendidikan'] = found_list[0]
        else:
            filters['bentukPendidikan'] = found_list
            
    # 5. Numeric Filters (Siswa, Guru, Fasilitas)
    numbers = extract_numbers_with_context(query)
    for val, field, op in numbers:
        filters[field] = {"op": op, "value": val}

    # 5b. Existential Checks (e.g., "punya lab", "ada perpustakaan")
    # If no numeric filter for these fields exists, we assume user wants > 0
    existential_map = {
        'lab': 'jml_lab',
        'laboratorium': 'jml_lab',
        'perpus': 'jml_perpus',
        'perpustakaan': 'jml_perpus',
        'komputer': 'jml_lab' # often implies lab computer
    }

    for kw, field in existential_map.items():
        if field not in filters and kw in query_lower:
             # Only add if keyword is present
             filters[field] = {"op": ">", "value": 0}

    # 6. Location (Kecamatan, Kab, Prov)
    locs = extract_location_entities(query)
    filters.update(locs)
    
    # 7. Explicit Name Filter ("nama [X]")
    # We use a special key 'name_contains' to signal partial/substring matching
    # re-use the logic from extract_school_name but strictly for the "nama" pattern
    nama_match = re.search(r'\bnama\s+([a-zA-Z0-9]+)(?:\s|$)', query_lower)
    if nama_match:
        extracted = nama_match.group(1).upper()
        if extracted not in ["SEKOLAH", "YANG", "DI", "DENGAN", "DAN", "ATAU"]:
            filters['name_contains'] = extracted

    return filters

def extract_limit_from_query(query: str) -> Optional[int]:
    """Extracts the requested number of results (limit) from the query."""
    text = normalize_text(query)

    # Pattern 1: Explicit "Top X" or "List X" or "Cari X"
    # e.g., "top 10", "list 5", "cari 3", "tampilkan 20"
    match_explicit = re.search(r'\b(top|list|daftar|cari|tampilkan|sebanyak|jumlah)\s+(\d+)', text)
    if match_explicit:
        return int(match_explicit.group(2))

    # Pattern 2: "X sekolah", "X smk", "X sma" (where X is a small number)
    # e.g., "10 sekolah terbaik", "5 smk di sidoarjo"
    match_implicit = re.search(r'\b(\d+)\s+(sekolah|smk|sma|sd|smp|slb|madrasah|hasil|data)', text)
    if match_implicit:
        val = int(match_implicit.group(1))
        # Threshold to avoid confusing "500 sekolah" (limit) with "500 siswa" (filter)
        # Assuming if user asks for > 100 schools, it might still be a limit request if phrased like "1000 sekolah".
        # But "500 siswa" is different because "siswa" is not in the list above.
        return val

    return None

def extract_sort_preference(query: str) -> Optional[List[Dict[str, str]]]:
    """
    Extracts sort preference from query.
    Returns list of dicts: [{'field': 'pd', 'order': 'desc'}]
    """
    text = normalize_text(query)
    sort_list = []

    # 1. Determine Direction
    # Default DESC (paling banyak, top, terbaik)
    order = 'desc'
    if any(x in text for x in ['sedikit', 'terkecil', 'rendah', 'bawah', 'min', 'kurang']):
        order = 'asc'

    # 2. Determine Field
    field = None

    if any(x in text for x in ['siswa', 'murid', 'anak', 'pd']):
        field = 'pd'
    elif any(x in text for x in ['guru', 'pengajar', 'ptk']):
        field = 'ptk'
    elif any(x in text for x in ['fasilitas', 'lab', 'laboratorium']):
        field = 'jml_lab'
    elif any(x in text for x in ['perpus', 'perpustakaan']):
        field = 'jml_perpus'
    elif 'akreditasi' in text:
        field = 'akreditasi'

    # "Paling banyak" without field context -> usually means students (pd) for schools
    if not field:
        if any(x in text for x in ['banyak', 'besar', 'ramai', 'sedikit', 'kecil', 'sepi']):
            field = 'pd'

    if field:
        sort_list.append({'field': field, 'order': order})
        return sort_list

    return None
