# app/enhanced_query_executor.py
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from app.enhanced_query_planner import QueryPlan, EnhancedQueryPlanner
from app.config import Config
from app.utils import build_qdrant_filter, normalize_query

class EnhancedQueryExecutor:
    """
    Enhanced query executor.
    Bertanggung jawab untuk:
    1. Menerima query user
    2. Menggunakan Planner untuk membuat strategi pencarian
    3. Menjalankan pencarian ke Qdrant (Structured/Semantic/Hybrid)
    4. Melakukan post-filtering dan sorting hasil
    """
    
    def __init__(self, qdrant_client, embedder, llm_client=None):
        self.qdrant = qdrant_client
        self.embedder = embedder
        self.llm_client = llm_client
        # Planner menerima llm_client yang sudah ditentukan (Gemini/Ollama) di api.py
        self.planner = EnhancedQueryPlanner(llm_client)
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_stats = {}
    
    def execute_query(self, user_query: str) -> Tuple[List[Dict], QueryPlan]:
        """
        Eksekusi query utama.
        Returns: (List data sekolah, Object QueryPlan)
        """
        start_time = time.time()
        
        try:
            # 1. Buat Rencana Query (Query Plan)
            plan = self.planner.plan_query(user_query)
            self.logger.info(f"Query Plan Created: intent={plan.intent}, routing={plan.routing}, filters={plan.filters}")
            
            # 2. Eksekusi Pencarian berdasarkan strategi Routing
            results = []
            
            if plan.routing == "structured":
                results = self._execute_structured_query_unlimited(plan)
            elif plan.routing == "semantic":
                results = self._execute_semantic_query_unlimited(plan, user_query)
            else:  # hybrid
                results = self._execute_hybrid_query_unlimited(plan, user_query)
            
            # 3. Post-Filtering (Manual Filter)
            # Penting: Filter manual memastikan akurasi logika (terutama numerik)
            if plan.filters and results:
                results = self._apply_manual_filtering(results, plan.filters)
            
            self.logger.info(f"Results after filtering: {len(results)}")
            
            # 4. Fallback Strategy
            # Jika hasil kosong, coba strategi fallback secara bertahap
            if not results:
                self.logger.info("Primary query returned no results, attempting fallback strategies...")
                results = self._execute_progressive_fallback_unlimited(user_query, plan)
            
            # 5. Sorting
            if results and len(results) > 1:
                results = self._sort_results(results, plan, user_query)
            
            # Log Performance
            execution_time = time.time() - start_time
            self._log_performance(user_query, plan, len(results), execution_time)
            
            return results, plan
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            # Return empty result dengan plan darurat
            basic_plan = QueryPlan(
                intent="search_school",
                routing="semantic",
                filters={},
                text_query=user_query,
                fields=["nama", "npsn"],
                limit=20,
                sort=None,
                confidence=0.1
            )
            return [], basic_plan

    def _apply_manual_filtering(self, results: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """Menerapkan filter manual Python untuk validasi ganda"""
        filtered_results = []
        for school in results:
            match = True
            for field, value in filters.items():
                # Handle Special Keys
                if field == "name_contains":
                    # Partial match for name
                    school_name = str(school.get("nama", "")).lower()
                    target_name = str(value).lower()
                    if target_name not in school_name:
                        match = False
                    continue

                school_value = school.get(field)
                
                # Handle numeric operators (e.g., {"op": ">", "value": 500})
                if isinstance(value, dict) and "op" in value and "value" in value:
                    op = value["op"]
                    filter_val = value["value"]
                    
                    # Konversi ke float untuk perbandingan aman
                    try:
                        s_val = float(school_value) if school_value is not None else 0
                        f_val = float(filter_val)
                        
                        if op == ">" and not (s_val > f_val): match = False
                        elif op == ">=" and not (s_val >= f_val): match = False
                        elif op == "<" and not (s_val < f_val): match = False
                        elif op == "<=" and not (s_val <= f_val): match = False
                        elif op == "==" and not (s_val == f_val): match = False
                    except (ValueError, TypeError):
                        match = False # Gagal konversi, anggap tidak match
                
                # Handle exact string match
                else:
                    # Skip if value is list (handled by build_qdrant_filter, but for manual filter we assume exact or skip)
                    # If it is a list, check if school_value is IN that list
                    if isinstance(value, list):
                        if school_value not in value:
                             match = False
                    elif str(school_value).lower() != str(value).lower():
                        match = False
            
            if match:
                filtered_results.append(school)
        return filtered_results
    
    def _execute_structured_query_unlimited(self, plan: QueryPlan) -> List[Dict]:
        """Eksekusi structured query (filter only)"""
        try:
            if not plan.filters:
                return []
            
            qdrant_filter = build_qdrant_filter(plan.filters)
            if not qdrant_filter:
                return []
            
            self.logger.info(f"Executing structured query with {len(plan.filters)} filters")
            
            # Gunakan scroll untuk mengambil semua data
            all_results = []
            next_offset = None
            
            while True:
                scroll_results, next_offset = self.qdrant.scroll(
                    collection_name=Config.COLLECTION_NAME,
                    scroll_filter=qdrant_filter,
                    limit=100,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False
                )
                all_results.extend([r.payload for r in scroll_results])
                if next_offset is None:
                    break
                
            return all_results
        except Exception as e:
            self.logger.error(f"Structured query failed: {e}")
            return []
        
    def _execute_semantic_query_unlimited(self, plan: QueryPlan, user_query: str) -> List[Dict]:
        """Eksekusi semantic query (vector only)"""
        try:
            query_text = plan.text_query or user_query
            self.logger.info(f"Executing semantic query: '{query_text[:30]}...'")
            
            query_vector = self.embedder.encode(query_text, convert_to_tensor=False).tolist()
            qdrant_filter = build_qdrant_filter(plan.filters) if plan.filters else None
            
            search_result = self.qdrant.search(
                collection_name=Config.COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=1000, # Max limit
                score_threshold=0.2, # Threshold relevansi
                with_payload=True
            )
            
            return [r.payload for r in search_result]
        except Exception as e:
            self.logger.error(f"Semantic query failed: {e}")
            return []
    
    def _execute_hybrid_query_unlimited(self, plan: QueryPlan, user_query: str) -> List[Dict]:
        """Eksekusi hybrid query (vector + filter)"""
        try:
            query_text = plan.text_query or user_query
            self.logger.info(f"Executing hybrid query. Filters: {len(plan.filters)}")
            
            query_vector = self.embedder.encode(query_text, convert_to_tensor=False).tolist()
            qdrant_filter = build_qdrant_filter(plan.filters) if plan.filters else None
            
            search_result = self.qdrant.search(
                collection_name=Config.COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=1000,
                score_threshold=0.15, # Sedikit lebih rendah untuk hybrid
                with_payload=True
            )
            
            return [r.payload for r in search_result]
        except Exception as e:
            self.logger.error(f"Hybrid query failed: {e}")
            return []
    
    def _execute_progressive_fallback_unlimited(self, user_query: str, original_plan: QueryPlan) -> List[Dict]:
        """Mencoba strategi fallback secara berurutan jika query utama gagal"""
        fallback_strategies = [
            ("relax_numeric", self._fallback_relax_numeric),
            ("location_only", self._fallback_location_only),  
            ("semantic_only", self._fallback_semantic_only),
            ("basic_search", self._fallback_basic_search),
            ("emergency", self._fallback_emergency)
        ]
        
        for name, func in fallback_strategies:
            try:
                self.logger.info(f"Trying fallback: {name}")
                results = func(user_query, original_plan)
                if results:
                    self.logger.info(f"Fallback '{name}' success: {len(results)} results")
                    return results
            except Exception as e:
                self.logger.warning(f"Fallback '{name}' failed: {e}")
                continue
        
        return []
    
    # --- Fallback Methods ---
    
    def _fallback_relax_numeric(self, user_query: str, plan: QueryPlan) -> List[Dict]:
        # Melonggarkan filter angka (misal > 500 jadi > 250)
        relaxed_filters = {}
        has_numeric = False
        
        for k, v in plan.filters.items():
            if isinstance(v, dict) and "op" in v and "value" in v:
                has_numeric = True
                val = v["value"]
                if v["op"] in [">", ">="]:
                    relaxed_filters[k] = {"op": ">=", "value": int(val * 0.5)}
                elif v["op"] in ["<", "<="]:
                    relaxed_filters[k] = {"op": "<=", "value": int(val * 1.5)}
                else:
                    relaxed_filters[k] = v
            else:
                relaxed_filters[k] = v
        
        if not has_numeric: return []
        
        new_plan = QueryPlan(intent=plan.intent, routing="structured", filters=relaxed_filters, 
                           text_query=None, fields=plan.fields, limit=1000, sort=None, confidence=0.5)
        return self._execute_structured_query_unlimited(new_plan)

    def _fallback_location_only(self, user_query: str, plan: QueryPlan) -> List[Dict]:
        # Hanya ambil filter lokasi
        loc_fields = ["namaProvinsi", "namaKabupaten", "namaKecamatan", "namaDesa"]
        loc_filters = {k: v for k, v in plan.filters.items() if k in loc_fields}
        
        if not loc_filters:
            loc_filters = {"namaKabupaten": "KAB. SIDOARJO"}
            
        new_plan = QueryPlan(intent=plan.intent, routing="hybrid", filters=loc_filters,
                           text_query=user_query, fields=plan.fields, limit=1000, sort=None, confidence=0.4)
        return self._execute_hybrid_query_unlimited(new_plan, user_query)

    def _fallback_semantic_only(self, user_query: str, plan: QueryPlan) -> List[Dict]:
        # Hanya pencarian vektor
        new_plan = QueryPlan(intent=plan.intent, routing="semantic", filters={},
                           text_query=user_query, fields=plan.fields, limit=1000, sort=None, confidence=0.3)
        return self._execute_semantic_query_unlimited(new_plan, user_query)

    def _fallback_basic_search(self, user_query: str, plan: QueryPlan) -> List[Dict]:
        # Pencarian vektor + filter Sidoarjo
        filters = {"namaKabupaten": "KAB. SIDOARJO"}
        new_plan = QueryPlan(intent="search_school", routing="semantic", filters=filters,
                           text_query=user_query, fields=plan.fields, limit=1000, sort=None, confidence=0.2)
        return self._execute_semantic_query_unlimited(new_plan, user_query)

    def _fallback_emergency(self, user_query: str, plan: QueryPlan) -> List[Dict]:
        # Ambil 200 sekolah pertama di Sidoarjo
        results, _ = self.qdrant.scroll(
            collection_name=Config.COLLECTION_NAME,
            scroll_filter=build_qdrant_filter({"namaKabupaten": "KAB. SIDOARJO"}),
            limit=200,
            with_payload=True
        )
        return [r.payload for r in results]

    # --- Sorting & Logging ---

    def _sort_results(self, results: List[Dict], plan: QueryPlan, user_query: str) -> List[Dict]:
        q_lower = user_query.lower()
        
        if plan.intent == "ranking_query":
            return self._sort_by_quality(results)
        elif plan.intent == "count_query" or "siswa" in q_lower:
            return sorted(results, key=lambda x: int(x.get('pd', 0) or 0), reverse=True)
        elif "guru" in q_lower:
            return sorted(results, key=lambda x: int(x.get('ptk', 0) or 0), reverse=True)
        elif "akreditasi" in q_lower:
            return self._sort_by_accreditation(results)
        
        return results

    def _sort_by_quality(self, results: List[Dict]) -> List[Dict]:
        def score(school):
            s = 0
            accred = school.get('akreditasi', '')
            if accred == 'A': s += 100
            elif accred == 'B': s += 70
            
            facilities = (int(school.get('jml_lab', 0) or 0) * 5) + \
                         (int(school.get('jml_perpus', 0) or 0) * 10)
            s += facilities
            return s
        
        return sorted(results, key=score, reverse=True)

    def _sort_by_accreditation(self, results: List[Dict]) -> List[Dict]:
        order = {'A': 3, 'B': 2, 'C': 1}
        return sorted(results, key=lambda x: order.get(x.get('akreditasi', ''), 0), reverse=True)

    def _log_performance(self, query: str, plan: QueryPlan, count: int, time_taken: float):
        stats = {
            "query": query,
            "intent": plan.intent,
            "routing": plan.routing,
            "filters": len(plan.filters),
            "results": count,
            "time": round(time_taken, 2)
        }
        self.logger.info(f"Performance: {stats}")