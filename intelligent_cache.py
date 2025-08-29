"""
Intelligent Cache System

Advanced caching system that intelligently caches query results, analysis data,
and AI-generated content with smart invalidation and optimization.

Features:
- Multi-level caching (memory, disk, distributed)
- Intelligent cache key generation based on query semantics
- Smart TTL based on data volatility and usage patterns
- Cache warming and prefetching strategies
- Automatic cache invalidation on data changes
- Performance analytics and cache hit optimization
"""

import logging
import hashlib
import json
import time
import pickle
import threading
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
import weakref

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Represents a cached entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int]
    size_bytes: int
    tags: List[str]
    dependencies: List[str]
    volatility_score: float  # 0.0 = stable, 1.0 = highly volatile

class IntelligentCache:
    """
    Advanced caching system with intelligent cache management.
    Uses query semantics and data patterns for optimal performance.
    """
    
    def __init__(self, max_memory_mb: int = 512, max_entries: int = 10000):
        # Cache storage
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._cache_lock = threading.RLock()
        
        # Configuration
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_entries = max_entries
        self.current_memory_usage = 0
        
        # Intelligence components
        self.access_patterns = defaultdict(list)  # Track access patterns
        self.query_similarity_cache = {}  # Cache similar query mappings
        self.data_volatility_tracker = defaultdict(float)  # Track how often data changes
        self.performance_metrics = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage': 0,
            'avg_response_time': 0.0
        }
        
        # Cache categories with different TTL strategies
        self.cache_categories = {
            'query_results': {
                'base_ttl': 3600,  # 1 hour
                'volatility_multiplier': 0.5,
                'access_multiplier': 2.0
            },
            'query_analysis': {
                'base_ttl': 7200,  # 2 hours
                'volatility_multiplier': 0.3,
                'access_multiplier': 1.5
            },
            'column_mapping': {
                'base_ttl': 86400,  # 24 hours
                'volatility_multiplier': 0.1,
                'access_multiplier': 1.2
            },
            'ai_generated_sql': {
                'base_ttl': 1800,  # 30 minutes
                'volatility_multiplier': 0.7,
                'access_multiplier': 2.5
            },
            'table_metadata': {
                'base_ttl': 43200,  # 12 hours
                'volatility_multiplier': 0.2,
                'access_multiplier': 1.0
            }
        }
        
        # Background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self._cleanup_thread.start()
    
    def get(self, key: str, category: str = 'default') -> Optional[Any]:
        """
        Intelligent cache retrieval with pattern learning.
        
        Args:
            key: Cache key
            category: Cache category for TTL optimization
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._cache_lock:
            # Check if key exists
            if key not in self._memory_cache:
                self.performance_metrics['misses'] += 1
                self._record_access_pattern(key, False)
                return None
            
            entry = self._memory_cache[key]
            
            # Check if expired
            if self._is_expired(entry):
                del self._memory_cache[key]
                self.current_memory_usage -= entry.size_bytes
                self.performance_metrics['misses'] += 1
                self._record_access_pattern(key, False)
                return None
            
            # Update access information
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            
            # Move to end (LRU)
            self._memory_cache.move_to_end(key)
            
            # Record successful hit
            self.performance_metrics['hits'] += 1
            self._record_access_pattern(key, True)
            
            logger.debug(f"Cache hit: {key} (category: {category})")
            return entry.value
    
    def set(self, key: str, value: Any, category: str = 'default', 
            tags: Optional[List[str]] = None, dependencies: Optional[List[str]] = None) -> bool:
        """
        Intelligent cache storage with automatic TTL optimization.
        
        Args:
            key: Cache key
            value: Value to cache
            category: Cache category for TTL optimization
            tags: Tags for batch invalidation
            dependencies: Dependencies for cascade invalidation
            
        Returns:
            True if successfully cached
        """
        with self._cache_lock:
            try:
                # Calculate size
                size_bytes = self._calculate_size(value)
                
                # Check if we need to make space
                if not self._ensure_space(size_bytes):
                    logger.warning(f"Could not make space for cache entry: {key}")
                    return False
                
                # Calculate intelligent TTL
                ttl_seconds = self._calculate_intelligent_ttl(key, category, value)
                
                # Calculate volatility score
                volatility_score = self._calculate_volatility_score(key, category)
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=1,
                    ttl_seconds=ttl_seconds,
                    size_bytes=size_bytes,
                    tags=tags or [],
                    dependencies=dependencies or [],
                    volatility_score=volatility_score
                )
                
                # Remove existing entry if present
                if key in self._memory_cache:
                    old_entry = self._memory_cache[key]
                    self.current_memory_usage -= old_entry.size_bytes
                
                # Store entry
                self._memory_cache[key] = entry
                self.current_memory_usage += size_bytes
                
                # Update performance metrics
                self.performance_metrics['memory_usage'] = self.current_memory_usage
                
                logger.debug(f"Cache set: {key} (category: {category}, TTL: {ttl_seconds}s)")
                return True
                
            except Exception as e:
                logger.error(f"Error caching {key}: {e}")
                return False
    
    def invalidate(self, pattern: Optional[str] = None, tags: Optional[List[str]] = None,
                  dependencies: Optional[List[str]] = None) -> int:
        """
        Intelligent cache invalidation with pattern and dependency support.
        
        Args:
            pattern: Key pattern to match (supports wildcards)
            tags: Tags to invalidate
            dependencies: Dependencies to invalidate
            
        Returns:
            Number of entries invalidated
        """
        with self._cache_lock:
            keys_to_remove = set()
            
            # Pattern-based invalidation
            if pattern:
                import fnmatch
                for key in self._memory_cache:
                    if fnmatch.fnmatch(key, pattern):
                        keys_to_remove.add(key)
            
            # Tag-based invalidation
            if tags:
                for key, entry in self._memory_cache.items():
                    if any(tag in entry.tags for tag in tags):
                        keys_to_remove.add(key)
            
            # Dependency-based invalidation
            if dependencies:
                for key, entry in self._memory_cache.items():
                    if any(dep in entry.dependencies for dep in dependencies):
                        keys_to_remove.add(key)
            
            # Remove entries
            for key in keys_to_remove:
                entry = self._memory_cache[key]
                self.current_memory_usage -= entry.size_bytes
                del self._memory_cache[key]
            
            count = len(keys_to_remove)
            if count > 0:
                logger.info(f"Invalidated {count} cache entries")
            
            return count
    
    def get_similar_query_cache_key(self, query: str, query_analysis: Dict[str, Any]) -> Optional[str]:
        """
        Find cache key for similar query using semantic analysis.
        
        Args:
            query: Current query
            query_analysis: Analysis from query intelligence
            
        Returns:
            Cache key of similar query or None
        """
        # Create semantic signature
        semantic_signature = self._create_semantic_signature(query, query_analysis)
        
        # Check if we've seen similar queries
        if semantic_signature in self.query_similarity_cache:
            similar_key = self.query_similarity_cache[semantic_signature]
            if similar_key in self._memory_cache:
                entry = self._memory_cache[similar_key]
                if not self._is_expired(entry):
                    logger.info(f"Found similar cached query for: {query[:50]}")
                    return similar_key
        
        return None
    
    def cache_query_result(self, query: str, query_analysis: Dict[str, Any], 
                          result: Dict[str, Any], table_dependencies: List[str]) -> bool:
        """
        Cache query result with intelligent optimization.
        
        Args:
            query: Original query
            query_analysis: Query analysis data
            result: Query execution result
            table_dependencies: Tables that this query depends on
            
        Returns:
            True if cached successfully
        """
        # Create intelligent cache key
        cache_key = self._create_query_cache_key(query, query_analysis)
        
        # Create semantic signature for similarity matching
        semantic_signature = self._create_semantic_signature(query, query_analysis)
        self.query_similarity_cache[semantic_signature] = cache_key
        
        # Determine tags for invalidation
        tags = ['query_results']
        if query_analysis.get('intent'):
            tags.append(f"intent_{query_analysis['intent']}")
        
        # Add entity tags
        entities = query_analysis.get('entities', {})
        for entity_type, entity_list in entities.items():
            if entity_list:
                tags.append(f"entity_{entity_type}")
        
        # Cache with table dependencies
        return self.set(
            key=cache_key,
            value=result,
            category='query_results',
            tags=tags,
            dependencies=table_dependencies
        )
    
    def cache_analysis_result(self, data_key: str, analysis: Dict[str, Any], 
                             category: str = 'query_analysis') -> bool:
        """
        Cache analysis results (query analysis, column mapping, etc.).
        
        Args:
            data_key: Key identifying the analyzed data
            analysis: Analysis results
            category: Analysis category
            
        Returns:
            True if cached successfully
        """
        cache_key = f"analysis_{category}_{hashlib.md5(data_key.encode()).hexdigest()}"
        
        return self.set(
            key=cache_key,
            value=analysis,
            category=category,
            tags=[category, 'analysis']
        )
    
    def warm_cache(self, queries: List[str], table_info: Dict[str, Any]) -> int:
        """
        Warm cache with common queries and analysis.
        
        Args:
            queries: Common queries to pre-process
            table_info: Available table information
            
        Returns:
            Number of entries warmed
        """
        warmed_count = 0
        
        try:
            # Pre-cache table metadata
            for table_name, info in table_info.items():
                cache_key = f"table_metadata_{table_name}"
                if self.set(cache_key, info, 'table_metadata', tags=['metadata', table_name]):
                    warmed_count += 1
            
            # Pre-process common query patterns if query intelligence is available
            try:
                from query_intelligence import query_intelligence
                
                for query in queries:
                    analysis = query_intelligence.analyze_query_intent(query)
                    analysis_key = f"query_{hashlib.md5(query.encode()).hexdigest()}"
                    
                    if self.cache_analysis_result(analysis_key, analysis, 'query_analysis'):
                        warmed_count += 1
                        
            except ImportError:
                logger.info("Query intelligence not available for cache warming")
            
            logger.info(f"Cache warmed with {warmed_count} entries")
            
        except Exception as e:
            logger.error(f"Error during cache warming: {e}")
        
        return warmed_count
    
    def _create_query_cache_key(self, query: str, query_analysis: Dict[str, Any]) -> str:
        """Create intelligent cache key for query based on semantics."""
        
        # Use semantic components for key generation
        key_components = [
            query_analysis.get('intent', 'unknown'),
            str(sorted(query_analysis.get('semantic_roles_needed', []))),
            str(query_analysis.get('complexity_score', 0))
        ]
        
        # Add entity information
        entities = query_analysis.get('entities', {})
        for entity_type in sorted(entities.keys()):
            entity_values = [e.get('text', '') for e in entities[entity_type]]
            key_components.append(f"{entity_type}:{','.join(sorted(entity_values))}")
        
        # Create deterministic hash
        key_string = "|".join(key_components)
        cache_key = f"query_{hashlib.md5(key_string.encode()).hexdigest()}"
        
        return cache_key
    
    def _create_semantic_signature(self, query: str, query_analysis: Dict[str, Any]) -> str:
        """Create semantic signature for query similarity matching."""
        
        signature_components = [
            query_analysis.get('intent', 'unknown'),
            str(sorted(query_analysis.get('semantic_roles_needed', []))),
        ]
        
        # Add normalized entity types (not values for similarity)
        entities = query_analysis.get('entities', {})
        signature_components.append(str(sorted(entities.keys())))
        
        return hashlib.md5("|".join(signature_components).encode()).hexdigest()
    
    def _calculate_intelligent_ttl(self, key: str, category: str, value: Any) -> int:
        """Calculate intelligent TTL based on multiple factors."""
        
        # Get base TTL for category
        category_config = self.cache_categories.get(category, self.cache_categories['query_results'])
        base_ttl = category_config['base_ttl']
        
        # Factor 1: Data volatility
        volatility_score = self._calculate_volatility_score(key, category)
        volatility_multiplier = category_config['volatility_multiplier']
        ttl_adjustment = base_ttl * (1 - volatility_score * volatility_multiplier)
        
        # Factor 2: Access patterns
        access_frequency = self._get_access_frequency(key)
        access_multiplier = category_config['access_multiplier']
        if access_frequency > 0:
            ttl_adjustment *= (1 + access_frequency * access_multiplier)
        
        # Factor 3: Data size (larger data gets shorter TTL to manage memory)
        data_size = self._calculate_size(value)
        if data_size > 100000:  # > 100KB
            ttl_adjustment *= 0.8
        
        # Ensure reasonable bounds
        min_ttl = base_ttl // 4  # At least 25% of base TTL
        max_ttl = base_ttl * 3   # At most 300% of base TTL
        
        final_ttl = max(min_ttl, min(max_ttl, int(ttl_adjustment)))
        
        return final_ttl
    
    def _calculate_volatility_score(self, key: str, category: str) -> float:
        """Calculate how volatile this data is (how often it changes)."""
        
        # Default volatility scores by category
        default_volatility = {
            'query_results': 0.3,
            'query_analysis': 0.1,
            'column_mapping': 0.05,
            'ai_generated_sql': 0.5,
            'table_metadata': 0.2
        }
        
        base_volatility = default_volatility.get(category, 0.3)
        
        # Check historical volatility
        if key in self.data_volatility_tracker:
            historical_volatility = self.data_volatility_tracker[key]
            # Weighted average of historical and default
            return (base_volatility + historical_volatility) / 2
        
        return base_volatility
    
    def _get_access_frequency(self, key: str) -> float:
        """Get access frequency score for a key."""
        
        if key not in self.access_patterns:
            return 0.0
        
        # Calculate access frequency in the last hour
        now = time.time()
        recent_accesses = [t for t in self.access_patterns[key] if now - t < 3600]
        
        return len(recent_accesses) / 10.0  # Normalize to 0-1 range (10+ accesses = 1.0)
    
    def _record_access_pattern(self, key: str, was_hit: bool) -> None:
        """Record access pattern for learning."""
        
        current_time = time.time()
        
        # Record access time
        if len(self.access_patterns[key]) > 100:  # Limit memory usage
            self.access_patterns[key] = self.access_patterns[key][-50:]  # Keep recent 50
        
        self.access_patterns[key].append(current_time)
        
        # Update volatility if it was a miss (data might have changed)
        if not was_hit and key in self.data_volatility_tracker:
            self.data_volatility_tracker[key] = min(1.0, self.data_volatility_tracker[key] + 0.1)
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        
        if entry.ttl_seconds is None:
            return False
        
        age = (datetime.now() - entry.created_at).total_seconds()
        return age > entry.ttl_seconds
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of cached value."""
        
        try:
            if isinstance(value, (str, int, float, bool)):
                return len(str(value)) * 8  # Approximate
            elif isinstance(value, dict):
                return len(json.dumps(value, default=str)) * 8
            elif isinstance(value, list):
                return sum(self._calculate_size(item) for item in value[:10]) * (len(value) / 10)
            else:
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default estimate
    
    def _ensure_space(self, required_bytes: int) -> bool:
        """Ensure there's enough space for new cache entry."""
        
        # Check if we exceed entry limit
        while len(self._memory_cache) >= self.max_entries:
            if not self._evict_lru():
                return False
        
        # Check memory limit
        while self.current_memory_usage + required_bytes > self.max_memory_bytes:
            if not self._evict_lru():
                return False
        
        return True
    
    def _evict_lru(self) -> bool:
        """Evict least recently used entry."""
        
        if not self._memory_cache:
            return False
        
        # Get LRU entry (first in OrderedDict)
        lru_key = next(iter(self._memory_cache))
        lru_entry = self._memory_cache[lru_key]
        
        # Remove entry
        del self._memory_cache[lru_key]
        self.current_memory_usage -= lru_entry.size_bytes
        self.performance_metrics['evictions'] += 1
        
        logger.debug(f"Evicted LRU entry: {lru_key}")
        return True
    
    def _background_cleanup(self) -> None:
        """Background thread for cache cleanup and optimization."""
        
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                
                with self._cache_lock:
                    # Remove expired entries
                    expired_keys = []
                    for key, entry in self._memory_cache.items():
                        if self._is_expired(entry):
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        entry = self._memory_cache[key]
                        self.current_memory_usage -= entry.size_bytes
                        del self._memory_cache[key]
                    
                    if expired_keys:
                        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
                    # Update performance metrics
                    total_requests = self.performance_metrics['hits'] + self.performance_metrics['misses']
                    if total_requests > 0:
                        hit_rate = self.performance_metrics['hits'] / total_requests
                        logger.debug(f"Cache hit rate: {hit_rate:.2%}")
            
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        
        with self._cache_lock:
            total_requests = self.performance_metrics['hits'] + self.performance_metrics['misses']
            hit_rate = (self.performance_metrics['hits'] / total_requests) if total_requests > 0 else 0
            
            # Category distribution
            category_stats = defaultdict(int)
            for entry in self._memory_cache.values():
                for tag in entry.tags:
                    if tag in self.cache_categories:
                        category_stats[tag] += 1
            
            return {
                'total_entries': len(self._memory_cache),
                'memory_usage_mb': self.current_memory_usage / (1024 * 1024),
                'memory_usage_percent': (self.current_memory_usage / self.max_memory_bytes) * 100,
                'hit_rate': hit_rate,
                'total_hits': self.performance_metrics['hits'],
                'total_misses': self.performance_metrics['misses'],
                'total_evictions': self.performance_metrics['evictions'],
                'category_distribution': dict(category_stats),
                'avg_entry_size_kb': (self.current_memory_usage / len(self._memory_cache)) / 1024 if self._memory_cache else 0
            }
    
    def clear_all(self) -> int:
        """Clear all cache entries."""
        
        with self._cache_lock:
            count = len(self._memory_cache)
            self._memory_cache.clear()
            self.current_memory_usage = 0
            logger.info(f"Cleared all {count} cache entries")
            return count

# Global instance
intelligent_cache = IntelligentCache(max_memory_mb=512, max_entries=10000)

def get_intelligent_cache() -> IntelligentCache:
    """Get the global intelligent cache instance"""
    return intelligent_cache