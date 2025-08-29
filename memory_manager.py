"""
Memory Manager Module for PHERS
Phase 4: Performance Optimization

This module provides intelligent memory optimization strategies including:
- Dynamic memory pool management
- Query result streaming for large datasets
- Memory usage monitoring and optimization
- Garbage collection optimization
- Memory leak detection and prevention
- Adaptive memory allocation based on workload patterns
"""

import gc
import sys
import threading
import time
import logging
import json
import psutil
from typing import Dict, List, Optional, Any, Iterator, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import weakref
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
import mmap
import pickle

@dataclass
class MemoryUsage:
    """Memory usage snapshot"""
    timestamp: datetime
    total_memory: int  # bytes
    available_memory: int
    process_memory: int
    python_objects: int
    gc_collections: Dict[str, int]
    memory_pools: Dict[str, int]
    cache_usage: int
    temp_data_usage: int

@dataclass
class MemoryPool:
    """Memory pool for efficient allocation/deallocation"""
    name: str
    pool_size: int
    block_size: int
    allocated_blocks: int
    free_blocks: int
    total_allocations: int
    total_deallocations: int
    peak_usage: int
    created_at: datetime

@dataclass
class StreamingConfig:
    """Configuration for streaming large result sets"""
    chunk_size: int
    max_memory_usage: int
    enable_compression: bool
    temporary_storage_path: str
    cleanup_threshold: float

class MemoryProfiler:
    """Memory profiling and monitoring component"""
    
    def __init__(self, enable_tracemalloc: bool = True):
        self.enable_tracemalloc = enable_tracemalloc
        self.snapshots = deque(maxlen=1440)  # 24 hours of minute-level data
        self.memory_leaks = []
        self.profiling_active = False
        
        if enable_tracemalloc:
            tracemalloc.start()
    
    def start_profiling(self):
        """Start memory profiling"""
        self.profiling_active = True
        if self.enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
    
    def stop_profiling(self):
        """Stop memory profiling"""
        self.profiling_active = False
        if tracemalloc.is_tracing():
            tracemalloc.stop()
    
    def capture_snapshot(self) -> MemoryUsage:
        """Capture current memory usage snapshot"""
        # System memory info
        system_memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # Python-specific memory info
        gc_stats = {}
        for i in range(3):  # GC generations 0, 1, 2
            gc_stats[f"generation_{i}"] = len(gc.get_objects(i))
        
        # Tracemalloc info if available
        python_objects = 0
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            current = tracemalloc.take_snapshot()
            python_objects = len(current.traces)
        
        snapshot = MemoryUsage(
            timestamp=datetime.now(),
            total_memory=system_memory.total,
            available_memory=system_memory.available,
            process_memory=process_memory.rss,
            python_objects=python_objects,
            gc_collections={
                'gen0': gc.get_count()[0],
                'gen1': gc.get_count()[1],
                'gen2': gc.get_count()[2]
            },
            memory_pools={},  # Will be populated by MemoryManager
            cache_usage=0,     # Will be populated by MemoryManager
            temp_data_usage=0  # Will be populated by MemoryManager
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def detect_memory_leaks(self, threshold_mb: float = 10.0) -> List[Dict[str, Any]]:
        """Detect potential memory leaks using tracemalloc"""
        if not self.enable_tracemalloc or not tracemalloc.is_tracing():
            return []
        
        try:
            # Take current snapshot
            current_snapshot = tracemalloc.take_snapshot()
            
            # Compare with previous snapshots
            if len(self.snapshots) < 10:
                return []  # Need more data points
            
            # Get top memory consumers
            top_stats = current_snapshot.statistics('lineno')
            
            leaks = []
            for stat in top_stats[:20]:  # Top 20 consumers
                size_mb = stat.size / 1024 / 1024
                if size_mb > threshold_mb:
                    leak_info = {
                        'filename': stat.traceback.format()[0] if stat.traceback else 'unknown',
                        'size_mb': size_mb,
                        'count': stat.count,
                        'detected_at': datetime.now().isoformat()
                    }
                    leaks.append(leak_info)
            
            self.memory_leaks.extend(leaks)
            return leaks
        
        except Exception as e:
            logging.error(f"Error detecting memory leaks: {e}")
            return []

class QueryResultStreamer:
    """Streaming interface for large query results"""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.active_streams = {}
        self.temp_files = set()
        
        # Ensure temp directory exists
        Path(config.temporary_storage_path).mkdir(exist_ok=True)
    
    @contextmanager
    def stream_results(self, query: str, db_path: str, stream_id: str = None):
        """Context manager for streaming query results"""
        if not stream_id:
            stream_id = f"stream_{int(time.time() * 1000000)}"
        
        temp_file = None
        try:
            # Check if result set is likely to be large
            estimated_size = self._estimate_result_size(query, db_path)
            
            if estimated_size > self.config.max_memory_usage:
                # Use temporary file for large results
                temp_file = Path(self.config.temporary_storage_path) / f"{stream_id}.tmp"
                self.temp_files.add(temp_file)
                
                yield self._file_based_streamer(query, db_path, temp_file)
            else:
                # Use memory-based streaming for smaller results
                yield self._memory_based_streamer(query, db_path)
        
        finally:
            # Cleanup
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
            
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                    self.temp_files.discard(temp_file)
                except:
                    pass  # Ignore cleanup errors
    
    def _estimate_result_size(self, query: str, db_path: str) -> int:
        """Estimate query result size"""
        try:
            # Use EXPLAIN QUERY PLAN to estimate
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Try to get row count estimate
                if 'COUNT' not in query.upper():
                    count_query = f"SELECT COUNT(*) FROM ({query})"
                    cursor.execute(count_query)
                    row_count = cursor.fetchone()[0]
                    
                    # Estimate average row size (rough approximation)
                    avg_row_size = 100  # bytes per row estimate
                    estimated_size = row_count * avg_row_size
                    
                    return estimated_size
        
        except Exception as e:
            logging.warning(f"Failed to estimate result size: {e}")
        
        return 0  # Conservative estimate
    
    def _memory_based_streamer(self, query: str, db_path: str) -> Iterator[List[Any]]:
        """Stream results from memory in chunks"""
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                
                while True:
                    chunk = cursor.fetchmany(self.config.chunk_size)
                    if not chunk:
                        break
                    yield chunk
        
        except Exception as e:
            logging.error(f"Error in memory-based streaming: {e}")
            raise
    
    def _file_based_streamer(self, query: str, db_path: str, temp_file: Path) -> Iterator[List[Any]]:
        """Stream results using temporary file storage"""
        try:
            # First, write results to temp file
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                
                with open(temp_file, 'wb') as f:
                    while True:
                        chunk = cursor.fetchmany(self.config.chunk_size)
                        if not chunk:
                            break
                        
                        # Compress if enabled
                        if self.config.enable_compression:
                            data = pickle.dumps(chunk)
                        else:
                            data = pickle.dumps(chunk)
                        
                        f.write(len(data).to_bytes(4, 'big'))  # Size prefix
                        f.write(data)
            
            # Now stream from temp file
            with open(temp_file, 'rb') as f:
                while True:
                    size_bytes = f.read(4)
                    if not size_bytes:
                        break
                    
                    size = int.from_bytes(size_bytes, 'big')
                    data = f.read(size)
                    chunk = pickle.loads(data)
                    yield chunk
        
        except Exception as e:
            logging.error(f"Error in file-based streaming: {e}")
            raise
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in list(self.temp_files):
            try:
                if temp_file.exists():
                    temp_file.unlink()
                self.temp_files.discard(temp_file)
            except Exception as e:
                logging.warning(f"Failed to cleanup temp file {temp_file}: {e}")

class MemoryManager:
    """
    Comprehensive memory management system that optimizes memory usage,
    prevents leaks, and provides intelligent allocation strategies.
    """
    
    def __init__(self, activity_log_path: str = "/data/activity.log"):
        self.activity_log_path = activity_log_path
        
        # Initialize components
        self.profiler = MemoryProfiler(enable_tracemalloc=True)
        self.memory_pools = {}
        self.weak_references = {}
        
        # Streaming configuration
        self.streaming_config = StreamingConfig(
            chunk_size=1000,
            max_memory_usage=100 * 1024 * 1024,  # 100MB
            enable_compression=True,
            temporary_storage_path="/tmp/phers_temp",
            cleanup_threshold=0.8  # Clean when 80% of temp space used
        )
        
        self.result_streamer = QueryResultStreamer(self.streaming_config)
        
        # Memory optimization settings
        self.gc_optimization_enabled = True
        self.memory_monitoring_enabled = True
        self.auto_cleanup_enabled = True
        
        # Monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Memory thresholds
        self.memory_warning_threshold = 0.8  # 80% of available memory
        self.memory_critical_threshold = 0.9  # 90% of available memory
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Start monitoring
        self.start_monitoring()
        
        self._log_activity("MemoryManager initialized", {
            "streaming_config": asdict(self.streaming_config),
            "gc_optimization": self.gc_optimization_enabled,
            "memory_thresholds": {
                "warning": self.memory_warning_threshold,
                "critical": self.memory_critical_threshold
            }
        })
    
    def _log_activity(self, activity: str, details: Dict[str, Any] = None):
        """Log activity to the activity log file"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "module": "MemoryManager",
                "activity": activity,
                "details": details or {}
            }
            
            Path(self.activity_log_path).parent.mkdir(exist_ok=True)
            with open(self.activity_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to log activity: {e}")
    
    def start_monitoring(self, interval: float = 60.0):
        """Start memory monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.profiler.start_profiling()
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Capture memory snapshot
                    snapshot = self.profiler.capture_snapshot()
                    
                    # Update snapshot with our data
                    snapshot.memory_pools = {name: pool.allocated_blocks * pool.block_size 
                                           for name, pool in self.memory_pools.items()}
                    
                    # Check memory thresholds
                    self._check_memory_thresholds(snapshot)
                    
                    # Perform automatic optimizations
                    if self.auto_cleanup_enabled:
                        self._auto_optimize_memory()
                    
                    # Detect memory leaks
                    leaks = self.profiler.detect_memory_leaks()
                    if leaks:
                        self._log_activity("Memory leaks detected", {"leaks": leaks})
                    
                    time.sleep(interval)
                
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(interval)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        self.profiler.stop_profiling()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
    
    def _check_memory_thresholds(self, snapshot: MemoryUsage):
        """Check if memory usage exceeds thresholds"""
        memory_usage_ratio = (snapshot.total_memory - snapshot.available_memory) / snapshot.total_memory
        
        if memory_usage_ratio >= self.memory_critical_threshold:
            self._log_activity("Critical memory usage detected", {
                "usage_ratio": memory_usage_ratio,
                "available_mb": snapshot.available_memory / 1024 / 1024,
                "process_mb": snapshot.process_memory / 1024 / 1024
            })
            
            # Trigger aggressive cleanup
            self.emergency_cleanup()
        
        elif memory_usage_ratio >= self.memory_warning_threshold:
            self._log_activity("High memory usage warning", {
                "usage_ratio": memory_usage_ratio,
                "available_mb": snapshot.available_memory / 1024 / 1024
            })
            
            # Trigger gentle cleanup
            self.optimize_memory_usage()
    
    def _auto_optimize_memory(self):
        """Perform automatic memory optimizations"""
        # Run garbage collection if enabled
        if self.gc_optimization_enabled:
            collected = gc.collect()
            if collected > 0:
                self._log_activity("Automatic garbage collection", {
                    "objects_collected": collected
                })
        
        # Clean up weak references
        self._cleanup_weak_references()
        
        # Clean up temporary files
        self.result_streamer.cleanup_temp_files()
    
    def create_memory_pool(self, name: str, pool_size: int, block_size: int) -> MemoryPool:
        """Create a memory pool for efficient allocation"""
        if name in self.memory_pools:
            return self.memory_pools[name]
        
        pool = MemoryPool(
            name=name,
            pool_size=pool_size,
            block_size=block_size,
            allocated_blocks=0,
            free_blocks=pool_size,
            total_allocations=0,
            total_deallocations=0,
            peak_usage=0,
            created_at=datetime.now()
        )
        
        self.memory_pools[name] = pool
        self._log_activity("Memory pool created", {
            "pool_name": name,
            "pool_size": pool_size,
            "block_size": block_size
        })
        
        return pool
    
    def allocate_from_pool(self, pool_name: str) -> Optional[int]:
        """Allocate a block from memory pool"""
        if pool_name not in self.memory_pools:
            return None
        
        pool = self.memory_pools[pool_name]
        
        if pool.free_blocks <= 0:
            return None  # Pool exhausted
        
        pool.allocated_blocks += 1
        pool.free_blocks -= 1
        pool.total_allocations += 1
        pool.peak_usage = max(pool.peak_usage, pool.allocated_blocks)
        
        return pool.allocated_blocks - 1  # Return block ID
    
    def deallocate_from_pool(self, pool_name: str, block_id: int):
        """Deallocate a block from memory pool"""
        if pool_name not in self.memory_pools:
            return
        
        pool = self.memory_pools[pool_name]
        
        if pool.allocated_blocks > 0:
            pool.allocated_blocks -= 1
            pool.free_blocks += 1
            pool.total_deallocations += 1
    
    def register_weak_reference(self, obj: Any, cleanup_callback: Callable = None) -> str:
        """Register an object with weak reference for automatic cleanup"""
        ref_id = f"ref_{int(time.time() * 1000000)}"
        
        def cleanup_wrapper(ref):
            if cleanup_callback:
                cleanup_callback()
            if ref_id in self.weak_references:
                del self.weak_references[ref_id]
        
        self.weak_references[ref_id] = weakref.ref(obj, cleanup_wrapper)
        return ref_id
    
    def _cleanup_weak_references(self):
        """Clean up dead weak references"""
        dead_refs = [ref_id for ref_id, ref in self.weak_references.items() 
                    if ref() is None]
        
        for ref_id in dead_refs:
            del self.weak_references[ref_id]
        
        if dead_refs:
            self._log_activity("Weak references cleaned", {
                "cleaned_count": len(dead_refs)
            })
    
    @contextmanager
    def stream_large_results(self, query: str, db_path: str):
        """Context manager for streaming large query results"""
        stream_id = f"stream_{int(time.time() * 1000000)}"
        
        with self.result_streamer.stream_results(query, db_path, stream_id) as streamer:
            yield streamer
    
    def optimize_gc_settings(self):
        """Optimize garbage collection settings based on workload"""
        if not self.gc_optimization_enabled:
            return
        
        # Get current GC stats
        gen0_count, gen1_count, gen2_count = gc.get_count()
        gen0_thresh, gen1_thresh, gen2_thresh = gc.get_threshold()
        
        # Analyze memory usage patterns
        recent_snapshots = list(self.profiler.snapshots)[-10:]  # Last 10 snapshots
        
        if len(recent_snapshots) >= 5:
            # Calculate memory growth rate
            memory_growth = []
            for i in range(1, len(recent_snapshots)):
                growth = recent_snapshots[i].process_memory - recent_snapshots[i-1].process_memory
                memory_growth.append(growth)
            
            avg_growth = sum(memory_growth) / len(memory_growth)
            
            # Adjust GC thresholds based on growth rate
            if avg_growth > 1024 * 1024:  # Growing by more than 1MB per snapshot
                # More aggressive GC for fast-growing memory
                new_gen0_thresh = max(100, int(gen0_thresh * 0.8))
                new_gen1_thresh = max(10, int(gen1_thresh * 0.8))
                new_gen2_thresh = max(5, int(gen2_thresh * 0.8))
            elif avg_growth < 0:  # Memory usage decreasing
                # Less aggressive GC when memory is stable/decreasing
                new_gen0_thresh = min(1000, int(gen0_thresh * 1.2))
                new_gen1_thresh = min(100, int(gen1_thresh * 1.2))
                new_gen2_thresh = min(50, int(gen2_thresh * 1.2))
            else:
                return  # No change needed
            
            gc.set_threshold(new_gen0_thresh, new_gen1_thresh, new_gen2_thresh)
            
            self._log_activity("GC thresholds optimized", {
                "old_thresholds": [gen0_thresh, gen1_thresh, gen2_thresh],
                "new_thresholds": [new_gen0_thresh, new_gen1_thresh, new_gen2_thresh],
                "avg_memory_growth": avg_growth
            })
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Perform comprehensive memory optimization"""
        optimization_results = {}
        
        # 1. Run garbage collection
        if self.gc_optimization_enabled:
            collected = gc.collect()
            optimization_results['gc_collected'] = collected
        
        # 2. Optimize GC settings
        self.optimize_gc_settings()
        
        # 3. Clean up weak references
        old_ref_count = len(self.weak_references)
        self._cleanup_weak_references()
        optimization_results['weak_refs_cleaned'] = old_ref_count - len(self.weak_references)
        
        # 4. Clean up temporary files
        self.result_streamer.cleanup_temp_files()
        
        # 5. Defragment memory pools
        for pool_name, pool in self.memory_pools.items():
            if pool.free_blocks > pool.pool_size * 0.5:  # More than 50% free
                # Consider pool defragmentation (simplified)
                pass
        
        # 6. Clear internal caches if available
        # This would integrate with intelligent_cache.py
        try:
            from intelligent_cache import get_intelligent_cache
            cache = get_intelligent_cache()
            cleared = cache.clear_expired_entries()
            optimization_results['cache_cleared'] = cleared
        except ImportError:
            pass
        
        self._log_activity("Memory optimization completed", optimization_results)
        return optimization_results
    
    def emergency_cleanup(self) -> Dict[str, Any]:
        """Perform emergency memory cleanup when critical threshold is reached"""
        cleanup_results = {}
        
        # 1. Force garbage collection multiple times
        total_collected = 0
        for _ in range(3):
            collected = gc.collect()
            total_collected += collected
        cleanup_results['emergency_gc_collected'] = total_collected
        
        # 2. Clear all non-essential caches
        try:
            from intelligent_cache import get_intelligent_cache
            cache = get_intelligent_cache()
            cleared = cache.emergency_clear()
            cleanup_results['emergency_cache_cleared'] = cleared
        except ImportError:
            pass
        
        # 3. Clean up all temporary data
        self.result_streamer.cleanup_temp_files()
        
        # 4. Reset memory pools (dangerous but necessary in emergency)
        for pool in self.memory_pools.values():
            pool.allocated_blocks = 0
            pool.free_blocks = pool.pool_size
        cleanup_results['memory_pools_reset'] = len(self.memory_pools)
        
        # 5. Clear weak references
        self.weak_references.clear()
        
        self._log_activity("Emergency memory cleanup performed", cleanup_results)
        return cleanup_results
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        current_snapshot = self.profiler.capture_snapshot()
        
        # Memory pool statistics
        pool_stats = {}
        for name, pool in self.memory_pools.items():
            pool_stats[name] = {
                'allocated_blocks': pool.allocated_blocks,
                'free_blocks': pool.free_blocks,
                'utilization': pool.allocated_blocks / pool.pool_size if pool.pool_size > 0 else 0,
                'total_allocations': pool.total_allocations,
                'total_deallocations': pool.total_deallocations,
                'peak_usage': pool.peak_usage
            }
        
        # Recent trends
        recent_snapshots = list(self.profiler.snapshots)[-10:]
        memory_trend = 'stable'
        if len(recent_snapshots) >= 3:
            first_memory = recent_snapshots[0].process_memory
            last_memory = recent_snapshots[-1].process_memory
            change_ratio = (last_memory - first_memory) / first_memory
            
            if change_ratio > 0.1:
                memory_trend = 'increasing'
            elif change_ratio < -0.1:
                memory_trend = 'decreasing'
        
        statistics = {
            'timestamp': current_snapshot.timestamp.isoformat(),
            'current_usage': {
                'total_memory_mb': current_snapshot.total_memory / 1024 / 1024,
                'available_memory_mb': current_snapshot.available_memory / 1024 / 1024,
                'process_memory_mb': current_snapshot.process_memory / 1024 / 1024,
                'python_objects': current_snapshot.python_objects,
                'memory_usage_ratio': (current_snapshot.total_memory - current_snapshot.available_memory) / current_snapshot.total_memory
            },
            'gc_statistics': current_snapshot.gc_collections,
            'memory_pools': pool_stats,
            'weak_references_count': len(self.weak_references),
            'memory_trend': memory_trend,
            'temp_files_count': len(self.result_streamer.temp_files),
            'potential_leaks': len(self.profiler.memory_leaks),
            'streaming_config': asdict(self.streaming_config)
        }
        
        return statistics
    
    def export_memory_report(self, filepath: str = None, include_snapshots: bool = False) -> Dict[str, Any]:
        """Export comprehensive memory usage report"""
        if not filepath:
            filepath = f"/data/memory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        statistics = self.get_memory_statistics()
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'monitoring_duration_hours': (datetime.now() - self.profiler.snapshots[0].timestamp).total_seconds() / 3600 if self.profiler.snapshots else 0,
                'include_snapshots': include_snapshots
            },
            'memory_statistics': statistics,
            'memory_leaks': self.profiler.memory_leaks
        }
        
        if include_snapshots:
            report['memory_snapshots'] = [asdict(snapshot) for snapshot in self.profiler.snapshots]
        
        try:
            Path(filepath).parent.mkdir(exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            self._log_activity("Memory report exported", {
                "filepath": filepath,
                "include_snapshots": include_snapshots
            })
            
            return {"success": True, "filepath": filepath, "report": report}
        
        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
            return {"success": False, "error": str(e), "report": report}
    
    def shutdown(self):
        """Shutdown memory manager"""
        self.stop_monitoring()
        self.result_streamer.cleanup_temp_files()
        
        self._log_activity("MemoryManager shutdown", {
            "memory_pools_count": len(self.memory_pools),
            "weak_references_count": len(self.weak_references)
        })

# Global instance for easy integration
memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get or create global memory manager instance"""
    global memory_manager
    if memory_manager is None:
        memory_manager = MemoryManager()
    return memory_manager

@contextmanager
def stream_large_query_results(query: str, db_path: str):
    """Convenience context manager for streaming large results"""
    manager = get_memory_manager()
    with manager.stream_large_results(query, db_path) as streamer:
        yield streamer

def optimize_memory():
    """Convenience function to optimize memory usage"""
    manager = get_memory_manager()
    return manager.optimize_memory_usage()

if __name__ == "__main__":
    # Example usage and testing
    manager = MemoryManager()
    
    # Create a memory pool
    pool = manager.create_memory_pool("query_cache", 1000, 1024)
    
    # Allocate some blocks
    for i in range(10):
        block_id = manager.allocate_from_pool("query_cache")
        print(f"Allocated block: {block_id}")
    
    # Get memory statistics
    stats = manager.get_memory_statistics()
    print(f"Memory usage: {stats['current_usage']['process_memory_mb']:.2f} MB")
    
    # Test streaming (would need actual database)
    # with manager.stream_large_results("SELECT * FROM large_table", "test.db") as streamer:
    #     for chunk in streamer:
    #         print(f"Processing chunk with {len(chunk)} rows")
    
    # Optimize memory
    optimization_results = manager.optimize_memory_usage()
    print(f"Optimization results: {optimization_results}")
    
    # Export report
    report = manager.export_memory_report()
    print(f"Report exported: {report.get('filepath', 'Failed')}")
    
    # Shutdown
    manager.shutdown()