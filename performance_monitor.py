"""
Performance Monitor Module for PHERS
Phase 4: Performance Optimization

This module provides real-time performance monitoring and analytics capabilities:
- Real-time query performance tracking
- Resource usage monitoring (CPU, Memory, I/O)
- Performance trend analysis and alerting
- Bottleneck detection and root cause analysis
- Performance dashboard and reporting
- Automatic performance baseline establishment
"""

import time
import psutil
import threading
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
import sqlite3
from pathlib import Path
import statistics
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class PerformanceSnapshot:
    """Single point-in-time performance measurement"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_io_read: float
    disk_io_write: float
    active_connections: int
    queries_per_second: float
    avg_query_time: float
    cache_hit_ratio: float
    error_rate: float

@dataclass
class QueryPerformanceMetric:
    """Performance metrics for individual queries"""
    query_id: str
    query_pattern: str
    execution_time: float
    cpu_time: float
    memory_used: float
    rows_processed: int
    cache_hits: int
    cache_misses: int
    timestamp: datetime
    optimization_score: float

@dataclass
class PerformanceAlert:
    """Performance alert definition"""
    alert_id: str
    severity: str  # 'critical', 'warning', 'info'
    metric_name: str
    threshold: float
    current_value: float
    message: str
    timestamp: datetime
    resolved: bool = False

@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison"""
    metric_name: str
    baseline_value: float
    acceptable_range: tuple  # (min, max)
    measurement_window: int  # minutes
    confidence_level: float
    last_updated: datetime

class ResourceMonitor:
    """System resource monitoring component"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None
        self.snapshots = deque(maxlen=1440)  # 24 hours of minute-level data
        
    def start_monitoring(self, interval: float = 60.0):
        """Start continuous resource monitoring"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                snapshot = self._capture_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(interval)
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _capture_snapshot(self) -> PerformanceSnapshot:
        """Capture current system performance snapshot"""
        # System-wide metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        
        # Process-specific metrics
        process_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return PerformanceSnapshot(
            timestamp=datetime.now(),
            cpu_usage=cpu_percent,
            memory_usage=process_memory,
            memory_available=memory.available / 1024 / 1024,  # MB
            disk_io_read=disk_io.read_bytes if disk_io else 0,
            disk_io_write=disk_io.write_bytes if disk_io else 0,
            active_connections=0,  # Would need connection pool integration
            queries_per_second=0,  # Updated by PerformanceMonitor
            avg_query_time=0,  # Updated by PerformanceMonitor
            cache_hit_ratio=0,  # Updated by PerformanceMonitor
            error_rate=0  # Updated by PerformanceMonitor
        )
    
    def get_recent_snapshots(self, minutes: int = 60) -> List[PerformanceSnapshot]:
        """Get recent performance snapshots"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [s for s in self.snapshots if s.timestamp >= cutoff]

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system that tracks query performance,
    resource usage, and provides real-time analytics and alerting.
    """
    
    def __init__(self, db_path: str = "data.db", activity_log_path: str = "/data/activity.log"):
        self.db_path = db_path
        self.activity_log_path = activity_log_path
        
        # Initialize components
        self.resource_monitor = ResourceMonitor()
        self.query_metrics = deque(maxlen=10000)  # Keep last 10k queries
        self.performance_alerts = []
        self.baselines = {}
        
        # Performance tracking
        self.query_count = 0
        self.error_count = 0
        self.start_time = time.time()
        self.last_metrics_update = time.time()
        
        # Alert thresholds (configurable)
        self.alert_thresholds = {
            'slow_query': 5.0,  # seconds
            'high_cpu': 80.0,   # percent
            'low_memory': 100.0,  # MB available
            'high_error_rate': 0.05,  # 5%
            'low_cache_hit_ratio': 0.7  # 70%
        }
        
        # Baseline calculation settings
        self.baseline_window = 60  # minutes
        self.baseline_confidence = 0.95
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        
        self._log_activity("PerformanceMonitor initialized", {
            "db_path": db_path,
            "alert_thresholds": self.alert_thresholds,
            "baseline_window": self.baseline_window
        })
    
    def _log_activity(self, activity: str, details: Dict[str, Any] = None):
        """Log activity to the activity log file"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "module": "PerformanceMonitor",
                "activity": activity,
                "details": details or {}
            }
            
            Path(self.activity_log_path).parent.mkdir(exist_ok=True)
            with open(self.activity_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to log activity: {e}")
    
    def track_query_performance(self, query: str, execution_time: float,
                              rows_processed: int = 0, cache_hits: int = 0,
                              cache_misses: int = 0) -> QueryPerformanceMetric:
        """Track individual query performance"""
        query_id = f"query_{int(time.time() * 1000000)}"  # Microsecond timestamp
        
        # Normalize query for pattern recognition
        query_pattern = self._normalize_query_pattern(query)
        
        # Calculate memory usage (simplified)
        estimated_memory = rows_processed * 50  # bytes per row estimate
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(
            execution_time, rows_processed, cache_hits, cache_misses
        )
        
        metric = QueryPerformanceMetric(
            query_id=query_id,
            query_pattern=query_pattern,
            execution_time=execution_time,
            cpu_time=0,  # Would need profiling integration
            memory_used=estimated_memory,
            rows_processed=rows_processed,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            timestamp=datetime.now(),
            optimization_score=optimization_score
        )
        
        self.query_metrics.append(metric)
        self.query_count += 1
        
        # Check for performance alerts
        self._check_query_alerts(metric)
        
        # Update real-time metrics
        self._update_realtime_metrics()
        
        return metric
    
    def _normalize_query_pattern(self, query: str) -> str:
        """Normalize query to identify patterns"""
        import re
        
        # Remove extra whitespace and normalize case
        normalized = re.sub(r'\s+', ' ', query.upper().strip())
        
        # Replace literals with placeholders
        normalized = re.sub(r"'[^']*'", "'?'", normalized)
        normalized = re.sub(r'\b\d+\b', '?', normalized)
        
        # Extract main operation
        if normalized.startswith('SELECT'):
            # Extract table name
            table_match = re.search(r'FROM\s+(\w+)', normalized)
            table = table_match.group(1) if table_match else 'unknown'
            return f"SELECT FROM {table}"
        elif normalized.startswith('INSERT'):
            table_match = re.search(r'INSERT\s+INTO\s+(\w+)', normalized)
            table = table_match.group(1) if table_match else 'unknown'
            return f"INSERT INTO {table}"
        elif normalized.startswith('UPDATE'):
            table_match = re.search(r'UPDATE\s+(\w+)', normalized)
            table = table_match.group(1) if table_match else 'unknown'
            return f"UPDATE {table}"
        elif normalized.startswith('DELETE'):
            table_match = re.search(r'DELETE\s+FROM\s+(\w+)', normalized)
            table = table_match.group(1) if table_match else 'unknown'
            return f"DELETE FROM {table}"
        else:
            return normalized[:50]  # First 50 chars for other queries
    
    def _calculate_optimization_score(self, execution_time: float,
                                    rows_processed: int, cache_hits: int,
                                    cache_misses: int) -> float:
        """Calculate query optimization score (0.0 to 1.0, higher is better)"""
        score = 1.0
        
        # Penalize slow execution
        if execution_time > 1.0:
            score -= min(0.4, execution_time / 10.0)
        
        # Penalize inefficient row processing
        if rows_processed > 0 and execution_time > 0:
            rows_per_second = rows_processed / execution_time
            if rows_per_second < 1000:  # Less than 1000 rows/sec
                score -= 0.2
        
        # Reward good cache usage
        total_cache_ops = cache_hits + cache_misses
        if total_cache_ops > 0:
            cache_ratio = cache_hits / total_cache_ops
            score += 0.3 * cache_ratio - 0.15  # Bonus for high cache hit ratio
        
        return max(0.0, min(1.0, score))
    
    def _check_query_alerts(self, metric: QueryPerformanceMetric):
        """Check if query performance triggers any alerts"""
        alerts = []
        
        # Slow query alert
        if metric.execution_time > self.alert_thresholds['slow_query']:
            alert = PerformanceAlert(
                alert_id=f"slow_query_{metric.query_id}",
                severity='warning',
                metric_name='query_execution_time',
                threshold=self.alert_thresholds['slow_query'],
                current_value=metric.execution_time,
                message=f"Slow query detected: {metric.query_pattern} took {metric.execution_time:.2f}s",
                timestamp=datetime.now()
            )
            alerts.append(alert)
        
        # Poor optimization alert
        if metric.optimization_score < 0.3:
            alert = PerformanceAlert(
                alert_id=f"poor_optimization_{metric.query_id}",
                severity='info',
                metric_name='optimization_score',
                threshold=0.3,
                current_value=metric.optimization_score,
                message=f"Poorly optimized query: {metric.query_pattern} (score: {metric.optimization_score:.2f})",
                timestamp=datetime.now()
            )
            alerts.append(alert)
        
        self.performance_alerts.extend(alerts)
        
        if alerts:
            self._log_activity("Performance alerts triggered", {
                "alerts_count": len(alerts),
                "query_pattern": metric.query_pattern,
                "execution_time": metric.execution_time
            })
    
    def _update_realtime_metrics(self):
        """Update real-time performance metrics"""
        current_time = time.time()
        
        # Update queries per second (every 10 seconds)
        if current_time - self.last_metrics_update >= 10:
            time_window = current_time - self.last_metrics_update
            recent_queries = len([m for m in self.query_metrics 
                                if (current_time - m.timestamp.timestamp()) <= time_window])
            
            queries_per_second = recent_queries / time_window
            
            # Update latest resource snapshot
            if self.resource_monitor.snapshots:
                latest_snapshot = self.resource_monitor.snapshots[-1]
                latest_snapshot.queries_per_second = queries_per_second
                
                # Calculate average query time for recent queries
                recent_metrics = [m for m in self.query_metrics 
                                if (current_time - m.timestamp.timestamp()) <= 60]  # Last minute
                if recent_metrics:
                    latest_snapshot.avg_query_time = statistics.mean(
                        [m.execution_time for m in recent_metrics]
                    )
                
                # Calculate cache hit ratio
                if recent_metrics:
                    total_hits = sum(m.cache_hits for m in recent_metrics)
                    total_misses = sum(m.cache_misses for m in recent_metrics)
                    total_cache_ops = total_hits + total_misses
                    
                    latest_snapshot.cache_hit_ratio = (
                        total_hits / total_cache_ops if total_cache_ops > 0 else 0
                    )
                
                # Calculate error rate
                runtime = current_time - self.start_time
                latest_snapshot.error_rate = (
                    self.error_count / self.query_count if self.query_count > 0 else 0
                )
            
            self.last_metrics_update = current_time
    
    def record_error(self, error_type: str, error_message: str, query: str = None):
        """Record query or system error"""
        self.error_count += 1
        
        self._log_activity("Error recorded", {
            "error_type": error_type,
            "error_message": error_message,
            "query": query[:100] if query else None,
            "total_errors": self.error_count
        })
        
        # Check error rate alert
        error_rate = self.error_count / max(1, self.query_count)
        if error_rate > self.alert_thresholds['high_error_rate']:
            alert = PerformanceAlert(
                alert_id=f"high_error_rate_{int(time.time())}",
                severity='critical',
                metric_name='error_rate',
                threshold=self.alert_thresholds['high_error_rate'],
                current_value=error_rate,
                message=f"High error rate detected: {error_rate:.3f} ({self.error_count}/{self.query_count})",
                timestamp=datetime.now()
            )
            self.performance_alerts.append(alert)
    
    def establish_performance_baselines(self):
        """Establish performance baselines based on historical data"""
        if len(self.query_metrics) < 100:  # Need sufficient data
            return
        
        # Calculate baselines for key metrics
        recent_metrics = list(self.query_metrics)[-1000:]  # Last 1000 queries
        
        # Query execution time baseline
        execution_times = [m.execution_time for m in recent_metrics]
        baseline_exec_time = statistics.median(execution_times)
        exec_time_std = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        
        self.baselines['execution_time'] = PerformanceBaseline(
            metric_name='execution_time',
            baseline_value=baseline_exec_time,
            acceptable_range=(
                max(0, baseline_exec_time - 2 * exec_time_std),
                baseline_exec_time + 3 * exec_time_std
            ),
            measurement_window=self.baseline_window,
            confidence_level=self.baseline_confidence,
            last_updated=datetime.now()
        )
        
        # Optimization score baseline
        opt_scores = [m.optimization_score for m in recent_metrics]
        baseline_opt_score = statistics.median(opt_scores)
        
        self.baselines['optimization_score'] = PerformanceBaseline(
            metric_name='optimization_score',
            baseline_value=baseline_opt_score,
            acceptable_range=(max(0, baseline_opt_score - 0.2), 1.0),
            measurement_window=self.baseline_window,
            confidence_level=self.baseline_confidence,
            last_updated=datetime.now()
        )
        
        # System resource baselines
        snapshots = self.resource_monitor.get_recent_snapshots(self.baseline_window)
        if snapshots:
            cpu_values = [s.cpu_usage for s in snapshots]
            memory_values = [s.memory_usage for s in snapshots]
            
            self.baselines['cpu_usage'] = PerformanceBaseline(
                metric_name='cpu_usage',
                baseline_value=statistics.median(cpu_values),
                acceptable_range=(0, self.alert_thresholds['high_cpu']),
                measurement_window=self.baseline_window,
                confidence_level=self.baseline_confidence,
                last_updated=datetime.now()
            )
            
            self.baselines['memory_usage'] = PerformanceBaseline(
                metric_name='memory_usage',
                baseline_value=statistics.median(memory_values),
                acceptable_range=(0, 1000),  # 1GB max
                measurement_window=self.baseline_window,
                confidence_level=self.baseline_confidence,
                last_updated=datetime.now()
            )
        
        self._log_activity("Performance baselines established", {
            "baselines_count": len(self.baselines),
            "baseline_names": list(self.baselines.keys())
        })
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        current_time = datetime.now()
        
        # Recent performance metrics
        recent_queries = [m for m in self.query_metrics 
                         if (current_time - m.timestamp).total_seconds() <= 3600]  # Last hour
        
        # Resource snapshots
        recent_snapshots = self.resource_monitor.get_recent_snapshots(60)  # Last hour
        
        # Active alerts
        active_alerts = [a for a in self.performance_alerts if not a.resolved]
        
        # Performance summary
        dashboard = {
            'timestamp': current_time.isoformat(),
            'summary': {
                'total_queries': self.query_count,
                'total_errors': self.error_count,
                'uptime_seconds': int(time.time() - self.start_time),
                'queries_last_hour': len(recent_queries),
                'active_alerts': len(active_alerts),
                'error_rate': self.error_count / max(1, self.query_count)
            },
            'query_performance': {
                'avg_execution_time': statistics.mean([m.execution_time for m in recent_queries]) if recent_queries else 0,
                'median_execution_time': statistics.median([m.execution_time for m in recent_queries]) if recent_queries else 0,
                'slowest_queries': sorted(recent_queries, key=lambda m: m.execution_time, reverse=True)[:5],
                'avg_optimization_score': statistics.mean([m.optimization_score for m in recent_queries]) if recent_queries else 0
            },
            'system_performance': {
                'current_cpu': recent_snapshots[-1].cpu_usage if recent_snapshots else 0,
                'current_memory': recent_snapshots[-1].memory_usage if recent_snapshots else 0,
                'avg_cpu_last_hour': statistics.mean([s.cpu_usage for s in recent_snapshots]) if recent_snapshots else 0,
                'avg_memory_last_hour': statistics.mean([s.memory_usage for s in recent_snapshots]) if recent_snapshots else 0,
                'queries_per_second': recent_snapshots[-1].queries_per_second if recent_snapshots else 0,
                'cache_hit_ratio': recent_snapshots[-1].cache_hit_ratio if recent_snapshots else 0
            },
            'alerts': [asdict(alert) for alert in active_alerts],
            'baselines': {name: asdict(baseline) for name, baseline in self.baselines.items()},
            'trends': self._calculate_performance_trends(recent_snapshots, recent_queries)
        }
        
        return dashboard
    
    def _calculate_performance_trends(self, snapshots: List[PerformanceSnapshot],
                                    queries: List[QueryPerformanceMetric]) -> Dict[str, str]:
        """Calculate performance trends (improving, degrading, stable)"""
        if len(snapshots) < 2 or len(queries) < 10:
            return {'overall': 'insufficient_data'}
        
        trends = {}
        
        # CPU trend
        cpu_values = [s.cpu_usage for s in snapshots[-30:]]  # Last 30 snapshots
        if len(cpu_values) >= 10:
            first_half = statistics.mean(cpu_values[:len(cpu_values)//2])
            second_half = statistics.mean(cpu_values[len(cpu_values)//2:])
            
            if second_half > first_half * 1.1:
                trends['cpu'] = 'degrading'
            elif second_half < first_half * 0.9:
                trends['cpu'] = 'improving'
            else:
                trends['cpu'] = 'stable'
        
        # Query performance trend
        exec_times = [q.execution_time for q in queries[-100:]]  # Last 100 queries
        if len(exec_times) >= 20:
            first_half = statistics.mean(exec_times[:len(exec_times)//2])
            second_half = statistics.mean(exec_times[len(exec_times)//2:])
            
            if second_half > first_half * 1.2:
                trends['query_performance'] = 'degrading'
            elif second_half < first_half * 0.8:
                trends['query_performance'] = 'improving'
            else:
                trends['query_performance'] = 'stable'
        
        # Overall trend
        degrading_metrics = sum(1 for trend in trends.values() if trend == 'degrading')
        improving_metrics = sum(1 for trend in trends.values() if trend == 'improving')
        
        if degrading_metrics > improving_metrics:
            trends['overall'] = 'degrading'
        elif improving_metrics > degrading_metrics:
            trends['overall'] = 'improving'
        else:
            trends['overall'] = 'stable'
        
        return trends
    
    def export_performance_report(self, filepath: str = None,
                                include_raw_data: bool = False) -> Dict[str, Any]:
        """Export comprehensive performance report"""
        if not filepath:
            filepath = f"/data/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        dashboard = self.get_performance_dashboard()
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_period': '24_hours',
                'include_raw_data': include_raw_data
            },
            'performance_dashboard': dashboard
        }
        
        if include_raw_data:
            report['raw_data'] = {
                'query_metrics': [asdict(m) for m in self.query_metrics],
                'resource_snapshots': [asdict(s) for s in self.resource_monitor.snapshots],
                'all_alerts': [asdict(a) for a in self.performance_alerts]
            }
        
        try:
            Path(filepath).parent.mkdir(exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            self._log_activity("Performance report exported", {
                "filepath": filepath,
                "include_raw_data": include_raw_data
            })
            
            return {"success": True, "filepath": filepath, "report": report}
        
        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
            return {"success": False, "error": str(e), "report": report}
    
    def cleanup_old_data(self, days_to_keep: int = 7):
        """Clean up old performance data to manage memory usage"""
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean query metrics
        original_count = len(self.query_metrics)
        self.query_metrics = deque(
            [m for m in self.query_metrics if m.timestamp >= cutoff_time],
            maxlen=self.query_metrics.maxlen
        )
        
        # Clean alerts
        original_alerts = len(self.performance_alerts)
        self.performance_alerts = [
            a for a in self.performance_alerts 
            if a.timestamp >= cutoff_time or not a.resolved
        ]
        
        cleaned_queries = original_count - len(self.query_metrics)
        cleaned_alerts = original_alerts - len(self.performance_alerts)
        
        self._log_activity("Old performance data cleaned", {
            "queries_cleaned": cleaned_queries,
            "alerts_cleaned": cleaned_alerts,
            "days_kept": days_to_keep
        })
    
    def shutdown(self):
        """Shutdown performance monitoring"""
        self.resource_monitor.stop_monitoring()
        self._log_activity("PerformanceMonitor shutdown", {
            "total_queries_tracked": self.query_count,
            "total_errors": self.error_count,
            "uptime_seconds": int(time.time() - self.start_time)
        })

# Global instance for easy integration
performance_monitor = None

def get_performance_monitor(db_path: str = None) -> PerformanceMonitor:
    """Get or create global performance monitor instance"""
    global performance_monitor
    if performance_monitor is None or (db_path and performance_monitor.db_path != db_path):
        performance_monitor = PerformanceMonitor(db_path or "data.db")
    return performance_monitor

def track_query(query: str, execution_time: float, **kwargs):
    """Convenience function to track query performance"""
    monitor = get_performance_monitor()
    return monitor.track_query_performance(query, execution_time, **kwargs)

def record_error(error_type: str, error_message: str, query: str = None):
    """Convenience function to record errors"""
    monitor = get_performance_monitor()
    return monitor.record_error(error_type, error_message, query)

if __name__ == "__main__":
    # Example usage and testing
    monitor = PerformanceMonitor("test.db")
    
    # Simulate some queries
    import random
    for i in range(100):
        query = f"SELECT * FROM test_table WHERE id = {i}"
        exec_time = random.uniform(0.1, 3.0)
        rows = random.randint(1, 1000)
        cache_hits = random.randint(0, 10)
        cache_misses = random.randint(0, 5)
        
        monitor.track_query_performance(query, exec_time, rows, cache_hits, cache_misses)
        
        if random.random() < 0.05:  # 5% error rate
            monitor.record_error("SQL_ERROR", "Simulated error", query)
        
        time.sleep(0.1)
    
    # Establish baselines
    monitor.establish_performance_baselines()
    
    # Get dashboard
    dashboard = monitor.get_performance_dashboard()
    print(f"Dashboard generated with {len(dashboard['alerts'])} active alerts")
    
    # Export report
    report = monitor.export_performance_report()
    print(f"Report exported: {report.get('filepath', 'Failed')}")
    
    # Cleanup
    monitor.cleanup_old_data(days_to_keep=1)
    monitor.shutdown()