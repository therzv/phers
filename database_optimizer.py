"""
Database Optimizer Module for PHERS
Phase 4: Performance Optimization

This module provides intelligent database optimization capabilities including:
- Dynamic index creation based on query patterns
- Query execution plan analysis
- Performance bottleneck identification
- Automatic optimization recommendations
- Index usage monitoring and maintenance
"""

import sqlite3
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
import threading
from pathlib import Path

@dataclass
class QueryProfile:
    """Represents the performance profile of a query"""
    query_hash: str
    execution_time: float
    rows_examined: int
    rows_returned: int
    index_usage: List[str]
    table_scans: int
    sort_operations: int
    join_operations: int
    complexity_score: float
    frequency: int
    last_executed: datetime
    optimization_applied: bool = False

@dataclass
class IndexRecommendation:
    """Represents an index optimization recommendation"""
    table_name: str
    columns: List[str]
    index_type: str  # 'single', 'composite', 'covering'
    estimated_benefit: float
    query_patterns: List[str]
    creation_cost: float
    maintenance_overhead: float
    priority: str  # 'high', 'medium', 'low'
    reasoning: str

@dataclass
class PerformanceMetrics:
    """Performance metrics for database operations"""
    avg_query_time: float
    slowest_queries: List[QueryProfile]
    most_frequent_queries: List[QueryProfile]
    index_hit_ratio: float
    table_scan_ratio: float
    memory_usage: int
    disk_io: int
    optimization_opportunities: int

class DatabaseOptimizer:
    """
    Intelligent database optimizer that analyzes query patterns and provides
    automatic optimization recommendations and implementations.
    """
    
    def __init__(self, db_path: str, activity_log_path: str = "/data/activity.log"):
        self.db_path = db_path
        self.activity_log_path = activity_log_path
        self.query_profiles: Dict[str, QueryProfile] = {}
        self.index_recommendations: List[IndexRecommendation] = []
        self.performance_history: List[PerformanceMetrics] = []
        
        # Optimization thresholds
        self.slow_query_threshold = 1.0  # seconds
        self.high_frequency_threshold = 10  # executions
        self.table_scan_threshold = 0.3  # 30% table scan ratio
        
        # Index patterns for automatic detection
        self.index_patterns = {
            'equality_filter': r'WHERE\s+(\w+)\s*=',
            'range_filter': r'WHERE\s+(\w+)\s*[><]=?',
            'order_by': r'ORDER\s+BY\s+(\w+)',
            'group_by': r'GROUP\s+BY\s+(\w+)',
            'join_condition': r'JOIN\s+\w+\s+ON\s+\w+\.(\w+)\s*=\s*\w+\.(\w+)'
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._log_activity("DatabaseOptimizer initialized", {
            "db_path": db_path,
            "thresholds": {
                "slow_query": self.slow_query_threshold,
                "high_frequency": self.high_frequency_threshold,
                "table_scan": self.table_scan_threshold
            }
        })
    
    def _log_activity(self, activity: str, details: Dict[str, Any] = None):
        """Log activity to the activity log file"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "module": "DatabaseOptimizer",
                "activity": activity,
                "details": details or {}
            }
            
            Path(self.activity_log_path).parent.mkdir(exist_ok=True)
            with open(self.activity_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to log activity: {e}")
    
    def _generate_query_hash(self, query: str) -> str:
        """Generate a hash for query pattern recognition"""
        # Normalize query for pattern matching
        normalized = re.sub(r'\s+', ' ', query.upper().strip())
        # Replace specific values with placeholders
        normalized = re.sub(r"'[^']*'", "'?'", normalized)
        normalized = re.sub(r'\b\d+\b', '?', normalized)
        return str(hash(normalized))
    
    def analyze_query_execution(self, query: str, execution_time: float, 
                              result_count: int) -> QueryProfile:
        """Analyze a query execution and create/update its profile"""
        query_hash = self._generate_query_hash(query)
        
        # Get execution plan
        execution_plan = self._get_execution_plan(query)
        
        # Extract metrics from execution plan
        rows_examined = execution_plan.get('rows_examined', result_count)
        index_usage = execution_plan.get('index_usage', [])
        table_scans = execution_plan.get('table_scans', 0)
        sort_operations = execution_plan.get('sort_operations', 0)
        join_operations = execution_plan.get('join_operations', 0)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(
            query, rows_examined, table_scans, sort_operations, join_operations
        )
        
        # Update or create query profile
        if query_hash in self.query_profiles:
            profile = self.query_profiles[query_hash]
            profile.frequency += 1
            profile.execution_time = (profile.execution_time + execution_time) / 2
            profile.last_executed = datetime.now()
        else:
            profile = QueryProfile(
                query_hash=query_hash,
                execution_time=execution_time,
                rows_examined=rows_examined,
                rows_returned=result_count,
                index_usage=index_usage,
                table_scans=table_scans,
                sort_operations=sort_operations,
                join_operations=join_operations,
                complexity_score=complexity_score,
                frequency=1,
                last_executed=datetime.now()
            )
            self.query_profiles[query_hash] = profile
        
        self._log_activity("Query execution analyzed", {
            "query_hash": query_hash,
            "execution_time": execution_time,
            "complexity_score": complexity_score,
            "frequency": profile.frequency
        })
        
        return profile
    
    def _get_execution_plan(self, query: str) -> Dict[str, Any]:
        """Get query execution plan from SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Use EXPLAIN QUERY PLAN to get execution details
                explain_query = f"EXPLAIN QUERY PLAN {query}"
                cursor.execute(explain_query)
                plan_rows = cursor.fetchall()
                
                # Parse execution plan
                execution_plan = {
                    'rows_examined': 0,
                    'index_usage': [],
                    'table_scans': 0,
                    'sort_operations': 0,
                    'join_operations': 0
                }
                
                for row in plan_rows:
                    plan_detail = row[3] if len(row) > 3 else str(row)
                    
                    if 'SCAN' in plan_detail:
                        execution_plan['table_scans'] += 1
                    elif 'INDEX' in plan_detail:
                        # Extract index name
                        index_match = re.search(r'INDEX\s+(\w+)', plan_detail)
                        if index_match:
                            execution_plan['index_usage'].append(index_match.group(1))
                    
                    if 'ORDER BY' in plan_detail or 'SORT' in plan_detail:
                        execution_plan['sort_operations'] += 1
                    
                    if 'JOIN' in plan_detail:
                        execution_plan['join_operations'] += 1
                
                return execution_plan
        
        except Exception as e:
            self.logger.warning(f"Failed to get execution plan: {e}")
            return {}
    
    def _calculate_complexity_score(self, query: str, rows_examined: int,
                                  table_scans: int, sort_operations: int,
                                  join_operations: int) -> float:
        """Calculate query complexity score (0.0 to 1.0)"""
        score = 0.0
        
        # Base complexity from SQL operations
        if 'JOIN' in query.upper():
            score += 0.3
        if 'ORDER BY' in query.upper():
            score += 0.2
        if 'GROUP BY' in query.upper():
            score += 0.2
        if 'HAVING' in query.upper():
            score += 0.1
        
        # Adjust for execution characteristics
        if table_scans > 0:
            score += 0.3 * table_scans
        if sort_operations > 0:
            score += 0.2 * sort_operations
        if join_operations > 1:
            score += 0.1 * (join_operations - 1)
        
        # Factor in data volume
        if rows_examined > 1000:
            score += 0.1
        if rows_examined > 10000:
            score += 0.1
        
        return min(score, 1.0)
    
    def generate_index_recommendations(self) -> List[IndexRecommendation]:
        """Generate intelligent index recommendations based on query patterns"""
        recommendations = []
        
        # Analyze query patterns for index opportunities
        column_usage = defaultdict(list)
        join_patterns = defaultdict(int)
        filter_patterns = defaultdict(int)
        sort_patterns = defaultdict(int)
        
        for query_hash, profile in self.query_profiles.items():
            if profile.frequency < 3:  # Skip infrequent queries
                continue
            
            # Reconstruct query patterns (simplified)
            for pattern_name, pattern_regex in self.index_patterns.items():
                # This would need actual query text, simplified for demonstration
                pass
        
        # Get actual table schema for recommendations
        tables_info = self._get_tables_info()
        
        for table_name, columns in tables_info.items():
            # Analyze slow queries on this table
            slow_queries = [p for p in self.query_profiles.values() 
                          if p.execution_time > self.slow_query_threshold 
                          and p.table_scans > 0]
            
            if slow_queries:
                # Recommend indexes for commonly filtered columns
                for column in columns:
                    if self._should_recommend_index(table_name, column):
                        recommendation = IndexRecommendation(
                            table_name=table_name,
                            columns=[column],
                            index_type='single',
                            estimated_benefit=self._estimate_index_benefit(
                                table_name, [column]
                            ),
                            query_patterns=[],
                            creation_cost=self._estimate_creation_cost(
                                table_name, [column]
                            ),
                            maintenance_overhead=0.05,
                            priority=self._calculate_priority(
                                table_name, [column]
                            ),
                            reasoning=f"Frequent filtering/sorting on {column}"
                        )
                        recommendations.append(recommendation)
        
        # Sort by estimated benefit
        recommendations.sort(key=lambda r: r.estimated_benefit, reverse=True)
        
        self.index_recommendations = recommendations
        self._log_activity("Index recommendations generated", {
            "recommendations_count": len(recommendations),
            "high_priority": len([r for r in recommendations if r.priority == 'high'])
        })
        
        return recommendations
    
    def _get_tables_info(self) -> Dict[str, List[str]]:
        """Get information about tables and their columns"""
        tables_info = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [col[1] for col in cursor.fetchall()]
                    tables_info[table_name] = columns
        
        except Exception as e:
            self.logger.error(f"Failed to get table info: {e}")
        
        return tables_info
    
    def _should_recommend_index(self, table_name: str, column_name: str) -> bool:
        """Determine if an index should be recommended for a column"""
        # Check if index already exists
        existing_indexes = self._get_existing_indexes(table_name)
        
        for index in existing_indexes:
            if column_name in index['columns']:
                return False
        
        # Analyze column usage patterns
        # This is simplified - would need actual query analysis
        return True
    
    def _get_existing_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """Get existing indexes for a table"""
        indexes = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA index_list({table_name})")
                
                for index_info in cursor.fetchall():
                    index_name = index_info[1]
                    cursor.execute(f"PRAGMA index_info({index_name})")
                    columns = [col[2] for col in cursor.fetchall()]
                    
                    indexes.append({
                        'name': index_name,
                        'columns': columns,
                        'unique': bool(index_info[2])
                    })
        
        except Exception as e:
            self.logger.error(f"Failed to get existing indexes: {e}")
        
        return indexes
    
    def _estimate_index_benefit(self, table_name: str, columns: List[str]) -> float:
        """Estimate the performance benefit of creating an index"""
        # Simplified benefit estimation
        # Would need actual row counts and query frequency analysis
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
        except:
            row_count = 1000  # Default assumption
        
        # Benefit is higher for larger tables and frequently queried columns
        base_benefit = min(row_count / 1000.0, 10.0)  # Cap at 10x
        
        # Adjust based on query patterns (simplified)
        frequent_queries = len([p for p in self.query_profiles.values() 
                              if p.frequency > self.high_frequency_threshold])
        
        return base_benefit * (1 + frequent_queries * 0.1)
    
    def _estimate_creation_cost(self, table_name: str, columns: List[str]) -> float:
        """Estimate the cost of creating an index"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
        except:
            row_count = 1000
        
        # Cost increases with table size and number of columns
        base_cost = row_count / 10000.0  # Normalized cost
        column_factor = 1 + (len(columns) - 1) * 0.3
        
        return base_cost * column_factor
    
    def _calculate_priority(self, table_name: str, columns: List[str]) -> str:
        """Calculate recommendation priority"""
        benefit = self._estimate_index_benefit(table_name, columns)
        cost = self._estimate_creation_cost(table_name, columns)
        
        if benefit / cost > 5.0:
            return 'high'
        elif benefit / cost > 2.0:
            return 'medium'
        else:
            return 'low'
    
    def apply_optimization_recommendations(self, max_recommendations: int = 5) -> Dict[str, Any]:
        """Apply the top optimization recommendations"""
        if not self.index_recommendations:
            self.generate_index_recommendations()
        
        applied_recommendations = []
        failed_recommendations = []
        
        high_priority_recs = [r for r in self.index_recommendations 
                             if r.priority == 'high'][:max_recommendations]
        
        for recommendation in high_priority_recs:
            try:
                success = self._create_index(recommendation)
                if success:
                    applied_recommendations.append(recommendation)
                else:
                    failed_recommendations.append(recommendation)
            except Exception as e:
                self.logger.error(f"Failed to apply recommendation: {e}")
                failed_recommendations.append(recommendation)
        
        result = {
            'applied': len(applied_recommendations),
            'failed': len(failed_recommendations),
            'recommendations': applied_recommendations,
            'errors': failed_recommendations
        }
        
        self._log_activity("Optimization recommendations applied", result)
        
        return result
    
    def _create_index(self, recommendation: IndexRecommendation) -> bool:
        """Create an index based on recommendation"""
        try:
            index_name = f"idx_{recommendation.table_name}_{'_'.join(recommendation.columns)}"
            columns_str = ', '.join(recommendation.columns)
            
            create_sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {recommendation.table_name} ({columns_str})"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(create_sql)
                conn.commit()
            
            self.logger.info(f"Created index: {index_name}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to create index: {e}")
            return False
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        if not self.query_profiles:
            return PerformanceMetrics(0, [], [], 0, 0, 0, 0, 0)
        
        # Calculate metrics
        avg_query_time = sum(p.execution_time for p in self.query_profiles.values()) / len(self.query_profiles)
        
        slowest_queries = sorted(
            self.query_profiles.values(),
            key=lambda p: p.execution_time,
            reverse=True
        )[:10]
        
        most_frequent_queries = sorted(
            self.query_profiles.values(),
            key=lambda p: p.frequency,
            reverse=True
        )[:10]
        
        # Calculate ratios
        total_queries = len(self.query_profiles)
        queries_with_scans = len([p for p in self.query_profiles.values() if p.table_scans > 0])
        queries_with_indexes = len([p for p in self.query_profiles.values() if p.index_usage])
        
        table_scan_ratio = queries_with_scans / total_queries if total_queries > 0 else 0
        index_hit_ratio = queries_with_indexes / total_queries if total_queries > 0 else 0
        
        optimization_opportunities = len([p for p in self.query_profiles.values() 
                                        if p.execution_time > self.slow_query_threshold 
                                        and not p.optimization_applied])
        
        metrics = PerformanceMetrics(
            avg_query_time=avg_query_time,
            slowest_queries=slowest_queries,
            most_frequent_queries=most_frequent_queries,
            index_hit_ratio=index_hit_ratio,
            table_scan_ratio=table_scan_ratio,
            memory_usage=0,  # Would need system-level monitoring
            disk_io=0,       # Would need system-level monitoring
            optimization_opportunities=optimization_opportunities
        )
        
        self.performance_history.append(metrics)
        self._log_activity("Performance metrics calculated", {
            "avg_query_time": avg_query_time,
            "slowest_query_count": len(slowest_queries),
            "table_scan_ratio": table_scan_ratio,
            "optimization_opportunities": optimization_opportunities
        })
        
        return metrics
    
    def export_optimization_report(self, filepath: str = None) -> Dict[str, Any]:
        """Export comprehensive optimization report"""
        if not filepath:
            filepath = f"/data/optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'database_path': self.db_path,
            'performance_metrics': asdict(self.get_performance_metrics()),
            'query_profiles': [asdict(p) for p in self.query_profiles.values()],
            'index_recommendations': [asdict(r) for r in self.index_recommendations],
            'optimization_summary': {
                'total_queries_analyzed': len(self.query_profiles),
                'slow_queries': len([p for p in self.query_profiles.values() 
                                   if p.execution_time > self.slow_query_threshold]),
                'high_priority_recommendations': len([r for r in self.index_recommendations 
                                                    if r.priority == 'high']),
                'potential_improvements': len(self.index_recommendations)
            }
        }
        
        try:
            Path(filepath).parent.mkdir(exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            self._log_activity("Optimization report exported", {"filepath": filepath})
            return {"success": True, "filepath": filepath, "report": report}
        
        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
            return {"success": False, "error": str(e), "report": report}

# Global instance for easy integration
database_optimizer = None

def get_database_optimizer(db_path: str = None) -> DatabaseOptimizer:
    """Get or create global database optimizer instance"""
    global database_optimizer
    if database_optimizer is None or (db_path and database_optimizer.db_path != db_path):
        database_optimizer = DatabaseOptimizer(db_path or "data.db")
    return database_optimizer

def analyze_query_performance(query: str, execution_time: float, result_count: int, db_path: str = None):
    """Convenience function to analyze query performance"""
    optimizer = get_database_optimizer(db_path)
    return optimizer.analyze_query_execution(query, execution_time, result_count)

def get_optimization_recommendations(db_path: str = None) -> List[IndexRecommendation]:
    """Convenience function to get optimization recommendations"""
    optimizer = get_database_optimizer(db_path)
    return optimizer.generate_index_recommendations()

if __name__ == "__main__":
    # Example usage and testing
    optimizer = DatabaseOptimizer("test.db")
    
    # Simulate some query executions
    test_queries = [
        ("SELECT * FROM users WHERE email = 'test@example.com'", 0.5, 1),
        ("SELECT * FROM orders WHERE user_id = 123 ORDER BY created_date", 1.2, 25),
        ("SELECT COUNT(*) FROM products WHERE category = 'electronics'", 2.1, 1),
    ]
    
    for query, exec_time, result_count in test_queries:
        optimizer.analyze_query_execution(query, exec_time, result_count)
    
    # Generate recommendations
    recommendations = optimizer.generate_index_recommendations()
    print(f"Generated {len(recommendations)} recommendations")
    
    # Get performance metrics
    metrics = optimizer.get_performance_metrics()
    print(f"Average query time: {metrics.avg_query_time:.2f}s")
    
    # Export report
    report = optimizer.export_optimization_report()
    print(f"Report exported: {report.get('filepath', 'Failed')}")